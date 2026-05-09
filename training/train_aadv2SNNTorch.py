# training/train_aadv2.py
"""
Training script for AdaptiveAxonalDelayNetV2 — snnTorch version.
Faithful reproduction of Sun et al. ICASSP 2023.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from configs import get_config
from utils.seed import set_seed
from experiments.results_logger import ResultsLogger
from adaptiveAxonalDelayV2SNNTorch import AdaptiveAxonalDelayNetV2SNNTorch


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SHDDataset(Dataset):
    def __init__(self, path, split="train",
                 time_steps=400, n_inputs=700, max_time=1.4):
        self.time_steps = time_steps
        self.n_inputs   = n_inputs
        self.max_time   = max_time
        self.hdf5_path  = os.path.join(path, f"shd_{split}.h5")
        print(f"  Loading SHD {split} from {self.hdf5_path}")
        with h5py.File(self.hdf5_path, "r") as f:
            self.labels    = f["labels"][:]
            self.n_samples = len(self.labels)
        print(f"  {split}: {self.n_samples} samples, "
              f"{len(set(self.labels))} classes")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            times = f["spikes"]["times"][idx]
            units = f["spikes"]["units"][idx]
        label = int(self.labels[idx])
        t_idx = np.floor(
            times / self.max_time * self.time_steps).astype(int)
        t_idx = np.clip(t_idx, 0, self.time_steps - 1)
        u_idx = np.clip(units.astype(int), 0, self.n_inputs - 1)
        spikes = torch.zeros(self.time_steps, self.n_inputs,
                             dtype=torch.float32)
        spikes[t_idx, u_idx] = 1.0
        return spikes, label


def get_loaders(cfg):
    return (
        DataLoader(
            SHDDataset(cfg.datasets_path, "train",
                       cfg.time_steps, cfg.n_inputs),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
        ),
        DataLoader(
            SHDDataset(cfg.datasets_path, "test",
                       cfg.time_steps, cfg.n_inputs),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────


def get_peak_gpu_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2) \
        if torch.cuda.is_available() else 0.0

def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# DELAY SCHEDULER — matches Pengfei exactly
# ─────────────────────────────────────────────────────────────────────────────

class DelayScheduler:
    def __init__(self, cfg):
        self.thea1         = float(cfg.delay_init_cap)
        self.thea2         = float(cfg.delay_init_cap)
        self.adapt_start   = cfg.delay_adapt_start
        self.adapt_window  = cfg.delay_adapt_window
        self.expand_margin = cfg.delay_expand_margin
        self.update1       = 0
        self.update2       = 0

    def step(self, model, epoch):
        if epoch <= self.adapt_start:
            model.clamp_delays(self.thea1, self.thea2)
            return

        self.update1 += 1
        self.update2 += 1
        stats = model.get_delay_stats()

        if self.update1 > self.adapt_window:
            if stats["delay1"]["p95"] > (self.thea1 - self.expand_margin):
                self.thea1 += 1.0
            self.update1 = 0

        if self.update2 > self.adapt_window:
            if stats["delay2"]["p95"] > (self.thea2 - self.expand_margin):
                self.thea2 += 1.0
            self.update2 = 0

        model.clamp_delays(self.thea1, self.thea2)

    def get_caps(self):
        return self.thea1, self.thea2

    def state_dict(self):
        return {"thea1": self.thea1, "thea2": self.thea2,
                "update1": self.update1, "update2": self.update2}

    def load_state_dict(self, state):
        self.thea1   = state["thea1"]
        self.thea2   = state["thea2"]
        self.update1 = state["update1"]
        self.update2 = state["update2"]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, results_dir):
    print(f"\n{'='*60}")
    print(f"  Model    : {cfg.model_name}")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Device   : {cfg.device}")
    print(f"  Results  : {results_dir}")
    print(f"{'='*60}\n")

    os.makedirs(results_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  Using device : {device}\n")

    model     = AdaptiveAxonalDelayNetV2SNNTorch(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {n_params:,}\n")
    train_loader, test_loader = get_loaders(cfg)

    # Pengfei uses lr=0.1, milestone=800
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma,
    )

    delay_sched = DelayScheduler(cfg)
    logger      = ResultsLogger(results_dir)

    train_losses, train_accs = [], []
    test_accs, epoch_times, peak_mems = [], [], []
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        model.train()
        reset_gpu_stats()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            spikes = spikes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(spikes)                    # (B, 20, T)
            loss   = model.compute_loss(output, labels, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimizer.step()
            delay_sched.step(model, epoch)

            pred        = model.predict(output)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device)
                labels = labels.to(device)
                output       = model(spikes)
                pred         = model.predict(output)
                test_correct += (pred == labels).sum().item()
                test_total   += labels.size(0)

        t_end      = time.time()
        peak_mem   = get_peak_gpu_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total
        test_acc   = test_correct / test_total
        avg_loss   = epoch_loss / len(train_loader)
        thea1, thea2 = delay_sched.get_caps()

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch"      : epoch + 1,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "delay_sched": delay_sched.state_dict(),
                "test_acc"   : test_acc,
                "thea1"      : thea1,
                "thea2"      : thea2,
            }, os.path.join(results_dir, f"{cfg.model_name}_best.pt"))

        print(
            f"  Epoch [{epoch+1:4d}/{cfg.epochs}] "
            f"Loss: {avg_loss:.6f} | "
            f"TrainAcc: {train_acc*100:.2f}% | "
            f"TestAcc: {test_acc*100:.2f}% | "
            f"Best: {best_acc*100:.2f}% | "
            f"Mem: {peak_mem:.1f}MB | "
            f"Time: {epoch_time:.1f}s | "
            f"Caps: [{thea1:.0f},{thea2:.0f}]"
        )

        logger.log_epoch(cfg.model_name, epoch + 1,
                         avg_loss, train_acc, test_acc,
                         peak_mem, epoch_time)

        if cfg.save_checkpoints and \
                (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save({
                "epoch"      : epoch + 1,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "delay_sched": delay_sched.state_dict(),
                "test_acc"   : test_acc,
                "thea1"      : thea1,
                "thea2"      : thea2,
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_epoch{epoch+1}.pt"))

    summary = {
        "model_name"  : cfg.model_name,
        "best_acc"    : float(best_acc),
        "n_params"    : n_params,
        "total_train_time_seconds": float(sum(epoch_times)),
        "train_losses": [float(x) for x in train_losses],
        "train_accs"  : [float(x) for x in train_accs],
        "test_accs"   : [float(x) for x in test_accs],
        "epoch_times" : [float(x) for x in epoch_times],
        "peak_mems"   : [float(x) for x in peak_mems],
        "final_thea1" : float(delay_sched.thea1),
        "final_thea2" : float(delay_sched.thea2),
        "hyperparams" : {
            "tau_s"    : cfg.tau_s,
            "tau_r"    : cfg.tau_r,
            "theta_u"  : cfg.theta_u,
            "lr"       : cfg.lr,
            "batch_size": cfg.batch_size,
            "epochs"   : cfg.epochs,
        }
    }

    summary_path = os.path.join(
        results_dir, f"{cfg.model_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.save()

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best accuracy  : {best_acc*100:.2f}%")
    print(f"  Summary saved  : {summary_path}")
    print(f"  Final caps     : [{delay_sched.thea1:.0f},{delay_sched.thea2:.0f}]")
    print(f"{'='*60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--results-dir",   type=str, default=None)
    parser.add_argument("--datasets-path", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config("aad_v2")
    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.datasets_path is not None: cfg.datasets_path = args.datasets_path

    results_dir = args.results_dir or os.path.join(
        cfg.results_dir, "aad_v2")
    train(cfg, results_dir)


if __name__ == "__main__":
    main()