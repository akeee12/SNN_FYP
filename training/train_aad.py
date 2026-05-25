# training/train_aad.py
"""
Standalone AAD training script.
Extracted from train.py — runs independently as a PBS job.
Results saved to: --results-dir (default: results/aad_pytorch/)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from configs import get_config
from utils.seed import set_seed
from CODE.SNN_FYP.experiments.plot_buffer_comparison import ResultsLogger
from axonal_delay_modelPYTorch import AdaptiveAxonalDelayNet


# ── Dataset ───────────────────────────────────────────────────────────────────
class SHDDataset(Dataset):
    def __init__(self, path, split="train", time_steps=400,
                 n_inputs=700, max_time=1.4):
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
            SHDDataset(cfg.datasets_path, "train", cfg.time_steps),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True),
        DataLoader(
            SHDDataset(cfg.datasets_path, "test", cfg.time_steps),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True),
    )


# ── Memory utils ──────────────────────────────────────────────────────────────
def get_param_mb(model):
    return sum(p.numel() * p.element_size()
               for p in model.parameters()) / (1024**2)

def get_peak_gpu_mb():
    return torch.cuda.max_memory_allocated() / (1024**2) \
           if torch.cuda.is_available() else 0.0

def reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ── Main training function ────────────────────────────────────────────────────
def train(cfg, results_dir):
    print(f"\n{'='*60}")
    print(f"  Model    : {cfg.model_name}")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Device   : {cfg.device}")
    print(f"  Results  : {results_dir}")
    print(f"{'='*60}")

    os.makedirs(results_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model     = AdaptiveAxonalDelayNet(cfg).to(device)
    param_mem = get_param_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {n_params:,}")
    print(f"  Param memory : {param_mem:.2f} MB")

    train_loader, test_loader = get_loaders(cfg)
    logger    = ResultsLogger(results_dir)
    optimizer = torch.optim.NAdam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)

    def spike_rate_loss(output, labels):
        B, C   = output.shape
        rates  = output / cfg.time_steps
        target = torch.full((B, C), cfg.target_false,
                            dtype=torch.float32, device=device)
        target[torch.arange(B), labels] = cfg.target_true
        return F.mse_loss(rates, target)

    thea1, thea2 = float(cfg.delay_init_cap), float(cfg.delay_init_cap)
    update_count = 0
    train_losses, test_accs, epoch_times, peak_mems = [], [], [], []

    for epoch in range(cfg.epochs):
        model.train()
        reset_gpu()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            spikes = spikes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(spikes)
            loss   = spike_rate_loss(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss   += loss.item()
            correct      += (output.argmax(1) == labels).sum().item()
            total        += labels.size(0)

            update_count += 1
            if epoch <= cfg.delay_adapt_start:
                model.clamp_delays(thea1, thea2)
            else:
                if update_count >= cfg.delay_adapt_window:
                    stats = model.get_delay_stats()
                    if stats["delay1"]["p95"] > (
                            thea1 - cfg.delay_expand_margin):
                        thea1 += 1
                    if stats["delay2"]["p95"] > (
                            thea2 - cfg.delay_expand_margin):
                        thea2 += 1
                    model.clamp_delays(thea1, thea2)
                    update_count = 0

        scheduler.step()

        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device)
                labels = labels.to(device)
                output = model(spikes)
                test_correct += (output.argmax(1) == labels).sum().item()
                test_total   += labels.size(0)

        t_end      = time.time()
        peak_mem   = get_peak_gpu_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total
        test_acc   = test_correct / test_total
        avg_loss   = epoch_loss / len(train_loader)

        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        print(f"  Epoch [{epoch+1:4d}/{cfg.epochs}] "
              f"Loss: {avg_loss:.6f} | "
              f"TrainAcc: {train_acc*100:.2f}% | "
              f"TestAcc: {test_acc*100:.2f}% | "
              f"PeakMem: {peak_mem:.1f}MB | "
              f"Time: {epoch_time:.1f}s | "
              f"Delays: [{thea1:.1f},{thea2:.1f}]")

        logger.log_epoch(cfg.model_name, epoch+1, avg_loss,
                         train_acc, test_acc, peak_mem, epoch_time)

        if cfg.save_checkpoints and (epoch+1) % cfg.checkpoint_every == 0:
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "test_acc": test_acc,
                "thea1": thea1, "thea2": thea2,
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_epoch{epoch+1}.pt"))

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "model_name"  : cfg.model_name,
        "best_acc"    : float(max(test_accs)),
        "n_params"    : n_params,
        "param_mem_mb": float(param_mem),
        "train_losses": [float(x) for x in train_losses],
        "test_accs"   : [float(x) for x in test_accs],
        "epoch_times" : [float(x) for x in epoch_times],
        "peak_mems"   : [float(x) for x in peak_mems],
    }
    summary_path = os.path.join(
        results_dir, f"{cfg.model_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.save()
    print(f"\n  ✓ Best accuracy : {max(test_accs)*100:.2f}%")
    print(f"  Summary saved  : {summary_path}")
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train AAD on SHD")
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--results-dir",   type=str, default=None)
    parser.add_argument("--datasets-path", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config("aad_pytorch")
    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.datasets_path is not None: cfg.datasets_path = args.datasets_path

    # Default results dir: results/SNN_FYP/aad_pytorch/
    results_dir = args.results_dir or os.path.join(
        cfg.results_dir, "aad_pytorch")

    train(cfg, results_dir)


if __name__ == "__main__":
    main()