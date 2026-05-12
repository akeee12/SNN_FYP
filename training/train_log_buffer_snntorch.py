# training/train_log_buffer_snntorch.py
"""
Training script for LogBufferAADSNNTorch on SHD.

Usage:
    python training/train_log_buffer_snntorch.py
    python training/train_log_buffer_snntorch.py --epochs 500
    python training/train_log_buffer_snntorch.py --datasets-path /path/to/shd
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
from experiments.results_logger import ResultsLogger
from log_buffer_aad_snntorch import LogBufferAADSNNTorch


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

        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"SHD not found: {self.hdf5_path}"
            )

        with h5py.File(self.hdf5_path, "r") as f:
            self.labels    = f["labels"][:]
            self.n_samples = len(self.labels)

        print(f"    {split}: {self.n_samples} samples")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, "r") as f:
            times = f["spikes"]["times"][idx]
            units = f["spikes"]["units"][idx]

        label = int(self.labels[idx])
        t_idx = np.floor(
            times / self.max_time * self.time_steps
        ).astype(int)
        t_idx = np.clip(t_idx, 0, self.time_steps - 1)
        u_idx = np.clip(units.astype(int), 0, self.n_inputs - 1)

        spikes = torch.zeros(
            self.time_steps, self.n_inputs, dtype=torch.float32
        )
        spikes[t_idx, u_idx] = 1.0
        return spikes, label


def get_loaders(cfg):
    return (
        DataLoader(
            SHDDataset(cfg.datasets_path, "train", cfg.time_steps),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
        ),
        DataLoader(
            SHDDataset(cfg.datasets_path, "test", cfg.time_steps),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_param_mb(model):
    return sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024**2)

def get_peak_gpu_mb():
    return torch.cuda.max_memory_allocated() / (1024**2) \
        if torch.cuda.is_available() else 0.0

def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# DELAY SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class DelayScheduler:
    def __init__(self, cfg):
        self.thea1         = float(cfg.delay_init_cap)
        self.thea2         = float(cfg.delay_init_cap)
        self.adapt_start   = cfg.delay_adapt_start
        self.adapt_window  = cfg.delay_adapt_window
        self.expand_margin = cfg.delay_expand_margin
        self.update_count  = 0

    def step(self, model, epoch):
        if epoch < self.adapt_start:
            model.clamp_delays(self.thea1, self.thea2)
            return
        self.update_count += 1
        model.clamp_delays(self.thea1, self.thea2)
        if self.update_count >= self.adapt_window:
            stats = model.get_delay_stats()
            if stats["delay1"]["p95"] > (self.thea1 - self.expand_margin):
                self.thea1 += 1.0
            if stats["delay2"]["p95"] > (self.thea2 - self.expand_margin):
                self.thea2 += 1.0
            model.clamp_delays(self.thea1, self.thea2)
            self.update_count = 0

    def get_caps(self):
        return self.thea1, self.thea2

    def state_dict(self):
        return {
            "thea1"       : self.thea1,
            "thea2"       : self.thea2,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state):
        self.thea1        = state["thea1"]
        self.thea2        = state["thea2"]
        self.update_count = state["update_count"]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, results_dir, resume_path=None):

    print(f"\n{'='*65}")
    print(f"  Model         : {cfg.model_name}")
    print(f"  buffer_size   : {cfg.buffer_size} (k)")
    print(f"  buffer_max_lag: {cfg.buffer_max_lag}")
    print(f"  tau_s         : {cfg.tau_s}")
    print(f"  tau_r         : {cfg.tau_r}")
    print(f"  threshold     : {cfg.threshold}")
    print(f"  Epochs        : {cfg.epochs}")
    print(f"  LR            : {cfg.lr}")
    print(f"  Results       : {results_dir}")
    print(f"{'='*65}\n")

    os.makedirs(results_dir, exist_ok=True)

    # Random seed — logged for reproducibility
    import random
    seed = random.randint(0, 99999)
    print(f"  Seed : {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device : {device}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model     = LogBufferAADSNNTorch(cfg).to(device)
    param_mem = get_param_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Params : {n_params:,}")
    print(f"  Memory : {param_mem:.4f} MB\n")

    # ── Data ──────────────────────────────────────────────────────────
    print("  Loading datasets...")
    train_loader, test_loader = get_loaders(cfg)

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr    = cfg.lr,
        betas = (0.9, 0.999),
        eps   = 1e-8,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones = cfg.lr_milestones,
        gamma      = cfg.lr_gamma,
    )

    delay_sched = DelayScheduler(cfg)
    logger      = ResultsLogger(results_dir)

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc    = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler"   in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "delay_sched" in ckpt:
            delay_sched.load_state_dict(ckpt["delay_sched"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc    = ckpt.get("test_acc", 0.0)
        print(f"  Resumed epoch {start_epoch}, "
              f"best={best_acc*100:.2f}%\n")

    train_losses, train_accs = [], []
    test_accs, epoch_times   = [], []
    peak_mems                = []

    for epoch in range(start_epoch, cfg.epochs):

        # ── TRAIN ─────────────────────────────────────────────────────
        model.train()
        reset_gpu_stats()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            spikes = spikes.to(device)    # (B, T, 700)
            labels = labels.to(device)    # (B,)

            optimizer.zero_grad()

            # Forward — returns (T, B, 20)
            output = model(spikes)

            # Loss
            loss = model.compute_loss(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            # Delay scheduling
            delay_sched.step(model, epoch)

            with torch.no_grad():
                pred     = model.predict(output)  # (B,)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)
                epoch_loss += loss.item()

        # ── EVAL ──────────────────────────────────────────────────────
        model.eval()
        test_correct = 0
        test_total   = 0

        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device)
                labels = labels.to(device)
                output       = model(spikes)
                pred         = model.predict(output)
                test_correct += (pred == labels).sum().item()
                test_total   += labels.size(0)

        scheduler.step()

        # ── Stats ─────────────────────────────────────────────────────
        t_end        = time.time()
        peak_mem     = get_peak_gpu_mb()
        epoch_time   = t_end - t_start
        train_acc    = correct / total
        test_acc     = test_correct / test_total
        avg_loss     = epoch_loss / len(train_loader)
        thea1, thea2 = delay_sched.get_caps()
        current_lr   = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch"      : epoch + 1,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "delay_sched": delay_sched.state_dict(),
                "test_acc"   : test_acc,
                "thea1"      : thea1,
                "thea2"      : thea2,
                "seed"       : seed,
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_best.pt"))

        print(
            f"  [{epoch+1:4d}/{cfg.epochs}] "
            f"Loss: {avg_loss:.5f} | "
            f"Tr: {train_acc*100:.2f}% | "
            f"Te: {test_acc*100:.2f}% | "
            f"Best: {best_acc*100:.2f}% | "
            f"LR: {current_lr:.5f} | "
            f"Mem: {peak_mem:.0f}MB | "
            f"t: {epoch_time:.1f}s | "
            f"caps:[{thea1:.0f},{thea2:.0f}]"
        )

        logger.log_epoch(
            cfg.model_name, epoch+1, avg_loss,
            train_acc, test_acc, peak_mem, epoch_time
        )

        if cfg.save_checkpoints and \
                (epoch+1) % cfg.checkpoint_every == 0:
            torch.save({
                "epoch"      : epoch + 1,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "delay_sched": delay_sched.state_dict(),
                "test_acc"   : test_acc,
            }, os.path.join(
                results_dir,
                f"{cfg.model_name}_epoch{epoch+1}.pt"
            ))

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "model_name"    : cfg.model_name,
        "best_acc"      : float(best_acc),
        "n_params"      : n_params,
        "param_mem_mb"  : float(param_mem),
        "seed"          : seed,
        "buffer_size"   : cfg.buffer_size,
        "buffer_max_lag": cfg.buffer_max_lag,
        "tau_s"         : cfg.tau_s,
        "tau_r"         : cfg.tau_r,
        "threshold"     : cfg.threshold,
        "train_losses"  : [float(x) for x in train_losses],
        "train_accs"    : [float(x) for x in train_accs],
        "test_accs"     : [float(x) for x in test_accs],
        "epoch_times"   : [float(x) for x in epoch_times],
        "peak_mems"     : [float(x) for x in peak_mems],
    }

    summary_path = os.path.join(
        results_dir, f"{cfg.model_name}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.save()

    print(f"\n{'='*65}")
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Summary saved : {summary_path}")
    print(f"{'='*65}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train LogBufferAADSNNTorch on SHD"
    )
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--results-dir",   type=str, default=None)
    parser.add_argument("--datasets-path", type=str, default=None)
    parser.add_argument("--buffer-size",   type=int, default=None)
    parser.add_argument("--max-lag",       type=int, default=None)
    parser.add_argument("--resume",        type=str, default=None)
    args = parser.parse_args()

    cfg = get_config("log_buffer_snntorch")

    if args.epochs        is not None: cfg.epochs         = args.epochs
    if args.datasets_path is not None: cfg.datasets_path  = args.datasets_path
    if args.buffer_size   is not None: cfg.buffer_size    = args.buffer_size
    if args.max_lag       is not None: cfg.buffer_max_lag = args.max_lag

    results_dir = args.results_dir or os.path.join(
        cfg.results_dir, "log_buffer_snntorch"
    )

    train(cfg, results_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()