# training/train_three_factor_delay.py
"""
Training script for Three-Factor Delay SNN on SHD.
Based on: Vassallo & Taherinejad, arXiv:2601.00668, Jan 2026

Usage:
    python training/train_three_factor_delay.py
    python training/train_three_factor_delay.py --epochs 1000
    python training/train_three_factor_delay.py --datasets-path /path/to/shd
    python training/train_three_factor_delay.py --resume results/three_factor_delay/v1/ThreeFactorDelay_best.pt
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
from three_factor_delay_model import ThreeFactorDelayNet


# ── Versioning ────────────────────────────────────────────────────────────────

def get_versioned_dir(base_dir: str) -> tuple:
    """
    Scans base_dir for existing vN subdirectories.
    Returns (next_version_dir, version_number).

    Example:
        base_dir/v1/ exists → returns (base_dir/v2/, 2)
        base_dir/v1/, v2/ exist → returns (base_dir/v3/, 3)
        nothing exists → returns (base_dir/v1/, 1)
    """
    existing = []
    if os.path.isdir(base_dir):
        for entry in os.scandir(base_dir):
            if entry.is_dir() and entry.name.startswith("v"):
                try:
                    existing.append(int(entry.name[1:]))
                except ValueError:
                    pass

    version     = max(existing) + 1 if existing else 1
    version_dir = os.path.join(base_dir, f"v{version}")
    return version_dir, version


# ── Dataset ───────────────────────────────────────────────────────────────────

class SHDDataset(Dataset):
    def __init__(self, path, split="train",
                 time_steps=400, n_inputs=700, max_time=1.4):
        self.time_steps = time_steps
        self.n_inputs   = n_inputs
        self.max_time   = max_time
        self.hdf5_path  = os.path.join(path, f"shd_{split}.h5")

        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(
                f"SHD file not found: {self.hdf5_path}\n"
                f"Download from: https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/"
            )

        with h5py.File(self.hdf5_path, "r") as f:
            self.labels    = f["labels"][:]
            self.n_samples = len(self.labels)

        print(f"    {split:5s}: {self.n_samples} samples")

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

        spikes = torch.zeros(self.time_steps, self.n_inputs,
                             dtype=torch.float32)
        spikes[t_idx, u_idx] = 1.0
        return spikes, label


def get_loaders(cfg):
    train_loader = DataLoader(
        SHDDataset(cfg.datasets_path, "train",
                   cfg.time_steps, cfg.n_inputs),
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        SHDDataset(cfg.datasets_path, "test",
                   cfg.time_steps, cfg.n_inputs),
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    return train_loader, test_loader


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_param_mb(model):
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)


def get_peak_gpu_mb():
    return (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_params_by_type(model):
    weight_params = 0
    delay_params  = 0
    other_params  = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if "delay" in name:
            delay_params  += n
        elif "weight" in name or "fc" in name:
            weight_params += n
        else:
            other_params  += n
    return {
        "weight_params": weight_params,
        "delay_params" : delay_params,
        "other_params" : other_params,
        "total"        : weight_params + delay_params + other_params,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg, base_dir, resume_path=None):

    # ── Versioning ────────────────────────────────────────────────────────
    # If resuming, use the existing version dir from the checkpoint path
    # Otherwise, create a new version dir
    if resume_path and os.path.exists(resume_path):
        results_dir = os.path.dirname(resume_path)
        version     = int(os.path.basename(results_dir)[1:])
    else:
        results_dir, version = get_versioned_dir(base_dir)

    os.makedirs(results_dir, exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Model      : {cfg.model_name}")
    print(f"  Version    : v{version}")
    print(f"  Results    : {results_dir}")
    print(f"  n_hidden   : {cfg.n_hidden}")
    print(f"  time_steps : {cfg.time_steps}")
    print(f"  max_syn_d  : {cfg.max_synaptic_delay}")
    print(f"  max_axn_d  : {cfg.max_axonal_delay}")
    print(f"  tau_mem    : {cfg.tau_mem}")
    print(f"  tau_syn    : {cfg.tau_syn}")
    print(f"  Epochs     : {cfg.epochs}")
    print(f"  LR         : {cfg.lr}")
    print(f"  Delay LR   : {cfg.learning_rate_d}")
    print(f"  Batch      : {cfg.batch_size}")
    print(f"{'='*65}\n")

    set_seed(cfg.seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device : {device}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = ThreeFactorDelayNet(cfg).to(device)
    param_mb  = get_param_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    breakdown = count_params_by_type(model)

    print(f"  Parameter breakdown:")
    print(f"    FC weights   : {breakdown['weight_params']:>8,}")
    print(f"    Delay params : {breakdown['delay_params']:>8,}")
    print(f"    Other        : {breakdown['other_params']:>8,}")
    print(f"    ─────────────────────────")
    print(f"    Total        : {n_params:>8,}")
    print(f"    Memory       : {param_mb:.4f} MB\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("  Loading datasets...")
    train_loader, test_loader = get_loaders(cfg)
    print()

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimiser — separate LR for weights vs delays ─────────────────────
    delay_params  = [p for n, p in model.named_parameters()
                     if "delay" in n]
    weight_params = [p for n, p in model.named_parameters()
                     if "delay" not in n]

    optimizer = torch.optim.Adam(
        [
            {"params": weight_params, "lr": cfg.lr},
            {"params": delay_params,  "lr": cfg.learning_rate_d},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.lr_milestones,
        gamma=cfg.lr_gamma,
    )

    logger = ResultsLogger(results_dir)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc    = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from : {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc    = ckpt.get("test_acc", 0.0)
        print(f"  Resumed epoch {start_epoch}, "
              f"best={best_acc*100:.2f}%\n")

    # ── History ───────────────────────────────────────────────────────────
    train_losses = []
    train_accs   = []
    test_accs    = []
    epoch_times  = []
    peak_mems    = []

    # ── Save config snapshot into version dir ─────────────────────────────
    config_snapshot = {
        "model_name"        : cfg.model_name,
        "version"           : version,
        "n_inputs"          : cfg.n_inputs,
        "n_hidden"          : cfg.n_hidden,
        "n_outputs"         : cfg.n_outputs,
        "time_steps"        : cfg.time_steps,
        "max_synaptic_delay": cfg.max_synaptic_delay,
        "max_axonal_delay"  : cfg.max_axonal_delay,
        "tau_mem"           : cfg.tau_mem,
        "tau_syn"           : cfg.tau_syn,
        "surrogate_sigma"   : cfg.surrogate_sigma,
        "lr"                : cfg.lr,
        "learning_rate_d"   : cfg.learning_rate_d,
        "epochs"            : cfg.epochs,
        "batch_size"        : cfg.batch_size,
        "seed"              : cfg.seed,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # ── Main Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):

        # ── Train ─────────────────────────────────────────────────────
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
            logits = model(spikes)
            loss   = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            # Clamp delays to valid range after every step
            with torch.no_grad():
                model.sdel1.delay.clamp_(0, cfg.max_synaptic_delay)
                model.sdel2.delay.clamp_(0, cfg.max_synaptic_delay)
                model.adel1.delay.clamp_(0, cfg.max_axonal_delay)
                model.adel2.delay.clamp_(0, cfg.max_axonal_delay)

            with torch.no_grad():
                preds      = logits.argmax(dim=1)
                correct   += (preds == labels).sum().item()
                total     += labels.size(0)
                epoch_loss += loss.item()

        # ── Eval ──────────────────────────────────────────────────────
        model.eval()
        test_correct = 0
        test_total   = 0

        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes       = spikes.to(device)
                labels       = labels.to(device)
                logits       = model(spikes)
                preds        = logits.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)

        scheduler.step()

        # ── Metrics ───────────────────────────────────────────────────
        t_end      = time.time()
        peak_mem   = get_peak_gpu_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total
        test_acc   = test_correct / test_total
        avg_loss   = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        # ── Best checkpoint ───────────────────────────────────────────
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch"      : epoch + 1,
                "version"    : version,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "test_acc"   : test_acc,
                "delay_stats": model.get_delay_stats(),
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_best.pt"))

        # ── Print ─────────────────────────────────────────────────────
        print(
            f"  [v{version}][{epoch+1:4d}/{cfg.epochs}] "
            f"Loss: {avg_loss:.5f} | "
            f"Tr: {train_acc*100:.2f}% | "
            f"Te: {test_acc*100:.2f}% | "
            f"Best: {best_acc*100:.2f}% | "
            f"LR: {current_lr:.5f} | "
            f"Mem: {peak_mem:.0f}MB | "
            f"t: {epoch_time:.1f}s"
        )

        logger.log_epoch(
            cfg.model_name, epoch+1, avg_loss,
            train_acc, test_acc, peak_mem, epoch_time
        )

        # ── Periodic checkpoint ───────────────────────────────────────
        if (cfg.save_checkpoints and
                (epoch + 1) % cfg.checkpoint_every == 0):
            torch.save({
                "epoch"      : epoch + 1,
                "version"    : version,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "test_acc"   : test_acc,
                "delay_stats": model.get_delay_stats(),
            }, os.path.join(
                results_dir,
                f"{cfg.model_name}_epoch{epoch+1}.pt"
            ))

    # ── Final delay statistics ─────────────────────────────────────────────
    delay_stats = model.get_delay_stats()
    print(f"\n  Learned delay statistics:")
    for layer, stats in delay_stats.items():
        print(f"    {layer:20s}: "
              f"mean={stats['mean']:.2f}  "
              f"max={stats['max']:.2f}  "
              f"p95={stats['p95']:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "model_name"     : cfg.model_name,
        "version"        : version,
        "results_dir"    : results_dir,
        "best_acc"       : float(best_acc),
        "n_params"       : n_params,
        "param_breakdown": {k: int(v) for k, v in breakdown.items()},
        "param_mem_mb"   : float(param_mb),
        "train_losses"   : [float(x) for x in train_losses],
        "train_accs"     : [float(x) for x in train_accs],
        "test_accs"      : [float(x) for x in test_accs],
        "epoch_times"    : [float(x) for x in epoch_times],
        "peak_mems"      : [float(x) for x in peak_mems],
        "delay_stats"    : delay_stats,
        "config"         : config_snapshot,
    }

    summary_path = os.path.join(
        results_dir, f"{cfg.model_name}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.save()

    print(f"\n{'='*65}")
    print(f"  Version       : v{version}")
    print(f"  Best accuracy : {best_acc*100:.2f}%")
    print(f"  Results dir   : {results_dir}")
    print(f"  Summary saved : {summary_path}")
    print(f"{'='*65}\n")

    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train Three-Factor Delay SNN on SHD"
    )
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--base-dir",      type=str, default=None,
                        help="Base results dir — versioning happens inside here")
    parser.add_argument("--datasets-path", type=str, default=None)
    parser.add_argument("--resume",        type=str, default=None,
                        help="Path to checkpoint — resumes in same version dir")
    args = parser.parse_args()

    cfg = get_config("three_factor_delay")

    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.datasets_path is not None: cfg.datasets_path = args.datasets_path

    # base_dir is the unversioned root e.g. results/three_factor_delay/
    # versioning creates results/three_factor_delay/v1/, v2/, etc.
    base_dir = args.base_dir or os.path.join(
        cfg.results_dir, "three_factor_delay"
    )

    train(cfg, base_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()