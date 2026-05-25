# training/train_dhlif.py
"""
Training script for DH-LIF (DH-SFNN) on SHD.
Paper: Zheng et al. 2024, Nature Communications 15:277

Usage:
    python training/train_dhlif.py
    python training/train_dhlif.py --epochs 200
    python training/train_dhlif.py --datasets-path /path/to/shd
    python training/train_dhlif.py --resume results/dhlif/v1/DHLIF_best.pt
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
from CODE.SNN_FYP.experiments.plot_buffer_comparison import ResultsLogger
from dhlif_model import DHLIFNet


# ── Versioning ────────────────────────────────────────────────────────────────

def get_versioned_dir(base_dir: str) -> tuple:
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
    """
    SHD dataset loader.
    Paper uses raw 700 channels, dt=1ms, T~250 timesteps.
    No channel binning (unlike cAdLIF which bins to 140).
    """

    def __init__(self, path, split="train",
                 time_steps=250, n_inputs=700, max_time=1.4):
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

        print(f"    {split:5s}: {self.n_samples} samples  "
              f"T={time_steps}  N={n_inputs}")

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
    return sum(p.numel() * p.element_size()
               for p in model.parameters()) / (1024 ** 2)

def get_peak_gpu_mb():
    return (torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available() else 0.0)

def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def count_params_by_type(model):
    timing_p = sum(
        p.numel() for n, p in model.named_parameters()
        if "alpha_hat" in n or "beta_hat" in n
    )
    weight_p = sum(
        p.numel() for n, p in model.named_parameters()
        if "alpha_hat" not in n and "beta_hat" not in n
    )
    total = timing_p + weight_p
    return {"weight_params" : weight_p,
            "timing_params" : timing_p,
            "total"         : total}


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg, base_dir, resume_path=None):

    # ── Versioning ────────────────────────────────────────────────────────
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
    print(f"  n_layers   : {cfg.n_layers}")
    print(f"  n_branches : {cfg.n_branches}")
    print(f"  time_steps : {cfg.time_steps}")
    print(f"  alpha_init : N({cfg.alpha_init_mean},{cfg.alpha_init_std})")
    print(f"  beta_init  : N({cfg.beta_init_mean},{cfg.beta_init_std})")
    print(f"  Epochs     : {cfg.epochs}")
    print(f"  LR         : {cfg.lr}")
    print(f"  Batch      : {cfg.batch_size}")
    print(f"{'='*65}\n")

    set_seed(cfg.seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"  Device : {device}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = DHLIFNet(cfg).to(device)
    param_mb  = get_param_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    breakdown = count_params_by_type(model)

    print(f"  Parameter breakdown:")
    print(f"    Weights + readout : {breakdown['weight_params']:>8,}")
    print(f"    Timing factors    : {breakdown['timing_params']:>8,}")
    print(f"    ─────────────────────────")
    print(f"    Total             : {n_params:>8,}")
    print(f"    Memory            : {param_mb:.4f} MB\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("  Loading datasets...")
    train_loader, test_loader = get_loaders(cfg)
    print()

    # ── Loss ──────────────────────────────────────────────────────────────
    # Model outputs log(sum_t softmax) — NLLLoss is correct
    criterion = nn.NLLLoss()

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr    = cfg.lr,
        betas = (0.9, 0.999),
        eps   = 1e-8,
    )

    # Step LR scheduler (Methods: step learning rate scheduler)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg.lr_step_size,
        gamma     = cfg.lr_gamma,
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

    # ── Config snapshot ───────────────────────────────────────────────────
    config_snapshot = {
        "model_name"      : cfg.model_name,
        "version"         : version,
        "n_inputs"        : cfg.n_inputs,
        "n_hidden"        : cfg.n_hidden,
        "n_outputs"       : cfg.n_outputs,
        "n_layers"        : cfg.n_layers,
        "n_branches"      : cfg.n_branches,
        "time_steps"      : cfg.time_steps,
        "v_threshold"     : cfg.v_threshold,
        "alpha_init_mean" : cfg.alpha_init_mean,
        "alpha_init_std"  : cfg.alpha_init_std,
        "beta_init_mean"  : cfg.beta_init_mean,
        "beta_init_std"   : cfg.beta_init_std,
        "surrogate"       : "multi_gaussian",
        "gamma"           : cfg.surrogate_gamma,
        "h"               : cfg.surrogate_h,
        "sigma"           : cfg.surrogate_sigma,
        "s"               : cfg.surrogate_s,
        "lr"              : cfg.lr,
        "lr_step_size"    : cfg.lr_step_size,
        "lr_gamma"        : cfg.lr_gamma,
        "epochs"          : cfg.epochs,
        "batch_size"      : cfg.batch_size,
        "seed"            : cfg.seed,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # ── History ───────────────────────────────────────────────────────────
    train_losses = []
    train_accs   = []
    test_accs    = []
    epoch_times  = []
    peak_mems    = []

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
                "epoch"        : epoch + 1,
                "version"      : version,
                "model_state"  : model.state_dict(),
                "optimizer"    : optimizer.state_dict(),
                "scheduler"    : scheduler.state_dict(),
                "test_acc"     : test_acc,
                "timing_stats" : model.get_timing_stats(),
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_best.pt"))

        print(
            f"  [v{version}][{epoch+1:4d}/{cfg.epochs}] "
            f"Loss: {avg_loss:.5f} | "
            f"Tr: {train_acc*100:.2f}% | "
            f"Te: {test_acc*100:.2f}% | "
            f"Best: {best_acc*100:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Mem: {peak_mem:.0f}MB | "
            f"t: {epoch_time:.1f}s"
        )

        logger.log_epoch(
            cfg.model_name, epoch+1, avg_loss,
            train_acc, test_acc, peak_mem, epoch_time
        )

        if (cfg.save_checkpoints and
                (epoch + 1) % cfg.checkpoint_every == 0):
            torch.save({
                "epoch"      : epoch + 1,
                "version"    : version,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "test_acc"   : test_acc,
            }, os.path.join(
                results_dir,
                f"{cfg.model_name}_epoch{epoch+1}.pt"
            ))

    # ── Final timing statistics ───────────────────────────────────────────
    timing_stats = model.get_timing_stats()
    print(f"\n  Learned timing factor statistics:")
    for layer, stats in timing_stats.items():
        print(f"    {layer}: "
              f"alpha mean={stats['alpha_mean']:.3f} "
              f"std={stats['alpha_std']:.3f}  "
              f"beta mean={stats['beta_mean']:.3f}")

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
        "timing_stats"   : timing_stats,
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
    print(f"  Summary       : {summary_path}")
    print(f"{'='*65}\n")

    return summary


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DH-LIF (DH-SFNN) on SHD"
    )
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--base-dir",      type=str, default=None)
    parser.add_argument("--datasets-path", type=str, default=None)
    parser.add_argument("--resume",        type=str, default=None)
    args = parser.parse_args()

    cfg = get_config("dhlif")

    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.datasets_path is not None: cfg.datasets_path = args.datasets_path

    base_dir = args.base_dir or os.path.join(
        cfg.results_dir, "dhlif"
    )

    train(cfg, base_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()