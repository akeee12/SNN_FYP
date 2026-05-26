# training/train_power_law_ring_buffer_ablation_snntorch.py
"""
Training script for PowerLawTapAADNetSNNTorch on SHD.
Ablation-ready: supports --num-taps (n=2..9), --network-size
(small/medium/large), and --gamma (power law exponent).

Tap rule:
    t_i = round( max_lag * (i / (n-1))^gamma )
    buffer_max_lag passed explicitly via --buffer-max-lag.
    Endpoints always pinned to 1 and max_lag.

Naming convention:
    results/ablation_power_law/vN/k32_n7_lag64_g2.0_medium/
    results/ablation_power_law/vN/k32_n4_lag8_g2.0_small/

Usage (ablation, called from PBS array job):
    python3 training/train_power_law_ring_buffer_ablation_snntorch.py \
        --epochs        100                                            \
        --datasets-path /path/to/SHD                                  \
        --base-dir      /path/to/results/ablation_power_law/v1        \
        --num-taps      7                                              \
        --network-size  medium                                         \
        --gamma         2.0                                            \
        --buffer-max-lag 64

Usage (production, defaults):
    python3 training/train_power_law_ring_buffer_ablation_snntorch.py \
        --epochs        500                                            \
        --datasets-path /path/to/SHD                                  \
        --base-dir      /path/to/results/power_law_ring_buffer
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
from power_law_tap_aad_snntorch import PowerLawTapAADNetSNNTorch


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class ResultsLogger:
    """
    Writes one CSV row per epoch to results_dir/training_log.csv.
    Columns: model, epoch, loss, train_acc, test_acc, peak_mem_mb, epoch_time_s
    """
    def __init__(self, results_dir: str):
        os.makedirs(results_dir, exist_ok=True)
        self.path = os.path.join(results_dir, "training_log.csv")
        with open(self.path, "w") as f:
            f.write("model,epoch,loss,train_acc,"
                    "test_acc,peak_mem_mb,epoch_time_s\n")

    def log_epoch(self, model_name: str, epoch: int,
                  loss: float, train_acc: float, test_acc: float,
                  peak_mem: float, epoch_time: float):
        with open(self.path, "a") as f:
            f.write(f"{model_name},{epoch},{loss:.6f},"
                    f"{train_acc:.6f},{test_acc:.6f},"
                    f"{peak_mem:.2f},{epoch_time:.2f}\n")

    def save(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# NETWORK SIZE MAP
# ─────────────────────────────────────────────────────────────────────────────

NETWORK_SIZES = {
    "small" : 64,
    "medium": 128,
    "large" : 256,
}


# ─────────────────────────────────────────────────────────────────────────────
# TAP GENERATION — power law spacing
# ─────────────────────────────────────────────────────────────────────────────

def make_power_law_taps(n: int, max_lag: int, gamma: float) -> list:
    """
    Generate exactly n power-law spaced taps up to max_lag.

        t_i = round( max_lag * (i / (n-1))^gamma )

    Endpoints pinned: t_0=1, t_{n-1}=max_lag.

    n=7, max_lag=64, gamma=2.0 → [1, 2, 7, 16, 28, 44, 64]
    n=5, max_lag=64, gamma=2.0 → [1, 4, 16, 36, 64]
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if n == 2:
        return [1, max_lag]

    taps = []
    for i in range(n):
        t = max_lag * ((i / (n - 1)) ** gamma)
        taps.append(max(1, round(t)))

    taps[0]  = 1
    taps[-1] = max_lag

    if len(set(taps)) != n:
        raise ValueError(
            f"Power-law taps contain duplicates: {taps}\n"
            f"max_lag={max_lag}, n_taps={n}, gamma={gamma}\n"
            f"Fix: decrease n_taps, increase max_lag, or adjust gamma."
        )
    return taps


# ─────────────────────────────────────────────────────────────────────────────
# RUN NAMING
# ─────────────────────────────────────────────────────────────────────────────

def make_run_name(cfg, network_size: str) -> str:
    n = len(cfg.taps)
    return (f"k{cfg.buffer_size}_n{n}"
            f"_lag{cfg.buffer_max_lag}"
            f"_g{cfg.gamma:.1f}"
            f"_{network_size}")


# ─────────────────────────────────────────────────────────────────────────────
# VERSIONING
# ─────────────────────────────────────────────────────────────────────────────

def get_run_dir(base_dir: str, run_name: str) -> str:
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


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


def resolve_results_dir(base_dir: str, run_name: str) -> str:
    base = base_dir.rstrip("/")
    bn   = os.path.basename(base)
    if bn.startswith("v") and bn[1:].isdigit():
        return get_run_dir(base, run_name)
    else:
        version_dir, _ = get_versioned_dir(base)
        return get_run_dir(version_dir, run_name)


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
            batch_size  = cfg.batch_size,
            shuffle     = True,
            num_workers = cfg.num_workers,
            pin_memory  = True,
        ),
        DataLoader(
            SHDDataset(cfg.datasets_path, "test",
                       cfg.time_steps, cfg.n_inputs),
            batch_size  = cfg.batch_size,
            shuffle     = False,
            num_workers = cfg.num_workers,
            pin_memory  = True,
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

def get_param_mb(model):
    return sum(p.numel() * p.element_size()
               for p in model.parameters()) / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# NaN UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def zero_nan_grads(model: nn.Module):
    for param in model.parameters():
        if param.grad is not None:
            if (torch.isnan(param.grad).any() or
                    torch.isinf(param.grad).any()):
                param.grad = torch.nan_to_num(
                    param.grad, nan=0.0, posinf=0.0, neginf=0.0)


def reset_nan_params(model: nn.Module):
    with torch.no_grad():
        for param in model.parameters():
            if (torch.isnan(param.data).any() or
                    torch.isinf(param.data).any()):
                param.data = torch.nan_to_num(
                    param.data, nan=0.0, posinf=1.0, neginf=0.0)


def safe_delay_stats(model) -> tuple:
    try:
        stats  = model.get_delay_stats()
        d1_p95 = stats['delay1']['p95']
        d2_p95 = stats['delay2']['p95']
        if not torch.isfinite(torch.tensor(float(d1_p95))):
            d1_p95 = 0.0
        if not torch.isfinite(torch.tensor(float(d2_p95))):
            d2_p95 = 0.0
    except Exception:
        d1_p95 = 0.0
        d2_p95 = 0.0
    return d1_p95, d2_p95


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
        self.update1       = 0
        self.update2       = 0

    def step(self, model, epoch):
        if epoch <= self.adapt_start:
            model.clamp_delays(self.thea1, self.thea2)
            return
        self.update1 += 1
        self.update2 += 1
        d1 = torch.nan_to_num(
            model.delay1.delay.detach().cpu(), nan=0.0)
        d2 = torch.nan_to_num(
            model.delay2.delay.detach().cpu(), nan=0.0)
        sorted1 = torch.sort(torch.floor(d1))[0]
        sorted2 = torch.sort(torch.floor(d2))[0]
        p84_idx1 = max(0, int(0.84 * len(sorted1)) - 1)
        p84_idx2 = max(0, int(0.84 * len(sorted2)) - 1)
        if self.update1 >= self.adapt_window:
            if sorted1[p84_idx1] > (self.thea1 - self.expand_margin):
                self.thea1 += 1.0
            self.update1 = 0
        if self.update2 >= self.adapt_window:
            if sorted2[p84_idx2] > (self.thea2 - self.expand_margin):
                self.thea2 += 1.0
            self.update2 = 0
        model.clamp_delays(self.thea1, self.thea2)

    def get_caps(self):
        return self.thea1, self.thea2

    def state_dict(self):
        return {
            "thea1"  : self.thea1,   "thea2"  : self.thea2,
            "update1": self.update1,  "update2": self.update2,
        }

    def load_state_dict(self, state):
        self.thea1   = state["thea1"];   self.thea2   = state["thea2"]
        self.update1 = state["update1"]; self.update2 = state["update2"]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, results_dir, network_size, resume_path=None):

    os.makedirs(results_dir, exist_ok=True)
    run_name = make_run_name(cfg, network_size)

    print(f"\n{'='*60}")
    print(f"  Run           : {run_name}")
    print(f"  Model         : {cfg.model_name}")
    print(f"  Strategy      : power_law  (ring buffer storage)")
    print(f"  Gamma         : {cfg.gamma}")
    print(f"  Network size  : {network_size}  (n_hidden={cfg.n_hidden})")
    print(f"  Taps (n={len(cfg.taps):2d}) : {cfg.taps}")
    print(f"  buffer_size   : {cfg.buffer_size}  (k)")
    print(f"  buffer_max_lag: {cfg.buffer_max_lag}")
    print(f"  Results dir   : {results_dir}")
    print(f"  Epochs        : {cfg.epochs}")
    print(f"  LR            : {cfg.lr}")
    print(f"  Batch         : {cfg.batch_size}")
    print(f"  tau_s         : {cfg.tau_s}")
    print(f"  tau_r         : {cfg.tau_r}")
    print(f"  threshold     : {cfg.threshold}")
    print(f"  theta_u       : {cfg.theta_u}")
    print(f"  target_true   : {cfg.target_true}")
    print(f"  target_false  : {cfg.target_false}")
    print(f"{'='*60}\n")

    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"  Using device  : {device}\n")

    model    = PowerLawTapAADNetSNNTorch(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = get_param_mb(model)
    buf_info = model.get_buffer_info()

    print(f"  Parameters    : {n_params:,}")
    print(f"  Memory        : {param_mb:.4f} MB")
    print(f"  Buffer k      : {buf_info['k']}")
    print(f"  Buffer L      : {buf_info['L']}")
    print(f"  Buffer taps   : {buf_info['taps']}")
    print(f"  Storage       : {buf_info['storage']}\n")

    train_loader, test_loader = get_loaders(cfg)

    optimizer = torch.optim.NAdam(
        model.parameters(),
        lr           = cfg.lr,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
        weight_decay = getattr(cfg, "weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones = cfg.lr_milestones,
        gamma      = cfg.lr_gamma,
    )

    delay_sched = DelayScheduler(cfg)
    logger      = ResultsLogger(results_dir)

    start_epoch = 0
    best_acc    = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from : {resume_path}")
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

    config_snapshot = {
        "run_name"          : run_name,
        "model_name"        : cfg.model_name,
        "network_size"      : network_size,
        "n_hidden"          : cfg.n_hidden,
        "tap_strategy"      : "power_law",
        "storage"           : "ring",
        "gamma"             : cfg.gamma,
        "taps"              : cfg.taps,
        "n_taps"            : len(cfg.taps),
        "buffer_size"       : cfg.buffer_size,
        "buffer_max_lag"    : cfg.buffer_max_lag,
        "tau_s"             : cfg.tau_s,
        "tau_r"             : cfg.tau_r,
        "threshold"         : cfg.threshold,
        "theta_u"           : cfg.theta_u,
        "target_true"       : cfg.target_true,
        "target_false"      : cfg.target_false,
        "lr"                : cfg.lr,
        "weight_decay"      : getattr(cfg, "weight_decay", 0.0),
        "dropout"           : getattr(cfg, "dropout", 0.0),
        "batch_size"        : cfg.batch_size,
        "epochs"            : cfg.epochs,
        "delay_init_cap"    : cfg.delay_init_cap,
        "delay_adapt_start" : cfg.delay_adapt_start,
        "delay_adapt_window": cfg.delay_adapt_window,
        "seed"              : cfg.seed,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    train_losses, train_accs, test_accs = [], [], []
    epoch_times, peak_mems = [], []

    for epoch in range(start_epoch, cfg.epochs):
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
            output = model(spikes)
            loss   = model.compute_loss(output, labels)
            loss.backward()
            zero_nan_grads(model)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0)
            optimizer.step()
            reset_nan_params(model)
            pred        = model.predict(output)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)
            epoch_loss += loss.item()

        delay_sched.step(model, epoch)
        scheduler.step()

        model.eval()
        test_correct = 0
        test_total   = 0
        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes        = spikes.to(device)
                labels        = labels.to(device)
                output        = model(spikes)
                pred          = model.predict(output)
                test_correct += (pred == labels).sum().item()
                test_total   += labels.size(0)

        t_end        = time.time()
        peak_mem     = get_peak_gpu_mb()
        epoch_time   = t_end - t_start
        train_acc    = correct / total
        test_acc     = test_correct / test_total
        avg_loss     = epoch_loss / len(train_loader)
        thea1, thea2 = delay_sched.get_caps()
        current_lr   = optimizer.param_groups[0]["lr"]
        d1_p95, d2_p95 = safe_delay_stats(model)

        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch"        : epoch + 1,
                "run_name"     : run_name,
                "network_size" : network_size,
                "model_state"  : model.state_dict(),
                "optimizer"    : optimizer.state_dict(),
                "scheduler"    : scheduler.state_dict(),
                "delay_sched"  : delay_sched.state_dict(),
                "test_acc"     : test_acc,
                "thea1"        : thea1,
                "thea2"        : thea2,
            }, os.path.join(results_dir, f"{run_name}_best.pt"))

        print(
            f"  [{run_name}][{epoch+1:4d}/{cfg.epochs}] "
            f"Loss: {avg_loss:.6f} | "
            f"Tr: {train_acc*100:.2f}% | "
            f"Te: {test_acc*100:.2f}% | "
            f"Best: {best_acc*100:.2f}% | "
            f"LR: {current_lr:.5f} | "
            f"Mem: {peak_mem:.1f}MB | "
            f"t: {epoch_time:.1f}s | "
            f"Caps: [{thea1:.0f},{thea2:.0f}] | "
            f"d1_p95: {d1_p95:.2f} | "
            f"d2_p95: {d2_p95:.2f}"
        )

        logger.log_epoch(
            run_name, epoch + 1,
            avg_loss, train_acc, test_acc,
            peak_mem, epoch_time
        )

        if (cfg.save_checkpoints and
                (epoch + 1) % cfg.checkpoint_every == 0):
            torch.save({
                "epoch"        : epoch + 1,
                "run_name"     : run_name,
                "network_size" : network_size,
                "model_state"  : model.state_dict(),
                "optimizer"    : optimizer.state_dict(),
                "scheduler"    : scheduler.state_dict(),
                "delay_sched"  : delay_sched.state_dict(),
                "test_acc"     : test_acc,
                "thea1"        : thea1,
                "thea2"        : thea2,
            }, os.path.join(
                results_dir,
                f"{run_name}_epoch{epoch+1}.pt"
            ))

    summary = {
        "run_name"                : run_name,
        "model_name"              : cfg.model_name,
        "network_size"            : network_size,
        "n_hidden"                : cfg.n_hidden,
        "tap_strategy"            : "power_law",
        "storage"                 : "ring",
        "gamma"                   : cfg.gamma,
        "taps"                    : cfg.taps,
        "n_taps"                  : len(cfg.taps),
        "buffer_size"             : cfg.buffer_size,
        "buffer_max_lag"          : cfg.buffer_max_lag,
        "best_acc"                : float(best_acc),
        "n_params"                : n_params,
        "param_mem_mb"            : float(param_mb),
        "buffer_k"                : buf_info["k"],
        "buffer_L"                : buf_info["L"],
        "buffer_taps"             : buf_info["taps"],
        "total_train_time_seconds": float(sum(epoch_times)),
        "train_losses"            : [float(x) for x in train_losses],
        "train_accs"              : [float(x) for x in train_accs],
        "test_accs"               : [float(x) for x in test_accs],
        "epoch_times"             : [float(x) for x in epoch_times],
        "peak_mems"               : [float(x) for x in peak_mems],
        "final_thea1"             : float(delay_sched.thea1),
        "final_thea2"             : float(delay_sched.thea2),
        "config"                  : config_snapshot,
    }
    summary_path = os.path.join(
        results_dir, f"{run_name}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.save()

    print(f"\n{'='*60}")
    print(f"  Run            : {run_name}")
    print(f"  Network size   : {network_size}  (n_hidden={cfg.n_hidden})")
    print(f"  Gamma          : {cfg.gamma}")
    print(f"  Taps (n={len(cfg.taps)})    : {cfg.taps}")
    print(f"  Best accuracy  : {best_acc*100:.2f}%")
    print(f"  Results dir    : {results_dir}")
    print(f"  Summary saved  : {summary_path}")
    print(f"  Final caps     : [{delay_sched.thea1:.0f},"
          f"{delay_sched.thea2:.0f}]")
    print(f"{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PowerLawTapAADNetSNNTorch — ablation ready"
    )
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--datasets-path",  type=str,   default=None)
    parser.add_argument("--base-dir",       type=str,   default=None)
    parser.add_argument("--resume",         type=str,   default=None)
    parser.add_argument("--buffer-size",    type=int,   default=None)
    parser.add_argument("--buffer-max-lag", type=int,   default=None,
                        help="max_lag for the ring buffer and tap generation. "
                             "MUST be passed explicitly from the PBS script.")
    parser.add_argument("--num-taps",       type=int,   default=None,
                        help="n — number of taps (2..9).")
    parser.add_argument("--network-size",   type=str,   default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--gamma",          type=float, default=None,
                        help="Power law exponent (default from config = 2.0)")
    parser.add_argument("--theta-u",        type=float, default=None)
    args = parser.parse_args()

    cfg = get_config("power_law_tap_snntorch")

    # ── Step 1: basic overrides ───────────────────────────────────────
    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.datasets_path is not None: cfg.datasets_path = args.datasets_path
    if args.buffer_size   is not None: cfg.buffer_size   = args.buffer_size
    if args.theta_u       is not None: cfg.theta_u       = args.theta_u

    # ── Step 2: network size ──────────────────────────────────────────
    cfg.n_hidden = NETWORK_SIZES[args.network_size]

    # ── Step 3: gamma ─────────────────────────────────────────────────
    if args.gamma is not None:
        cfg.gamma = args.gamma

    # ── Step 4: max_lag — MUST be set before tap computation ─────────
    # The PBS script passes --buffer-max-lag explicitly so we always
    # use that value. Never auto-compute from n here — that was the bug.
    if args.buffer_max_lag is not None:
        cfg.buffer_max_lag = args.buffer_max_lag
        print(f"  [max_lag] buffer_max_lag={cfg.buffer_max_lag}")
    else:
        # fallback for production runs without --buffer-max-lag
        print(f"  [max_lag] using config default: {cfg.buffer_max_lag}")

    # ── Step 5: taps — computed AFTER max_lag is finalised ────────────
    if args.num_taps is not None:
        cfg.taps = make_power_law_taps(
            args.num_taps, cfg.buffer_max_lag, cfg.gamma
        )
        print(f"  [tap gen] n={args.num_taps} gamma={cfg.gamma} "
              f"max_lag={cfg.buffer_max_lag} → taps={cfg.taps}")

    # ── Step 6: resolve output directory ─────────────────────────────
    run_name = make_run_name(cfg, args.network_size)

    if args.base_dir is not None:
        results_dir = resolve_results_dir(args.base_dir, run_name)
    else:
        default_base = os.path.join(
            cfg.results_dir, "power_law_ring_buffer_ablation"
        )
        results_dir = resolve_results_dir(default_base, run_name)

    train(cfg, results_dir, args.network_size, resume_path=args.resume)


if __name__ == "__main__":
    main()
    