"""
Training script for PowerLawTapAADNetSNNTorch on DVS Gesture.
Ablation-ready: supports --num-taps (n=2..9), --network-size, --gamma,
and --buffer-max-lag override.

DVS Gesture:
    2048 input channels (32x32x2 downsampled from 128x128x2)
    11 classes, time_steps=300, max_time=3704997.0 microseconds
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from configs import get_config
from utils.seed import set_seed
from experiments.results_logger import ResultsLogger
from power_law_tap_aad_snntorch import PowerLawTapAADNetSNNTorch

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAME = "dvs"
N_INPUTS     = 2048
N_OUTPUTS    = 11
TIME_STEPS   = 300
MAX_TIME     = 3704997.0

NETWORK_SIZES = {
    "small" : 64,
    "medium": 128,
    "large" : 256,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DVSGestureDataset(Dataset):
    GRID  = 32
    N_POL = 2

    def __init__(self, path, split="train",
                 time_steps=TIME_STEPS, n_inputs=N_INPUTS,
                 max_time=MAX_TIME):
        self.time_steps = time_steps
        self.n_inputs   = n_inputs
        self.max_time   = max_time
        self.hdf5_path  = os.path.join(path, f"dvs_{split}.h5")
        print(f"  Loading DVS {split} from {self.hdf5_path}")
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
        label  = int(self.labels[idx])
        t_idx  = np.floor(
            times / self.max_time * self.time_steps).astype(int)
        t_idx  = np.clip(t_idx, 0, self.time_steps - 1)
        units  = units.astype(int)
        p      = units // (128 * 128)
        rem    = units %  (128 * 128)
        y      = rem   // 128
        x      = rem   %  128
        x_ds   = x // 4
        y_ds   = y // 4
        p      = np.clip(p, 0, self.N_POL - 1)
        u_idx  = (p * self.GRID * self.GRID
                  + y_ds * self.GRID + x_ds)
        u_idx  = np.clip(u_idx, 0, self.n_inputs - 1)
        spikes = torch.zeros(self.time_steps, self.n_inputs,
                             dtype=torch.float32)
        spikes[t_idx, u_idx] = 1.0
        return spikes, label


def get_loaders(cfg):
    return (
        DataLoader(
            DVSGestureDataset(cfg.datasets_path, "train"),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
        ),
        DataLoader(
            DVSGestureDataset(cfg.datasets_path, "test"),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAP GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_power_law_taps(n, max_lag, gamma):
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
            f"Duplicate taps: {taps}  n={n} max_lag={max_lag} gamma={gamma}"
        )
    return taps


def max_lag_for_n(n):
    return 2 ** (n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# RUN NAMING
# ─────────────────────────────────────────────────────────────────────────────

def make_run_name(cfg, network_size):
    n = len(cfg.taps)
    return (f"k{cfg.buffer_size}_n{n}"
            f"_lag{cfg.buffer_max_lag}"
            f"_g{cfg.gamma:.1f}"
            f"_{network_size}")


# ─────────────────────────────────────────────────────────────────────────────
# VERSIONING
# ─────────────────────────────────────────────────────────────────────────────

def get_run_dir(base_dir, run_name):
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_versioned_dir(base_dir):
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


def resolve_results_dir(base_dir, run_name):
    base = base_dir.rstrip("/")
    bn   = os.path.basename(base)
    if bn.startswith("v") and bn[1:].isdigit():
        return get_run_dir(base, run_name)
    else:
        version_dir, _ = get_versioned_dir(base)
        return get_run_dir(version_dir, run_name)


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
            "thea1": self.thea1, "thea2": self.thea2,
            "update1": self.update1, "update2": self.update2,
        }

    def load_state_dict(self, state):
        self.thea1   = state["thea1"];  self.thea2   = state["thea2"]
        self.update1 = state["update1"]; self.update2 = state["update2"]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, results_dir, network_size, resume_path=None):
    os.makedirs(results_dir, exist_ok=True)
    run_name = make_run_name(cfg, network_size)

    # Apply DVS overrides
    cfg.n_inputs   = N_INPUTS
    cfg.n_outputs  = N_OUTPUTS
    cfg.time_steps = TIME_STEPS

    print(f"\n{'='*60}")
    print(f"  Run           : {run_name}")
    print(f"  Dataset       : DVS Gesture")
    print(f"  n_inputs      : {cfg.n_inputs}")
    print(f"  n_outputs     : {cfg.n_outputs}")
    print(f"  time_steps    : {cfg.time_steps}")
    print(f"  Strategy      : power_law  (ring buffer)")
    print(f"  Gamma         : {cfg.gamma}")
    print(f"  Network size  : {network_size}  (n_hidden={cfg.n_hidden})")
    print(f"  Taps (n={len(cfg.taps):2d}) : {cfg.taps}")
    print(f"  buffer_size   : {cfg.buffer_size}")
    print(f"  buffer_max_lag: {cfg.buffer_max_lag}")
    print(f"  Results dir   : {results_dir}")
    print(f"  Epochs        : {cfg.epochs}")
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
    print(f"  Buffer taps   : {buf_info['taps']}\n")

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
        "dataset"           : DATASET_NAME,
        "n_inputs"          : N_INPUTS,
        "n_outputs"         : N_OUTPUTS,
        "time_steps"        : TIME_STEPS,
        "network_size"      : network_size,
        "n_hidden"          : cfg.n_hidden,
        "tap_strategy"      : "power_law",
        "storage"           : "ring",
        "gamma"             : cfg.gamma,
        "taps"              : cfg.taps,
        "n_taps"            : len(cfg.taps),
        "buffer_size"       : cfg.buffer_size,
        "buffer_max_lag"    : cfg.buffer_max_lag,
        "lr"                : cfg.lr,
        "batch_size"        : cfg.batch_size,
        "epochs"            : cfg.epochs,
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
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()
            pred        = model.predict(output)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)
            epoch_loss += loss.item()

        delay_sched.step(model, epoch)
        scheduler.step()

        model.eval()
        test_correct = test_total = 0
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
        stats        = model.get_delay_stats()

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
                "dataset"      : DATASET_NAME,
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
            f"  [{run_name}][DVS][{epoch+1:4d}/{cfg.epochs}] "
            f"Loss: {avg_loss:.6f} | "
            f"Tr: {train_acc*100:.2f}% | "
            f"Te: {test_acc*100:.2f}% | "
            f"Best: {best_acc*100:.2f}% | "
            f"LR: {current_lr:.5f} | "
            f"Mem: {peak_mem:.1f}MB | "
            f"t: {epoch_time:.1f}s | "
            f"Caps: [{thea1:.0f},{thea2:.0f}] | "
            f"d1_p95: {stats['delay1']['p95']:.2f} | "
            f"d2_p95: {stats['delay2']['p95']:.2f}"
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
                "dataset"      : DATASET_NAME,
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
        "dataset"                 : DATASET_NAME,
        "n_inputs"                : N_INPUTS,
        "n_outputs"               : N_OUTPUTS,
        "time_steps"              : TIME_STEPS,
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
    print(f"  Dataset        : DVS Gesture")
    print(f"  Best accuracy  : {best_acc*100:.2f}%")
    print(f"  Summary saved  : {summary_path}")
    print(f"{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PowerLawTapAADNetSNNTorch on DVS Gesture — ablation"
    )
    parser.add_argument("--epochs",         type=int,   default=None)
    parser.add_argument("--datasets-path",  type=str,   default=None)
    parser.add_argument("--base-dir",       type=str,   default=None)
    parser.add_argument("--resume",         type=str,   default=None)
    parser.add_argument("--buffer-size",    type=int,   default=None)
    parser.add_argument("--buffer-max-lag", type=int,   default=None)
    parser.add_argument("--num-taps",       type=int,   default=None)
    parser.add_argument("--network-size",   type=str,   default="medium",
                        choices=["small", "medium", "large"])
    parser.add_argument("--gamma",          type=float, default=None)
    parser.add_argument("--theta-u",        type=float, default=None)
    args = parser.parse_args()

    cfg = get_config("power_law_tap_snntorch")

    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.datasets_path is not None: cfg.datasets_path = args.datasets_path
    if args.buffer_size   is not None: cfg.buffer_size   = args.buffer_size
    if args.theta_u       is not None: cfg.theta_u       = args.theta_u

    cfg.n_hidden = NETWORK_SIZES[args.network_size]

    if args.gamma is not None:
        cfg.gamma = args.gamma

    if args.num_taps is not None:
        cfg.buffer_max_lag = max_lag_for_n(args.num_taps)
        cfg.taps = make_power_law_taps(
            args.num_taps, cfg.buffer_max_lag, cfg.gamma
        )
        print(f"  [tap gen] n={args.num_taps} gamma={cfg.gamma} -> "
              f"taps={cfg.taps}  max_lag={cfg.buffer_max_lag}")

    # buffer_max_lag override — must come after num_taps
    if args.buffer_max_lag is not None:
        cfg.buffer_max_lag = args.buffer_max_lag
        print(f"  [max_lag override] buffer_max_lag={cfg.buffer_max_lag}")

    run_name = make_run_name(cfg, args.network_size)

    if args.base_dir is not None:
        results_dir = resolve_results_dir(args.base_dir, run_name)
    else:
        default_base = os.path.join(
            cfg.results_dir, "power_law_ring_buffer_ablation_dvs"
        )
        results_dir = resolve_results_dir(default_base, run_name)

    train(cfg, results_dir, args.network_size, resume_path=args.resume)


if __name__ == "__main__":
    main()