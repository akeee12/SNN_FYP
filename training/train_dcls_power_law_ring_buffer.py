# training/train_dcls_power_law_ring_buffer.py
"""
Training script for DCLSPowerLawRingBufferNet on SHD / SSC / DVS.

Identical control flow to train_dcls_log_ring_buffer.py with these changes:
    1. Imports DCLSPowerLawRingBufferNet instead of DCLSLogRingBufferNet.
    2. Config key is "dcls_power_law_ring_buffer".
    3. Config snapshot includes gamma + power-law tap hyperparameters.

Usage:
    python train_dcls_power_law_ring_buffer.py --dataset shd
    python train_dcls_power_law_ring_buffer.py --dataset ssc --epochs 200
    python train_dcls_power_law_ring_buffer.py --dataset dvs --gamma 1.5
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
from experiments.results_logger import ResultsLogger

try:
    from dcls_power_law_ring_buffer_model import DCLSPowerLawRingBufferNet
    DCLS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] DCLS not available: {e}")
    DCLS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# VERSIONING
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# DATASETS  (identical to train_dcls_log_ring_buffer.py)
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


class SSCDataset(Dataset):
    def __init__(self, path, split="train",
                 time_steps=400, n_inputs=700, max_time=1.0):
        self.time_steps = time_steps
        self.n_inputs   = n_inputs
        self.max_time   = max_time
        self.hdf5_path  = os.path.join(path, f"ssc_{split}.h5")
        print(f"  Loading SSC {split} from {self.hdf5_path}")
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


class DVSGestureDataset(Dataset):
    GRID  = 32
    N_POL = 2

    def __init__(self, path, split="train",
                 time_steps=300, n_inputs=2048,
                 max_time=3704997.0):
        self.time_steps = time_steps
        self.n_inputs   = n_inputs
        self.max_time   = max_time
        self.hdf5_path  = os.path.join(path, f"dvs_{split}.h5")
        print(f"  Loading DVS Gesture {split} from {self.hdf5_path}")
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


DATASET_DEFAULTS = {
    "shd": dict(n_inputs=700,  n_outputs=20,
                time_steps=400, max_time=1.4),
    "ssc": dict(n_inputs=700,  n_outputs=35,
                time_steps=400, max_time=1.0),
    "dvs": dict(n_inputs=2048, n_outputs=11,
                time_steps=300, max_time=3704997.0),
}

DATASET_CLASSES = {
    "shd": SHDDataset,
    "ssc": SSCDataset,
    "dvs": DVSGestureDataset,
}

DATASET_SPLITS = {
    "shd": ("train", "test"),
    "ssc": ("train", "test"),
    "dvs": ("train", "test"),
}


def get_loaders(cfg, dataset_name: str):
    dataset_name            = dataset_name.lower()
    cls                     = DATASET_CLASSES[dataset_name]
    defaults                = DATASET_DEFAULTS[dataset_name]
    train_split, test_split = DATASET_SPLITS[dataset_name]

    train_ds = cls(
        cfg.datasets_path, train_split,
        defaults["time_steps"],
        defaults["n_inputs"],
        defaults["max_time"],
    )
    test_ds = cls(
        cfg.datasets_path, test_split,
        defaults["time_steps"],
        defaults["n_inputs"],
        defaults["max_time"],
    )

    return (
        DataLoader(train_ds,
                   batch_size  = cfg.batch_size,
                   shuffle     = True,
                   num_workers = cfg.num_workers,
                   pin_memory  = True),
        DataLoader(test_ds,
                   batch_size  = cfg.batch_size,
                   shuffle     = False,
                   num_workers = cfg.num_workers,
                   pin_memory  = True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_param_mb(model):
    return sum(p.numel() * p.element_size()
               for p in model.parameters()) / (1024 ** 2)

def get_peak_gpu_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2) \
           if torch.cuda.is_available() else 0.0

def reset_gpu():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, base_dir, dataset_name: str, resume_path=None):
    if not DCLS_AVAILABLE:
        print("[ERROR] DCLS not available — exiting.")
        return None

    # ── Versioning ────────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        results_dir = os.path.dirname(resume_path)
        version     = int(os.path.basename(results_dir)[1:])
    else:
        results_dir, version = get_versioned_dir(base_dir)

    os.makedirs(results_dir, exist_ok=True)

    # ── Apply dataset-specific overrides ─────────────────────────────
    defaults       = DATASET_DEFAULTS[dataset_name]
    cfg.n_inputs   = defaults["n_inputs"]
    cfg.n_outputs  = defaults["n_outputs"]
    cfg.time_steps = defaults["time_steps"]

    buf_info_str = (f"k={cfg.buffer_size} L={cfg.n_taps} "
                    f"taps={cfg.taps} max_lag={cfg.buffer_max_lag} "
                    f"gamma={cfg.gamma}")

    print(f"\n{'='*60}")
    print(f"  Model        : {cfg.model_name}")
    print(f"  Dataset      : {dataset_name.upper()}")
    print(f"  n_inputs     : {cfg.n_inputs}")
    print(f"  n_outputs    : {cfg.n_outputs}")
    print(f"  time_steps   : {cfg.time_steps}")
    print(f"  Buffer       : {buf_info_str}")
    print(f"  theta_u      : {cfg.theta_u}")
    print(f"  Version      : v{version}")
    print(f"  Results      : {results_dir}")
    print(f"  Epochs       : {cfg.epochs}")
    print(f"  lr_w         : {cfg.lr_w}")
    print(f"  lr_pos       : {cfg.lr_pos}")
    print(f"  Batch        : {cfg.batch_size}")
    print(f"  max_delay    : {cfg.max_delay}")
    print(f"  v_threshold  : {cfg.v_threshold}")
    print(f"{'='*60}\n")

    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"  Using device : {device}\n")

    model     = DCLSPowerLawRingBufferNet(cfg).to(device)
    param_mem = get_param_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    buf_info  = model.get_buffer_info()

    print(f"  Parameters   : {n_params:,}")
    print(f"  Memory       : {param_mem:.4f} MB")
    print(f"  Buffer k     : {buf_info['k']}")
    print(f"  Buffer L     : {buf_info['L']}")
    print(f"  Buffer taps  : {buf_info['taps']}")
    print(f"  Storage      : {buf_info['storage']}\n")

    train_loader, test_loader = get_loaders(cfg, dataset_name)
    logger = ResultsLogger(results_dir)

    # ── Dual optimisers ───────────────────────────────────────────────
    delay_params  = model.get_delay_params()
    weight_params = model.get_weight_params()

    optimizer_w = torch.optim.Adam(
        weight_params,
        lr           = cfg.lr_w,
        weight_decay = cfg.weight_decay,
    )
    optimizer_p = torch.optim.Adam(
        delay_params,
        lr           = cfg.lr_pos,
        weight_decay = 0,
    )
    scheduler_w = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_w,
        max_lr          = cfg.max_lr_w,
        epochs          = cfg.epochs,
        steps_per_epoch = len(train_loader),
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc    = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from : {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer_w.load_state_dict(ckpt["optimizer_w"])
        optimizer_p.load_state_dict(ckpt["optimizer_p"])
        if "scheduler_w" in ckpt:
            scheduler_w.load_state_dict(ckpt["scheduler_w"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc    = ckpt.get("test_acc", 0.0)
        model.current_epoch = start_epoch
        print(f"  Resumed epoch {start_epoch}, "
              f"best={best_acc*100:.2f}%\n")

    # ── Config snapshot ───────────────────────────────────────────────
    config_snapshot = {
        "model_name"      : cfg.model_name,
        "dataset"         : dataset_name,
        "n_inputs"        : cfg.n_inputs,
        "n_outputs"       : cfg.n_outputs,
        "time_steps"      : cfg.time_steps,
        "version"         : version,
        "buffer_size"     : cfg.buffer_size,
        "buffer_max_lag"  : cfg.buffer_max_lag,
        "taps"            : cfg.taps,
        "n_taps"          : cfg.n_taps,
        "gamma"           : cfg.gamma,
        "theta_u"         : cfg.theta_u,
        "n_hidden_neurons": cfg.n_hidden_neurons,
        "max_delay"       : cfg.max_delay,
        "v_threshold"     : cfg.v_threshold,
        "lr_w"            : cfg.lr_w,
        "lr_pos"          : cfg.lr_pos,
        "weight_decay"    : cfg.weight_decay,
        "dropout_p"       : cfg.dropout_p,
        "batch_size"      : cfg.batch_size,
        "epochs"          : cfg.epochs,
        "seed"            : cfg.seed,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    train_losses, train_accs, test_accs = [], [], []
    epoch_times, peak_mems = [], []

    # ── Main loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        reset_gpu()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            # DCLS expects (B, n_inputs, T) — permute from (B, T, n_inputs)
            spikes = spikes.to(device).permute(0, 2, 1)
            labels = labels.to(device)

            optimizer_w.zero_grad()
            optimizer_p.zero_grad()
            model.reset()

            output = model(spikes)                    # (B, n_outputs)
            loss   = F.cross_entropy(output, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(delay_params,  max_norm=1.0)

            optimizer_w.step()
            optimizer_p.step()
            scheduler_w.step()

            epoch_loss += loss.item()
            correct    += (output.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        sigma = model.update_sigma()

        # ── Eval ──────────────────────────────────────────────────────
        model.eval()
        test_correct = 0
        test_total   = 0
        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device).permute(0, 2, 1)
                labels = labels.to(device)
                model.reset()
                output        = model(spikes)
                test_correct += (output.argmax(1) == labels).sum().item()
                test_total   += labels.size(0)

        # ── Metrics ───────────────────────────────────────────────────
        t_end      = time.time()
        peak_mem   = get_peak_gpu_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total
        test_acc   = test_correct / test_total
        avg_loss   = epoch_loss / len(train_loader)

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
                "dataset"    : dataset_name,
                "model_state": model.state_dict(),
                "optimizer_w": optimizer_w.state_dict(),
                "optimizer_p": optimizer_p.state_dict(),
                "scheduler_w": scheduler_w.state_dict(),
                "test_acc"   : test_acc,
                "sigma"      : sigma,
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_best.pt"))

        print(
            f"  [v{version}][{dataset_name.upper()}]"
            f"[{epoch+1:4d}/{cfg.epochs}] "
            f"Loss: {avg_loss:.4f} | "
            f"Tr: {train_acc*100:.2f}% | "
            f"Te: {test_acc*100:.2f}% | "
            f"Best: {best_acc*100:.2f}% | "
            f"Mem: {peak_mem:.1f}MB | "
            f"t: {epoch_time:.1f}s | "
            f"σ: {sigma:.3f}"
        )

        logger.log_epoch(
            cfg.model_name, epoch + 1, avg_loss,
            train_acc, test_acc, peak_mem, epoch_time
        )

        if cfg.save_checkpoints and \
                (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save({
                "epoch"      : epoch + 1,
                "version"    : version,
                "dataset"    : dataset_name,
                "model_state": model.state_dict(),
                "optimizer_w": optimizer_w.state_dict(),
                "optimizer_p": optimizer_p.state_dict(),
                "scheduler_w": scheduler_w.state_dict(),
                "test_acc"   : test_acc,
                "sigma"      : sigma,
            }, os.path.join(
                results_dir,
                f"{cfg.model_name}_epoch{epoch+1}.pt"
            ))

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "model_name"              : cfg.model_name,
        "dataset"                 : dataset_name,
        "n_inputs"                : cfg.n_inputs,
        "n_outputs"               : cfg.n_outputs,
        "time_steps"              : cfg.time_steps,
        "version"                 : version,
        "best_acc"                : float(best_acc),
        "n_params"                : n_params,
        "param_mem_mb"            : float(param_mem),
        "buffer_k"                : buf_info["k"],
        "buffer_L"                : buf_info["L"],
        "buffer_taps"             : buf_info["taps"],
        "gamma"                   : cfg.gamma,
        "total_train_time_seconds": float(sum(epoch_times)),
        "train_losses"            : [float(x) for x in train_losses],
        "train_accs"              : [float(x) for x in train_accs],
        "test_accs"               : [float(x) for x in test_accs],
        "epoch_times"             : [float(x) for x in epoch_times],
        "peak_mems"               : [float(x) for x in peak_mems],
        "config"                  : config_snapshot,
    }
    summary_path = os.path.join(
        results_dir, f"{cfg.model_name}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.save()

    print(f"\n{'='*60}")
    print(f"  Version        : v{version}")
    print(f"  Dataset        : {dataset_name.upper()}")
    print(f"  Best accuracy  : {best_acc*100:.2f}%")
    print(f"  Results dir    : {results_dir}")
    print(f"  Summary saved  : {summary_path}")
    print(f"{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train DCLSPowerLawRingBufferNet on SHD/SSC/DVS"
    )
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--datasets-path", type=str,   default=None)
    parser.add_argument("--base-dir",      type=str,   default=None)
    parser.add_argument("--resume",        type=str,   default=None)
    parser.add_argument("--dataset",       type=str,   default="shd",
                        choices=["shd", "ssc", "dvs"],
                        help="Dataset to train on (default: shd)")
    parser.add_argument("--buffer-size",   type=int,   default=None,
                        help="k — bottleneck width")
    parser.add_argument("--theta-u",       type=float, default=None)
    parser.add_argument("--gamma",         type=float, default=None,
                        help="Power-law exponent (default: 2.0)")
    parser.add_argument("--n-taps",        type=int,   default=None,
                        help="Number of taps (default: 7)")
    args = parser.parse_args()

    cfg = get_config("dcls_power_law_ring_buffer")

    if args.epochs        is not None: cfg.epochs         = args.epochs
    if args.datasets_path is not None: cfg.datasets_path  = args.datasets_path
    if args.buffer_size   is not None: cfg.buffer_size    = args.buffer_size
    if args.theta_u       is not None: cfg.theta_u        = args.theta_u
    if args.gamma         is not None:
        cfg.gamma  = args.gamma
        cfg.taps   = cfg._compute_power_law_taps()
        cfg.n_taps = len(cfg.taps)
    if args.n_taps        is not None:
        cfg.n_taps_target = args.n_taps
        cfg.taps          = cfg._compute_power_law_taps()
        cfg.n_taps        = len(cfg.taps)

    base_dir = args.base_dir or os.path.join(
        cfg.results_dir, f"dcls_power_law_ring_buffer_{args.dataset}"
    )

    train(cfg, base_dir, dataset_name=args.dataset,
          resume_path=args.resume)


if __name__ == "__main__":
    main()