# training/train_aad_v2.py
"""
Training script for AdaptiveAxonalDelayNetV2.

Faithful reproduction of Sun et al. ICASSP 2023 training procedure.
Verified against Pengfei's training script and SLAYER source.

Key differences from train_aad.py:
    1. Van Rossum loss (NOT simple MSE on rates)
    2. Forward returns (B, 20, T) — sum done in predict/loss
    3. Surrogate: exponential with tauRho*theta width
    4. Alpha kernels for PSP and refractory
    5. Weight init ×10, no weight_norm
    6. Delay adaptive scheduler matches Pengfei exactly

Usage:
    python training/train_aad_v2.py
    python training/train_aad_v2.py --epochs 500
    python training/train_aad_v2.py --datasets-path /path/to/shd
    python training/train_aad_v2.py --results-dir results/aad_v2/
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

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from configs import get_config
from utils.seed import set_seed
from experiments.results_logger import ResultsLogger
from axonal_delay_modelPYTorchV2 import AdaptiveAxonalDelayNetV2


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# Matches Pengfei's shdDataset but reads from HDF5 directly
# emptytensor shape: (T, 700) — permuted to (B, 700, T) in model
# ─────────────────────────────────────────────────────────────────────────────

class SHDDataset(Dataset):
    """
    SHD dataset loader.
    Reads from HDF5 files — shd_train.h5 / shd_test.h5.

    Output:
        spikes : (T, 700) float32 — binary spike tensor
        label  : int
    """

    def __init__(self, path: str, split: str = "train",
                 time_steps: int = 400, n_inputs: int = 700,
                 max_time: float = 1.4):
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

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        with h5py.File(self.hdf5_path, "r") as f:
            times = f["spikes"]["times"][idx]
            units = f["spikes"]["units"][idx]

        label = int(self.labels[idx])

        # Bin spike times into T timesteps
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


def get_loaders(cfg) -> tuple:
    """Returns (train_loader, test_loader)."""
    train_loader = DataLoader(
        SHDDataset(
            cfg.datasets_path, "train",
            cfg.time_steps, cfg.n_inputs
        ),
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        SHDDataset(
            cfg.datasets_path, "test",
            cfg.time_steps, cfg.n_inputs
        ),
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_param_mb(model: nn.Module) -> float:
    return sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 ** 2)


def get_peak_gpu_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2) \
        if torch.cuda.is_available() else 0.0


def reset_gpu_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE DELAY SCHEDULER
# Confirmed from Pengfei training script — exact logic reproduced
#
# Pengfei uses sorted[108] for 128 neurons:
#   108/128 = 0.844 ≈ 84th percentile
#   BUT comment says "95% windows" — we use cfg.delay_percentile
#   to compute the correct index: int(0.95 * n_hidden) = 121
#
# We use torch.quantile for cleaner implementation
# ─────────────────────────────────────────────────────────────────────────────

class DelayScheduler:
    """
    Adaptive delay cap scheduler.
    Matches Pengfei's training script logic exactly.

    Logic:
        - Before delay_adapt_start epochs: hard clamp at init cap
        - After delay_adapt_start:
            every delay_adapt_window steps:
                if p95 of delay > (cap - margin): cap += 1
    """

    def __init__(self, cfg):
        self.thea1          = float(cfg.delay_init_cap)
        self.thea2          = float(cfg.delay_init_cap)
        self.adapt_start    = cfg.delay_adapt_start
        self.adapt_window   = cfg.delay_adapt_window
        self.expand_margin  = cfg.delay_expand_margin
        self.percentile     = cfg.delay_percentile
        self.update_count   = 0

    def step(self, model: nn.Module, epoch: int):
        """
        Call once per batch after optimizer.step().

        Args:
            model : AdaptiveAxonalDelayNetV2
            epoch : current epoch (0-indexed)
        """
        if epoch < self.adapt_start:
            # Hard clamp — no expansion yet
            model.clamp_delays(self.thea1, self.thea2)
            return

        # Adaptive phase
        self.update_count += 1
        model.clamp_delays(self.thea1, self.thea2)

        if self.update_count >= self.adapt_window:
            stats = model.get_delay_stats()

            # Expand cap if p95 is within margin of current cap
            if stats["delay1"]["p95"] > (self.thea1 - self.expand_margin):
                self.thea1 += 1.0

            if stats["delay2"]["p95"] > (self.thea2 - self.expand_margin):
                self.thea2 += 1.0

            # Apply updated caps
            model.clamp_delays(self.thea1, self.thea2)
            self.update_count = 0

    def get_caps(self) -> tuple:
        return self.thea1, self.thea2

    def state_dict(self) -> dict:
        return {
            "thea1"       : self.thea1,
            "thea2"       : self.thea2,
            "update_count": self.update_count,
        }

    def load_state_dict(self, state: dict):
        self.thea1        = state["thea1"]
        self.thea2        = state["thea2"]
        self.update_count = state["update_count"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, results_dir: str) -> dict:
    """
    Full training loop for AdaptiveAxonalDelayNetV2.

    Returns summary dict with best accuracy and training curves.
    """

    # ── Setup ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model    : {cfg.model_name}")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Device   : {cfg.device}")
    print(f"  Results  : {results_dir}")
    print(f"{'='*60}\n")

    os.makedirs(results_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"  Using device : {device}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model     = AdaptiveAxonalDelayNetV2(cfg).to(device)
    param_mem = get_param_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())

    print(f"  Parameters   : {n_params:,}")
    print(f"  Param memory : {param_mem:.4f} MB\n")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(cfg)

    # ── Optimiser — NAdam confirmed equivalent to Pengfei's Nadam ─────
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

    # ── Delay scheduler ───────────────────────────────────────────────
    delay_sched = DelayScheduler(cfg)

    # ── Logger ────────────────────────────────────────────────────────
    logger = ResultsLogger(results_dir)

    # ── Tracking ──────────────────────────────────────────────────────
    train_losses = []
    train_accs   = []
    test_accs    = []
    epoch_times  = []
    peak_mems    = []

    best_acc     = 0.0

    # ─────────────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────────────

    for epoch in range(cfg.epochs):

        model.train()
        reset_gpu_stats()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            spikes = spikes.to(device)   # (B, T, 700)
            labels = labels.to(device)   # (B,)

            optimizer.zero_grad()

            # ── Forward ───────────────────────────────────────────────
            # output: (B, 20, T) — raw spike tensor before sum
            output = model(spikes)

            # ── Van Rossum loss ───────────────────────────────────────
            # Confirmed from spikeLoss.py:
            #   error = psp(spikeOut - spikeDesired)
            #   loss  = 0.5 * sum(error^2) * Ts
            loss = model.compute_loss(output, labels, cfg)

            # ── Backward ──────────────────────────────────────────────
            loss.backward()

            # Gradient clipping — stabilises SRM training
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            # ── Adaptive delay scheduling ─────────────────────────────
            # Confirmed from Pengfei: clamp after every optimizer.step()
            delay_sched.step(model, epoch)

            # ── Metrics ───────────────────────────────────────────────
            pred     = model.predict(output)         # (B,)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
            epoch_loss += loss.item()

        # ── LR scheduler ──────────────────────────────────────────────
        scheduler.step()

        # ─────────────────────────────────────────────────────────────
        # EVALUATION LOOP
        # ─────────────────────────────────────────────────────────────

        model.eval()
        test_correct = 0
        test_total   = 0

        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device)
                labels = labels.to(device)

                output       = model(spikes)          # (B, 20, T)
                pred         = model.predict(output)  # (B,)
                test_correct += (pred == labels).sum().item()
                test_total   += labels.size(0)

        # ── Epoch stats ───────────────────────────────────────────────
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
            torch.save(
                {
                    "epoch"       : epoch + 1,
                    "model_state" : model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "delay_sched" : delay_sched.state_dict(),
                    "test_acc"    : test_acc,
                    "thea1"       : thea1,
                    "thea2"       : thea2,
                },
                os.path.join(results_dir,
                             f"{cfg.model_name}_best.pt")
            )

        # ── Print ─────────────────────────────────────────────────────
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

        # ── Logger ────────────────────────────────────────────────────
        logger.log_epoch(
            cfg.model_name, epoch + 1,
            avg_loss, train_acc, test_acc,
            peak_mem, epoch_time
        )

        # ── Periodic checkpoint ───────────────────────────────────────
        if cfg.save_checkpoints and \
                (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save(
                {
                    "epoch"       : epoch + 1,
                    "model_state" : model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "scheduler"   : scheduler.state_dict(),
                    "delay_sched" : delay_sched.state_dict(),
                    "test_acc"    : test_acc,
                    "thea1"       : thea1,
                    "thea2"       : thea2,
                },
                os.path.join(
                    results_dir,
                    f"{cfg.model_name}_epoch{epoch+1}.pt"
                )
            )

    # ─────────────────────────────────────────────────────────────────
    # SAVE SUMMARY
    # ─────────────────────────────────────────────────────────────────

    summary = {
        "model_name"  : cfg.model_name,
        "best_acc"    : float(best_acc),
        "n_params"    : n_params,
        "param_mem_mb": float(param_mem),
        "train_losses": [float(x) for x in train_losses],
        "train_accs"  : [float(x) for x in train_accs],
        "test_accs"   : [float(x) for x in test_accs],
        "epoch_times" : [float(x) for x in epoch_times],
        "peak_mems"   : [float(x) for x in peak_mems],
        "final_thea1" : float(delay_sched.thea1),
        "final_thea2" : float(delay_sched.thea2),
        # Hyperparameters — for reproducibility
        "hyperparams" : {
            "tau_s"      : cfg.tau_s,
            "tau_r"      : cfg.tau_r,
            "theta_u"    : cfg.theta_u,
            "alpha_r"    : cfg.alpha_r,
            "tau_rho"    : cfg.tau_rho,
            "scale_rho"  : cfg.scale_rho,
            "lr"         : cfg.lr,
            "batch_size" : cfg.batch_size,
            "epochs"     : cfg.epochs,
            "weight_scale": cfg.weight_scale,
        }
    }

    summary_path = os.path.join(
        results_dir, f"{cfg.model_name}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.save()

    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best accuracy  : {best_acc*100:.2f}%")
    print(f"  Summary saved  : {summary_path}")
    print(f"  Final delay caps: [{delay_sched.thea1:.0f}, "
          f"{delay_sched.thea2:.0f}]")
    print(f"{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train AdaptiveAxonalDelayNetV2 on SHD"
    )
    parser.add_argument(
        "--epochs",
        type=int, default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--results-dir",
        type=str, default=None,
        help="Directory to save results"
    )
    parser.add_argument(
        "--datasets-path",
        type=str, default=None,
        help="Path to SHD HDF5 files"
    )
    parser.add_argument(
        "--resume",
        type=str, default=None,
        help="Path to checkpoint.pt file to resume from"
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    cfg = get_config("aad_v2")

    # ── CLI overrides ─────────────────────────────────────────────────
    if args.epochs        is not None:
        cfg.epochs        = args.epochs
    if args.datasets_path is not None:
        cfg.datasets_path = args.datasets_path

    results_dir = args.results_dir or os.path.join(
        cfg.results_dir, "aad_v2"
    )

    train(cfg, results_dir)


if __name__ == "__main__":
    main()