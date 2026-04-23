# training/train.py
"""
Unified benchmark training harness.
Trains AdaptiveAxonalDelay (PyTorch) and DCLS on SHD.

Fixes applied v2:
    1. AAD uses spike rate MSE loss (original paper)
    2. DCLS uses CrossEntropy on raw FC logits (no lif_out overflow)
    3. plot_results confirmed at module level
    4. AAD delay_adapt_start reduced to work within 100 epochs

    Fixes:
1. Single shared SHDDataset for both models
2. Unified CrossEntropy loss on spike counts for both
3. Gradient clipping added to both runners
4. Same time binning, same preprocessing for fair comparison
5. DCLS input permuted (B,T,700) → (B,700,T)
6. DCLS uses cfg.time_steps not int(1000/time_step)

NOTE on loss:
AAD paper uses spike-rate MSE loss.
DCLS paper uses CrossEntropy on summed spike counts.
We use CrossEntropy for BOTH for fair comparison.
Documented deviation — justified in report.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset, DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from configs import get_config, list_models
from utils.seed import set_seed
from experiments.results_logger import ResultsLogger

# ── Model imports ─────────────────────────────────────────────────────────────
from axonal_delay_modelPYTorch import AdaptiveAxonalDelayNet

try:
    from dcls_model import build_dcls_model
    DCLS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] DCLS not available: {e}")
    DCLS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATASET — lazy loading to avoid OOM on login node
# ─────────────────────────────────────────────────────────────────────────────

class SHDDataset(Dataset):
    """
    Spiking Heidelberg Digits — lazy HDF5 loader.
    Only loads labels at init — reads spikes on demand.
    Prevents OOM on nodes with limited RAM.
    """

    def __init__(
        self,
        path      : str,
        split     : str   = "train",
        time_steps: int   = 400,
        n_inputs  : int   = 700,
        max_time  : float = 1.4,
    ):
        self.time_steps = time_steps
        self.n_inputs   = n_inputs
        self.max_time   = max_time
        self.hdf5_path  = os.path.join(path, f"shd_{split}.h5")

        print(f"  Loading SHD {split} from {self.hdf5_path}")

        # Only load labels at init — tiny memory footprint
        with h5py.File(self.hdf5_path, "r") as f:
            self.labels    = f["labels"][:]
            self.n_samples = len(self.labels)

        print(f"  {split}: {self.n_samples} samples, "
              f"{len(set(self.labels))} classes")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Read one sample at a time from HDF5
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
            self.time_steps, self.n_inputs, dtype=torch.float32)
        spikes[t_idx, u_idx] = 1.0

        return spikes, label


def get_loaders(cfg, time_steps: int) -> tuple:
    """Single shared loader for both models."""
    train_loader = DataLoader(
        SHDDataset(cfg.datasets_path, "train", time_steps),
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        SHDDataset(cfg.datasets_path, "test", time_steps),
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_param_memory_mb(model: nn.Module) -> float:
    return sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / (1024 ** 2)


def get_peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER 1 — ADAPTIVE AXONAL DELAY (PYTORCH)
# ─────────────────────────────────────────────────────────────────────────────

def run_aad_pytorch(cfg, logger: ResultsLogger) -> dict:
    """
    Trains AdaptiveAxonalDelayNet (pure PyTorch) on SHD.

    Loss: Spike rate MSE — original paper loss
        target_true  = 0.8 for correct class
        target_false = 0.0 for all others
    Optimiser: NAdam lr=0.001 (reduced from 0.1)
    """
    print(f"\n{'='*60}")
    print(f"  Training : {cfg.model_name}")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Device   : {cfg.device}")
    print(f"{'='*60}")

    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = AdaptiveAxonalDelayNet(cfg).to(device)
    param_mem = get_param_memory_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {n_params:,}")
    print(f"  Param memory : {param_mem:.2f} MB")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(cfg, cfg.time_steps)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.NAdam(
        model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones = cfg.lr_milestones,
        gamma      = cfg.lr_gamma,
    )

    # ── Loss: spike rate MSE — original paper loss ────────────────────────────
    def spike_rate_loss(output, labels):
        """
        MSE between normalised spike rates and target rates.
        Correct class target  = cfg.target_true  (0.8)
        All other classes     = cfg.target_false (0.0)
        """
        B         = output.shape[0]
        n_classes = output.shape[1]
        # Normalise spike counts to rates
        rates  = output / cfg.time_steps
        target = torch.full(
            (B, n_classes), cfg.target_false,
            dtype=torch.float32, device=device)
        target[torch.arange(B), labels] = cfg.target_true
        return F.mse_loss(rates, target)

    # ── Adaptive delay state ───────────────────────────────────────────────────
    thea1        = float(cfg.delay_init_cap)
    thea2        = float(cfg.delay_init_cap)
    update_count = 0

    train_losses, test_accs, epoch_times, peak_mems = [], [], [], []

    for epoch in range(cfg.epochs):
        model.train()
        reset_gpu_memory()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            spikes = spikes.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(spikes)                    # (B, 20)
            loss   = spike_rate_loss(output, labels)  # ✅ MSE loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds       = output.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

            # ── Adaptive delay clamping ────────────────────────────────────────
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

        # ── Evaluation ────────────────────────────────────────────────────────
        model.eval()
        test_correct = 0
        test_total   = 0
        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device)
                labels = labels.to(device)
                output = model(spikes)
                preds  = output.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)

        t_end      = time.time()
        peak_mem   = get_peak_gpu_memory_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total
        test_acc   = test_correct / test_total
        avg_loss   = epoch_loss / len(train_loader)

        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        print(f"  Epoch [{epoch+1:3d}/{cfg.epochs}] "
              f"Loss: {avg_loss:.6f} | "
              f"TrainAcc: {train_acc*100:.2f}% | "
              f"TestAcc: {test_acc*100:.2f}% | "
              f"PeakMem: {peak_mem:.1f}MB | "
              f"Time: {epoch_time:.1f}s | "
              f"Delays: [{thea1:.1f}, {thea2:.1f}]")

        logger.log_epoch(
            cfg.model_name, epoch + 1, avg_loss,
            train_acc, test_acc, peak_mem, epoch_time)

        if cfg.save_checkpoints and (epoch + 1) % cfg.checkpoint_every == 0:
            os.makedirs(cfg.results_dir, exist_ok=True)
            torch.save({
                "epoch"      : epoch + 1,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "test_acc"   : test_acc,
                "thea1"      : thea1,
                "thea2"      : thea2,
            }, os.path.join(
                cfg.results_dir,
                f"{cfg.model_name}_epoch{epoch+1}.pt"))

    return {
        "model_name"  : cfg.model_name,
        "train_losses": train_losses,
        "test_accs"   : test_accs,
        "epoch_times" : epoch_times,
        "peak_mems"   : peak_mems,
        "param_mem_mb": param_mem,
        "best_acc"    : max(test_accs),
        "n_params"    : n_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER 2 — DCLS
# ─────────────────────────────────────────────────────────────────────────────

def run_dcls(cfg, logger: ResultsLogger) -> dict:
    """
    Trains DCLS on SHD.
    Loss: CrossEntropy on raw FC logits (lif_out removed from model)
    """
    print(f"\n{'='*60}")
    print(f"  Training : {cfg.model_name}")
    print(f"  Epochs   : {cfg.epochs}")
    print(f"  Device   : {cfg.device}")
    print(f"{'='*60}")

    if not DCLS_AVAILABLE:
        print("[ERROR] DCLS not available — skipping.")
        return None

    set_seed(cfg.seed)
    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_dcls_model(cfg).to(device)
    param_mem = get_param_memory_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {n_params:,}")
    print(f"  Param memory : {param_mem:.2f} MB")

    # ── Shared loader ─────────────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(cfg, cfg.time_steps)

    # ── Dual optimisers ───────────────────────────────────────────────────────
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

    train_losses, test_accs, epoch_times, peak_mems = [], [], [], []

    for epoch in range(cfg.epochs):
        model.train()
        reset_gpu_memory()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            spikes = spikes.to(device)
            spikes = spikes.permute(0, 2, 1)    # (B,T,700) → (B,700,T)
            labels = labels.to(device)

            optimizer_w.zero_grad()
            optimizer_p.zero_grad()

            model.reset()
            output = model(spikes)              # (B, 20) raw logits
            
# output is softmax probabilities from lif_out
# CrossEntropy applies softmax AGAIN internally
# → double softmax → garbage gradients

# Fix:
            loss = F.cross_entropy(output, labels)   # ✅ raw logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(delay_params,  max_norm=1.0)

            optimizer_w.step()
            optimizer_p.step()
            scheduler_w.step()

            epoch_loss += loss.item()
            preds       = output.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        sigma = model.update_sigma()

        # ── Evaluation ────────────────────────────────────────────────────────
        model.eval()
        test_correct = 0
        test_total   = 0
        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device)
                spikes = spikes.permute(0, 2, 1)
                labels = labels.to(device)
                model.reset()
                output = model(spikes)
                preds  = output.argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)

        t_end      = time.time()
        peak_mem   = get_peak_gpu_memory_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total
        test_acc   = test_correct / test_total
        avg_loss   = epoch_loss / len(train_loader)

        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        print(f"  Epoch [{epoch+1:3d}/{cfg.epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"TrainAcc: {train_acc*100:.2f}% | "
              f"TestAcc: {test_acc*100:.2f}% | "
              f"PeakMem: {peak_mem:.1f}MB | "
              f"Time: {epoch_time:.1f}s | "
              f"Sigma: {sigma:.3f}")

        logger.log_epoch(
            cfg.model_name, epoch + 1, avg_loss,
            train_acc, test_acc, peak_mem, epoch_time)

        if cfg.save_checkpoints and (epoch + 1) % cfg.checkpoint_every == 0:
            os.makedirs(cfg.results_dir, exist_ok=True)
            torch.save({
                "epoch"      : epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_w": optimizer_w.state_dict(),
                "optimizer_p": optimizer_p.state_dict(),
                "test_acc"   : test_acc,
                "sigma"      : sigma,
            }, os.path.join(
                cfg.results_dir,
                f"{cfg.model_name}_epoch{epoch+1}.pt"))

    return {
        "model_name"  : cfg.model_name,
        "train_losses": train_losses,
        "test_accs"   : test_accs,
        "epoch_times" : epoch_times,
        "peak_mems"   : peak_mems,
        "param_mem_mb": param_mem,
        "best_acc"    : max(test_accs),
        "n_params"    : n_params,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING — defined at module level
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results: list, out_dir: str):
    """Generates three benchmark comparison plots."""
    os.makedirs(out_dir, exist_ok=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    # ── Plot 1: Training Loss vs Epochs ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(all_results):
        ax.plot(
            range(1, len(res["train_losses"]) + 1),
            res["train_losses"],
            label=res["model_name"],
            color=colors[i], linewidth=2,
        )
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss", fontsize=13)
    ax.set_title("Training Loss vs Epochs — SHD", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_loss_vs_epochs.png"), dpi=150)
    plt.close(fig)
    print("  Saved: plot_loss_vs_epochs.png")

    # ── Plot 2: Accuracy vs Memory ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, res in enumerate(all_results):
        best_acc  = res["best_acc"] * 100
        param_mem = res["param_mem_mb"]
        peak_mem  = max(res["peak_mems"])
        ax.scatter(param_mem, best_acc, s=200,
                   color=colors[i], zorder=5,
                   label=f"{res['model_name']} param")
        ax.scatter(peak_mem, best_acc, s=200,
                   color=colors[i], zorder=5, marker="^",
                   label=f"{res['model_name']} peak GPU")
        ax.annotate(res["model_name"], (param_mem, best_acc),
                    textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Memory Footprint (MB)", fontsize=13)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Memory Footprint — SHD", fontsize=15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_acc_vs_memory.png"), dpi=150)
    plt.close(fig)
    print("  Saved: plot_acc_vs_memory.png")

    # ── Plot 3: Accuracy vs Time ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(all_results):
        cum_times = np.cumsum(res["epoch_times"]) / 60
        ax.plot(cum_times,
                [a * 100 for a in res["test_accs"]],
                label=res["model_name"],
                color=colors[i], linewidth=2)
    ax.set_xlabel("Cumulative Training Time (minutes)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Test Accuracy vs Training Time — SHD", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_acc_vs_time.png"), dpi=150)
    plt.close(fig)
    print("  Saved: plot_acc_vs_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

RUNNERS = {
    "aad_pytorch" : run_aad_pytorch,
    "dcls"        : run_dcls,
}


def main():
    parser = argparse.ArgumentParser(
        description="SNN Benchmark — SHD Dataset")
    parser.add_argument(
        "--models", nargs="+",
        default=["aad_pytorch", "dcls"],
        help=f"Models to run. Available: {list(RUNNERS.keys())}",
    )
    parser.add_argument(
        "--epochs", type=int, default=None)
    parser.add_argument(
        "--results-dir", type=str, default=None)
    parser.add_argument(
        "--datasets-path", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        from configs.base_config import BenchmarkBaseConfig
        results_dir = BenchmarkBaseConfig.results_dir
    os.makedirs(results_dir, exist_ok=True)

    logger      = ResultsLogger(results_dir)
    all_results = []

    for model_name in args.models:
        if model_name not in RUNNERS:
            print(f"[WARNING] Unknown model '{model_name}' — skipping.")
            print(f"  Available: {list(RUNNERS.keys())}")
            continue

        cfg = get_config(model_name)
        if args.epochs        is not None: cfg.epochs        = args.epochs
        if args.datasets_path is not None: cfg.datasets_path = args.datasets_path
        if args.results_dir   is not None: cfg.results_dir   = args.results_dir

        result = RUNNERS[model_name](cfg, logger)

        if result is not None:
            all_results.append(result)
            with open(os.path.join(
                    results_dir,
                    f"{model_name}_summary.json"), "w") as f:
                json.dump({
                    k: ([float(x) for x in v]
                        if isinstance(v, list) else v)
                    for k, v in result.items()
                }, f, indent=2)
            print(f"\n  ✓ {model_name} best accuracy: "
                  f"{result['best_acc']*100:.2f}%")

    if all_results:
        print(f"\n{'='*60}")
        print("  Generating plots...")
        plot_results(all_results, results_dir)
        logger.save()
        print(f"  Done. Results at: {results_dir}")
    else:
        print("[ERROR] No models completed.")


if __name__ == "__main__":
    main()