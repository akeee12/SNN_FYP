# training/train_dcls.py
"""
Standalone DCLS training script.
Run independently as a PBS job.
Results saved to: --results-dir (default: results/SNN_FYP/dcls/)
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
    from dcls_model import build_dcls_model
    DCLS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] DCLS not available: {e}")
    DCLS_AVAILABLE = False


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
    if not DCLS_AVAILABLE:
        print("[ERROR] DCLS not available — exiting.")
        return None

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

    model     = build_dcls_model(cfg).to(device)
    param_mem = get_param_mb(model)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"  Parameters   : {n_params:,}")
    print(f"  Param memory : {param_mem:.2f} MB")

    train_loader, test_loader = get_loaders(cfg)
    logger = ResultsLogger(results_dir)

    # ── Dual optimisers ───────────────────────────────────────────────────
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
        reset_gpu()
        t_start    = time.time()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for spikes, labels in train_loader:
            spikes = spikes.to(device)
            spikes = spikes.permute(0, 2, 1)   # (B,T,700)→(B,700,T)
            labels = labels.to(device)

            optimizer_w.zero_grad()
            optimizer_p.zero_grad()
            model.reset()

            output = model(spikes)             # (B, 20) raw logits
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

        # ── Evaluation ────────────────────────────────────────────────────
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for spikes, labels in test_loader:
                spikes = spikes.to(device)
                spikes = spikes.permute(0, 2, 1)
                labels = labels.to(device)
                model.reset()
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
              f"Loss: {avg_loss:.4f} | "
              f"TrainAcc: {train_acc*100:.2f}% | "
              f"TestAcc: {test_acc*100:.2f}% | "
              f"PeakMem: {peak_mem:.1f}MB | "
              f"Time: {epoch_time:.1f}s | "
              f"Sigma: {sigma:.3f}")

        logger.log_epoch(cfg.model_name, epoch+1, avg_loss,
                         train_acc, test_acc, peak_mem, epoch_time)

        if cfg.save_checkpoints and (epoch+1) % cfg.checkpoint_every == 0:
            torch.save({
                "epoch"      : epoch+1,
                "model_state": model.state_dict(),
                "optimizer_w": optimizer_w.state_dict(),
                "optimizer_p": optimizer_p.state_dict(),
                "test_acc"   : test_acc,
                "sigma"      : sigma,
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_epoch{epoch+1}.pt"))

    # ── Save summary JSON ─────────────────────────────────────────────────
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
        description="Train DCLS on SHD")
    parser.add_argument("--epochs",        type=int, default=None)
    parser.add_argument("--results-dir",   type=str, default=None)
    parser.add_argument("--datasets-path", type=str, default=None)
    args = parser.parse_args()

    cfg = get_config("dcls")
    if args.epochs        is not None: cfg.epochs        = args.epochs
    if args.datasets_path is not None: cfg.datasets_path = args.datasets_path

    # Default results dir: results/SNN_FYP/dcls/
    results_dir = args.results_dir or os.path.join(
        cfg.results_dir, "dcls")

    train(cfg, results_dir)


if __name__ == "__main__":
    main()