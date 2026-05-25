# training/train_log_buffer_aad.py
"""
Training script for LogBufferAADNetSNNTorch on SHD.

Identical control flow to train_aadv2.py with three changes:
  1. Imports LogBufferAADNetSNNTorch instead of
     AdaptiveAxonalDelayNetV2SNNTorch.
  2. Config key is "log_buffer_aad_snntorch" → LogBufferAADConfigSNNTorch.
  3. Config snapshot includes buffer_size, buffer_max_lag, theta_u,
     target_true, target_false.

All three FIXes from train_aadv2.py are preserved:
  FIX 1 — DelayScheduler thea1/thea2 only ever increment.
  FIX 2 — delay_sched.step() called once per epoch after train loop.
  FIX 3 — dropout exposed as config knob (default 0.0).
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
from CODE.SNN_FYP.experiments.plot_buffer_comparison import ResultsLogger
from log_buffer_aad_snntorch import LogBufferAADNetSNNTorch


# ─────────────────────────────────────────────────────────────────────────────
# VERSIONING  (identical to train_aadv2.py)
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
# DATASET  (identical to train_aadv2.py)
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
# MEMORY UTILITIES  (identical to train_aadv2.py)
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
# DELAY SCHEDULER  (FIX 1 + FIX 2 — identical to train_aadv2.py)
# ─────────────────────────────────────────────────────────────────────────────

class DelayScheduler:
    """
    Monotonically non-decreasing delay caps.
    step() must be called once per epoch after the training loop.
    """

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

        d1 = model.delay1.delay.detach().cpu()
        d2 = model.delay2.delay.detach().cpu()

        sorted1 = torch.sort(torch.floor(d1))[0]
        sorted2 = torch.sort(torch.floor(d2))[0]

        if self.update1 >= self.adapt_window:
            if sorted1[108] > (self.thea1 - self.expand_margin):
                self.thea1 += 1.0
            self.update1 = 0

        if self.update2 >= self.adapt_window:
            if sorted2[108] > (self.thea2 - self.expand_margin):
                self.thea2 += 1.0
            self.update2 = 0

        model.clamp_delays(self.thea1, self.thea2)

    def get_caps(self):
        return self.thea1, self.thea2

    def state_dict(self):
        return {
            "thea1"  : self.thea1,
            "thea2"  : self.thea2,
            "update1": self.update1,
            "update2": self.update2,
        }

    def load_state_dict(self, state):
        self.thea1   = state["thea1"]
        self.thea2   = state["thea2"]
        self.update1 = state["update1"]
        self.update2 = state["update2"]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, base_dir, resume_path=None):

    # ── Versioning ────────────────────────────────────────────────────
    if resume_path and os.path.exists(resume_path):
        results_dir = os.path.dirname(resume_path)
        version     = int(os.path.basename(results_dir)[1:])
    else:
        results_dir, version = get_versioned_dir(base_dir)

    os.makedirs(results_dir, exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Model        : {cfg.model_name}")
    print(f"  Version      : v{version}")
    print(f"  Results      : {results_dir}")
    print(f"  Epochs       : {cfg.epochs}")
    print(f"  LR           : {cfg.lr}")
    print(f"  Batch        : {cfg.batch_size}")
    print(f"  tau_s        : {cfg.tau_s}")
    print(f"  tau_r        : {cfg.tau_r}")
    print(f"  threshold    : {cfg.threshold}")
    print(f"  buffer_size  : {cfg.buffer_size}  (k)")
    print(f"  buffer_max_lag: {cfg.buffer_max_lag}")
    print(f"  theta_u      : {cfg.theta_u}")
    print(f"  target_true  : {cfg.target_true}")
    print(f"  target_false : {cfg.target_false}")
    print(f"{'='*60}\n")

    set_seed(cfg.seed)

    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    print(f"  Using device : {device}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model    = LogBufferAADNetSNNTorch(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = get_param_mb(model)

    buf_info = model.get_buffer_info()
    print(f"  Parameters   : {n_params:,}")
    print(f"  Memory       : {param_mb:.4f} MB")
    print(f"  Buffer k     : {buf_info['k']}")
    print(f"  Buffer L     : {buf_info['L']}  lags={buf_info['lags']}\n")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, test_loader = get_loaders(cfg)

    # ── Optimiser ─────────────────────────────────────────────────────
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

    # ── Resume ────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc    = 0.0

    if resume_path and os.path.exists(resume_path):
        print(f"  Resuming from : {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "delay_sched" in ckpt:
            delay_sched.load_state_dict(ckpt["delay_sched"])
        start_epoch = ckpt.get("epoch", 0)
        best_acc    = ckpt.get("test_acc", 0.0)
        print(f"  Resumed epoch {start_epoch}, "
              f"best={best_acc*100:.2f}%\n")

    # ── Config snapshot ───────────────────────────────────────────────
    config_snapshot = {
        "model_name"        : cfg.model_name,
        "version"           : version,
        # neuron
        "tau_s"             : cfg.tau_s,
        "tau_r"             : cfg.tau_r,
        "threshold"         : cfg.threshold,
        # log buffer
        "buffer_size"       : cfg.buffer_size,
        "buffer_max_lag"    : cfg.buffer_max_lag,
        "theta_u"           : cfg.theta_u,
        "target_true"       : cfg.target_true,
        "target_false"      : cfg.target_false,
        # training
        "lr"                : cfg.lr,
        "weight_decay"      : getattr(cfg, "weight_decay", 0.0),
        "dropout"           : getattr(cfg, "dropout", 0.0),
        "batch_size"        : cfg.batch_size,
        "epochs"            : cfg.epochs,
        # delay
        "delay_init_cap"    : cfg.delay_init_cap,
        "delay_adapt_start" : cfg.delay_adapt_start,
        "delay_adapt_window": cfg.delay_adapt_window,
        "seed"              : cfg.seed,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_snapshot, f, indent=2)

    # ── History ───────────────────────────────────────────────────────
    train_losses = []
    train_accs   = []
    test_accs    = []
    epoch_times  = []
    peak_mems    = []

    # ── Main Loop ─────────────────────────────────────────────────────
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
            loss   = model.compute_loss(output, labels, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            optimizer.step()

            pred        = model.predict(output)
            correct    += (pred == labels).sum().item()
            total      += labels.size(0)
            epoch_loss += loss.item()

        # FIX 2: delay scheduler once per epoch
        delay_sched.step(model, epoch)
        scheduler.step()

        # ── Eval ──────────────────────────────────────────────────────
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

        # ── Metrics ───────────────────────────────────────────────────
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

        # ── Best checkpoint ───────────────────────────────────────────
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch"      : epoch + 1,
                "version"    : version,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "delay_sched": delay_sched.state_dict(),
                "test_acc"   : test_acc,
                "thea1"      : thea1,
                "thea2"      : thea2,
            }, os.path.join(results_dir,
                            f"{cfg.model_name}_best.pt"))

        print(
            f"  [v{version}][{epoch+1:4d}/{cfg.epochs}] "
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
            cfg.model_name, epoch + 1,
            avg_loss, train_acc, test_acc,
            peak_mem, epoch_time
        )

        if (cfg.save_checkpoints and
                (epoch + 1) % cfg.checkpoint_every == 0):
            torch.save({
                "epoch"      : epoch + 1,
                "version"    : version,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "scheduler"  : scheduler.state_dict(),
                "delay_sched": delay_sched.state_dict(),
                "test_acc"   : test_acc,
                "thea1"      : thea1,
                "thea2"      : thea2,
            }, os.path.join(
                results_dir,
                f"{cfg.model_name}_epoch{epoch+1}.pt"
            ))

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "model_name"              : cfg.model_name,
        "version"                 : version,
        "results_dir"             : results_dir,
        "best_acc"                : float(best_acc),
        "n_params"                : n_params,
        "param_mem_mb"            : float(param_mb),
        "buffer_k"                : buf_info["k"],
        "buffer_L"                : buf_info["L"],
        "buffer_lags"             : buf_info["lags"],
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
        results_dir, f"{cfg.model_name}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.save()

    print(f"\n{'='*60}")
    print(f"  Version        : v{version}")
    print(f"  Best accuracy  : {best_acc*100:.2f}%")
    print(f"  Results dir    : {results_dir}")
    print(f"  Summary saved  : {summary_path}")
    print(f"  Final caps     : [{delay_sched.thea1:.0f},{delay_sched.thea2:.0f}]")
    print(f"{'='*60}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train LogBufferAADNet (snnTorch) on SHD"
    )
    parser.add_argument("--epochs",         type=int, default=None)
    parser.add_argument("--base-dir",       type=str, default=None)
    parser.add_argument("--datasets-path",  type=str, default=None)
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--buffer-size",    type=int, default=None,
                        help="Override cfg.buffer_size (k)")
    parser.add_argument("--buffer-max-lag", type=int, default=None,
                        help="Override cfg.buffer_max_lag")
    parser.add_argument("--theta-u",        type=float, default=None,
                        help="Override cfg.theta_u")
    args = parser.parse_args()

    cfg = get_config("log_buffer_snntorch")

    if args.epochs         is not None: cfg.epochs         = args.epochs
    if args.datasets_path  is not None: cfg.datasets_path  = args.datasets_path
    if args.buffer_size    is not None: cfg.buffer_size    = args.buffer_size
    if args.buffer_max_lag is not None: cfg.buffer_max_lag = args.buffer_max_lag
    if args.theta_u        is not None: cfg.theta_u        = args.theta_u

    base_dir = args.base_dir or os.path.join(
        cfg.results_dir, "log_buffer_aad"
    )

    train(cfg, base_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()