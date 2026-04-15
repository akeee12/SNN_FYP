# training/train.py
"""
Unified benchmark training harness.
Trains both DCLS and AdaptiveAxonalDelay on SHD with identical conditions.
Measures: training loss/epoch, test accuracy, GPU memory footprint, wall-clock time.
Outputs: CSV logs + matplotlib plots.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external_models", "Adaptive_axonal_delay"))
sys.path.insert(0, os.path.join(ROOT, "model_notebooks", "SNN-delays"))

from configs.model_configs import get_config
from utils.seed import set_seed
from experiments.results_logger import ResultsLogger


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_model_memory_mb(model: torch.nn.Module) -> float:
    """
    Returns the static parameter memory footprint of a model in MB.
    Formula: sum of (numel * bytes_per_element) for all parameters.
    """
    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    return total_bytes / (1024 ** 2)


def get_gpu_peak_memory_mb() -> float:
    """Returns peak GPU memory allocated (MB) since last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def reset_gpu_memory_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ─────────────────────────────────────────────────────────────────────────────
# DCLS MODEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_dcls(cfg, logger: ResultsLogger):
    """
    Trains the DCLS/SNN-delays model using its native main.py logic,
    wrapped to capture per-epoch metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {cfg.model_name}")
    print(f"{'='*60}")

    set_seed(cfg.seed)

    # Import DCLS internals (from SNN-delays submodule)
    try:
        from spikingjelly.activation_based import surrogate
        import importlib.util, shutil

        # Dynamically load config into SNN-delays expected format
        snn_delays_path = os.path.join(ROOT, "model_notebooks", "SNN-delays")
        spec = importlib.util.spec_from_file_location(
            "snn_delays_main",
            os.path.join(snn_delays_path, "main.py")
        )
        # We'll use the model and dataset classes directly
        sys.path.insert(0, snn_delays_path)
        from model import SNN, SNNDelays
        from datasets import load_shd_or_ssc
        from snn_delays import SNNDelaysModel

    except ImportError as e:
        print(f"[DCLS] Import error: {e}")
        print("  → Make sure SNN-delays submodule is initialised: "
              "git submodule update --init --recursive")
        return None

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Build model
    model = SNNDelays(cfg).to(device)
    param_memory = get_model_memory_mb(model)
    print(f"  Parameter memory: {param_memory:.2f} MB")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset
    train_loader, test_loader = load_shd_or_ssc(cfg)

    # Optimisers (dual: weights + delay positions)
    optimizer_w   = torch.optim.Adam(
        [p for n, p in model.named_parameters() if "pos" not in n],
        lr=cfg.lr_w, weight_decay=cfg.weight_decay
    )
    optimizer_pos = torch.optim.Adam(
        [p for n, p in model.named_parameters() if "pos" in n],
        lr=cfg.lr_pos, weight_decay=0
    )
    scheduler_w   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_w, max_lr=cfg.max_lr_w,
        epochs=cfg.epochs, steps_per_epoch=len(train_loader)
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    train_losses, test_accs, epoch_times, peak_mems = [], [], [], []

    for epoch in range(cfg.epochs):
        model.train()
        reset_gpu_memory_stats()
        t_start = time.time()
        epoch_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_w.zero_grad()
            optimizer_pos.zero_grad()

            outputs = model(inputs)

            if cfg.loss == "sum":
                loss = loss_fn(outputs.sum(dim=-1), labels)
            elif cfg.loss == "mean":
                loss = loss_fn(outputs.mean(dim=-1), labels)
            else:
                loss = loss_fn(outputs.max(dim=-1).values, labels)

            loss.backward()
            optimizer_w.step()
            optimizer_pos.step()
            scheduler_w.step()

            epoch_loss += loss.item()
            preds = outputs.sum(dim=-1).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        t_end      = time.time()
        peak_mem   = get_gpu_peak_memory_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total

        # ── Evaluation ──
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.sum(dim=-1).argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total   += labels.size(0)
        test_acc = test_correct / test_total

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        print(f"  Epoch [{epoch+1:3d}/{cfg.epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"TrainAcc: {train_acc*100:.2f}% | "
              f"TestAcc: {test_acc*100:.2f}% | "
              f"PeakMem: {peak_mem:.1f}MB | "
              f"Time: {epoch_time:.1f}s")

        logger.log_epoch(cfg.model_name, epoch+1, avg_loss,
                         train_acc, test_acc, peak_mem, epoch_time)

    return {
        "model_name":    cfg.model_name,
        "train_losses":  train_losses,
        "test_accs":     test_accs,
        "epoch_times":   epoch_times,
        "peak_mems":     peak_mems,
        "param_mem_mb":  param_memory,
        "best_acc":      max(test_accs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE AXONAL DELAY MODEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_adaptive_axonal_delay(cfg, logger: ResultsLogger):
    """
    Trains the Adaptive Axonal Delay model (Sun et al.) on SHD.
    Wraps the SLAYER-based training loop to capture per-epoch metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training: {cfg.model_name}")
    print(f"{'='*60}")

    set_seed(cfg.seed)

    try:
        aad_path = os.path.join(ROOT, "external_models", "Adaptive_axonal_delay")
        sys.path.insert(0, aad_path)
        import slayerSNN as snn
        from slayerSNN import spikeLoss
        from torch.utils.data import DataLoader, Dataset
    except ImportError as e:
        print(f"[AdaptiveAxonalDelay] Import error: {e}")
        print("  → Ensure SLAYER is installed: "
              "https://github.com/bamsumit/slayerPytorch")
        return None

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ── Dataset ──────────────────────────────────────────────────────────────
    class SHDDataset(Dataset):
        def __init__(self, data_path):
            self.data_path = data_path
            self.samples   = np.loadtxt(data_path + ".txt").astype("int")

        def __len__(self):
            return self.samples.shape[0]

        def __getitem__(self, idx):
            input_idx = self.samples[idx, 0]
            label     = self.samples[idx, 1]
            path      = f"{self.data_path}/{input_idx}.npy"
            spikes    = torch.from_numpy(np.load(path))
            empty     = torch.zeros((1, 1, cfg.time_steps, cfg.n_inputs))
            yy        = spikes.transpose(1, 0).nonzero(as_tuple=False)
            empty[0, 0, yy[:, 0], yy[:, 1]] = 1
            desired   = torch.zeros((cfg.n_outputs, 1, 1, 1))
            desired[label, ...] = 1
            return empty, desired, torch.tensor(label)

    train_set    = SHDDataset(os.path.join(cfg.datasets_path, "train"))
    test_set     = SHDDataset(os.path.join(cfg.datasets_path, "test"))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
                              shuffle=True,  num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)

    # ── Network ───────────────────────────────────────────────────────────────
    class Network(torch.nn.Module):
        def __init__(self, netParams):
            super().__init__()
            slayer    = snn.layer(netParams["neuron"], netParams["simulation"])
            self.slayer = slayer
            self.fc1  = torch.nn.utils.weight_norm(
                slayer.dense(cfg.n_inputs, cfg.n_hidden), name="weight")
            self.fc2  = torch.nn.utils.weight_norm(
                slayer.dense(cfg.n_hidden, cfg.n_hidden), name="weight")
            self.fc3  = torch.nn.utils.weight_norm(
                slayer.dense(cfg.n_hidden, cfg.n_outputs), name="weight")
            self.delay1 = slayer.delay(cfg.n_hidden)
            self.delay2 = slayer.delay(cfg.n_hidden)

        def forward(self, spike):
            spike = spike.reshape(
                (spike.shape[0], -1, 1, 1, spike.shape[-1]))
            spike = self.slayer.spike(self.fc1(self.slayer.psp(spike)))
            spike = self.delay1(spike)
            spike = self.slayer.spike(self.fc2(self.slayer.psp(spike)))
            spike = self.delay2(spike)
            spike = self.slayer.spike(self.fc3(self.slayer.psp(spike)))
            return spike

        def clamp_init(self):
            self.delay1.delay.data.clamp_(0, cfg.delay_init_cap)
            self.delay2.delay.data.clamp_(0, cfg.delay_init_cap)

        def clamp(self, thea1, thea2):
            self.delay1.delay.data.clamp_(0, thea1)
            self.delay2.delay.data.clamp_(0, thea2)

    netParams = {
        "neuron":     cfg.neuron_params,
        "simulation": cfg.simulation_params,
    }
    net    = Network(netParams).to(device)
    module = net

    param_memory = get_model_memory_mb(net)
    print(f"  Parameter memory: {param_memory:.2f} MB")
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")

    error     = spikeLoss(netParams).to(device)
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)

    # Adaptive delay state
    thea1, thea2   = cfg.delay_init_cap, cfg.delay_init_cap
    update1, update2 = 0, 0

    train_losses, test_accs, epoch_times, peak_mems = [], [], [], []

    for epoch in range(cfg.epochs):
        net.train()
        reset_gpu_memory_stats()
        t_start   = time.time()
        epoch_loss = 0.0
        correct, total = 0, 0

        for inputs, targets, labels in train_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            output  = net(inputs)

            correct += torch.sum(
                snn.predict.getClass(output) == labels).item()
            total   += len(labels)

            loss = error.spikeRate(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item()

            # Adaptive delay clamping
            if epoch <= cfg.delay_adapt_start:
                module.clamp_init()
            else:
                update1 += 1
                update2 += 1
                for name, params in net.named_parameters():
                    if name == "delay1.delay" and update1 > cfg.delay_adapt_window:
                        sorted1, _ = torch.sort(
                            torch.floor(params.detach().squeeze()))
                        thea1 = torch.max(
                            torch.floor(params.detach().squeeze()))
                        idx = int(cfg.delay_percentile * len(sorted1))
                        if sorted1[idx] > (thea1 - cfg.delay_expand_margin):
                            thea1 += 1
                        update1 = 0
                    elif name == "delay2.delay" and update2 > cfg.delay_adapt_window:
                        sorted2, _ = torch.sort(
                            torch.floor(params.detach().squeeze()))
                        thea2 = torch.max(
                            torch.floor(params.detach().squeeze()))
                        idx = int(cfg.delay_percentile * len(sorted2))
                        if sorted2[idx] > (thea2 - cfg.delay_expand_margin):
                            thea2 += 1
                        update2 = 0
                module.clamp(thea1=thea1, thea2=thea2)

        scheduler.step()
        t_end      = time.time()
        peak_mem   = get_gpu_peak_memory_mb()
        epoch_time = t_end - t_start
        train_acc  = correct / total

        # ── Evaluation ──
        net.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for inputs, targets, labels in test_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                output  = net(inputs)
                test_correct += torch.sum(
                    snn.predict.getClass(output) == labels).item()
                test_total   += len(labels)
        test_acc = test_correct / test_total

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        epoch_times.append(epoch_time)
        peak_mems.append(peak_mem)

        print(f"  Epoch [{epoch+1:3d}/{cfg.epochs}] "
              f"Loss: {avg_loss:.4f} | "
              f"TrainAcc: {train_acc*100:.2f}% | "
              f"TestAcc: {test_acc*100:.2f}% | "
              f"PeakMem: {peak_mem:.1f}MB | "
              f"Time: {epoch_time:.1f}s")

        logger.log_epoch(cfg.model_name, epoch+1, avg_loss,
                         train_acc, test_acc, peak_mem, epoch_time)

    return {
        "model_name":   cfg.model_name,
        "train_losses": train_losses,
        "test_accs":    test_accs,
        "epoch_times":  epoch_times,
        "peak_mems":    peak_mems,
        "param_mem_mb": param_memory,
        "best_acc":     max(test_accs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results: list, out_dir: str):
    """
    Generates three benchmark plots:
      1. Training Loss vs Epochs
      2. Test Accuracy vs Peak GPU Memory Footprint
      3. Test Accuracy vs Cumulative Training Time
    """
    os.makedirs(out_dir, exist_ok=True)
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    # ── Plot 1: Training Loss vs Epochs ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(all_results):
        ax.plot(range(1, len(res["train_losses"]) + 1),
                res["train_losses"],
                label=res["model_name"], color=colors[i], linewidth=2)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss", fontsize=13)
    ax.set_title("Training Loss vs Epochs — SHD Dataset", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_loss_vs_epochs.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: plot_loss_vs_epochs.png")

    # ── Plot 2: Accuracy vs Memory Footprint ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, res in enumerate(all_results):
        best_acc   = res["best_acc"] * 100
        param_mem  = res["param_mem_mb"]
        peak_mem   = max(res["peak_mems"])
        ax.scatter(param_mem, best_acc,
                   s=200, color=colors[i], zorder=5,
                   label=f"{res['model_name']} (param)")
        ax.scatter(peak_mem, best_acc,
                   s=200, color=colors[i], zorder=5,
                   marker="^",
                   label=f"{res['model_name']} (peak GPU)")
        ax.annotate(res["model_name"],
                    (param_mem, best_acc),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Memory Footprint (MB)", fontsize=13)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Memory Footprint — SHD Dataset", fontsize=15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_acc_vs_memory.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: plot_acc_vs_memory.png")

    # ── Plot 3: Accuracy vs Cumulative Training Time ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(all_results):
        cum_times = np.cumsum(res["epoch_times"]) / 60  # convert to minutes
        ax.plot(cum_times, [a * 100 for a in res["test_accs"]],
                label=res["model_name"], color=colors[i], linewidth=2)
    ax.set_xlabel("Cumulative Training Time (minutes)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Test Accuracy vs Training Time — SHD Dataset", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_acc_vs_time.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: plot_acc_vs_time.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified SNN Benchmark — SHD Dataset")
    parser.add_argument(
        "--models", nargs="+",
        default=["dcls", "adaptive_axonal_delay"],
        help="Models to benchmark. Options: dcls, adaptive_axonal_delay")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs (default: from config)")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save results and plots")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    logger = ResultsLogger(args.results_dir)

    all_results = []
    runners = {
        "dcls":                  (run_dcls,                  "DCLSConfig"),
        "adaptive_axonal_delay": (run_adaptive_axonal_delay, "AdaptiveAxonalDelayConfig"),
    }

    for model_name in args.models:
        if model_name not in runners:
            print(f"[WARNING] Unknown model '{model_name}', skipping.")
            continue

        cfg = get_config(model_name)
        if args.epochs is not None:
            cfg.epochs = args.epochs

        runner_fn, _ = runners[model_name]
        result = runner_fn(cfg, logger)

        if result is not None:
            all_results.append(result)
            # Save per-model JSON summary
            summary_path = os.path.join(
                args.results_dir, f"{model_name}_summary.json")
            with open(summary_path, "w") as f:
                json.dump({
                    k: (v if not isinstance(v, list) else v)
                    for k, v in result.items()
                }, f, indent=2)
            print(f"\n  ✓ {model_name} best accuracy: {result['best_acc']*100:.2f}%")

    if all_results:
        print("\n" + "="*60)
        print("  Generating comparison plots...")
        plot_results(all_results, args.results_dir)
        logger.save()
        print("  All done! Results saved to:", args.results_dir)
    else:
        print("[ERROR] No models completed training.")


if __name__ == "__main__":
    main()