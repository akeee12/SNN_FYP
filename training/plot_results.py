# training/plot_results.py
"""
Scalable combined plotting script.
Automatically discovers all model result folders and plots them together.

Design principles:
    1. Auto-discovery — scans results directory for *_summary.json files
    2. No hardcoding — works with any number of models
    3. Extensible — add a new model folder → plots update automatically

Usage:
    # Plot all models found in results/SNN_FYP/
    python3 training/plot_results.py \
        --results-root /rds/general/user/ak2721/home/results/SNN_FYP \
        --output-dir   /rds/general/user/ak2721/home/results/SNN_FYP/plots

    # Plot specific models only
    python3 training/plot_results.py \
        --results-root /rds/general/user/ak2721/home/results/SNN_FYP \
        --models aad_pytorch dcls \
        --output-dir   /rds/general/user/ak2721/home/results/SNN_FYP/plots
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colour palette — add more as you add models
COLORS = [
    "#2196F3",   # blue      — aad_pytorch
    "#FF5722",   # orange    — dcls
    "#4CAF50",   # green     — aad_log_buffer
    "#9C27B0",   # purple    — future model 4
    "#F44336",   # red       — future model 5
    "#009688",   # teal      — future model 6
    "#FF9800",   # amber     — future model 7
    "#607D8B",   # blue-grey — future model 8
]


# ─────────────────────────────────────────────────────────────────────────────
# DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_results(results_root: str,
                     model_filter: list = None) -> list:
    """
    Automatically finds all *_summary.json files under results_root.
    Each subdirectory is treated as one model's results.

    Args:
        results_root  : root directory to search
        model_filter  : optional list of model names to include
                        e.g. ["aad_pytorch", "dcls"]
                        None = include all found

    Returns:
        list of dicts, one per model, sorted by model_name
    """
    results = []

    # Walk all subdirectories
    for entry in sorted(os.scandir(results_root)):
        if not entry.is_dir():
            continue

        # Skip the combined plots output folder
        if entry.name == "plots":
            continue

        # Apply model filter if specified
        if model_filter and entry.name not in model_filter:
            continue

        # Look for summary JSON
        summary_path = None
        for fname in os.listdir(entry.path):
            if fname.endswith("_summary.json"):
                summary_path = os.path.join(entry.path, fname)
                break

        if summary_path is None:
            print(f"  [SKIP] No summary found in {entry.path}")
            continue

        try:
            with open(summary_path) as f:
                summary = json.load(f)
            results.append(summary)
            print(f"  [FOUND] {summary['model_name']} — "
                  f"best acc: {summary['best_acc']*100:.2f}%")
        except Exception as e:
            print(f"  [ERROR] Could not load {summary_path}: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_vs_epochs(results, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        ax.plot(
            range(1, len(res["train_losses"]) + 1),
            res["train_losses"],
            label=res["model_name"],
            color=COLORS[i % len(COLORS)],
            linewidth=2,
        )
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss", fontsize=13)
    ax.set_title("Training Loss vs Epochs — SHD", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "plot_loss_vs_epochs.png")


def plot_test_acc_vs_epochs(results, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        ax.plot(
            range(1, len(res["test_accs"]) + 1),
            [a * 100 for a in res["test_accs"]],
            label=res["model_name"],
            color=COLORS[i % len(COLORS)],
            linewidth=2,
        )
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Test Accuracy vs Epochs — SHD", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "plot_test_acc_vs_epochs.png")


def plot_acc_vs_memory(results, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, res in enumerate(results):
        best_acc  = res["best_acc"] * 100
        param_mem = res["param_mem_mb"]
        peak_mem  = max(res["peak_mems"])
        color     = COLORS[i % len(COLORS)]

        ax.scatter(param_mem, best_acc, s=200, color=color,
                   zorder=5, label=f"{res['model_name']} (param)")
        ax.scatter(peak_mem, best_acc, s=200, color=color,
                   zorder=5, marker="^",
                   label=f"{res['model_name']} (peak GPU)")
        ax.annotate(res["model_name"], (param_mem, best_acc),
                    textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Memory Footprint (MB)", fontsize=13)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Memory Footprint — SHD", fontsize=15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "plot_acc_vs_memory.png")


def plot_acc_vs_time(results, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, res in enumerate(results):
        cum_times = np.cumsum(res["epoch_times"]) / 60
        ax.plot(
            cum_times,
            [a * 100 for a in res["test_accs"]],
            label=res["model_name"],
            color=COLORS[i % len(COLORS)],
            linewidth=2,
        )
    ax.set_xlabel("Cumulative Training Time (minutes)", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Test Accuracy vs Training Time — SHD", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, "plot_acc_vs_time.png")


def print_summary_table(results):
    """Prints a clean comparison table to stdout."""
    print("\n" + "="*70)
    print(f"  {'Model':<30} {'Best Acc':>10} "
          f"{'Params':>12} {'Param MB':>9} {'Peak GPU MB':>12}")
    print("="*70)
    for res in results:
        print(f"  {res['model_name']:<30} "
              f"{res['best_acc']*100:>9.2f}% "
              f"{res['n_params']:>12,} "
              f"{res['param_mem_mb']:>8.2f} "
              f"{max(res['peak_mems']):>11.1f}")
    print("="*70)


def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Auto-discover results and generate combined plots")
    parser.add_argument(
        "--results-root", type=str, required=True,
        help="Root directory containing model subdirectories "
             "e.g. /rds/.../results/SNN_FYP")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save plots (default: results-root/plots/)")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Optional: only plot these models "
             "e.g. --models aad_pytorch dcls")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(
        args.results_root, "plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nSearching for results in: {args.results_root}")
    results = discover_results(args.results_root, args.models)

    if not results:
        print("[ERROR] No results found.")
        return

    print(f"\nFound {len(results)} model(s). Generating plots...")
    plot_loss_vs_epochs(results, out_dir)
    plot_test_acc_vs_epochs(results, out_dir)
    plot_acc_vs_memory(results, out_dir)
    plot_acc_vs_time(results, out_dir)
    print_summary_table(results)
    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()