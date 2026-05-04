# training/plot_results.py
"""
Plotting script for SNN training results — single model.

Generates three plots:
    1. Test accuracy vs epochs
    2. Loss vs epochs (linear scale)
    3. Log loss vs epochs (log scale)

Usage:
    python3 training/plot_results.py \
        --results-root /rds/general/user/ak2721/home/results/SNN_FYP \
        --model log_buffer \
        --output-dir /rds/general/user/ak2721/home/results/SNN_FYP/plots
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2196F3"
ORANGE = "#FF5722"
GREEN  = "#4CAF50"
GREY   = "#607D8B"

plt.rcParams.update({
    "font.family"       : "sans-serif",
    "font.size"         : 12,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
})


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_summary(results_root: str, model_name: str) -> dict:
    """Load summary JSON for a single model."""
    model_dir = os.path.join(results_root, model_name)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}"
        )

    for fname in os.listdir(model_dir):
        if fname.endswith("_summary.json"):
            path = os.path.join(model_dir, fname)
            with open(path) as f:
                summary = json.load(f)
            print(f"  Loaded : {path}")
            print(f"  Model  : {summary['model_name']}")
            print(f"  Best   : {summary['best_acc']*100:.2f}%")
            print(f"  Params : {summary['n_params']:,}")
            return summary

    raise FileNotFoundError(
        f"No *_summary.json found in {model_dir}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Test Accuracy vs Epochs
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_acc_vs_epochs(summary: dict, out_dir: str):
    """
    Test accuracy vs epochs — single model.

    Legend placed outside plot on the right.
    Vertical dashed line marks best epoch.
    Horizontal dotted line marks best accuracy.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs   = range(1, len(summary["test_accs"]) + 1)
    accs     = [a * 100 for a in summary["test_accs"]]
    best_acc = max(accs)
    best_ep  = accs.index(best_acc) + 1

    ax.plot(
        epochs, accs,
        color=BLUE, linewidth=2,
        label=f"{summary['model_name']}\n"
              f"best: {best_acc:.2f}% @ epoch {best_ep}\n"
              f"params: {summary['n_params']:,}"
    )
    ax.axvline(
        best_ep, color=BLUE,
        linestyle="--", alpha=0.4, linewidth=1
    )
    ax.axhline(
        best_acc, color=GREY,
        linestyle=":", alpha=0.4, linewidth=1
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(
        f"Test Accuracy vs Epochs — "
        f"{summary['model_name']} on SHD"
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)

    # Legend outside plot — right side
    ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    _save(fig, out_dir, "plot_test_acc_vs_epochs.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Loss vs Epochs (linear)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_vs_epochs(summary: dict, out_dir: str):
    """
    Training loss vs epochs — linear scale.

    Legend placed outside plot on the right.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(summary["train_losses"]) + 1)
    losses = summary["train_losses"]

    ax.plot(
        epochs, losses,
        color=ORANGE, linewidth=2,
        label=f"{summary['model_name']}\n"
              f"final loss: {losses[-1]:.4f}\n"
              f"params: {summary['n_params']:,}"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (Van Rossum)")
    ax.set_title(
        f"Training Loss vs Epochs — "
        f"{summary['model_name']} on SHD"
    )
    ax.grid(True, alpha=0.2)

    # Legend outside plot — right side
    ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    _save(fig, out_dir, "plot_loss_vs_epochs.png")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Log Loss vs Epochs
# ─────────────────────────────────────────────────────────────────────────────

def plot_log_loss_vs_epochs(summary: dict, out_dir: str):
    """
    Training loss vs epochs — log scale.

    Suitability:
        Van Rossum loss spans multiple orders of magnitude
        during SNN training. Log scale reveals early training
        dynamics hidden in linear scale and makes instabilities
        (loss spikes) more visible.

    Legend placed outside plot on the right.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(summary["train_losses"]) + 1)
    eps    = 1e-8
    losses = np.maximum(summary["train_losses"], eps)

    ax.semilogy(
        epochs, losses,
        color=GREEN, linewidth=2,
        label=f"{summary['model_name']}\n"
              f"final loss: {losses[-1]:.4f}\n"
              f"params: {summary['n_params']:,}"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (log scale)")
    ax.set_title(
        f"Log Training Loss vs Epochs — "
        f"{summary['model_name']} on SHD"
    )
    ax.grid(True, alpha=0.2, which="both")

    # Minor gridlines for log scale
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(subs=np.arange(2, 10) * 0.1)
    )
    ax.grid(True, which="minor", alpha=0.1)

    # Legend outside plot — right side
    ax.legend(
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    _save(fig, out_dir, "plot_log_loss_vs_epochs.png")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(summary: dict):
    print("\n" + "="*55)
    print(f"  {'Field':<25} {'Value':>25}")
    print("="*55)
    print(f"  {'Model':<25} {summary['model_name']:>25}")
    print(f"  {'Best Accuracy':<25} "
          f"{summary['best_acc']*100:>24.2f}%")
    print(f"  {'Parameters':<25} "
          f"{summary['n_params']:>25,}")
    print(f"  {'Param Memory (MB)':<25} "
          f"{summary['param_mem_mb']:>25.4f}")
    print(f"  {'Epochs trained':<25} "
          f"{len(summary['test_accs']):>25}")
    print("="*55 + "\n")


def _save(fig, out_dir: str, filename: str):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot training results for a single SNN model"
    )
    parser.add_argument(
        "--results-root", type=str, required=True,
        help="Root directory containing model subdirectories"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model folder name e.g. log_buffer or aad_v2"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to save plots "
             "(default: results-root/plots/)"
    )
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(
        args.results_root, "plots"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nResults root : {args.results_root}")
    print(f"Model        : {args.model}")
    print(f"Output dir   : {out_dir}\n")

    # Load summary
    summary = load_summary(args.results_root, args.model)

    # Print table
    print_summary_table(summary)

    # Generate plots
    print("Generating plots...")
    plot_test_acc_vs_epochs(summary, out_dir)
    plot_loss_vs_epochs(summary, out_dir)
    plot_log_loss_vs_epochs(summary, out_dir)

    print(f"\nAll plots saved to: {out_dir}\n")


if __name__ == "__main__":
    main()