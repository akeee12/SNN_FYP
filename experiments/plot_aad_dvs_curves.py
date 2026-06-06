"""
AAD V2 SNNTorch — DVS Gesture training curves.
Test accuracy and train loss per epoch.

Outputs:
    plots/dvs/aad_dvs_accuracy.png
    plots/dvs/aad_dvs_loss.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"
SUMMARY_PATH = os.path.join(
    BASE_RESULTS,
    "aad_v2_dvs_500e/v1/AdaptiveAxonalDelayNetV2SNNTorch_summary.json"
)
OUTPUT_DIR = os.path.join(BASE_RESULTS, "plots", "dvs")


def set_style():
    plt.rcParams.update({
        "font.family"       : "serif",
        "font.size"         : 11,
        "axes.titlesize"    : 12,
        "axes.labelsize"    : 11,
        "xtick.labelsize"   : 10,
        "ytick.labelsize"   : 10,
        "legend.fontsize"   : 9,
        "figure.dpi"        : 150,
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
        "axes.grid"         : True,
        "grid.alpha"        : 0.3,
        "grid.linestyle"    : "--",
        "grid.linewidth"    : 0.6,
    })


def smooth(values, window=7):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def plot_single(raw, ylabel, title, output_path,
                color="#2166ac", is_acc=True):
    fig, ax = plt.subplots(figsize=(9, 5))

    epochs   = list(range(1, len(raw) + 1))
    smoothed = smooth(np.array(raw), window=7)
    best_val = max(raw) if is_acc else None
    best_ep  = int(np.argmax(raw)) + 1 if is_acc else None

    # Faint raw
    ax.plot(epochs, raw,
            color=color, alpha=0.15, linewidth=0.8, zorder=2)

    # Smoothed bold
    label = (f"AAD V2 SNNTorch  (best={best_val:.1f}% @ ep{best_ep})"
             if is_acc else "AAD V2 SNNTorch")
    ax.plot(epochs, smoothed,
            color=color, linewidth=2.2,
            linestyle="-", marker="o",
            markevery=max(1, len(epochs) // 10),
            markersize=5,
            markeredgewidth=0.5,
            markeredgecolor="white",
            label=label,
            zorder=3)

    if is_acc and best_val is not None:
        ax.axhline(y=best_val, color=color, alpha=0.3,
                   linewidth=0.8, linestyle="--")
        ax.axvline(x=best_ep, color=color, alpha=0.3,
                   linewidth=0.8, linestyle="--")
        ax.annotate(
            f"Best: {best_val:.1f}%\n@ epoch {best_ep}",
            xy=(best_ep, best_val),
            xytext=(best_ep + max(5, len(epochs)//20), best_val - 5),
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_xlim(1, len(epochs))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")

    ax.legend(
        loc="lower right",
        frameon=True, framealpha=0.95,
        edgecolor="#cccccc",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_style()

    print(f"\n{'='*60}")
    print(f"  AAD DVS training curves")
    print(f"  Summary : {SUMMARY_PATH}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    with open(SUMMARY_PATH) as f:
        s = json.load(f)

    test_accs    = [a * 100.0 for a in s.get("test_accs", [])]
    train_losses = s.get("train_losses", [])
    best_acc     = s.get("best_acc", 0.0) * 100.0
    n_epochs     = len(test_accs)

    print(f"  Epochs      : {n_epochs}")
    print(f"  Best acc    : {best_acc:.2f}%")
    print(f"  Best epoch  : {int(np.argmax(test_accs)) + 1}")

    print("\n  Accuracy curve ...")
    plot_single(
        test_accs,
        ylabel     = "Test Accuracy (%)",
        title      = "AAD V2 SNNTorch — DVS Gesture\nTest Accuracy per Epoch",
        output_path= os.path.join(OUTPUT_DIR, "aad_dvs_accuracy.png"),
        color      = "#2166ac",
        is_acc     = True,
    )

    print("  Loss curve ...")
    plot_single(
        train_losses,
        ylabel     = "Training Loss",
        title      = "AAD V2 SNNTorch — DVS Gesture\nTraining Loss per Epoch",
        output_path= os.path.join(OUTPUT_DIR, "aad_dvs_loss.png"),
        color      = "#d6604d",
        is_acc     = False,
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
