"""
Per-epoch test accuracy and train loss curves for all buffer strategies
on DVS Gesture, with AAD V2 SNNTorch overlaid as the baseline.
Reads from *_summary.json files. Plots up to 100 epochs.

Outputs:
    plots/dvs/buffer_accuracy_curves.png
    plots/dvs/buffer_loss_curves.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"

# (relative path to summary.json, label, color, linestyle, marker)
BASELINE = (
    "aad_v2_dvs_500e/v1/AdaptiveAxonalDelayNetV2SNNTorch_summary.json",
    "AAD (baseline)",
    "#333333", "--", "o",
)

BUFFER_MODELS = [
    ("log_ring_buffer_dvs/v1/LogRingBufferAADSNNTorch_summary.json",
     "Log buffer",               "#2166ac", "-",       "s"),
    ("linear_tap_dvs/v1/LinearTapAADSNNTorch_summary.json",
     "Linear buffer",            "#d6604d", "--",      "^"),
    ("exponential_tap_dvs/v2/ExponentialTapAADSNNTorch_summary.json",
     "Exponential buffer",       "#4dac26", "-.",      "D"),
    ("power_law_tap_dvs/v1/PowerLawTapAADSNNTorch_summary.json",
     "Power-law buffer (\u03b3=2)", "#8073ac", ":",    "v"),
    ("random_tap_dvs/v1/RandomTapAADSNNTorch_summary.json",
     "Random buffer",            "#e08214", (0,(5,2)), "x"),
]


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


def load_summary(rel_path, label):
    full_path = os.path.join(BASE_RESULTS, rel_path)
    if not os.path.isfile(full_path):
        print(f"  [SKIP] {label:<40} -- not found: {full_path}")
        return None, None
    try:
        with open(full_path) as f:
            s = json.load(f)
        test_accs    = [a * 100.0 for a in s.get("test_accs", [])]
        train_losses = s.get("train_losses", [])
        best   = max(test_accs) if test_accs else 0.0
        epochs = len(test_accs)
        print(f"  [LOAD] {label:<40}  best={best:.2f}%  epochs={epochs}")
        return test_accs, train_losses
    except Exception as e:
        print(f"  [SKIP] {label:<40} -- read error: {e}")
        return None, None


def smooth(values, window=5):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def plot_curves(key, ylabel, title, output_path, n_epochs=100):
    fig, ax = plt.subplots(figsize=(9, 5))

    all_entries = [BASELINE] + list(BUFFER_MODELS)

    for i, entry in enumerate(all_entries):
        rel_path, label, color, ls, marker = entry
        is_baseline = (i == 0)

        test_accs, train_losses = load_summary(rel_path, label)
        if test_accs is None:
            continue

        raw = test_accs if key == "test_accs" else train_losses
        raw = raw[:n_epochs]
        if not raw:
            print(f"  [WARN] {label} -- no {key} data")
            continue

        epochs   = list(range(1, len(raw) + 1))
        smoothed = smooth(np.array(raw), window=7)
        me       = max(1, len(epochs) // 8)
        best_val = max(raw) if key == "test_accs" else None
        leg_label = (f"{label}  (best={best_val:.1f}%)"
                     if key == "test_accs" else label)

        # Faint raw
        ax.plot(epochs, raw,
                color=color, alpha=0.15, linewidth=0.8,
                linestyle="-", zorder=2)

        # Smoothed bold
        lw = 1.8 if is_baseline else 2.2
        ax.plot(epochs, smoothed,
                color=color, linewidth=lw,
                linestyle=ls,
                marker=marker,
                markevery=me, markersize=5,
                markeredgewidth=0.5,
                markeredgecolor="white",
                label=leg_label,
                zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_xlim(1, n_epochs)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")

    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=2.5,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    DATASET    = "dvs"
    N_EPOCHS   = 100
    OUTPUT_DIR = os.path.join(BASE_RESULTS, "plots", DATASET)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_style()

    print(f"\n{'='*60}")
    print(f"  Buffer curve plots  --  dataset : DVS Gesture")
    print(f"  Output dir          : {OUTPUT_DIR}")
    print(f"  Epochs plotted      : up to {N_EPOCHS}")
    print(f"{'='*60}\n")

    print("  Accuracy curves ...")
    plot_curves(
        key        = "test_accs",
        ylabel     = "Test Accuracy (%)",
        title      = "Test Accuracy per Epoch — DVS Gesture\nBuffer strategies vs AAD baseline",
        output_path= os.path.join(OUTPUT_DIR, "buffer_accuracy_curves.png"),
        n_epochs   = N_EPOCHS,
    )

    print("\n  Loss curves ...")
    plot_curves(
        key        = "train_losses",
        ylabel     = "Training Loss",
        title      = "Training Loss per Epoch — DVS Gesture\nBuffer strategies vs AAD baseline",
        output_path= os.path.join(OUTPUT_DIR, "buffer_loss_curves.png"),
        n_epochs   = N_EPOCHS,
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
