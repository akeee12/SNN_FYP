"""
Per-epoch test accuracy and train loss curves for all buffer strategies
on SSC, with AAD V2 SNNTorch overlaid as the baseline.
Reads from training_log.csv files. Plots up to 50 epochs.

Outputs:
    plots/ssc/buffer_accuracy_curves.png
    plots/ssc/buffer_loss_curves.png
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"

# (relative path to training_log.csv, label, color, linestyle, marker)
BASELINE = (
    "aad_v2_ssc_500e/v2/training_log.csv",
    "AAD (baseline)",
    "#333333", "--", "o",
)

BUFFER_MODELS = [
    ("log_ring_buffer_ssc/v2/training_log.csv",
     "Log buffer",              "#2166ac", "-",       "s"),
    ("linear_tap_ssc/v2/training_log.csv",
     "Linear buffer",           "#d6604d", "--",      "^"),
    ("exponential_tap_ssc/v3/training_log.csv",
     "Exponential buffer",      "#4dac26", "-.",      "D"),
    ("power_law_tap_ssc/v2/training_log.csv",
     "Power-law buffer (\u03b3=2)", "#8073ac", ":",   "v"),
    ("random_tap_ssc/v2/training_log.csv",
     "Random buffer",           "#e08214", (0,(5,2)), "x"),
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


def load_csv(rel_path, label):
    full_path = os.path.join(BASE_RESULTS, rel_path)
    if not os.path.isfile(full_path):
        print(f"  [SKIP] {label:<40} -- not found: {full_path}")
        return None, None
    try:
        test_accs    = []
        train_losses = []
        with open(full_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_accs.append(float(row["test_acc"]) * 100.0)
                train_losses.append(float(row["loss"]))
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


def plot_curves(key, ylabel, title, output_path, n_epochs=50):
    fig, ax = plt.subplots(figsize=(9, 5))

    all_entries = [BASELINE] + list(BUFFER_MODELS)

    for i, entry in enumerate(all_entries):
        rel_path, label, color, ls, marker = entry
        is_baseline = (i == 0)

        test_accs, train_losses = load_csv(rel_path, label)
        if test_accs is None:
            continue

        raw = test_accs if key == "test_accs" else train_losses
        raw = raw[:n_epochs]
        if not raw:
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
    DATASET    = "ssc"
    N_EPOCHS   = 50
    OUTPUT_DIR = os.path.join(BASE_RESULTS, "plots", DATASET)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    set_style()

    print(f"\n{'='*60}")
    print(f"  Buffer curve plots  --  dataset : {DATASET.upper()}")
    print(f"  Output dir          : {OUTPUT_DIR}")
    print(f"  Epochs plotted      : up to {N_EPOCHS}")
    print(f"{'='*60}\n")

    print("  Accuracy curves ...")
    plot_curves(
        key        = "test_accs",
        ylabel     = "Test Accuracy (%)",
        title      = "Test Accuracy per Epoch — SSC\nBuffer strategies vs AAD baseline",
        output_path= os.path.join(OUTPUT_DIR, "buffer_accuracy_curves.png"),
        n_epochs   = N_EPOCHS,
    )

    print("\n  Loss curves ...")
    plot_curves(
        key        = "train_losses",
        ylabel     = "Training Loss",
        title      = "Training Loss per Epoch — SSC\nBuffer strategies vs AAD baseline",
        output_path= os.path.join(OUTPUT_DIR, "buffer_loss_curves.png"),
        n_epochs   = N_EPOCHS,
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
