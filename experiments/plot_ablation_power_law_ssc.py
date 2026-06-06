"""
Ablation plotting script — power law ring buffer, SSC.
Reads from training_log.csv files in ablation_power_law_ssc/v1/

Produces:
    plots/ablation_power_law_ssc/
        ablation_pl_ssc_accuracy_{small,medium,large}.png
        ablation_pl_ssc_loss_{small,medium,large}.png
        ablation_pl_ssc_scatter.png
        ablation_pl_ssc_summary_table.png
"""

import os
import csv
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"
ABLATION_DIR = os.path.join(BASE_RESULTS, "ablation_power_law_ssc", "v1")
OUTPUT_DIR   = os.path.join(BASE_RESULTS, "plots", "ablation_power_law_ssc")

N_COLOURS = {
    5: "#2166ac",
    6: "#d6604d",
    7: "#4dac26",
    8: "#8073ac",
}

SIZE_MARKERS = {"small": "o", "medium": "s", "large": "^"}
SIZE_LS      = {"small": ":", "medium": "--", "large": "-"}
SIZE_ORDER   = {"small": 0, "medium": 1, "large": 2}


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


def parse_run_name(run_dir_name):
    """
    Parse k32_n5_lag16_g2.0_small into components.
    Returns (buf_size, n, max_lag, gamma, size) or None.
    """
    m = re.match(
        r"k(?P<k>\d+)_n(?P<n>\d+)_lag(?P<lag>\d+)_g(?P<g>[\d.]+)_(?P<size>\w+)",
        run_dir_name
    )
    if not m:
        return None
    return (
        int(m.group("k")),
        int(m.group("n")),
        int(m.group("lag")),
        float(m.group("g")),
        m.group("size"),
    )


def load_csv(csv_path):
    """Read training_log.csv → (test_accs, train_losses)"""
    test_accs    = []
    train_losses = []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_accs.append(float(row["test_acc"]) * 100.0)
                train_losses.append(float(row["loss"]))
    except Exception as e:
        print(f"    [WARN] CSV read error: {e}")
    return test_accs, train_losses


def load_all_runs(ablation_dir):
    if not os.path.isdir(ablation_dir):
        print(f"  [ERROR] Not found: {ablation_dir}")
        return []

    runs = []
    for run_dir_name in sorted(os.listdir(ablation_dir)):
        run_dir = os.path.join(ablation_dir, run_dir_name)
        if not os.path.isdir(run_dir):
            continue

        parsed = parse_run_name(run_dir_name)
        if parsed is None:
            continue
        buf_size, n, max_lag, gamma, size = parsed

        csv_path = os.path.join(run_dir, "training_log.csv")
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {run_dir_name} -- no training_log.csv")
            continue

        test_accs, train_losses = load_csv(csv_path)
        if not test_accs:
            print(f"  [SKIP] {run_dir_name} -- empty CSV")
            continue

        best_acc   = max(test_accs)
        best_epoch = int(np.argmax(test_accs)) + 1

        runs.append(dict(
            n=n, network_size=size, buf_size=buf_size,
            gamma=gamma, max_lag=max_lag,
            best_acc=best_acc, best_epoch=best_epoch,
            test_accs=test_accs, train_losses=train_losses,
            color=N_COLOURS.get(n, "#888888"),
            marker=SIZE_MARKERS.get(size, "o"),
            ls=SIZE_LS.get(size, "-"),
            run_name=run_dir_name,
        ))
        print(f"  [LOAD] {run_dir_name:<35}  "
              f"epochs={len(test_accs):3d}  "
              f"best={best_acc:.2f}% @ep{best_epoch}")

    runs.sort(key=lambda r: (r["n"], SIZE_ORDER.get(r["network_size"], 9)))
    return runs


def smooth(values, window=5):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def add_legend(ax, runs):
    seen_n    = sorted(set(r["n"] for r in runs))
    seen_size = [s for s in ["small", "medium", "large"]
                 if any(r["network_size"] == s for r in runs)]
    sep = Line2D([0], [0], color="none")

    n_handles = [
        Line2D([0], [0], color="none", label="Tap count (n)")
    ] + [
        Line2D([0], [0], color=N_COLOURS.get(n, "#888"),
               linewidth=2.0, label=f"n = {n}")
        for n in seen_n
    ]

    size_handles = [
        Line2D([0], [0], color="none", label="Network size")
    ] + [
        Line2D([0], [0], color="grey", linewidth=1.5,
               linestyle=SIZE_LS.get(sz, "-"),
               marker=SIZE_MARKERS.get(sz, "o"),
               markersize=5, label=sz.capitalize())
        for sz in seen_size
    ]

    leg = ax.legend(
        handles=n_handles + [sep] + size_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=2.0,
    )
    for text in leg.get_texts():
        if text.get_text() in ("Tap count (n)", "Network size"):
            text.set_color("grey")
            text.set_style("italic")
            text.set_fontsize(8)


def plot_curves_by_size(runs, key, ylabel, size, output_path):
    size_runs = [r for r in runs if r["network_size"] == size]
    if not size_runs:
        print(f"  [SKIP] {size} -- no runs")
        return

    valid = [r for r in size_runs if r[key]]
    if not valid:
        print(f"  [SKIP] {size} {key} -- no data")
        return

    n_epochs = max(len(r[key]) for r in valid)
    fig, ax  = plt.subplots(figsize=(9, 5))

    for r in valid:
        raw      = r[key]
        epochs   = list(range(1, len(raw) + 1))
        smoothed = smooth(np.array(raw), window=5)
        me       = max(1, len(epochs) // 8)
        best_val = max(raw) if key == "test_accs" else None
        leg_label = (f"n={r['n']}  (best={best_val:.1f}%)"
                     if key == "test_accs" else f"n={r['n']}")

        ax.plot(epochs, raw,
                color=r["color"], alpha=0.12,
                linewidth=0.8, zorder=2)
        ax.plot(epochs, smoothed,
                color=r["color"], linewidth=2.0,
                linestyle=r["ls"],
                marker=r["marker"],
                markevery=me, markersize=5,
                markeredgewidth=0.5,
                markeredgecolor="white",
                label=leg_label,
                zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{size.capitalize()} network — SSC")
    ax.set_xlim(1, n_epochs)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")

    ax.legend(
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=2.0,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scatter(runs, output_path):
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    seen_n    = set()
    seen_size = set()

    for r in runs:
        ax.scatter(
            r["n"], r["best_acc"],
            color=r["color"], marker=r["marker"],
            s=120, zorder=4,
            edgecolors="black", linewidths=0.6,
        )
        seen_n.add(r["n"])
        seen_size.add(r["network_size"])

    ax.set_xlabel("Number of Taps (n)")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Tap Count vs Best Test Accuracy — SSC")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    ax.grid(True, which="minor", alpha=0.1)

    sep = Line2D([0], [0], color="none")

    n_handles = [
        Line2D([0], [0], color="none", label="Tap count (n)")
    ] + [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=N_COLOURS.get(n, "#888"),
               markeredgecolor="black", markeredgewidth=0.6,
               markersize=8, label=f"n = {n}", linestyle="None")
        for n in sorted(seen_n)
    ]

    size_handles = [
        Line2D([0], [0], color="none", label="Network size")
    ] + [
        Line2D([0], [0], marker=SIZE_MARKERS[sz], color="w",
               markerfacecolor="#888888",
               markeredgecolor="black", markeredgewidth=0.6,
               markersize=8, label=sz.capitalize(), linestyle="None")
        for sz in ["small", "medium", "large"]
        if sz in seen_size
    ]

    leg = ax.legend(
        handles=n_handles + [sep] + size_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=1.5, fontsize=9,
    )
    for text in leg.get_texts():
        if text.get_text() in ("Tap count (n)", "Network size"):
            text.set_color("grey")
            text.set_style("italic")
            text.set_fontsize(8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_table(runs, output_path):
    if not runs:
        return

    def top_k_avg(accs, k=10):
        if not accs:
            return 0.0
        return float(np.mean(sorted(accs, reverse=True)[:k]))

    rows = []
    for r in sorted(runs, key=lambda x: x["best_acc"], reverse=True):
        top10 = top_k_avg(r["test_accs"], k=10)
        rows.append([
            str(r["n"]),
            r["network_size"].capitalize(),
            str(r["gamma"]),
            str(r["max_lag"]),
            f"{r['best_acc']:.2f}%",
            f"{top10:.2f}%",
            f"ep {r['best_epoch']}",
            str(len(r["test_accs"])),
        ])

    col_labels = [
        "n", "Size", "gamma", "max_lag",
        "Best acc (%)", "Top-10 avg (%)",
        "Best epoch", "Epochs logged",
    ]

    n_rows  = len(rows)
    fig, ax = plt.subplots(figsize=(16, max(2.5, 0.5 * n_rows + 2.0)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1a6b3a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    if n_rows >= 1:
        for j in range(len(col_labels)):
            tbl[1, j].set_facecolor("#fff3b0")
            tbl[1, j].set_text_props(fontweight="bold")

    for i in range(2, n_rows + 1):
        fc = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fc)

    ax.set_title(
        "Ablation Summary — Power-law Ring Buffer — SSC",
        fontsize=11, pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_style()

    print(f"\n{'='*60}")
    print(f"  Power-law ablation plots — SSC")
    print(f"  Ablation dir : {ABLATION_DIR}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    runs = load_all_runs(ABLATION_DIR)
    if not runs:
        print("ERROR: no runs found.")
        return

    print(f"\n  {len(runs)} runs loaded\n")

    for size in ["small", "medium", "large"]:
        plot_curves_by_size(
            runs, "test_accs", "Test Accuracy (%)", size,
            os.path.join(OUTPUT_DIR, f"ablation_pl_ssc_accuracy_{size}.png"),
        )
    for size in ["small", "medium", "large"]:
        plot_curves_by_size(
            runs, "train_losses", "Training Loss", size,
            os.path.join(OUTPUT_DIR, f"ablation_pl_ssc_loss_{size}.png"),
        )

    plot_scatter(
        runs,
        os.path.join(OUTPUT_DIR, "ablation_pl_ssc_scatter.png"),
    )
    plot_summary_table(
        runs,
        os.path.join(OUTPUT_DIR, "ablation_pl_ssc_summary_table.png"),
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
