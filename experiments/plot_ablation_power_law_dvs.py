"""
Ablation plotting script — power law ring buffer, DVS Gesture.
Includes all buffer sizes (k=32, k=64, k=128), labelled in legend.

Produces:
    plots/ablation_power_law_dvs/
        ablation_pl_dvs_accuracy_{small,medium,large}.png
        ablation_pl_dvs_loss_{small,medium,large}.png
        ablation_pl_dvs_scatter.png
        ablation_pl_dvs_summary_table.png
"""

import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"
ABLATION_DIR = os.path.join(BASE_RESULTS, "ablation_power_law_dvs")
OUTPUT_DIR   = os.path.join(BASE_RESULTS, "plots", "ablation_power_law_dvs")

# n → colour
N_COLOURS = {
    5: "#2166ac",
    6: "#d6604d",
    7: "#4dac26",
    8: "#8073ac",
}

# buffer_size k → linestyle
K_LINESTYLES = {
    32 : "-",
    64 : "--",
    128: ":",
}

# buffer_size k → marker
K_MARKERS = {
    32 : "o",
    64 : "s",
    128: "^",
}

SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}


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


def load_all_runs(ablation_dir):
    if not os.path.isdir(ablation_dir):
        print(f"  [ERROR] Not found: {ablation_dir}")
        return []

    # Collect ALL vN directories
    version_dirs = []
    for entry in os.scandir(ablation_dir):
        if entry.is_dir() and entry.name.startswith("v"):
            try:
                version_dirs.append((int(entry.name[1:]), entry.path))
            except ValueError:
                pass

    if not version_dirs:
        print(f"  [ERROR] No vN dirs in {ablation_dir}")
        return []

    version_dirs.sort(key=lambda x: x[0])
    print(f"  Found versions: {[v for v,_ in version_dirs]}")

    # Collect all summaries across all versions
    all_summaries = []
    for _, vpath in version_dirs:
        sums = glob.glob(
            os.path.join(vpath, "**", "*_summary.json"), recursive=True
        )
        all_summaries.extend(sums)

    # Deduplicate by (n, network_size, buffer_size, max_lag) —
    # keep the most recently modified file for each unique combo
    seen     = {}
    for path in all_summaries:
        try:
            with open(path) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        n           = s.get("n_taps", None)
        size        = s.get("network_size", "?")
        buf_size    = int(s.get("buffer_size", s.get("buffer_k", 32)))
        max_lag     = s.get("buffer_max_lag", "?")
        key         = (n, size, buf_size, max_lag)
        mtime       = os.path.getmtime(path)
        if key not in seen or mtime > seen[key][1]:
            seen[key] = (path, mtime, s)

    runs = []
    for (n, size, buf_size, max_lag), (path, _, s) in seen.items():
        if n is None:
            continue

        gamma        = s.get("gamma", 2.0)
        taps         = s.get("taps", s.get("buffer_taps", []))
        best_acc     = s.get("best_acc", 0.0) * 100.0
        n_params     = s.get("n_params", 0)
        n_hidden     = s.get("n_hidden", "?")
        test_accs    = [a * 100.0 for a in s.get("test_accs", [])]
        train_losses = s.get("train_losses", [])
        epoch_times  = s.get("epoch_times", [])
        peak_mems    = s.get("peak_mems", [])

        color  = N_COLOURS.get(int(n), "#888888")
        ls     = K_LINESTYLES.get(buf_size, "-")
        marker = K_MARKERS.get(buf_size, "o")
        label  = f"n={n} k={buf_size}"

        runs.append(dict(
            n=int(n), network_size=size, n_hidden=n_hidden,
            gamma=gamma, max_lag=max_lag, taps=taps,
            buffer_size=buf_size,
            best_acc=best_acc, n_params=n_params,
            test_accs=test_accs, train_losses=train_losses,
            epoch_times=epoch_times, peak_mems=peak_mems,
            color=color, ls=ls, marker=marker, label=label,
            path=path,
        ))
        print(f"  [LOAD] n={n} {size:<8} k={buf_size:<4} "
              f"lag={max_lag:<4} best={best_acc:.2f}%  "
              f"params={n_params:,}")

    runs.sort(key=lambda r: (
        r["n"], r["buffer_size"], SIZE_ORDER.get(r["network_size"], 9)
    ))
    return runs


def smooth(values, window=7):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def add_legend(ax, runs):
    seen_n = sorted(set(r["n"] for r in runs))
    seen_k = sorted(set(r["buffer_size"] for r in runs))

    sep = Line2D([0], [0], color="none")

    n_handles = [
        Line2D([0], [0], color="none", label="Tap count (n)")
    ] + [
        Line2D([0], [0],
               color=N_COLOURS.get(n, "#888"),
               linewidth=2.0, label=f"n = {n}")
        for n in seen_n
    ]

    k_handles = [
        Line2D([0], [0], color="none", label="Buffer size (k)")
    ] + [
        Line2D([0], [0],
               color="grey", linewidth=1.5,
               linestyle=K_LINESTYLES.get(k, "-"),
               marker=K_MARKERS.get(k, "o"),
               markersize=5,
               label=f"k = {k}")
        for k in seen_k
    ]

    all_handles = n_handles + [sep] + k_handles

    leg = ax.legend(
        handles=all_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=2.0,
    )
    for text in leg.get_texts():
        if text.get_text() in ("Tap count (n)", "Buffer size (k)"):
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
        smoothed = smooth(np.array(raw), window=7)
        me       = max(1, len(epochs) // 8)
        best_val = max(raw) if key == "test_accs" else None
        leg_label = (f"n={r['n']} k={r['buffer_size']}  "
                     f"(best={best_val:.1f}%)"
                     if key == "test_accs"
                     else f"n={r['n']} k={r['buffer_size']}")

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
    ax.set_title(f"{size.capitalize()} network — DVS Gesture")
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
    fig, ax = plt.subplots(figsize=(9, 5.5))

    seen_n = set()
    seen_k = set()

    for r in runs:
        ax.scatter(
            r["n_params"], r["best_acc"],
            color=r["color"], marker=r["marker"],
            s=120, zorder=4,
            edgecolors="black", linewidths=0.6,
        )
        ax.annotate(
            f"n={r['n']} k={r['buffer_size']}\n{r['network_size']}",
            xy=(r["n_params"], r["best_acc"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=6, color=r["color"],
        )
        seen_n.add(r["n"])
        seen_k.add(r["buffer_size"])

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Parameters vs Best Test Accuracy — DVS Gesture")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
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

    k_handles = [
        Line2D([0], [0], color="none", label="Buffer size (k)")
    ] + [
        Line2D([0], [0], marker=K_MARKERS.get(k, "o"), color="w",
               markerfacecolor="#888888",
               markeredgecolor="black", markeredgewidth=0.6,
               markersize=8, label=f"k = {k}", linestyle="None")
        for k in sorted(seen_k)
    ]

    leg = ax.legend(
        handles=n_handles + [sep] + k_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=1.5, fontsize=9,
    )
    for text in leg.get_texts():
        if text.get_text() in ("Tap count (n)", "Buffer size (k)"):
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
        avg_mem = float(np.mean(r["peak_mems"]))   if r["peak_mems"]   else 0.0
        avg_t   = float(np.mean(r["epoch_times"])) if r["epoch_times"] else 0.0
        top10   = top_k_avg(r["test_accs"], k=10)
        rows.append([
            str(r["n"]),
            str(r["buffer_size"]),
            r["network_size"].capitalize(),
            str(r["n_hidden"]),
            str(r["gamma"]),
            str(r["max_lag"]),
            str(r["taps"]),
            f"{r['n_params']:,}",
            f"{r['best_acc']:.2f}%",
            f"{top10:.2f}%",
            f"{avg_mem:.1f}" if avg_mem > 0 else "—",
            f"{avg_t:.1f}"   if avg_t   > 0 else "—",
        ])

    col_labels = [
        "n", "k", "Size", "n_hidden", "gamma", "max_lag", "Taps",
        "Parameters", "Best acc (%)", "Top-10 avg (%)",
        "Avg mem (MB)", "Avg epoch (s)",
    ]

    n_rows  = len(rows)
    fig, ax = plt.subplots(figsize=(24, max(2.5, 0.5 * n_rows + 2.0)))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
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
        "Ablation Summary — Power-law Ring Buffer — DVS Gesture  |  All buffer sizes",
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
    print(f"  Power-law ablation plots — DVS Gesture (all k)")
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
            os.path.join(OUTPUT_DIR, f"ablation_pl_dvs_accuracy_{size}.png"),
        )
    for size in ["small", "medium", "large"]:
        plot_curves_by_size(
            runs, "train_losses", "Training Loss", size,
            os.path.join(OUTPUT_DIR, f"ablation_pl_dvs_loss_{size}.png"),
        )

    plot_scatter(
        runs,
        os.path.join(OUTPUT_DIR, "ablation_pl_dvs_scatter.png"),
    )
    plot_summary_table(
        runs,
        os.path.join(OUTPUT_DIR, "ablation_pl_dvs_summary_table.png"),
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
