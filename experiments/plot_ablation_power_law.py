"""
Ablation plotting script — power law ring buffer.

Plots:
    accuracy_small/medium/large.png  — test accuracy per epoch per network size
    loss_small/medium/large.png      — train loss per epoch per network size
    ablation_pl_scatter.png          — params vs best acc (linear scale)
    ablation_pl_summary_table.png    — full results table
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
ABLATION_DIR = os.path.join(BASE_RESULTS, "ablation_power_law")
OUTPUT_DIR   = os.path.join(BASE_RESULTS, "plots", "ablation_power_law")

# n value → colour
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


def load_all_runs(ablation_dir):
    if not os.path.isdir(ablation_dir):
        print(f"  [ERROR] Not found: {ablation_dir}")
        return []

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

    version_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_dir = version_dirs[0][1]
    print(f"  Using: {latest_dir}")

    summaries = glob.glob(
        os.path.join(latest_dir, "**", "*_summary.json"), recursive=True
    )

    runs = []
    for path in sorted(summaries):
        try:
            with open(path) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [SKIP] {path} -- {e}")
            continue

        n            = s.get("n_taps", None)
        network_size = s.get("network_size", "?")
        n_hidden     = s.get("n_hidden", "?")
        gamma        = s.get("gamma", 2.0)
        max_lag      = s.get("buffer_max_lag", "?")
        taps         = s.get("taps", [])
        best_acc     = s.get("best_acc", 0.0) * 100.0
        n_params     = s.get("n_params", 0)
        test_accs    = [a * 100.0 for a in s.get("test_accs", [])]
        train_losses = s.get("train_losses", [])
        epoch_times  = s.get("epoch_times", [])
        peak_mems    = s.get("peak_mems", [])

        if n is None:
            continue

        runs.append(dict(
            n=int(n), network_size=network_size, n_hidden=n_hidden,
            gamma=gamma, max_lag=max_lag, taps=taps,
            best_acc=best_acc, n_params=n_params,
            test_accs=test_accs, train_losses=train_losses,
            epoch_times=epoch_times, peak_mems=peak_mems,
            color=N_COLOURS.get(int(n), "#888888"),
            marker=SIZE_MARKERS.get(network_size, "o"),
            ls=SIZE_LS.get(network_size, "-"),
        ))
        print(f"  [LOAD] n={n} {network_size:<8} best={best_acc:.2f}%  "
              f"params={n_params:,}  taps={taps}")

    runs.sort(key=lambda r: (r["n"], SIZE_ORDER.get(r["network_size"], 9)))
    return runs


def smooth(values, window=7):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


# ─────────────────────────────────────────────────────────────────────────────
# PLOT CURVES PER NETWORK SIZE
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves_by_size(runs, key, ylabel, size, output_path):
    """One plot per network size — all tap counts (n values) overlaid."""
    size_runs = [r for r in runs if r["network_size"] == size]
    if not size_runs:
        print(f"  [SKIP] {size} — no runs")
        return

    data_key = "test_accs" if key == "test_accs" else "train_losses"
    valid    = [r for r in size_runs if r[data_key]]
    if not valid:
        print(f"  [SKIP] {size} {key} — no data")
        return

    n_epochs = max(len(r[data_key]) for r in valid)
    fig, ax  = plt.subplots(figsize=(8, 4.5))

    for r in valid:
        raw    = r[data_key]
        if key == "test_accs":
            raw = raw  # already in %
        epochs   = list(range(1, len(raw) + 1))
        smoothed = smooth(np.array(raw), window=7)
        me       = max(1, len(epochs) // 8)
        best_val = max(raw) if key == "test_accs" else None
        leg_label = (f"n={r['n']}  (best={best_val:.1f}%)"
                     if key == "test_accs" else f"n={r['n']}")

        # Faint raw
        ax.plot(epochs, raw,
                color=r["color"], alpha=0.12,
                linewidth=0.8, zorder=2)
        # Smoothed
        ax.plot(epochs, smoothed,
                color=r["color"], linewidth=2.0,
                linestyle="-",
                marker=r["marker"],
                markevery=me, markersize=5,
                markeredgewidth=0.5,
                markeredgecolor="white",
                label=leg_label,
                zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{size.capitalize()} network")
    ax.set_xlim(1, n_epochs)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        handlelength=2.0,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER — linear scale
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(runs, output_path):
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))

    seen_n    = set()
    seen_size = set()

    for r in runs:
        ax.scatter(
            r["n_params"], r["best_acc"],
            color=r["color"], marker=r["marker"],
            s=120, zorder=4,
            edgecolors="black", linewidths=0.6,
        )
        seen_n.add(r["n"])
        seen_size.add(r["network_size"])

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Parameters vs Best Test Accuracy")
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
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0,
        frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=1.5,
        fontsize=9,
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


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

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
        "n", "Size", "n_hidden", "gamma", "max_lag", "Taps",
        "Parameters", "Best acc (%)", "Top-10 avg (%)",
        "Avg mem (MB)", "Avg epoch (s)",
    ]

    n_rows  = len(rows)
    fig, ax = plt.subplots(figsize=(22, max(2.5, 0.5 * n_rows + 2.0)))
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
        "Ablation Summary — Power-law Ring Buffer",
        fontsize=11, pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_style()

    print(f"\n{'='*60}")
    print(f"  Power-law ablation plots")
    print(f"  Ablation dir : {ABLATION_DIR}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    runs = load_all_runs(ABLATION_DIR)
    if not runs:
        print("ERROR: no runs found.")
        return

    print(f"\n  {len(runs)} runs loaded\n")

    # ── Accuracy curves — one plot per network size ───────────────────
    for size in ["small", "medium", "large"]:
        plot_curves_by_size(
            runs, "test_accs", "Test Accuracy (%)", size,
            os.path.join(OUTPUT_DIR, f"ablation_pl_accuracy_{size}.png"),
        )

    # ── Loss curves — one plot per network size ───────────────────────
    for size in ["small", "medium", "large"]:
        plot_curves_by_size(
            runs, "train_losses", "Training Loss", size,
            os.path.join(OUTPUT_DIR, f"ablation_pl_loss_{size}.png"),
        )

    # ── Scatter — linear scale ────────────────────────────────────────
    plot_scatter(
        runs,
        os.path.join(OUTPUT_DIR, "ablation_pl_scatter.png"),
    )

    # ── Summary table ─────────────────────────────────────────────────
    plot_summary_table(
        runs,
        os.path.join(OUTPUT_DIR, "ablation_pl_summary_table.png"),
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
