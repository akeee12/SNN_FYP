"""
Ablation plotting script — power law ring buffer, SSC.
Parses per-epoch data directly from PBS log files.

Log pattern:
  [k32_n5_lag16_g2.0_small][SSC][   1/100] Loss: X | Tr: X% | Te: X% | ...

Log files:
  /rds/general/user/ak2721/home/logs/ablation_pl_ring_ssc_{1..12}.log

Produces:
    plots/ablation_power_law_ssc/
        ablation_pl_ssc_accuracy_{small,medium,large}.png
        ablation_pl_ssc_loss_{small,medium,large}.png
        ablation_pl_ssc_scatter.png
        ablation_pl_ssc_summary_table.png
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"
LOGS_DIR     = "/rds/general/user/ak2721/home/logs"
OUTPUT_DIR   = os.path.join(BASE_RESULTS, "plots", "ablation_power_law_ssc")

LOG_PATTERN  = os.path.join(LOGS_DIR, "ablation_pl_ring_ssc_*.log")

# Regex to parse epoch lines
# [k32_n5_lag16_g2.0_small][SSC][   1/100] Loss: 6.60 | Tr: 24.57% | Te: 33.31% | ...
EPOCH_RE = re.compile(
    r"\[(?P<run>[^\]]+)\]\[SSC\]\[\s*(?P<epoch>\d+)/\d+\]"
    r".*?Loss:\s*(?P<loss>[\d.]+)"
    r".*?Tr:\s*(?P<train>[\d.]+)%"
    r".*?Te:\s*(?P<test>[\d.]+)%"
)

N_COLOURS = {
    5: "#2166ac",
    6: "#d6604d",
    7: "#4dac26",
    8: "#8073ac",
}

K_LINESTYLES = {32: "-", 64: "--", 128: ":"}
K_MARKERS    = {32: "o", 64: "s", 128: "^"}
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


def parse_run_name(run_name):
    """
    Parse k32_n5_lag16_g2.0_small into components.
    Returns (buf_size, n, max_lag, gamma, network_size) or None.
    """
    m = re.match(
        r"k(?P<k>\d+)_n(?P<n>\d+)_lag(?P<lag>\d+)_g(?P<g>[\d.]+)_(?P<size>\w+)",
        run_name
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


def load_all_runs():
    """
    Parse all SSC ablation log files.
    Returns dict keyed by run_name, each value is a dict with
    test_accs, train_losses, epochs, metadata.
    """
    log_files = sorted(glob.glob(LOG_PATTERN))
    print(f"  Found {len(log_files)} log files")

    # Accumulate per run
    run_data = {}

    for log_path in log_files:
        print(f"  Parsing: {os.path.basename(log_path)}")
        try:
            with open(log_path) as f:
                lines = f.readlines()
        except OSError as e:
            print(f"    [WARN] Could not read: {e}")
            continue

        for line in lines:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            run_name = m.group("run")
            epoch    = int(m.group("epoch"))
            loss     = float(m.group("loss"))
            train    = float(m.group("train"))
            test     = float(m.group("test"))

            if run_name not in run_data:
                parsed = parse_run_name(run_name)
                if parsed is None:
                    continue
                buf_size, n, max_lag, gamma, size = parsed
                run_data[run_name] = dict(
                    run_name     = run_name,
                    buf_size     = buf_size,
                    n            = n,
                    max_lag      = max_lag,
                    gamma        = gamma,
                    network_size = size,
                    epochs       = [],
                    test_accs    = [],
                    train_losses = [],
                    train_accs   = [],
                )

            run_data[run_name]["epochs"].append(epoch)
            run_data[run_name]["test_accs"].append(test)
            run_data[run_name]["train_losses"].append(loss)
            run_data[run_name]["train_accs"].append(train)

    # Convert to list and add derived fields
    runs = []
    for run_name, d in run_data.items():
        if not d["test_accs"]:
            continue

        # Sort by epoch in case logs interleaved
        order = np.argsort(d["epochs"])
        d["epochs"]       = [d["epochs"][i]       for i in order]
        d["test_accs"]    = [d["test_accs"][i]    for i in order]
        d["train_losses"] = [d["train_losses"][i] for i in order]
        d["train_accs"]   = [d["train_accs"][i]   for i in order]

        d["best_acc"]   = max(d["test_accs"])
        d["best_epoch"] = d["epochs"][np.argmax(d["test_accs"])]
        d["color"]      = N_COLOURS.get(d["n"], "#888888")
        d["ls"]         = K_LINESTYLES.get(d["buf_size"], "-")
        d["marker"]     = K_MARKERS.get(d["buf_size"], "o")

        runs.append(d)
        print(f"  [LOAD] {run_name:<35}  "
              f"epochs={len(d['epochs'])}  "
              f"best={d['best_acc']:.2f}% @ep{d['best_epoch']}")

    runs.sort(key=lambda r: (
        r["n"], r["buf_size"], SIZE_ORDER.get(r["network_size"], 9)
    ))
    return runs


def smooth(values, window=5):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def plot_curves_by_size(runs, key, ylabel, size, output_path):
    size_runs = [r for r in runs if r["network_size"] == size]
    if not size_runs:
        print(f"  [SKIP] {size} -- no runs")
        return

    valid = [r for r in size_runs if r[key]]
    if not valid:
        print(f"  [SKIP] {size} {key} -- no data")
        return

    n_epochs = max(r["epochs"][-1] for r in valid)
    fig, ax  = plt.subplots(figsize=(9, 5))

    for r in valid:
        raw      = r[key]
        epochs   = r["epochs"]
        smoothed = smooth(np.array(raw), window=5)
        me       = max(1, len(epochs) // 8)
        best_val = max(raw) if key == "test_accs" else None
        leg_label = (f"n={r['n']} k={r['buf_size']}  "
                     f"(best={best_val:.1f}%)"
                     if key == "test_accs"
                     else f"n={r['n']} k={r['buf_size']}")

        ax.plot(epochs, raw,
                color=r["color"], alpha=0.12,
                linewidth=0.8, zorder=2)
        ax.plot(epochs, smoothed,
                color=r["color"], linewidth=2.0,
                linestyle=r["ls"], marker=r["marker"],
                markevery=me, markersize=5,
                markeredgewidth=0.5, markeredgecolor="white",
                label=leg_label, zorder=3)

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
    fig, ax = plt.subplots(figsize=(9, 5.5))

    seen_n = set()
    seen_k = set()

    for r in runs:
        ax.scatter(
            r["n"], r["best_acc"],
            color=r["color"], marker=r["marker"],
            s=120, zorder=4,
            edgecolors="black", linewidths=0.6,
        )
        ax.annotate(
            f"k={r['buf_size']}\n{r['network_size']}",
            xy=(r["n"], r["best_acc"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7, color=r["color"],
        )
        seen_n.add(r["n"])
        seen_k.add(r["buf_size"])

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
        top10 = top_k_avg(r["test_accs"], k=10)
        rows.append([
            str(r["n"]),
            str(r["buf_size"]),
            r["network_size"].capitalize(),
            str(r["gamma"]),
            str(r["max_lag"]),
            f"{r['best_acc']:.2f}%",
            f"{top10:.2f}%",
            f"ep {r['best_epoch']}",
            str(len(r["epochs"])),
        ])

    col_labels = [
        "n", "k", "Size", "gamma", "max_lag",
        "Best acc (%)", "Top-10 avg (%)",
        "Best epoch", "Epochs logged",
    ]

    n_rows  = len(rows)
    fig, ax = plt.subplots(figsize=(18, max(2.5, 0.5 * n_rows + 2.0)))
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
        "Ablation Summary — Power-law Ring Buffer — SSC  "
        "(parsed from PBS logs)",
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
    print(f"  Power-law ablation plots — SSC (from logs)")
    print(f"  Logs dir     : {LOGS_DIR}")
    print(f"  Log pattern  : ablation_pl_ring_ssc_*.log")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    runs = load_all_runs()
    if not runs:
        print("ERROR: no runs parsed from logs.")
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
