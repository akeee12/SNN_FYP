"""
DCLS SHD sweep plots — baseline vs power_law_buf across hidden sizes.
All series truncated to 100 epochs for fair comparison.

Produces:
    plots/dcls_shd_sweep/
        dcls_shd_accuracy_curves.png
        dcls_shd_scatter.png
        dcls_shd_summary_table.png
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

RESULTS_JSON = "/tmp/dcls_sweep_results.json"
SWEEP_ROOT   = "/rds/general/user/ak2721/home/results/SNN_FYP/dcls_shd_pl_sweep"
OUTPUT_DIR   = "/rds/general/user/ak2721/home/results/SNN_FYP/plots/dcls_shd_sweep"
EPOCH_CAP    = 100

MODEL_COLOURS = {
    "baseline"      : "#d6604d",
    "power_law_buf" : "#2166ac",
}
MODEL_LABELS = {
    "baseline"      : "DCLS Baseline",
    "power_law_buf" : "Power-law Buffer",
}
HIDDEN_LINESTYLES = {64: ":", 128: "--", 256: "-"}
HIDDEN_MARKERS    = {64: "^", 128: "s",  256: "o"}


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


def load_runs():
    runs = []

    if os.path.isfile(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            raw = json.load(f)
        for r in raw:
            r.setdefault("test_accs",   [])
            r.setdefault("peak_mems",   [])
            r.setdefault("epoch_times", [])
            runs.append(r)
        for r in runs:
            if not r["test_accs"]:
                _backfill(r)
    else:
        for model_type in ["baseline", "power_law_buf"]:
            for h in [64, 128, 256]:
                run_dir   = os.path.join(SWEEP_ROOT, f"{model_type}_h{h}")
                summaries = sorted(glob.glob(
                    os.path.join(run_dir, "v*", "*_summary.json")
                ))
                if not summaries:
                    print(f"  [MISSING] {model_type} h={h}")
                    continue
                with open(summaries[-1]) as f:
                    s = json.load(f)
                runs.append({
                    "model_type" : model_type,
                    "n_hidden"   : h,
                    "best_acc"   : s["best_acc"],
                    "n_params"   : s["n_params"],
                    "test_accs"  : s.get("test_accs",   []),
                    "peak_mems"  : s.get("peak_mems",   []),
                    "epoch_times": s.get("epoch_times", []),
                })

    # normalise to % and cap at EPOCH_CAP
    for r in runs:
        if r["best_acc"] <= 1.0:
            r["best_acc"] *= 100.0
        accs = r["test_accs"]
        if accs and max(accs) <= 1.0:
            accs = [a * 100.0 for a in accs]
        r["test_accs"] = accs[:EPOCH_CAP]

    runs.sort(key=lambda r: (r["model_type"], r["n_hidden"]))
    for r in runs:
        print(f"  [LOAD] {r['model_type']:15s} h={r['n_hidden']:3d}  "
              f"best={r['best_acc']:.2f}%  params={r['n_params']:,}  "
              f"epochs(capped)={len(r['test_accs'])}")
    return runs


def _backfill(r):
    run_dir   = os.path.join(SWEEP_ROOT, f"{r['model_type']}_h{r['n_hidden']}")
    summaries = sorted(glob.glob(
        os.path.join(run_dir, "v*", "*_summary.json")
    ))
    if not summaries:
        return
    with open(summaries[-1]) as f:
        s = json.load(f)
    raw = s.get("test_accs", [])
    r["test_accs"]   = [a * 100.0 for a in raw] if raw and max(raw) <= 1.0 else raw
    r["peak_mems"]   = s.get("peak_mems",   [])
    r["epoch_times"] = s.get("epoch_times", [])


def smooth(values, window=7):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _combined_legend(ax, runs):
    seen_models = sorted(set(r["model_type"] for r in runs))
    seen_hidden = sorted(set(r["n_hidden"]   for r in runs))
    sep = Line2D([0], [0], color="none")

    model_handles = [Line2D([0], [0], color="none", label="Model")] + [
        Line2D([0], [0], color=MODEL_COLOURS.get(m, "#888"),
               linewidth=2.0, label=MODEL_LABELS.get(m, m))
        for m in seen_models
    ]
    h_handles = [Line2D([0], [0], color="none", label="Hidden size")] + [
        Line2D([0], [0], color="grey", linewidth=1.5,
               linestyle=HIDDEN_LINESTYLES.get(h, "-"),
               marker=HIDDEN_MARKERS.get(h, "o"),
               markersize=5, label=f"h = {h}")
        for h in seen_hidden
    ]

    leg = ax.legend(
        handles=model_handles + [sep] + h_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0, frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=2.0,
    )
    for txt in leg.get_texts():
        if txt.get_text() in ("Model", "Hidden size"):
            txt.set_color("grey")
            txt.set_style("italic")
            txt.set_fontsize(8)


def plot_accuracy_curves(runs, output_path):
    valid = [r for r in runs if r["test_accs"]]
    if not valid:
        print("  [SKIP] accuracy curves — no data")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for r in valid:
        raw      = r["test_accs"]
        epochs   = list(range(1, len(raw) + 1))
        smoothed = smooth(raw, window=7)
        me       = max(1, len(epochs) // 8)
        color    = MODEL_COLOURS.get(r["model_type"], "#888")
        ls       = HIDDEN_LINESTYLES.get(r["n_hidden"], "-")
        marker   = HIDDEN_MARKERS.get(r["n_hidden"], "o")
        lbl      = (f"{MODEL_LABELS.get(r['model_type'], r['model_type'])} "
                    f"h={r['n_hidden']}  (best={max(raw):.1f}%)")

        ax.plot(epochs, raw,      color=color, alpha=0.12, linewidth=0.8, zorder=2)
        ax.plot(epochs, smoothed, color=color, linewidth=2.0,
                linestyle=ls, marker=marker, markevery=me,
                markersize=5, markeredgewidth=0.5,
                markeredgecolor="white", label=lbl, zorder=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"SHD — Test Accuracy (first {EPOCH_CAP} epochs)")
    ax.set_xlim(1, EPOCH_CAP)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    _combined_legend(ax, valid)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_scatter(runs, output_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for r in runs:
        color  = MODEL_COLOURS.get(r["model_type"], "#888")
        marker = HIDDEN_MARKERS.get(r["n_hidden"], "o")
        ax.scatter(r["n_params"], r["best_acc"],
                   color=color, marker=marker,
                   s=140, zorder=4, edgecolors="black", linewidths=0.7)
        ax.annotate(
            f"h={r['n_hidden']}\n{r['best_acc']:.2f}%",
            xy=(r["n_params"], r["best_acc"]),
            xytext=(7, 4), textcoords="offset points",
            fontsize=7.5, color=color,
        )

    for model_type in ["baseline", "power_law_buf"]:
        subset = sorted(
            [r for r in runs if r["model_type"] == model_type],
            key=lambda r: r["n_params"],
        )
        if subset:
            ax.plot(
                [r["n_params"] for r in subset],
                [r["best_acc"]  for r in subset],
                color=MODEL_COLOURS[model_type],
                linewidth=1.2, linestyle="--", alpha=0.45, zorder=2,
            )

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Best Test Accuracy (%)")
    ax.set_title("Parameter Efficiency — DCLS SHD Sweep")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    seen_models = sorted(set(r["model_type"] for r in runs))
    seen_hidden = sorted(set(r["n_hidden"]   for r in runs))
    sep = Line2D([0], [0], color="none")

    model_handles = [Line2D([0], [0], color="none", label="Model")] + [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=MODEL_COLOURS.get(m, "#888"),
               markeredgecolor="black", markeredgewidth=0.7,
               markersize=9, linestyle="None",
               label=MODEL_LABELS.get(m, m))
        for m in seen_models
    ]
    h_handles = [Line2D([0], [0], color="none", label="Hidden size")] + [
        Line2D([0], [0], marker=HIDDEN_MARKERS.get(h, "o"), color="w",
               markerfacecolor="#888888", markeredgecolor="black",
               markeredgewidth=0.7, markersize=9, linestyle="None",
               label=f"h = {h}")
        for h in seen_hidden
    ]

    leg = ax.legend(
        handles=model_handles + [sep] + h_handles,
        loc="lower right", frameon=True, framealpha=0.95,
        edgecolor="#cccccc", handlelength=1.5, fontsize=9,
    )
    for txt in leg.get_texts():
        if txt.get_text() in ("Model", "Hidden size"):
            txt.set_color("grey")
            txt.set_style("italic")
            txt.set_fontsize(8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_table(runs, output_path):
    def top_k_avg(accs, k=10):
        return float(np.mean(sorted(accs, reverse=True)[:k])) if accs else 0.0

    baseline_acc = {
        r["n_hidden"]: r["best_acc"]
        for r in runs if r["model_type"] == "baseline"
    }

    rows = []
    for r in sorted(runs, key=lambda x: x["best_acc"], reverse=True):
        bl      = baseline_acc.get(r["n_hidden"])
        is_pl   = r["model_type"] == "power_law_buf"
        delta   = f"+{r['best_acc'] - bl:.2f}%" if (is_pl and bl) else "—"
        top10   = top_k_avg(r["test_accs"], k=10)
        avg_mem = float(np.mean(r["peak_mems"]))   if r["peak_mems"]   else 0.0
        avg_t   = float(np.mean(r["epoch_times"])) if r["epoch_times"] else 0.0
        rows.append({
            "is_pl": is_pl,
            "delta_pos": delta.startswith("+"),
            "cols": [
                MODEL_LABELS.get(r["model_type"], r["model_type"]),
                str(r["n_hidden"]),
                f"{r['n_params']:,}",
                f"{r['best_acc']:.2f}%",
                f"{top10:.2f}%" if r["test_accs"] else "—",
                f"{avg_mem:.1f}" if avg_mem > 0 else "—",
                f"{avg_t:.1f}"   if avg_t   > 0 else "—",
                delta,
            ],
        })

    col_labels = [
        "Model", "n_hidden", "Parameters",
        "Best acc", "Top-10 avg", "Avg mem (MB)", "Avg epoch (s)", "Δ vs baseline",
    ]

    n_rows  = len(rows)
    fig, ax = plt.subplots(figsize=(18, max(2.5, 0.55 * n_rows + 2.0)))
    ax.axis("off")

    tbl = ax.table(
        cellText=[r["cols"] for r in rows],
        colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1a3a6b")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows, start=1):
        base_fc = "#dce9f7" if row["is_pl"] else ("#fdf3f2" if i % 2 == 0 else "#fff8f7")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(base_fc)
        if row["delta_pos"]:
            tbl[i, len(col_labels) - 1].set_facecolor("#c8f0c8")
            tbl[i, len(col_labels) - 1].set_text_props(fontweight="bold")

    ax.set_title(
        "DCLS SHD Sweep — Baseline vs Power-law Buffer",
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
    print(f"  DCLS SHD sweep — baseline vs power_law_buf")
    print(f"  Epoch cap    : {EPOCH_CAP}")
    print(f"  Results JSON : {RESULTS_JSON}")
    print(f"  Output dir   : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    runs = load_runs()
    if not runs:
        print("ERROR: no runs found.")
        return

    print(f"\n  {len(runs)} runs loaded\n")

    plot_accuracy_curves(
        runs,
        os.path.join(OUTPUT_DIR, "dcls_shd_accuracy_curves.png"),
    )
    plot_scatter(
        runs,
        os.path.join(OUTPUT_DIR, "dcls_shd_scatter.png"),
    )
    plot_summary_table(
        runs,
        os.path.join(OUTPUT_DIR, "dcls_shd_summary_table.png"),
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()