# experiments/plot_benchmark_scatter.py
"""
Benchmark scatter plot — parameters (x, linear scale) vs best test accuracy (y).

Exact RDS paths as found 25/05/2026:

    aad_v2_snntorch_500e/v3   → AAD SNNTorch baseline     (69.26%)
    cadlif/v2                 → CaDLIF                     (87.15%)
    dcls/                     → DCLS                       (88.74%)
    dhlif/v1                  → DHLIF                      (87.01%)
    event_ssm/v1              → Event SSM                  (81.58%)
    exponential_tap/v1        → Exponential buffer         (83.88%)
    linear_tap/v1             → Linear buffer              (88.74%)
    log_buffer_snntorch/v2    → Log buffer                 (90.72%)
    power_law_tap/v1          → Power-law buffer (γ=2)     (91.17%) ← best buffer
    random_tap/v1             → Random buffer              (84.36%)

three_factor_delay excluded — no trained results.

Change DATASET in main() to switch shd / ssc / dvs.

Outputs:
    plots/{dataset}/benchmark_scatter.png
    plots/{dataset}/benchmark_scatter_table.png
"""

import os
import subprocess
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# EXACT PATH MAP — hardcoded from diagnostic output
# Each entry: (exact_relative_path, display_label, marker, color, group)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH_MAP = {
    "shd": [
        # ── Benchmark models (circles) ────────────────────────────────
        ("aad_v2_snntorch_500e/v3/AdaptiveAxonalDelayNetV2SNNTorch_summary.json",
         "AAD SNNTorch",          "o", "black", "benchmark"),
        ("cadlif/v2/cAdLIF_summary.json",
         "CaDLIF",                "o", "black", "benchmark"),
        ("dcls/DCLS_summary.json",
         "DCLS",                  "o", "black", "benchmark"),
        ("dhlif/v1/DHLIF_summary.json",
         "DHLIF",                 "o", "black", "benchmark"),
        ("event_ssm/v1/event_ssm_summary.json",
         "Event SSM",             "o", "black", "benchmark"),
        # ── Buffer strategies (crosses) ───────────────────────────────
        ("exponential_tap/v1/ExponentialTapAADSNNTorch_summary.json",
         "Exponential buffer",    "x", "black", "buffer"),
        ("linear_tap/v1/LinearTapAADSNNTorch_summary.json",
         "Linear buffer",         "x", "black", "buffer"),
        ("log_buffer_snntorch/v2/LogBufferAADSNNTorch_summary.json",
         "Log buffer",            "x", "black", "buffer"),
        ("power_law_tap/v1/PowerLawTapAADSNNTorch_summary.json",
         "Power-law buffer (γ=2)","x", "black", "buffer"),
        ("random_tap/v1/RandomTapAADSNNTorch_summary.json",
         "Random buffer",         "x", "black", "buffer"),
    ],
    # ── Add ssc / dvs entries here when trained ───────────────────────
    "ssc": [],
    "dvs": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models(results_root: str, dataset: str) -> list:
    entries = MODEL_PATH_MAP.get(dataset, [])
    if not entries:
        print(f"  [WARN] No path map defined for dataset={dataset}")
        return []

    models = []
    print(f"\n  Loading models for dataset={dataset.upper()} ...")

    for rel_path, label, marker, color, group in entries:
        full_path = os.path.join(results_root, rel_path)
        if not os.path.isfile(full_path):
            print(f"  [SKIP] {label:<35}  — file not found:")
            print(f"         {full_path}")
            continue
        try:
            with open(full_path) as f:
                summary = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [SKIP] {label:<35}  — could not read: {e}")
            continue

        n_params = summary.get("n_params", 0)
        best_acc = summary.get("best_acc", 0.0) * 100.0
        print(f"  [LOAD] {label:<35}  "
              f"best={best_acc:.2f}%  params={n_params:,}")

        models.append(dict(
            label    = label,
            n_params = n_params,
            best_acc = best_acc,
            marker   = marker,
            color    = color,
            group    = group,
            summary  = summary,
        ))

    return models


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_top_k_avg(summary: dict, k: int = 10) -> float:
    accs = summary.get("test_accs", [])
    if not accs:
        return 0.0
    return float(np.mean(sorted(accs, reverse=True)[:k])) * 100.0


def get_avg_peak_mem(summary: dict) -> float:
    mems = summary.get("peak_mems", [])
    return float(np.mean(mems)) if mems else 0.0


def get_avg_epoch_time(summary: dict) -> float:
    times = summary.get("epoch_times", [])
    return float(np.mean(times)) if times else 0.0


def identify_best_buffer(models: list) -> str | None:
    buf = [m for m in models if m["group"] == "buffer"]
    if not buf:
        return None
    return max(buf, key=lambda m: m["best_acc"])["label"]


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — SCATTER
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_scatter(models: list,
                           dataset: str,
                           output_path: str):
    """
    X : number of parameters (linear scale)
    Y : highest test accuracy (%)

    Benchmark models  — filled circles (○), size 100
    Buffer strategies — crosses (×), size 100
    Best buffer       — annotated with ★

    Publication-quality styling: black/white, Times New Roman, no colour.
    """
    best_buf_label = identify_best_buffer(models)

    # ── Publication style ──────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         11,
        "axes.linewidth":    0.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon":    True,
        "legend.edgecolor":  "black",
        "legend.framealpha": 1.0,
        "figure.dpi":        300,
    })

    fig, ax = plt.subplots(figsize=(7, 5))

    proxy_handles = {"benchmark": [], "buffer": []}

    for m in models:
        is_best_buf   = (m["label"] == best_buf_label)
        is_benchmark  = (m["group"] == "benchmark")

        # Marker styling
        if is_benchmark:
            scatter_kw = dict(
                marker     = "o",
                facecolors = "black",
                edgecolors = "black",
                s          = 60,
                linewidths = 0.8,
                zorder     = 5,
            )
        else:
            scatter_kw = dict(
                marker     = "x",
                facecolors = "none",     # 'x' ignores facecolor but kept for clarity
                edgecolors = "black",
                s          = 60,
                linewidths = 1.2,
                zorder     = 4,
            )
            if is_best_buf:
                scatter_kw["s"]          = 80
                scatter_kw["linewidths"] = 1.8

        ax.scatter(m["n_params"], m["best_acc"], **scatter_kw)

        # Annotation
        ann_label = f"{m['label']} ★" if is_best_buf else m["label"]
        ax.annotate(
            ann_label,
            xy         = (m["n_params"], m["best_acc"]),
            xytext     = (8, 4),
            textcoords = "offset points",
            fontsize   = 8,
            color      = "black",
            fontweight = "bold" if is_benchmark else "normal",
            clip_on    = False,
        )

        # Proxy for legend
        if is_benchmark:
            proxy = Line2D(
                [], [],
                marker          = "o",
                color           = "black",
                markerfacecolor = "black",
                markeredgecolor = "black",
                markersize      = 6,
                label           = m["label"],
                linestyle       = "None",
            )
        else:
            proxy = Line2D(
                [], [],
                marker          = "x",
                color           = "black",
                markerfacecolor = "none",
                markeredgecolor = "black",
                markersize      = 6,
                label           = ann_label,
                linestyle       = "None",
                markeredgewidth = 1.2,
            )
        proxy_handles[m["group"]].append(proxy)

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xscale("linear")
    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Highest Test Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Comparison of Algorithm Performance on {dataset.upper()} Dataset",
        fontsize=12,
        pad=10,
    )

    # Format x-axis ticks as integers with commas
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    ax.grid(False)  # No grid — cleaner publication look

    # Light horizontal dashed reference lines at y-axis major ticks only
    ax.set_axisbelow(True)

    # ── Legend ────────────────────────────────────────────────────────
    sep          = Line2D([0], [0], color="none")
    section_bm   = Line2D([0], [0], color="none", label="Benchmark models")
    section_buf  = Line2D([0], [0], color="none", label="Buffer strategies")

    # Group headers as italic separators
    from matplotlib.patches import Patch
    header_bm  = Patch(facecolor="none", edgecolor="none",
                        label="Benchmark models")
    header_buf = Patch(facecolor="none", edgecolor="none",
                        label="Buffer strategies")

    all_handles = (
        [header_bm]  + proxy_handles["benchmark"] +
        [sep, header_buf] + proxy_handles["buffer"]
    )

    leg = ax.legend(
        handles        = all_handles,
        loc            = "lower right",
        fontsize       = 8,
        frameon        = True,
        framealpha     = 1.0,
        edgecolor      = "black",
        handlelength   = 1.2,
        borderpad      = 0.6,
        labelspacing   = 0.4,
    )
    for text in leg.get_texts():
        if text.get_text() in ("Benchmark models", "Buffer strategies"):
            text.set_style("italic")
            text.set_color("grey")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    # Reset rcParams to avoid affecting other plots
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — SUMMARY TABLE  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_table(models: list,
                         dataset: str,
                         top_k: int,
                         output_path: str):
    best_buf_label = identify_best_buffer(models)

    rows_raw = []
    for m in models:
        summary  = m.get("summary", {})
        top_avg  = get_top_k_avg(summary, top_k)
        accs_raw = summary.get("test_accs", [])
        best_ep  = int(np.argmax(accs_raw)) + 1 if accs_raw else "—"
        avg_mem  = get_avg_peak_mem(summary)
        avg_t    = get_avg_epoch_time(summary)

        rows_raw.append((m["best_acc"], m["color"], m["label"],
                         m["group"], [
            "",
            m["label"],
            m["group"].capitalize(),
            f"{m['n_params']:,}",
            f"{m['best_acc']:.2f}%",
            f"{top_avg:.2f}%",
            str(best_ep),
            f"{avg_mem:.1f}" if avg_mem > 0 else "—",
            f"{avg_t:.1f}"   if avg_t  > 0 else "—",
        ]))

    rows_raw.sort(key=lambda x: x[0], reverse=True)

    for rank, row in enumerate(rows_raw, 1):
        row[4][0] = str(rank)

    col_labels = [
        "Rank", "Model", "Group", "Parameters",
        "Best acc (%)", f"Top-{top_k} avg (%)",
        "Epoch of best", "Avg peak mem (MB)", "Avg epoch time (s)",
    ]
    table_rows = [r[4] for r in rows_raw]
    n_rows     = len(table_rows)

    fig, ax = plt.subplots(figsize=(18, max(2.5, 0.55 * n_rows + 2.2)))
    ax.axis("off")

    tbl = ax.table(
        cellText  = table_rows,
        colLabels = col_labels,
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1a6b3a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, color, label, group, _) in enumerate(rows_raw):
        row_idx = i + 1
        if i == 0:
            fc = "#fff3b0"
        elif label == best_buf_label:
            fc = "#ffe0e0"
        else:
            fc = "#f5f5f5" if i % 2 == 0 else "white"

        for j in range(len(col_labels)):
            tbl[row_idx, j].set_facecolor(fc)

        if i == 0:
            for j in range(len(col_labels)):
                tbl[row_idx, j].set_text_props(fontweight="bold")

        tbl[row_idx, 1].set_text_props(color=color, fontweight="bold")

    ax.set_title(
        f"Benchmark Comparison — {dataset.upper()}  ·  "
        f"Sorted by best test accuracy  ·  "
        f"Gold = overall best  ·  Pink = best buffer  ·  "
        f"Top-{top_k} avg = mean of {top_k} highest epoch accuracies",
        fontsize=11,
        pad=14,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(models: list, top_k: int):
    print(f"\n  {'Rank':<5} {'Model':<35} {'Group':<12} "
          f"{'Params':>12} {'Best acc':>10} {f'Top-{top_k} avg':>12}")
    print("  " + "-" * 92)
    for rank, m in enumerate(
            sorted(models, key=lambda x: x["best_acc"], reverse=True), 1):
        top_avg = get_top_k_avg(m.get("summary", {}), top_k)
        print(f"  {rank:<5} {m['label']:<35} {m['group']:<12} "
              f"{m['n_params']:>12,} {m['best_acc']:>9.2f}% "
              f"{top_avg:>11.2f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    DATASET = "shd"
    TOP_K   = 10

    BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"
    OUTPUT_DIR   = os.path.join(
        "/rds/general/user/ak2721/home/results/SNN_FYP/plots",
        DATASET,
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Benchmark scatter — dataset : {DATASET.upper()}")
    print(f"  Results root      : {BASE_RESULTS}")
    print(f"  Output dir        : {OUTPUT_DIR}")
    print(f"{'='*60}")

    models = load_all_models(BASE_RESULTS, DATASET)

    if not models:
        print("\nERROR: no models loaded. Check BASE_RESULTS and MODEL_PATH_MAP.")
        return

    print_summary(models, TOP_K)

    print("  Generating plots ...")

    plot_benchmark_scatter(
        models, DATASET,
        os.path.join(OUTPUT_DIR, "benchmark_scatter.png"),
    )
    plot_benchmark_table(
        models, DATASET, TOP_K,
        os.path.join(OUTPUT_DIR, "benchmark_scatter_table.png"),
    )

    REPO_PLOTS = "/rds/general/user/ak2721/home/SNN_FYP/results_plots"
    os.makedirs(REPO_PLOTS, exist_ok=True)

    for fname in ["benchmark_scatter.png", "benchmark_scatter_table.png"]:
        src = os.path.join(OUTPUT_DIR, fname)
        dst = os.path.join(REPO_PLOTS, f"{DATASET}_{fname}")
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied to repo: {dst}")

    subprocess.run(["git", "-C", "/rds/general/user/ak2721/home/SNN_FYP",
                    "add", "results_plots/"], check=False)
    subprocess.run(["git", "-C", "/rds/general/user/ak2721/home/SNN_FYP",
                    "commit", "-m", f"add benchmark scatter plots [{DATASET}]"],
                check=False)
    subprocess.run(["git", "-C", "/rds/general/user/ak2721/home/SNN_FYP",
                    "push", "origin", "main"], check=False)
    print("  Plots pushed to GitHub.")

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()