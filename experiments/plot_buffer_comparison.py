# experiments/plot_benchmark_scatter.py
"""
Benchmark scatter plot — parameters (x, log scale) vs best test accuracy (y).

Exact RDS paths as found 25/05/2026:

    aad_v2_snntorch_500e/v3   → AAD V2 SNNTorch baseline  (69.26%)
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
        # ── Benchmark models ──────────────────────────────────────────
        ("aad_v2_snntorch_500e/v3/AdaptiveAxonalDelayNetV2SNNTorch_summary.json",
         "AAD V2 SNNTorch",      "o",  "#2c7bb6", "benchmark"),
        ("cadlif/v2/cAdLIF_summary.json",
         "CaDLIF",               "P",  "#e6ab02", "benchmark"),
        ("dcls/DCLS_summary.json",
         "DCLS",                 "*",  "#f46d43", "benchmark"),
        ("dhlif/v1/DHLIF_summary.json",
         "DHLIF",                "h",  "#4dac26", "benchmark"),
        ("event_ssm/v1/event_ssm_summary.json",
         "Event SSM",            "H",  "#1b7837", "benchmark"),
        # ── Buffer strategies ─────────────────────────────────────────
        ("exponential_tap/v1/ExponentialTapAADSNNTorch_summary.json",
         "Exponential buffer",   "v",  "#abdda4", "buffer"),
        ("linear_tap/v1/LinearTapAADSNNTorch_summary.json",
         "Linear buffer",        "^",  "#fdae61", "buffer"),
        ("log_buffer_snntorch/v2/LogBufferAADSNNTorch_summary.json",
         "Log buffer",           "s",  "#d7191c", "buffer"),
        ("power_law_tap/v1/PowerLawTapAADSNNTorch_summary.json",
         "Power-law buffer (γ=2)","<", "#9e0142", "buffer"),
        ("random_tap/v1/RandomTapAADSNNTorch_summary.json",
         "Random buffer",        ">",  "#756bb1", "buffer"),
    ],
    # ── Add ssc / dvs entries here when trained ───────────────────────
    "ssc": [],
    "dvs": [],
}


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_all_models(results_root: str, dataset: str) -> list:
    """
    Loads each model from its exact hardcoded path.
    Returns list of model dicts.
    """
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
    """Returns the label of the highest-accuracy buffer strategy."""
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
    X : number of parameters (log scale)
    Y : best test accuracy (%)

    Benchmark models  — larger markers (size 280), bold annotation
    Buffer strategies — smaller markers (size 160), normal annotation
    Best buffer       — outlined with a gold ring to highlight it

    Legend outside right, split by group.
    """
    best_buf_label = identify_best_buffer(models)

    fig, ax = plt.subplots(figsize=(13, 7))

    proxy_handles = {"benchmark": [], "buffer": []}

    for m in models:
        is_best_buf = (m["label"] == best_buf_label)
        size        = 280 if m["group"] == "benchmark" else 160
        zorder      = 6  if m["group"] == "benchmark" else 5
        fontweight  = "bold" if m["group"] == "benchmark" else "normal"
        edgecolor   = "gold"  if is_best_buf else "black"
        linewidth   = 2.0     if is_best_buf else 0.7

        ax.scatter(
            m["n_params"],
            m["best_acc"],
            marker     = m["marker"],
            color      = m["color"],
            s          = size,
            zorder     = zorder,
            edgecolors = edgecolor,
            linewidths = linewidth,
        )

        # Annotation — best buffer gets a star suffix
        ann_label = f"{m['label']} ★" if is_best_buf else m["label"]
        ax.annotate(
            ann_label,
            xy         = (m["n_params"], m["best_acc"]),
            xytext     = (10, 6),
            textcoords = "offset points",
            fontsize   = 8,
            color      = m["color"],
            fontweight = fontweight,
            clip_on    = False,
        )

        # Proxy artist for legend
        proxy = Line2D(
            [], [],
            marker          = m["marker"],
            color           = "w",
            markerfacecolor = m["color"],
            markeredgecolor = edgecolor,
            markeredgewidth = linewidth,
            markersize      = 10 if m["group"] == "benchmark" else 8,
            label           = ann_label,
            linestyle       = "None",
        )
        proxy_handles[m["group"]].append(proxy)

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters (log scale)", fontsize=13)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=13)
    ax.set_title(
        f"Parameters vs Best Test Accuracy — {dataset.upper()}\n"
        f"★ = best buffer strategy",
        fontsize=14,
        pad=14,
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, which="major", alpha=0.35, linestyle="--")
    ax.grid(True, which="minor", alpha=0.12)

    # Faint horizontal guide lines
    for m in models:
        ax.axhline(
            y         = m["best_acc"],
            color     = m["color"],
            alpha     = 0.10,
            linewidth = 0.7,
            linestyle = ":",
            zorder    = 1,
        )

    # ── Legend — split by group ───────────────────────────────────────
    sep = Line2D([0], [0], color="none")

    section_bm  = Line2D([0], [0], color="none",
                         label="── Benchmark models ──")
    section_buf = Line2D([0], [0], color="none",
                         label="── Buffer strategies ──")

    all_handles = (
        [section_bm]  + proxy_handles["benchmark"] +
        [sep, section_buf] + proxy_handles["buffer"]
    )

    leg = ax.legend(
        handles        = all_handles,
        loc            = "center left",
        bbox_to_anchor = (1.01, 0.5),
        borderaxespad  = 0,
        fontsize       = 9,
        frameon        = True,
        framealpha     = 0.95,
        edgecolor      = "#cccccc",
        handlelength   = 1.5,
    )
    for text in leg.get_texts():
        if text.get_text().startswith("──"):
            text.set_color("grey")
            text.set_style("italic")
            text.set_fontsize(8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_table(models: list,
                         dataset: str,
                         top_k: int,
                         output_path: str):
    """
    Columns:
        Rank | Model | Group | Parameters | Best acc | Top-k avg
             | Epoch of best | Avg peak mem (MB) | Avg epoch time (s)

    Sorted by best accuracy descending.
    Best model row — gold. Best buffer row — light pink outline.
    """
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
            "",                              # rank — filled below
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

    # Fill rank
    for rank, row in enumerate(rows_raw, 1):
        row[4][0] = str(rank)

    col_labels = [
        "Rank",
        "Model",
        "Group",
        "Parameters",
        "Best acc (%)",
        f"Top-{top_k} avg (%)",
        "Epoch of best",
        "Avg peak mem (MB)",
        "Avg epoch time (s)",
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

    # Header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1a6b3a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, color, label, group, _) in enumerate(rows_raw):
        row_idx = i + 1

        # Background
        if i == 0:
            fc = "#fff3b0"   # gold — overall best
        elif label == best_buf_label:
            fc = "#ffe0e0"   # light pink — best buffer
        else:
            fc = "#f5f5f5" if i % 2 == 0 else "white"

        for j in range(len(col_labels)):
            tbl[row_idx, j].set_facecolor(fc)

        # Bold for top row
        if i == 0:
            for j in range(len(col_labels)):
                tbl[row_idx, j].set_text_props(fontweight="bold")

        # Model name column coloured to match scatter
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

    # ── CONFIGURATION ─────────────────────────────────────────────────
    DATASET = "shd"    # "shd" | "ssc" | "dvs"
    TOP_K   = 10       # top-k epochs for average accuracy

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

    # ── LOAD ──────────────────────────────────────────────────────────
    models = load_all_models(BASE_RESULTS, DATASET)

    if not models:
        print("\nERROR: no models loaded. Check BASE_RESULTS and MODEL_PATH_MAP.")
        return

    print_summary(models, TOP_K)

    # ── PLOT ──────────────────────────────────────────────────────────
    print("  Generating plots ...")

    plot_benchmark_scatter(
        models, DATASET,
        os.path.join(OUTPUT_DIR, "benchmark_scatter.png"),
    )
    plot_benchmark_table(
        models, DATASET, TOP_K,
        os.path.join(OUTPUT_DIR, "benchmark_scatter_table.png"),
    )

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()