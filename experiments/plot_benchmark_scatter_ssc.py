"""
Benchmark scatter plot — SSC dataset.
Parameters (x, linear scale) vs best test accuracy (y).

All data hardcoded from PBS logs and summary JSONs as of 07/06/2026.

Note: CaDLIF and DHLIF hit walltime before completing any epochs on SSC.
They are included in the table with params only, excluded from scatter.

Outputs:
    plots/ssc/benchmark_scatter.png
    plots/ssc/benchmark_scatter_table.png
"""

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_RESULTS = "/rds/general/user/ak2721/home/results/SNN_FYP"
OUTPUT_DIR   = os.path.join(BASE_RESULTS, "plots", "ssc")

# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED DATA — from PBS logs and summary JSONs
# (label, n_params, best_acc, top5, top10, avg_mem, avg_time, group,
#  marker, color, has_acc)
# has_acc=False → exclude from scatter, include in table only
# ─────────────────────────────────────────────────────────────────────────────

MODELS = [
    # ── Benchmark models ──────────────────────────────────────────────
    dict(label="AAD SNNTorch",   n_params=111_011,  best_acc=60.81,
         top5=60.44,  top10=60.27, avg_mem=548.2,  avg_time=830.7,
         group="benchmark", marker="o", color="#2166ac", has_acc=True,
         rel_path="aad_v2_ssc_500e/v2/training_log.csv"),
    dict(label="Event SSM",      n_params=294_307,  best_acc=76.62,
         top5=76.25,  top10=76.10, avg_mem=234.5,  avg_time=1694.5,
         group="benchmark", marker="o", color="#e08214", has_acc=True,
         rel_path="event_ssm_ssc/v1/training_log.csv"),
    dict(label="DCLS + buffer",  n_params=482_979,  best_acc=76.30,
         top5=76.15,  top10=76.09, avg_mem=1813.7, avg_time=109.4,
         group="benchmark", marker="o", color="#4dac26", has_acc=True,
         rel_path="dcls_power_law_ring_buffer_ssc/v1/DCLSPowerLawRingBuffer_summary.json"),
    dict(label="CaDLIF",         n_params=501_027,  best_acc=None,
         top5=None,   top10=None,  avg_mem=2950.8, avg_time=1000.1,
         group="benchmark", marker="o", color="#d6604d", has_acc=False,
         rel_path=None),
    dict(label="DHLIF",          n_params=323_107,  best_acc=None,
         top5=None,   top10=None,  avg_mem=2416.6, avg_time=2442.1,
         group="benchmark", marker="o", color="#8073ac", has_acc=False,
         rel_path=None),
    # ── Buffer strategies ─────────────────────────────────────────────
    dict(label="Power-law buffer (\u03b3=2)", n_params=72_515, best_acc=66.05,
         top5=65.92,  top10=65.68, avg_mem=636.5,  avg_time=1083.9,
         group="buffer", marker="*", color="#555555", has_acc=True,
         rel_path="power_law_tap_ssc/v2/training_log.csv"),
    dict(label="Exponential buffer",          n_params=72_515, best_acc=65.74,
         top5=65.64,  top10=65.53, avg_mem=636.5,  avg_time=1048.3,
         group="buffer", marker="^", color="#555555", has_acc=True,
         rel_path="exponential_tap_ssc/v3/training_log.csv"),
    dict(label="Log buffer",                  n_params=72_515, best_acc=65.80,
         top5=65.44,  top10=65.21, avg_mem=636.5,  avg_time=1089.5,
         group="buffer", marker="D", color="#555555", has_acc=True,
         rel_path="log_ring_buffer_ssc/v2/training_log.csv"),
    dict(label="Linear buffer",               n_params=72_515, best_acc=65.17,
         top5=64.98,  top10=64.82, avg_mem=636.5,  avg_time=1083.6,
         group="buffer", marker="s", color="#555555", has_acc=True,
         rel_path="linear_tap_ssc/v2/training_log.csv"),
    dict(label="Random buffer",               n_params=72_515, best_acc=64.09,
         top5=63.90,  top10=63.69, avg_mem=636.5,  avg_time=1086.2,
         group="buffer", marker="P", color="#555555", has_acc=True,
         rel_path="random_tap_ssc/v2/training_log.csv"),
]


def get_best_buffer_label():
    buf = [m for m in MODELS if m["group"] == "buffer" and m["has_acc"]]
    return max(buf, key=lambda m: m["best_acc"])["label"] if buf else None


def plot_scatter(output_path):
    plt.rcParams.update({
        "font.family"         : "serif",
        "font.size"           : 11,
        "axes.linewidth"      : 0.8,
        "xtick.direction"     : "in",
        "ytick.direction"     : "in",
        "xtick.minor.visible" : True,
        "ytick.minor.visible" : True,
        "legend.frameon"      : True,
        "legend.edgecolor"    : "black",
        "legend.framealpha"   : 1.0,
        "figure.dpi"          : 300,
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    best_buf_label = get_best_buffer_label()
    proxy_bm, proxy_buf = [], []

    for m in MODELS:
        if not m["has_acc"]:
            continue  # skip from scatter — no accuracy data

        is_bm       = m["group"] == "benchmark"
        is_best_buf = m["label"] == best_buf_label

        if is_bm:
            ax.scatter(
                m["n_params"], m["best_acc"],
                marker=m["marker"], facecolors=m["color"],
                edgecolors=m["color"], s=80,
                linewidths=0.8, zorder=5,
            )
            proxy_bm.append(Line2D(
                [], [], marker="o", color="w",
                markerfacecolor=m["color"],
                markeredgecolor=m["color"],
                markersize=7, label=m["label"], linestyle="None",
            ))
        else:
            ec = "gold" if is_best_buf else m["color"]
            lw = 1.8   if is_best_buf else 0.8
            sz = 100   if is_best_buf else 70
            ax.scatter(
                m["n_params"], m["best_acc"],
                marker=m["marker"],
                facecolors="none" if m["marker"] == "*" else m["color"],
                edgecolors=ec, s=sz,
                linewidths=lw, zorder=4,
            )
            ann = f"{m['label']} \u2605" if is_best_buf else m["label"]
            proxy_buf.append(Line2D(
                [], [], marker=m["marker"], color="w",
                markerfacecolor="none" if m["marker"] == "*" else m["color"],
                markeredgecolor=ec,
                markeredgewidth=lw,
                markersize=7 if m["marker"] != "*" else 9,
                label=ann, linestyle="None",
            ))

    ax.set_xlabel("Number of Parameters", fontsize=12)
    ax.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Comparison of Algorithm Performance on SSC Dataset\n"
        "(CaDLIF and DHLIF excluded — hit walltime before training)",
        fontsize=11, pad=10,
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(False)

    sep        = Line2D([0], [0], color="none", label=" ")
    header_bm  = Patch(facecolor="none", edgecolor="none",
                       label="Benchmark models")
    header_buf = Patch(facecolor="none", edgecolor="none",
                       label="Buffer strategies")

    leg = ax.legend(
        handles        = [header_bm] + proxy_bm + [sep, header_buf] + proxy_buf,
        loc            = "upper left",
        bbox_to_anchor = (1.02, 1.0),
        borderaxespad  = 0,
        fontsize       = 8.5,
        frameon        = True,
        framealpha     = 1.0,
        edgecolor      = "black",
        handlelength   = 1.2,
        borderpad      = 0.7,
        labelspacing   = 0.5,
    )
    for text in leg.get_texts():
        if text.get_text() in ("Benchmark models", "Buffer strategies"):
            text.set_style("italic")
            text.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Saved: {output_path}")


def plot_table(output_path):
    best_buf_label = get_best_buffer_label()

    rows_raw = []
    for m in MODELS:
        best_str  = f"{m['best_acc']:.2f}%" if m["best_acc"] is not None else "—"
        top5_str  = f"{m['top5']:.2f}%"     if m["top5"]     is not None else "—"
        top10_str = f"{m['top10']:.2f}%"    if m["top10"]    is not None else "—"
        note      = "" if m["has_acc"] else " (walltime)"

        rows_raw.append((
            m["best_acc"] if m["best_acc"] is not None else -1,
            m["color"], m["label"], m["group"], [
                "",
                m["label"] + note,
                m["group"].capitalize(),
                f"{m['n_params']:,}",
                best_str,
                top5_str,
                top10_str,
                f"{m['avg_mem']:.0f}" if m["avg_mem"] > 0 else "—",
                f"{m['avg_time']:.1f}" if m["avg_time"] > 0 else "—",
            ]
        ))

    rows_raw.sort(key=lambda x: x[0], reverse=True)
    for rank, row in enumerate(rows_raw, 1):
        row[4][0] = str(rank)

    col_labels = [
        "Rank", "Model", "Group", "Parameters",
        "Best acc (%)", "Top-5 avg (%)", "Top-10 avg (%)",
        "Avg mem (MB)", "Avg epoch (s)",
    ]
    table_rows = [r[4] for r in rows_raw]
    n_rows     = len(table_rows)

    fig, ax = plt.subplots(figsize=(20, max(2.5, 0.55 * n_rows + 2.2)))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(col_labels))))

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1a6b3a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, (_, color, label, group, _) in enumerate(rows_raw):
        row_idx = i + 1
        m_data  = next(m for m in MODELS if m["label"] == label.replace(" (walltime)", ""))

        if i == 0 and m_data["has_acc"]:
            fc = "#fff3b0"
        elif label.replace(" (walltime)", "") == best_buf_label:
            fc = "#ffe0e0"
        elif not m_data["has_acc"]:
            fc = "#f0f0f0"   # grey for no-data rows
        else:
            fc = "#f5f5f5" if i % 2 == 0 else "white"

        for j in range(len(col_labels)):
            tbl[row_idx, j].set_facecolor(fc)
        if i == 0 and m_data["has_acc"]:
            for j in range(len(col_labels)):
                tbl[row_idx, j].set_text_props(fontweight="bold")
        tbl[row_idx, 1].set_text_props(color=color, fontweight="bold")

    ax.set_title(
        "Benchmark Comparison — SSC  |  "
        "Sorted by best test accuracy  |  "
        "Gold = overall best  |  Pink = best buffer  |  "
        "Grey = no accuracy data (hit walltime)  |  "
        "Top-5/10 avg = mean of 5/10 highest epoch accuracies",
        fontsize=10, pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SSC benchmark scatter")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"{'='*60}")

    print(f"\n  Models (sorted by best accuracy):")
    for m in sorted(MODELS, key=lambda x: x["best_acc"] or -1, reverse=True):
        acc_str = f"{m['best_acc']:.2f}%" if m["best_acc"] else "no data"
        print(f"    {m['label']:<35}  {acc_str}  params={m['n_params']:,}")

    print(f"\n  Generating plots ...")
    plot_scatter(os.path.join(OUTPUT_DIR, "benchmark_scatter.png"))
    plot_table(os.path.join(OUTPUT_DIR, "benchmark_scatter_table.png"))

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()