"""
Benchmark scatter plot — DVS Gesture dataset.
Parameters (x, linear scale) vs best test accuracy (y).

Outputs:
    plots/dvs/benchmark_scatter.png
    plots/dvs/benchmark_scatter_table.png
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
OUTPUT_DIR   = os.path.join(BASE_RESULTS, "plots", "dvs")

MODEL_PATH_MAP = [
    ("cadlif_dvs/v2/cAdLIF_summary.json",
     "CaDLIF",               "o", "#2166ac", "benchmark"),
    ("event_ssm_dvs/v1/event_ssm_summary.json",
     "Event SSM",            "o", "#e08214", "benchmark"),
    ("dcls_power_law_ring_buffer_dvs/v2/DCLSPowerLawRingBuffer_summary.json",
     "DCLS + buffer",        "o", "#4dac26", "benchmark"),
    ("dhlif_dvs/v2/DHLIF_summary.json",
     "DHLIF",                "o", "#8073ac", "benchmark"),
    ("aad_v2_dvs_500e/v1/AdaptiveAxonalDelayNetV2SNNTorch_summary.json",
     "AAD SNNTorch",         "o", "#d6604d", "benchmark"),
    ("power_law_tap_dvs/v1/PowerLawTapAADSNNTorch_summary.json",
     "Power-law buffer (\u03b3=2)", "*", "#555555", "buffer"),
    ("exponential_tap_dvs/v2/ExponentialTapAADSNNTorch_summary.json",
     "Exponential buffer",   "^", "#555555", "buffer"),
    ("log_ring_buffer_dvs/v1/LogRingBufferAADSNNTorch_summary.json",
     "Log buffer",           "D", "#555555", "buffer"),
    ("linear_tap_dvs/v1/LinearTapAADSNNTorch_summary.json",
     "Linear buffer",        "s", "#555555", "buffer"),
    ("random_tap_dvs/v1/RandomTapAADSNNTorch_summary.json",
     "Random buffer",        "P", "#555555", "buffer"),
]

# Hardcoded from PBS logs — models whose summary JSON lacks peak_mems/epoch_times
HARDCODED_STATS = {
    "CaDLIF" : dict(avg_mem=2142.0, avg_time=23.5),
    "DHLIF"  : dict(avg_mem=536.0,  avg_time=33.9),
}


def load_model(rel_path, label):
    full_path = os.path.join(BASE_RESULTS, rel_path)
    if not os.path.isfile(full_path):
        print(f"  [SKIP] {label:<35} -- not found: {full_path}")
        return None

    if full_path.endswith(".json"):
        with open(full_path) as f:
            s = json.load(f)
        test_accs = [a * 100.0 for a in s.get("test_accs", [])]
        if not test_accs:
            test_accs = [s.get("best_acc", 0.0) * 100.0]
        return dict(
            test_accs   = test_accs,
            n_params    = s.get("n_params", 0),
            peak_mems   = s.get("peak_mems", []),
            epoch_times = s.get("epoch_times", []),
        )
    else:
        test_accs, mems, times = [], [], []
        with open(full_path) as f:
            for row in csv.DictReader(f):
                try: test_accs.append(float(row["test_acc"]) * 100.0)
                except: pass
                try: mems.append(float(row["peak_mem_mb"]))
                except: pass
                try: times.append(float(row["epoch_time_s"]))
                except: pass
        cfg_path = os.path.join(os.path.dirname(full_path), "config.json")
        n_params = 0
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                c = json.load(f)
            n_params = c.get("n_params", c.get("num_params", 0))
        return dict(
            test_accs   = test_accs,
            n_params    = n_params,
            peak_mems   = mems,
            epoch_times = times,
        )


def load_all_models():
    models = []
    print(f"\n  Loading DVS models ...")
    for rel_path, label, marker, color, group in MODEL_PATH_MAP:
        data = load_model(rel_path, label)
        if data is None:
            continue

        accs     = data["test_accs"]
        best_acc = max(accs) if accs else 0.0
        top5     = float(np.mean(sorted(accs, reverse=True)[:5]))  if accs else 0.0
        top10    = float(np.mean(sorted(accs, reverse=True)[:10])) if accs else 0.0
        avg_mem  = float(np.mean(data["peak_mems"]))   if data["peak_mems"]   else 0.0
        avg_time = float(np.mean(data["epoch_times"])) if data["epoch_times"] else 0.0

        # Apply hardcoded stats from logs if summary JSON lacks them
        if label in HARDCODED_STATS and avg_mem == 0.0:
            avg_mem  = HARDCODED_STATS[label]["avg_mem"]
            avg_time = HARDCODED_STATS[label]["avg_time"]

        print(f"  [LOAD] {label:<35}  best={best_acc:.2f}%  "
              f"top5={top5:.2f}%  top10={top10:.2f}%  "
              f"params={data['n_params']:,}  "
              f"mem={avg_mem:.0f}MB  time={avg_time:.1f}s")

        models.append(dict(
            label     = label,
            marker    = marker,
            color     = color,
            group     = group,
            n_params  = data["n_params"],
            best_acc  = best_acc,
            top5      = top5,
            top10     = top10,
            avg_mem   = avg_mem,
            avg_time  = avg_time,
            test_accs = accs,
        ))
    return models


def get_best_buffer_label(models):
    buf = [m for m in models if m["group"] == "buffer"]
    return max(buf, key=lambda m: m["best_acc"])["label"] if buf else None


def plot_scatter(models, output_path):
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
    best_buf_label = get_best_buffer_label(models)
    proxy_bm, proxy_buf = [], []

    for m in models:
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
        "Comparison of Algorithm Performance on DVS Gesture Dataset",
        fontsize=12, pad=10,
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(False)

    sep        = Line2D([0], [0], color="none", label=" ")
    header_bm  = Patch(facecolor="none", edgecolor="none", label="Benchmark models")
    header_buf = Patch(facecolor="none", edgecolor="none", label="Buffer strategies")

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


def plot_table(models, output_path):
    best_buf_label = get_best_buffer_label(models)

    rows_raw = []
    for m in models:
        accs    = m["test_accs"]
        best_ep = int(np.argmax(accs)) + 1 if accs else "—"
        rows_raw.append((m["best_acc"], m["color"], m["label"], m["group"], [
            "",
            m["label"],
            m["group"].capitalize(),
            f"{m['n_params']:,}",
            f"{m['best_acc']:.2f}%",
            f"{m['top5']:.2f}%",
            f"{m['top10']:.2f}%",
            str(best_ep),
            f"{m['avg_mem']:.0f}" if m["avg_mem"] > 0 else "—",
            f"{m['avg_time']:.1f}" if m["avg_time"] > 0 else "—",
        ]))

    rows_raw.sort(key=lambda x: x[0], reverse=True)
    for rank, row in enumerate(rows_raw, 1):
        row[4][0] = str(rank)

    col_labels = [
        "Rank", "Model", "Group", "Parameters",
        "Best acc (%)", "Top-5 avg (%)", "Top-10 avg (%)",
        "Epoch of best", "Avg mem (MB)", "Avg epoch (s)",
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
        "Benchmark Comparison — DVS Gesture  |  "
        "Sorted by best test accuracy  |  "
        "Gold = overall best  |  Pink = best buffer  |  "
        "Top-5/10 avg = mean of 5/10 highest epoch accuracies  |  "
        "CaDLIF and DHLIF mem/time from PBS logs",
        fontsize=10, pad=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DVS Gesture benchmark scatter")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"{'='*60}")

    models = load_all_models()
    if not models:
        print("ERROR: no models loaded.")
        return

    print(f"\n  {len(models)} models loaded")
    print(f"\n  Sorted by best accuracy:")
    for m in sorted(models, key=lambda x: x["best_acc"], reverse=True):
        print(f"    {m['label']:<35}  {m['best_acc']:.2f}%  "
              f"params={m['n_params']:,}")

    print(f"\n  Generating plots ...")
    plot_scatter(models, os.path.join(OUTPUT_DIR, "benchmark_scatter.png"))
    plot_table(models,  os.path.join(OUTPUT_DIR, "benchmark_scatter_table.png"))

    print(f"\n{'='*60}")
    print(f"  Done. Outputs: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
