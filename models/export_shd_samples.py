# scripts/export_shd_samples_small.py
"""
Export N smallest SHD test samples (by event count) to samples.h.
Selects smallest samples so the packed array fits in DE10-Lite on-chip RAM.

Run on HPC:
    module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
    python3 -u models/export_shd_samples_small.py
"""

import os
import sys
import h5py
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_PATH = "/rds/general/user/ak2721/home/datasets/SHD"
SPLIT         = "test"
TIME_STEPS    = 400
N_INPUTS      = 700
MAX_TIME      = 1.4

N_SAMPLES        = 2        # 2 smallest samples
CANDIDATE_POOL   = 200      # scan first 200 random samples, pick N smallest
SEED             = 42

OUTPUT_DIR  = "/rds/general/user/ak2721/home/fpga_export/power_law_small"
HEADER_PATH = os.path.join(OUTPUT_DIR, "samples.h")


def encode_sample(times, units, time_steps, n_inputs, max_time):
    t_idx = np.floor(times / max_time * time_steps).astype(int)
    t_idx = np.clip(t_idx, 0, time_steps - 1)
    u_idx = np.clip(units.astype(int), 0, n_inputs - 1)
    pairs = np.unique(np.stack([t_idx, u_idx], axis=1), axis=0)
    return pairs


def main():
    rng = np.random.default_rng(SEED)
    h5_path = os.path.join(DATASETS_PATH, f"shd_{SPLIT}.h5")
    print(f"Loading {h5_path}")

    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:]
        n_total = len(labels)
        print(f"  {n_total} samples total")

        # Scan a candidate pool, measure event counts, pick N smallest
        candidates = rng.choice(n_total, size=min(CANDIDATE_POOL, n_total),
                                replace=False)
        print(f"  Scanning {len(candidates)} candidates for smallest...")

        cand_events = []
        for idx in candidates:
            times = f["spikes"]["times"][idx]
            units = f["spikes"]["units"][idx]
            pairs = encode_sample(times, units, TIME_STEPS, N_INPUTS, MAX_TIME)
            cand_events.append((len(pairs), int(idx), int(labels[idx]), pairs))

        # Sort by event count ascending, pick N smallest with DISTINCT labels
        cand_events.sort(key=lambda x: x[0])
        chosen = []
        seen_labels = set()
        for ec, idx, lbl, pairs in cand_events:
            if lbl not in seen_labels:
                chosen.append((ec, idx, lbl, pairs))
                seen_labels.add(lbl)
                if len(chosen) == N_SAMPLES:
                    break

        if len(chosen) < N_SAMPLES:
            print(f"  WARNING: only {len(chosen)} distinct-label samples in pool")

    counts = [c[0] for c in chosen]
    sample_labels = [c[2] for c in chosen]
    sample_indices = [c[1] for c in chosen]
    all_events = [c[3] for c in chosen]

    print("\n  Selected samples (smallest, distinct labels):")
    for i, (ec, idx, lbl, _) in enumerate(chosen):
        print(f"    pos {i}  test_idx={idx:4d}  label={lbl:2d}  events={ec:5d}  "
              f"bytes={ec*4:6d}")
    total_events = sum(counts)
    total_kb = total_events * 4 / 1024
    print(f"\n  TOTAL events: {total_events}  ({total_kb:.2f} KB unpacked)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(HEADER_PATH, "w") as f:
        f.write("// Auto-generated — SHD test samples (smallest N) for FPGA\n")
        f.write(f"// N={len(chosen)}  time_steps={TIME_STEPS}  "
                f"n_inputs={N_INPUTS}  total_events={total_events}\n")
        f.write(f"// test_indices={sample_indices}  labels={sample_labels}\n\n")
        f.write("#ifndef SAMPLES_H\n#define SAMPLES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define N_SAMPLES {len(chosen)}\n")
        f.write(f"#define SAMPLE_TIME_STEPS {TIME_STEPS}\n\n")

        f.write(f"static const int sample_labels[N_SAMPLES] = {{")
        f.write(", ".join(str(l) for l in sample_labels))
        f.write("};\n\n")

        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)
        f.write(f"static const uint32_t sample_offsets[N_SAMPLES + 1] = {{")
        f.write(", ".join(str(o) for o in offsets))
        f.write("};\n\n")

        flat_events = []
        for pairs in all_events:
            for t, u in pairs:
                flat_events.append((int(t) << 16) | int(u))

        f.write(f"static const uint32_t spike_events[{len(flat_events)}] = {{\n    ")
        for i, ev in enumerate(flat_events):
            f.write(f"0x{ev:08X}")
            if i < len(flat_events) - 1:
                f.write(",")
                if (i + 1) % 8 == 0:
                    f.write("\n    ")
                else:
                    f.write(" ")
        f.write("\n};\n\n")
        f.write("#endif // SAMPLES_H\n")

    print(f"\n  Wrote {HEADER_PATH}  ({total_kb:.2f} KB)")


if __name__ == "__main__":
    main()