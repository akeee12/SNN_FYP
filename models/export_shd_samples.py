# scripts/export_shd_samples.py
"""
Export N real SHD test samples to samples.h as sparse spike events,
encoded identically to SHDDataset.__getitem__ (training-time encoding).

Run on HPC login node (needs h5py + the SHD test h5 file, no GPU needed).

Usage:
    python3 scripts/export_shd_samples.py
"""

import os
import sys
import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change these lines only
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_PATH = "/rds/general/user/ak2721/home/SNN_FYP/data"  # <-- confirm/edit
SPLIT         = "test"
TIME_STEPS    = 400      # cfg.time_steps, confirmed from aad_v2_snntorch_config.py
N_INPUTS      = 700
MAX_TIME      = 1.4
N_SAMPLES     = 20       # how many to pull — script reports real size, decide after
BALANCED      = True     # try to spread across classes rather than first-N
SEED          = 42

OUTPUT_DIR = "/rds/general/user/ak2721/home/fpga_export/power_law_small"
HEADER_PATH = os.path.join(OUTPUT_DIR, "samples.h")


# ─────────────────────────────────────────────────────────────────────────────
# Encoding — identical to SHDDataset.__getitem__
# ─────────────────────────────────────────────────────────────────────────────

def encode_sample(times, units, time_steps, n_inputs, max_time):
    """Returns sorted list of (timestep, channel) events, matching the
    exact binning SHDDataset uses (no event collapsing — duplicate
    (t,channel) pairs in the source are deduped to a single spike,
    same as the dense tensor assignment spikes[t_idx, u_idx] = 1.0)."""
    t_idx = np.floor(times / max_time * time_steps).astype(int)
    t_idx = np.clip(t_idx, 0, time_steps - 1)
    u_idx = np.clip(units.astype(int), 0, n_inputs - 1)

    # dedupe exactly like the dense assignment would (same (t,u) -> one spike)
    pairs = np.unique(np.stack([t_idx, u_idx], axis=1), axis=0)
    return pairs  # shape (n_events, 2), columns = [timestep, channel]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)
    h5_path = os.path.join(DATASETS_PATH, f"shd_{SPLIT}.h5")
    print(f"Loading {h5_path}")

    with h5py.File(h5_path, "r") as f:
        labels = f["labels"][:]
        n_total = len(labels)
        print(f"  {n_total} samples total, {len(set(labels))} classes")

        if BALANCED:
            classes = sorted(set(labels.tolist()))
            per_class = max(1, N_SAMPLES // len(classes))
            chosen = []
            for c in classes:
                idxs = np.where(labels == c)[0]
                rng.shuffle(idxs)
                chosen.extend(idxs[:per_class].tolist())
            rng.shuffle(chosen)
            chosen = chosen[:N_SAMPLES]
        else:
            chosen = rng.choice(n_total, size=N_SAMPLES, replace=False).tolist()

        print(f"  Selected {len(chosen)} samples (balanced={BALANCED})")

        all_events = []   # list of (t,u) arrays, one per sample
        sample_labels = []
        for idx in chosen:
            times = f["spikes"]["times"][idx]
            units = f["spikes"]["units"][idx]
            pairs = encode_sample(times, units, TIME_STEPS, N_INPUTS, MAX_TIME)
            all_events.append(pairs)
            sample_labels.append(int(labels[idx]))

    # ── Report real per-sample sizes ────────────────────────────────────
    counts = [len(e) for e in all_events]
    print("\n  Per-sample event counts:")
    for i, (c, lbl) in enumerate(zip(counts, sample_labels)):
        print(f"    sample {i:2d}  label={lbl:2d}  events={c:5d}  "
              f"bytes(uint32)={c*4:6d}")

    total_events = sum(counts)
    total_bytes_u32 = total_events * 4
    total_bytes_packed = total_events * 3  # if we switch to 3-byte packing
    print(f"\n  TOTAL events       : {total_events}")
    print(f"  TOTAL size (uint32): {total_bytes_u32/1024:.2f} KB")
    print(f"  TOTAL size (packed 3B): {total_bytes_packed/1024:.2f} KB")
    print(f"  Avg events/sample  : {total_events/len(counts):.1f}")
    print(f"  Max events/sample  : {max(counts)}")
    print(f"  Min events/sample  : {min(counts)}")

    # ── Write samples.h ──────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(HEADER_PATH, "w") as f:
        f.write("// Auto-generated — SHD test samples for FPGA demo\n")
        f.write(f"// N={len(chosen)}  time_steps={TIME_STEPS}  "
                f"n_inputs={N_INPUTS}  total_events={total_events}\n\n")
        f.write("#ifndef SAMPLES_H\n#define SAMPLES_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define N_SAMPLES {len(chosen)}\n")
        f.write(f"#define SAMPLE_TIME_STEPS {TIME_STEPS}\n\n")

        # labels
        f.write(f"static const int sample_labels[N_SAMPLES] = {{")
        f.write(", ".join(str(l) for l in sample_labels))
        f.write("};\n\n")

        # offsets (prefix sum into flat spike_events array)
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)
        f.write(f"static const uint32_t sample_offsets[N_SAMPLES + 1] = {{")
        f.write(", ".join(str(o) for o in offsets))
        f.write("};\n\n")

        # flat event array: (timestep << 16) | channel, packed uint32
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

    print(f"\n  Wrote {HEADER_PATH}")
    print(f"  ({total_bytes_u32/1024:.2f} KB as uint32 events, "
          f"vs {total_bytes_packed/1024:.2f} KB if packed to 3 bytes)")


if __name__ == "__main__":
    main()