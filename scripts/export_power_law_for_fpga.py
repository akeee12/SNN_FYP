# scripts/export_power_law_for_fpga.py
"""
Freeze trained PowerLawTapAADNetSNNTorch, extract weights,
quantise to INT8, and export to formats suitable for FPGA.

Run on HPC login node — no GPU needed.

Usage:
    python3 scripts/export_power_law_for_fpga.py
"""

import os
import sys
import json
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models"))

from configs import get_config
from power_law_tap_aad_snntorch import PowerLawTapAADNetSNNTorch


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change these lines only
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT = (
    "/rds/general/user/ak2721/home/results/SNN_FYP"
    "/power_law_tap_shd/v1/PowerLawTapAADSNNTorch_best.pt"
)
OUTPUT_DIR = "/rds/general/user/ak2721/home/fpga_export/power_law"
BITS       = 8


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    cfg = get_config("power_law_tap_snntorch")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Restore any runtime overrides that were active during training
    # These are stored in the config snapshot inside the summary JSON
    # If the checkpoint has them, apply them
    if "taps" in ckpt:
        cfg.taps = ckpt["taps"]
    if "buffer_max_lag" in ckpt:
        cfg.buffer_max_lag = ckpt["buffer_max_lag"]
    if "n_hidden" in ckpt:
        cfg.n_hidden = ckpt["n_hidden"]
    if "gamma" in ckpt:
        cfg.gamma = ckpt["gamma"]

    model = PowerLawTapAADNetSNNTorch(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model        : PowerLawTapAADNetSNNTorch")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Epoch        : {ckpt.get('epoch')}")
    print(f"  Test acc     : {ckpt.get('test_acc', 0)*100:.2f}%")
    print(f"  Dataset      : {ckpt.get('dataset', 'unknown')}")
    print(f"  Gamma        : {cfg.gamma}")
    print(f"  Taps         : {cfg.taps}")
    print(f"  Buffer k     : {cfg.buffer_size}")
    print(f"  Buffer lag   : {cfg.buffer_max_lag}")
    print(f"  n_hidden     : {cfg.n_hidden}")

    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Freeze
# ─────────────────────────────────────────────────────────────────────────────

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    print("\n  All layers frozen")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Extract weights
# Handles both old weight_norm API (weight_v/weight_g)
# and new parametrizations API (PyTorch 2.x)
# ─────────────────────────────────────────────────────────────────────────────

def get_effective_weight(module) -> np.ndarray:
    if hasattr(module, "weight_v"):
        wv = module.weight_v.data
        wg = module.weight_g.data
        w  = wg * (wv / wv.norm(dim=0, keepdim=True).clamp(min=1e-8))
        return w.detach().numpy()
    elif hasattr(module, "parametrizations"):
        return module.weight.detach().numpy()
    else:
        return module.weight.detach().numpy()


def extract_weights(model, cfg) -> dict:
    weights = {}

    # ── Buffer weights ────────────────────────────────────────────────
    # PowerLawTapAADNetSNNTorch uses self.tap_buffer
    weights["fc_proj"]   = get_effective_weight(model.tap_buffer.fc_proj)
    weights["fc_expand"] = get_effective_weight(model.tap_buffer.fc_expand)

    # ── Hidden layer weights ──────────────────────────────────────────
    weights["fc2"] = get_effective_weight(model.fc2)
    weights["fc3"] = get_effective_weight(model.fc3)

    # ── Axonal delays ─────────────────────────────────────────────────
    weights["delay1"] = model.delay1.delay.detach().numpy()
    weights["delay2"] = model.delay2.delay.detach().numpy()

    # ── Neuron parameters ─────────────────────────────────────────────
    lif1 = model.lif1
    weights["alpha"]         = np.array([lif1.alpha.item()
                                          if hasattr(lif1.alpha, "item")
                                          else float(lif1.alpha)])
    weights["beta"]          = np.array([lif1.beta.item()
                                          if hasattr(lif1.beta, "item")
                                          else float(lif1.beta)])
    weights["threshold"]     = np.array([float(lif1.threshold)])
    weights["threshold_out"] = np.array([float(model.lif3.threshold)])

    # ── Buffer config ─────────────────────────────────────────────────
    buf = model.tap_buffer
    weights["taps"]           = np.array(buf.taps, dtype=np.int32)
    weights["buffer_max_lag"] = np.array([buf.max_lag], dtype=np.int32)
    weights["buffer_k"]       = np.array([buf.k], dtype=np.int32)
    weights["theta_u"]        = np.array([buf.theta_u])
    weights["gamma"]          = np.array([cfg.gamma])
    weights["n_hidden"]       = np.array([cfg.n_hidden], dtype=np.int32)
    weights["n_outputs"]      = np.array([cfg.n_outputs], dtype=np.int32)
    weights["n_inputs"]       = np.array([cfg.n_inputs], dtype=np.int32)

    print("\n  Extracted weight shapes:")
    for k, v in weights.items():
        print(f"    {k:20s} : {v.shape}  dtype={v.dtype}")

    return weights


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Quantise to INT8
# Symmetric per-tensor quantisation
# scale = max(|W|) / 127
# W_q   = clip(round(W / scale), -127, 127)
# ─────────────────────────────────────────────────────────────────────────────

WEIGHT_KEYS = ["fc_proj", "fc_expand", "fc2", "fc3"]


def quantise(weights: dict, bits: int = 8) -> tuple:
    max_val   = 2 ** (bits - 1) - 1
    quantised = {}
    scales    = {}

    print(f"\n  Quantising to INT{bits}:")
    for k, v in weights.items():
        if k in WEIGHT_KEYS:
            abs_max    = np.max(np.abs(v))
            scale      = float(abs_max / max_val) if abs_max > 0 else 1.0
            w_q        = np.clip(np.round(v / scale), -max_val, max_val)
            quantised[k] = w_q.astype(np.int8)
            scales[k]    = np.float32(scale)
            print(f"    {k:20s}  scale={scale:.8f}  "
                  f"range=[{w_q.min():.0f}, {w_q.max():.0f}]  "
                  f"size={w_q.nbytes/1024:.1f} KB")
        else:
            quantised[k] = v
            scales[k]    = np.float32(1.0)

    total_kb = sum(quantised[k].nbytes for k in WEIGHT_KEYS) / 1024
    print(f"\n    Total weight storage (INT8): {total_kb:.1f} KB")
    print(f"    MAX 10 on-chip RAM         : 204 KB")
    print(f"    Utilisation                : {total_kb/204*100:.1f}%")

    return quantised, scales


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Verify quantisation
# ─────────────────────────────────────────────────────────────────────────────

def verify_quantisation(weights: dict, quantised: dict,
                        scales: dict, model, cfg):
    print("\n  Verifying quantisation error...")

    x      = (torch.rand(4, cfg.time_steps, cfg.n_inputs) < 0.05).float()
    labels = torch.randint(0, cfg.n_outputs, (4,))

    # Float32 reference
    with torch.no_grad():
        out_float  = model(x)
        pred_float = out_float.sum(dim=0).argmax(dim=-1) \
                     if out_float.dim() == 3 \
                     else out_float.argmax(dim=-1)

    # Substitute dequantised weights
    def set_weight(module, w_np):
        w_t = torch.from_numpy(w_np.astype(np.float32))
        if hasattr(module, "parametrizations"):
            with torch.no_grad():
                module.weight.copy_(w_t)
        elif hasattr(module, "weight_v"):
            with torch.no_grad():
                norm = w_t.norm(dim=0, keepdim=True).clamp(min=1e-8)
                module.weight_v.data.copy_(w_t / norm)
                module.weight_g.data.copy_(norm)
        else:
            with torch.no_grad():
                module.weight.data.copy_(w_t)

    modules = {
        "fc_proj"  : model.tap_buffer.fc_proj,
        "fc_expand": model.tap_buffer.fc_expand,
        "fc2"      : model.fc2,
        "fc3"      : model.fc3,
    }

    originals = {k: weights[k].copy() for k in WEIGHT_KEYS}

    for k in WEIGHT_KEYS:
        dq = quantised[k].astype(np.float32) * scales[k]
        set_weight(modules[k], dq)

    with torch.no_grad():
        out_quant  = model(x)
        pred_quant = out_quant.sum(dim=0).argmax(dim=-1) \
                     if out_quant.dim() == 3 \
                     else out_quant.argmax(dim=-1)

    for k in WEIGHT_KEYS:
        set_weight(modules[k], originals[k])

    max_diff = (out_float - out_quant).abs().max().item()
    agree    = (pred_float == pred_quant).float().mean().item() * 100.0

    print(f"    Max logit diff (float vs INT8) : {max_diff:.6f}")
    print(f"    Prediction agreement           : {agree:.1f}%")
    if max_diff < 1.0:
        print(f"    Quantisation error             : ACCEPTABLE ✓")
    else:
        print(f"    Quantisation error             : HIGH — consider bits=16")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(weights: dict, quantised: dict,
                 scales: dict, output_dir: str, cfg):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Float32 reference weights ─────────────────────────────────
    float_dir = os.path.join(output_dir, "float32")
    os.makedirs(float_dir, exist_ok=True)
    for k, v in weights.items():
        np.save(os.path.join(float_dir, f"{k}.npy"), v)

    # ── 2. INT8 quantised weights ─────────────────────────────────────
    int8_dir = os.path.join(output_dir, "int8")
    os.makedirs(int8_dir, exist_ok=True)
    for k, v in quantised.items():
        np.save(os.path.join(int8_dir, f"{k}.npy"), v)

    # ── 3. C header file ──────────────────────────────────────────────
    header_path = os.path.join(output_dir, "weights.h")
    with open(header_path, "w") as f:
        f.write("// Auto-generated — Power Law Tap AAD SNN INT8 weights\n")
        f.write("// gamma={:.1f}  taps={}  k={}  H={}\n\n".format(
            float(weights["gamma"][0]),
            weights["taps"].tolist(),
            int(weights["buffer_k"][0]),
            int(weights["n_hidden"][0])
        ))
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        # Scales
        for k in WEIGHT_KEYS:
            f.write(f"#define {k.upper()}_SCALE {scales[k]:.10f}f\n")
        f.write("\n")

        # Network dimensions
        f.write(f"#define N_INPUTS    {int(weights['n_inputs'][0])}\n")
        f.write(f"#define HIDDEN      {int(weights['n_hidden'][0])}\n")
        f.write(f"#define N_OUTPUTS   {int(weights['n_outputs'][0])}\n")
        f.write(f"#define BUFFER_K    {int(weights['buffer_k'][0])}\n")
        f.write(f"#define BUFFER_MAX_LAG {int(weights['buffer_max_lag'][0])}\n")
        f.write(f"#define N_TAPS      {len(weights['taps'])}\n")
        f.write(f"#define THETA_U     {float(weights['theta_u'][0]):.6f}f\n")
        f.write(f"#define THRESHOLD   {float(weights['threshold'][0]):.6f}f\n")
        f.write(f"#define THRESHOLD_OUT {float(weights['threshold_out'][0]):.6f}f\n")
        f.write(f"#define TAU_S_ALPHA {float(weights['alpha'][0]):.10f}f\n")
        f.write(f"#define TAU_R_BETA  {float(weights['beta'][0]):.10f}f\n\n")

        # Taps array
        taps = weights["taps"].tolist()
        f.write(f"static const int TAPS[N_TAPS] = {{")
        f.write(", ".join(str(t) for t in taps))
        f.write("};\n\n")

        # Weight matrices as INT8 arrays
        for k in WEIGHT_KEYS:
            w    = quantised[k]
            flat = w.flatten()
            f.write(f"// {k}  shape={w.shape}\n")
            f.write(f"static const int8_t {k}[{len(flat)}] = {{\n    ")
            for i, val in enumerate(flat):
                f.write(f"{int(val):4d}")
                if i < len(flat) - 1:
                    f.write(",")
                    if (i + 1) % 16 == 0:
                        f.write("\n    ")
                    else:
                        f.write(" ")
            f.write("\n};\n\n")

        # Delays as float arrays
        for dname in ["delay1", "delay2"]:
            d = weights[dname]
            f.write(f"static const float {dname}[{len(d)}] = {{")
            f.write(", ".join(f"{v:.6f}f" for v in d))
            f.write("};\n\n")

        f.write("#endif // WEIGHTS_H\n")

    # ── 4. Deployment summary JSON ────────────────────────────────────
    summary = {
        "model"          : "PowerLawTapAADNetSNNTorch",
        "bits"           : BITS,
        "gamma"          : float(weights["gamma"][0]),
        "taps"           : weights["taps"].tolist(),
        "buffer_k"       : int(weights["buffer_k"][0]),
        "buffer_max_lag" : int(weights["buffer_max_lag"][0]),
        "n_inputs"       : int(weights["n_inputs"][0]),
        "n_hidden"       : int(weights["n_hidden"][0]),
        "n_outputs"      : int(weights["n_outputs"][0]),
        "threshold"      : float(weights["threshold"][0]),
        "threshold_out"  : float(weights["threshold_out"][0]),
        "alpha"          : float(weights["alpha"][0]),
        "beta"           : float(weights["beta"][0]),
        "theta_u"        : float(weights["theta_u"][0]),
        "scales"         : {k: float(v) for k, v in scales.items()
                            if k in WEIGHT_KEYS},
        "weight_shapes"  : {k: list(quantised[k].shape)
                            for k in WEIGHT_KEYS},
        "weight_kb"      : {k: quantised[k].nbytes / 1024
                            for k in WEIGHT_KEYS},
        "total_weight_kb": sum(quantised[k].nbytes
                               for k in WEIGHT_KEYS) / 1024,
    }
    with open(os.path.join(output_dir, "deployment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Outputs saved to: {output_dir}")
    print(f"    float32/              — reference weights")
    print(f"    int8/                 — quantised weights")
    print(f"    weights.h             — C header for FPGA/HLS")
    print(f"    deployment_summary.json")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FPGA Export — Power Law Tap AAD SNN")
    print("=" * 60)

    print(f"\n  Checkpoint : {CHECKPOINT}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print(f"  Bits       : {BITS}")

    model, cfg = load_model(CHECKPOINT)
    model      = freeze_model(model)

    print("\n  Extracting weights...")
    weights = extract_weights(model, cfg)

    print(f"\n  Quantising to INT{BITS}...")
    quantised, scales = quantise(weights, bits=BITS)

    verify_quantisation(weights, quantised, scales, model, cfg)

    print("\n  Saving outputs...")
    save_outputs(weights, quantised, scales, OUTPUT_DIR, cfg)

    print("\n" + "=" * 60)
    print("  Export complete")
    print("  Next: copy weights.h to your HLS/Quartus project")
    print("=" * 60)


if __name__ == "__main__":
    main()