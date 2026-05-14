# configs/cadlif_config.py
"""
Constrained Adaptive LIF (cAdLIF) with Synaptic Delays.
Paper: Deckers et al. 2024
"Advancing Spatio-Temporal Processing in Spiking Neural Networks"
Frontiers in Neuroscience.

SHD-specific hyperparameters from Table 1 / Section 4.2.
"""

from configs.base_config import BenchmarkBaseConfig


class cAdLIFConfig(BenchmarkBaseConfig):

    # ── Identity ──────────────────────────────────────────────────────────
    model_name = "cAdLIF"

    # ── Architecture ──────────────────────────────────────────────────────
    n_inputs    = 140       # 700 channels binned → 140 (Section 4.1)
    n_hidden    = 128       # Table 1
    n_outputs   = 20        # SHD classes
    time_steps  = 100       # 1s → 100 timesteps at 10ms bins (Section 4.1)

    # ── cAdLIF Neuron Parameter Bounds ────────────────────────────────────
    # All four are learnable, per-neuron, uniform init (Section 4.2)
    alpha_min   = 0.36      # membrane leak lower bound
    alpha_max   = 0.96      # membrane leak upper bound
    beta_min    = 0.96      # adaptation leak lower bound
    beta_max    = 0.99      # adaptation leak upper bound
    a_min       = 0.0       # sub-threshold coupling — MUST stay >= 0
    a_max       = 1.0
    b_min       = 0.0       # spike-triggered adaptation — MUST stay >= 0
    b_max       = 2.0

    # Fixed spike threshold (dimensionless)
    v_threshold = 1.0

    # ── Surrogate Gradient ────────────────────────────────────────────────
    # Boxcar, Equation 6 — width fixed at 0.5
    surrogate_width = 0.5

    # ── Synaptic Delays ───────────────────────────────────────────────────
    max_delay   = 25        # delay cap [0, 25] timesteps (Table 1)
                            # context_c removed — gradient is implicit

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer   = "adam"
    lr          = 0.01      # weight + neuron param LR (Table 1 SHD)
    lr_delays   = 0.1       # delay LR = 10 * lr_weights (Section 4.2)

    # ReduceLROnPlateau (Section 4.2)
    lr_scheduler = "plateau"
    lr_factor    = 0.7
    lr_patience  = 5

    # ── Regularisation ────────────────────────────────────────────────────
    dropout_p   = 0.5       # SHD dropout rate (Table 1)

    # ── DataLoader ────────────────────────────────────────────────────────
    batch_size  = 128       # Table 1
    num_workers = 4

    # ── Training ──────────────────────────────────────────────────────────
    epochs      = 100       # Table 1