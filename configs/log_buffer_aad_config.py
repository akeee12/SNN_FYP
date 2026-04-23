# configs/log_buffer_aad_config.py
"""
Configuration for AAD-SNN with Logarithmic Spike Buffer extension.

Extends AdaptiveAxonalDelayPyTorchConfig with log buffer parameters.
All baseline parameters remain identical for fair ablation comparison.

Key difference from baseline:
    FC1 input: n_taps * n_inputs = 8 * 700 = 5600
    vs baseline: 700
    → more parameters, richer temporal representation
"""

from configs.base_config import BenchmarkBaseConfig


class LogBufferAADConfig(BenchmarkBaseConfig):
    """
    AAD + Logarithmic Spike Buffer configuration.

    Identical to AdaptiveAxonalDelayPyTorchConfig except for
    the log buffer parameters added below.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    model_name = "AAD_LogBuffer"

    # ── Architecture (same as baseline) ───────────────────────────────────
    n_inputs    = 700
    n_hidden    = 128
    n_outputs   = 20
    time_steps  = 400

    # ── SRM Neuron Parameters (same as baseline) ──────────────────────────
    tau_s       = 1.0
    tau_r       = 1.0
    theta_u     = 0.1          # tuned for real SHD input scale
    alpha_r     = 2.0
    tau_rho     = 1.0
    scale_rho   = 1.0
    kernel_size = 32

    # ── Adaptive Delay Scheduler (same as baseline) ───────────────────────
    delay_init_cap       = 64
    delay_adapt_start    = 250
    delay_adapt_window   = 150
    delay_percentile     = 0.95
    delay_expand_margin  = 5

    # ── Loss (same as baseline) ───────────────────────────────────────────
    target_true  = 0.8
    target_false = 0.0

    # ── Optimiser (same as baseline) ──────────────────────────────────────
    optimizer      = "nadam"
    lr             = 0.001
    lr_milestones  = [800]
    lr_gamma       = 0.5

    # ── DataLoader (same as baseline) ─────────────────────────────────────
    batch_size     = 128
    num_workers    = 4

    # ── Regularisation (same as baseline) ─────────────────────────────────
    use_weight_norm = True

    # ─────────────────────────────────────────────────────────────────────
    # LOG BUFFER PARAMETERS — new extension
    # ─────────────────────────────────────────────────────────────────────

    # Number of log-spaced taps K
    # More taps = richer temporal context but more parameters
    # 8 taps: τ = [0, 1, 3, 6, 12, 22, 39, 64]
    # 16 taps: finer resolution
    log_buffer_n_taps    = 8

    # Maximum delay D in timesteps
    # Matches delay_init_cap for consistency
    log_buffer_max_delay = 64

    # Whether tap positions are learnable parameters
    # False = fixed log-spaced (biologically motivated)
    # True  = learned positions (more flexible, more parameters)
    log_buffer_learnable = False