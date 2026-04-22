# configs/adaptive_axonal_delay_config.py
"""
Adaptive Axonal Delay SNN configuration for SHD.

Paper  : Sun et al., ICASSP 2023
Repo   : https://github.com/sunpengfei1122/Adaptive_axonal_delay
"""

from configs.base_config import BenchmarkBaseConfig


class AdaptiveAxonalDelayPyTorchConfig(BenchmarkBaseConfig):
    """
    Model-specific config for Adaptive Axonal Delay on SHD.
    Inherits all shared benchmark settings from BenchmarkBaseConfig.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    model_name = "AdaptiveAxonalDelayPyTorch"

    # ─────────────────────────────────────────────────────────────────────────
    # ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────

    n_inputs    = 700          # raw SHD channels — no binning
    n_hidden    = 128          # 2 hidden layers of 128 neurons
    n_outputs   = 20           # SHD classes
    time_steps  = 400          # T = 400 timesteps

    # ─────────────────────────────────────────────────────────────────────────
    # SRM NEURON PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────

    tau_s       = 1.0          # synaptic response time constant
    tau_r       = 1.0          # refractory time constant
    theta_u     = 0.1         # firing threshold
    alpha_r     = 2.0          # refractory scale

    # ─────────────────────────────────────────────────────────────────────────
    # SURROGATE GRADIENT (SLAYER triangle function)
    # ─────────────────────────────────────────────────────────────────────────

    tau_rho     = 1.0          # surrogate gradient width
    scale_rho   = 1.0          # surrogate gradient scale

    # ─────────────────────────────────────────────────────────────────────────
    # PSP + REFRACTORY KERNEL
    # ─────────────────────────────────────────────────────────────────────────

    kernel_size = 32           # capped at 32 — not full time window

    # ─────────────────────────────────────────────────────────────────────────
    # ADAPTIVE DELAY SCHEDULER
    # ─────────────────────────────────────────────────────────────────────────

    delay_init_cap       = 64  # initial hard cap θd = 64 for SHD
    delay_adapt_start    = 250 # step at which adaptive scheduling begins
    delay_adapt_window   = 150 # steps between cap update checks
    delay_percentile     = 0.95  # 95th percentile threshold
    delay_expand_margin  = 5   # expand cap if top neuron within 5 of cap

    # ─────────────────────────────────────────────────────────────────────────
    # LOSS
    # spike rate loss — MSE between output spike count and target rate
    # ─────────────────────────────────────────────────────────────────────────

    target_true  = 0.8         # desired neurons fire at 80%
    target_false = 0.0         # other neurons silent

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMISER
    # ─────────────────────────────────────────────────────────────────────────

    optimizer      = "nadam"   # torch.optim.NAdam — built into PyTorch 2.x
    lr             = 0.001
    lr_milestones  = [800]     # MultiStepLR decay milestones
    lr_gamma       = 0.5

    # ─────────────────────────────────────────────────────────────────────────
    # DATALOADER OVERRIDE
    # ─────────────────────────────────────────────────────────────────────60───

    batch_size     = 128       # overrides BenchmarkBaseConfig.batch_size = 256
    num_workers    = 4

    # ─────────────────────────────────────────────────────────────────────────
    # REGULARISATION
    # ─────────────────────────────────────────────────────────────────────────

    use_weight_norm = True     # weight_norm on all FC layers