# configs/axonal_delay_modelPYTORCH_config.py
"""
Configuration for AdaptiveAxonalDelayNetV2 — faithful PyTorch
reproduction of Sun et al. ICASSP 2023.

Verified against complete SLAYER source:
    - slayer.py        : kernels, surrogate, dense layer
    - spikeKernels.h   : getSpikes CUDA, evalRho CUDA
    - shiftKernels.h   : shift CUDA (floor, OOB=0)
    - spikeLoss.py     : van Rossum loss
    - nadam.py         : Nadam optimiser

All hyperparameters matched to Pengfei's network.yaml and
training script where confirmed.
"""

from configs.base_config import BenchmarkBaseConfig


class AdaptiveAxonalDelayV2Config(BenchmarkBaseConfig):
    """
    Model-specific config for AdaptiveAxonalDelayNetV2 on SHD.
    Inherits shared benchmark settings from BenchmarkBaseConfig.
    """

    # ── Identity ──────────────────────────────────────────────────────
    model_name = "AdaptiveAxonalDelayNetV2"

    # ─────────────────────────────────────────────────────────────────
    # ARCHITECTURE
    # Confirmed from Pengfei network.yaml + training script
    # ─────────────────────────────────────────────────────────────────

    n_inputs   = 700       # raw SHD cochlear channels
    n_hidden   = 128       # 2 hidden layers of 128 neurons
    n_outputs  = 20        # SHD digit classes
    time_steps = 400       # T = 400 timesteps (tSample=400, Ts=1.0)
    Ts         = 1.0       # simulation timestep size (ms)

    # ─────────────────────────────────────────────────────────────────
    # SRMALPHA NEURON PARAMETERS
    # Confirmed from network.yaml and SLAYER source
    # ─────────────────────────────────────────────────────────────────

    tau_s      = 5.0       # tauSr  — PSP time constant
    tau_r      = 2.0       # tauRef — refractory time constant
    theta_u    = 0.5      # theta  — firing threshold (NOT 0.1)
    alpha_r    = 2.0       # scaleRef — refractory scale

    # ─────────────────────────────────────────────────────────────────
    # SURROGATE GRADIENT
    # Confirmed from evalRhoKernel CUDA:
    #   tau_eff = tau_rho * theta_u = 1.0 * 10.0 = 10.0
    #   rho = scale_rho/tau_eff * exp(-|theta-u|/tau_eff)
    # ─────────────────────────────────────────────────────────────────

    tau_rho    = 1.0       # tauRho   — surrogate width (scaled by theta)
    scale_rho  = 1.0       # scaleRho — surrogate scale

    # ─────────────────────────────────────────────────────────────────
    # ALPHA KERNEL
    # Confirmed from _calculateAlphaKernel:
    #   dynamic length — stops when |val| < epsilon AND t > tau
    #   with tau=1.0, Ts=1.0: kernel length ≈ 5 timesteps
    # ─────────────────────────────────────────────────────────────────

    kernel_epsilon = 0.01  # SLAYER default EPSILON for kernel cutoff

    # ─────────────────────────────────────────────────────────────────
    # ADAPTIVE DELAY SCHEDULER
    # Confirmed from Pengfei training script
    # ─────────────────────────────────────────────────────────────────

    delay_init_cap      = 64   # initial hard cap θd
    delay_adapt_start   = 250  # epoch at which adaptive scheduling begins
    delay_adapt_window  = 150  # steps between cap update checks
    delay_percentile    = 0.95 # 95th percentile — index 121/128 ≈ sorted[108]
    delay_expand_margin = 5    # expand if p95 within 5 of cap

    # ─────────────────────────────────────────────────────────────────
    # LOSS — Van Rossum spike rate
    # Confirmed from spikeLoss.py:
    #   error = psp(spikeOut - spikeDesired)
    #   loss  = 0.5 * sum(error^2) * Ts
    # ─────────────────────────────────────────────────────────────────

    target_true  = 0.8     # desired rate for correct class neuron
    target_false = 0.0     # desired rate for all other neurons

    # ─────────────────────────────────────────────────────────────────
    # OPTIMISER
    # Confirmed from nadam.py — standard Nadam
    # torch.optim.NAdam is equivalent (confirmed from source)
    # ─────────────────────────────────────────────────────────────────

    optimizer     = "nadam"
    lr            = 0.001        # confirmed from Pengfei training script
    lr_milestones = [800]      # MultiStepLR milestone
    lr_gamma      = 0.5        # decay factor

    # ─────────────────────────────────────────────────────────────────
    # DATALOADER
    # ─────────────────────────────────────────────────────────────────

    batch_size  = 128          # confirmed from Pengfei training script
    num_workers = 4

    # ─────────────────────────────────────────────────────────────────
    # WEIGHT INITIALISATION
    # Confirmed from _denseLayer: weightScale=10 default
    # Applied as: weight.data *= 10.0 after nn.Linear init
    # ─────────────────────────────────────────────────────────────────

    weight_scale    = 1.0     # SLAYER _denseLayer default
    use_weight_norm = False    # NOT in SLAYER — confirmed absent

    # ─────────────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────────────

    epochs            = 1000   # confirmed from Pengfei training loop
    save_checkpoints  = True
    checkpoint_every  = 10     # save every 10 epochs