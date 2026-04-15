# configs/adaptive_axonal_delay_config.py
"""
Adaptive Axonal Delay SNN configuration for SHD.

Paper  : Sun et al., ICASSP 2023
Repo   : https://github.com/sunpengfei1122/Adaptive_axonal_delay
Branch : eb56c6b3199d1bdab60fcabbafa1dec5c5ec27ab
Ref    : main
"""

from configs.base_config import BenchmarkBaseConfig


class AdaptiveAxonalDelayConfig(BenchmarkBaseConfig):
    """
    Model-specific config for Adaptive Axonal Delay on SHD.
    Inherits all shared benchmark settings from BenchmarkBaseConfig
    and overrides only model-specific parameters.

    Framework: SLAYER (bamsumit/slayerPytorch)
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    model_name = "AdaptiveAxonalDelay"

    # ─────────────────────────────────────────────────────────────────────────
    # ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────

    n_inputs    = 700              # raw SHD neuron channels (no binning)
    n_hidden    = 128              # neurons per hidden layer (2 hidden layers)
    n_outputs   = 20              # SHD classes (explicit override for clarity)
    time_steps  = 400             # T dimension in SLAYER simulation grid

    # ─────────────────────────────────────────────────────────────────────────
    # SLAYER NEURON PARAMETERS
    # Mirrors the netParams / network.yaml used in the original main file
    # ─────────────────────────────────────────────────────────────────────────

    neuron_params = {
        "type"    : "SRNN",
        "theta"   : 10,            # firing threshold
        "tauSr"   : 1.0,           # synaptic rise time constant
        "tauRef"  : 1.0,           # refractory time constant
        "scaleRef": 1,
        "tauRho"  : 1,             # surrogate gradient width
        "scaleRho": 1,
    }

    simulation_params = {
        "Ts"      : 1.0,           # simulation time step (ms)
        "tSample" : 400,           # total simulation window = time_steps
        "nSample" : 1,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # ADAPTIVE DELAY SCHEDULER
    # Controls the dynamic cap on learnable axonal delays
    # ─────────────────────────────────────────────────────────────────────────

    delay_init_cap       = 64     # initial hard cap on delay values (time steps)
    delay_adapt_start    = 250    # training step at which adaptive scheduling begins
    delay_adapt_window   = 150    # steps between cap update checks
    delay_percentile     = 0.95   # 95th-percentile threshold for cap expansion
    delay_expand_margin  = 5      # expand cap if top-5% is within this margin of cap

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMISER
    # ─────────────────────────────────────────────────────────────────────────

    optimizer       = "nadam"
    lr              = 0.1
    lr_milestones   = [800]        # MultiStepLR decay milestones (in epochs)
    lr_gamma        = 0.5          # LR multiplied by gamma at each milestone

    # ─────────────────────────────────────────────────────────────────────────
    # DATALOADER OVERRIDE
    # ─────────────────────────────────────────────────────────────────────────

    # SLAYER processes spike tensors differently → smaller batch fits in VRAM
    batch_size      = 128          # overrides BenchmarkBaseConfig.batch_size = 256

    # ─────────────────────────────────────────────────────────────────────────
    # REGULARISATION
    # ─────────────────────────────────────────────────────────────────────────

    use_weight_norm = True         # weight_norm applied to all fc layers