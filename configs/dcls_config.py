# configs/dcls_config.py
"""
DCLS (Dilated Convolutions with Learnable Spacings) configuration for SHD.

Paper : Hammouamri et al., ICLR 2024
Repo  : https://github.com/Thvnvtos/SNN-delays
Ref   : best_config_SHD.py
"""

from configs.base_config import BenchmarkBaseConfig


class DCLSConfig(BenchmarkBaseConfig):
    """
    Model-specific config for DCLS/SNN-delays on SHD.
    Inherits all shared benchmark settings from BenchmarkBaseConfig
    and overrides only DCLS-specific parameters.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    model_name = "DCLS"

    # ─────────────────────────────────────────────────────────────────────────
    # ARCHITECTURE
    # ─────────────────────────────────────────────────────────────────────────

    # Model variant:
    #   'snn_delays'     → full learnable delay model
    #   'snn_delays_lr0' → delays frozen after init (ablation)
    #   'snn'            → plain SNN baseline (no delays)
    model_type          = "snn_delays"

    spiking_neuron_type = "lif"        # 'lif' | 'plif' (parametric LIF)

    # Input binning: SHD has 700 channels → binned into n_bins groups
    n_bins              = 5
    n_inputs            = 700 // n_bins    # = 140

    n_hidden_layers     = 2
    n_hidden_neurons    = 256

    # ─────────────────────────────────────────────────────────────────────────
    # NEURON PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────

    time_step           = 10               # ms per simulation step
    init_tau            = (10.05 + 1e-9) / time_step

    # Stateful synapse (optional membrane-level filtering)
    stateful_synapse             = False
    stateful_synapse_learnable   = False
    stateful_synapse_tau         = (10.0 + 1e-9) / time_step

    # Regularisation
    sparsity_p          = 0
    dropout_p           = 0.4
    use_batchnorm       = True
    bias                = False
    detach_reset        = True

    # ─────────────────────────────────────────────────────────────────────────
    # LOSS
    # ─────────────────────────────────────────────────────────────────────────

    # Spike aggregation across time before softmax:
    #   'sum'         → sum spikes over T  (use output_v_threshold = 1e9)
    #   'mean'        → mean spikes over T
    #   'max'         → max spike over T
    #   'spike_count' → count spikes
    loss             = "sum"
    loss_fn          = "CEloss"
    v_threshold      = 1.0
    output_v_threshold = 1e9           # effectively disables output threshold
    alpha            = 5.0             # surrogate gradient steepness

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIMISER  (dual: weights + delay positions get separate LRs)
    # ─────────────────────────────────────────────────────────────────────────

    optimizer_w         = "adam"
    optimizer_pos       = "adam"
    weight_decay        = 1e-5

    lr_w                = 1e-3
    lr_pos              = 100 * lr_w       # delay positions need higher LR

    # Schedulers
    scheduler_w         = "one_cycle"
    scheduler_pos       = "cosine_a"

    max_lr_w            = 5 * lr_w
    max_lr_pos          = 5 * lr_pos

    t_max_w             = BenchmarkBaseConfig.epochs
    t_max_pos           = BenchmarkBaseConfig.epochs

    # ─────────────────────────────────────────────────────────────────────────
    # DCLS DELAY PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────

    DCLSversion         = "gauss"          # Gaussian-weighted DCLS kernel
    decrease_sig_method = "exp"            # sigma annealing schedule

    kernel_count        = 1

    # Max delay in time steps (250 ms / 10 ms per step = 25 steps)
    # Kernel size must be odd for symmetric padding
    _max_delay_raw      = 250 // time_step
    max_delay           = _max_delay_raw if _max_delay_raw % 2 == 1 \
                          else _max_delay_raw + 1

    sigInit             = max_delay // 2   # initial Gaussian sigma

    # Epoch at which sigma annealing completes (25% of total epochs)
    final_epoch         = BenchmarkBaseConfig.epochs // 4

    # Causal padding: left = max_delay-1, right = (max_delay-1)//2
    left_padding        = max_delay - 1
    right_padding       = (max_delay - 1) // 2

    # Delay position initialisation
    init_pos_method     = "uniform"
    init_pos_a          = -(max_delay // 2)
    init_pos_b          =  (max_delay // 2)

    # ─────────────────────────────────────────────────────────────────────────
    # AUGMENTATION & LOGGING
    # ─────────────────────────────────────────────────────────────────────────

    augment             = False
    use_wandb           = False            # disabled for HPC batch runs