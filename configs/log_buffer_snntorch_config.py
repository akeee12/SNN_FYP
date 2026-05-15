# configs/log_buffer_aad_snntorch_config.py
"""
Configuration for LogBufferAADNetSNNTorch — 500-epoch production run.

Inherits ALL neuron/training/delay settings from
AdaptiveAxonalDelayV2ConfigSNNTorch and adds only the log-buffer
hyperparameters.

Log buffer replaces fc1 (700→128) with:
    fc_proj   : Linear(700, k,   bias=False)   — bottleneck
    fc_expand : Linear(k*L, 128, bias=False)   — temporal expansion

Lag set  : {1, 2, 4, …, buffer_max_lag}  → L = log2(buffer_max_lag)+1
           buffer_max_lag=64 → L=7  lags={1,2,4,8,16,32,64}

Parameter comparison (k=32, max_lag=64, L=7):
    Baseline fc1 : 700*128        = 89,600
    Log buffer   : 700*32+224*128 = 51,072  (-43%)

target_true / target_false:
    Override the V2 tgt_count_true/tgt_count_false fractions.
    target_true=0.8  → correct  neuron fires 80% of timesteps
    target_false=0.0 → incorrect neuron fires  0% of timesteps
    These are passed directly to SF.mse_count_loss as rates.

theta_u:
    Log buffer fc_proj+fc_expand produces u_ff max ~26 vs AAD fc1 ~8.
    theta_u=0.5 keeps hidden layer out of saturation.
    (theta_u=0.2 → rate≈0.48 SATURATED; theta_u=5.0 → rate≈0.04 too sparse)
"""

from configs.aad_v2_snntorch_config import AdaptiveAxonalDelayV2ConfigSNNTorch


class LogBufferAADConfigSNNTorch(AdaptiveAxonalDelayV2ConfigSNNTorch):
    """
    Log Buffer AAD config — inherits all V2 SNNTorch settings.
    Only adds / overrides buffer-specific hyperparameters.
    """

    # ── Identity ──────────────────────────────────────────────────────
    model_name = "LogBufferAADSNNTorch" 

    # ── Log buffer hyperparameters ────────────────────────────────────
    buffer_size    = 32   # k — bottleneck projection dimension
                          # options: 16 (-71%), 32 (-43%), 64 (-14%)

    buffer_max_lag = 64   # max lag — should match delay_init_cap
                          # lags = {1,2,4,…,buffer_max_lag} → L=log2+1
                          # buffer_max_lag=64 → L=7

    # ── Loss targets (rates, not counts) ─────────────────────────────
    # Passed directly to SF.mse_count_loss(correct_rate, incorrect_rate)
    # Override V2's tgt_count_true/tgt_count_false fraction approach.
    target_true  = 0.8   # correct   class fires 80% of timesteps
    target_false = 0.0   # incorrect class fires  0% of timesteps

    # ── Input current scaling ─────────────────────────────────────────
    # fc_proj+fc_expand produces larger u_ff than a single fc1.
    # theta_u gates the current into lif1 to prevent saturation.
    theta_u = 0.5

    # ── Parameter count reference ─────────────────────────────────────
    # k=16, L=7: 700*16 + 112*128 = 25,536  (-71% vs baseline)
    # k=32, L=7: 700*32 + 224*128 = 51,072  (-43% vs baseline)
    # k=64, L=7: 700*64 + 448*128 = 102,144 (-14% vs baseline)

    def __init__(self):
        # Recompute V2 count targets (keeps __init__ contract intact)
        # target_true/target_false are rates and bypass this.
        super().__init__()