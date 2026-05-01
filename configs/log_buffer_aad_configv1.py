# configs/log_buffer_aad_config.py
"""
Configuration for LogBufferAADNet.

Extends AdaptiveAxonalDelayV2Config with log buffer
hyperparameters. All neuron/training params inherited.

New hyperparameters:
    buffer_size    : k — bottleneck dimension
    buffer_max_lag : max lag for log spacing (determines L)

Parameter comparison (k=32, max_lag=64, L=7):
    Baseline FC1:  700*128        = 89,600
    Log buffer:    700*32+224*128 = 51,072  (-43%)
"""

from configs.axonal_delay_modelPYTORCH_config import (
    AdaptiveAxonalDelayV2Config
)


class LogBufferAADConfigv1(AdaptiveAxonalDelayV2Config):
    """
    Log Buffer AAD config — inherits all V2 settings.
    Only adds buffer-specific hyperparameters.
    """

    # ── Identity ──────────────────────────────────────────────────────
    model_name = "LogBufferAAD"

    # ── Log buffer hyperparameters ────────────────────────────────────
    # These are the ONLY new hyperparameters vs V2

    buffer_size    = 32    # k — bottleneck dimension
                           # try: 16, 32, 64
                           # smaller k = fewer params but less capacity

    buffer_max_lag = 64    # max lag — matches delay_init_cap
                           # lags = {1,2,4,8,16,32,64} → L=7
                           # try: 16 (L=5), 32 (L=6), 64 (L=7)

    # ── Parameter count reference ─────────────────────────────────────
    # k=16, L=7: 700*16 + 112*128 = 11,200 + 14,336 = 25,536  (-71%)
    # k=32, L=7: 700*32 + 224*128 = 22,400 + 28,672 = 51,072  (-43%)
    # k=64, L=7: 700*64 + 448*128 = 44,800 + 57,344 = 102,144 (-14%)