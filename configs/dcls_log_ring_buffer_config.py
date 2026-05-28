# configs/dcls_log_ring_buffer_config.py
"""
DCLS + Log Ring Buffer configuration for SHD / SSC / DVS.

Replaces the binning + fc_in input stage of DCLSNet with a
log-spaced ring buffer front-end:

    Baseline DCLS input path:
        700 → bin(n_bins=5) → 140 → fc_in → 256

    Log ring buffer input path:
        700 → fc_proj(k=32) → ring buffer → fc_expand(k*L → 256)

The DCLS delay layers (Dcls1d), LIF neurons, BN, dropout, and
fc_out are all unchanged.

Parameter comparison (k=32, L=7, max_lag=64):
    Baseline fc_in : 140 * 256 = 35,840
    Log buffer     : 700*32 + 224*256 = 22,400 + 57,344 = 79,744

Note: the buffer operates on the raw 700-channel input so it
captures the full spatial resolution before binning. This is
intentional — the buffer's fc_proj learns its own spatial
compression rather than using the fixed n_bins grouping.

Registry key: "dcls_log_ring_buffer"
"""

import math
from configs.dcls_config import DCLSConfig


class DCLSLogRingBufferConfig(DCLSConfig):

    # ── Identity ──────────────────────────────────────────────────────
    model_name = "DCLSLogRingBuffer"

    # ── Log ring buffer hyperparameters ──────────────────────────────
    buffer_size    = 32    # k — bottleneck projection dimension
    buffer_max_lag = 64    # max lag — lags = {1,2,4,8,16,32,64} → L=7
    theta_u        = 0.5   # current scaling into first DCLS layer

    # ── n_inputs override ─────────────────────────────────────────────
    # The buffer takes raw 700-channel input directly.
    # n_bins and the derived n_inputs=140 are no longer used for
    # the input projection — but n_bins is kept for compatibility
    # with any code that reads it.
    n_inputs = 700         # buffer fc_proj input dimension

    def __init__(self):
        super().__init__()
        self.taps   = self._compute_log_taps()
        self.n_taps = len(self.taps)
        print(f"  [DCLSLogRingBuffer] taps={self.taps}  "
              f"k={self.buffer_size}  storage=ring")

    def _compute_log_taps(self) -> list:
        max_lag = self.buffer_max_lag
        if math.log2(max_lag) != int(math.log2(max_lag)):
            raise ValueError(
                f"buffer_max_lag={max_lag} must be a power of two."
            )
        n_levels = int(math.log2(max_lag))
        taps     = [2 ** i for i in range(n_levels + 1)]
        return taps