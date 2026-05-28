# configs/dcls_power_law_ring_buffer_config.py
"""
DCLS + Power-Law Ring Buffer configuration for SHD / SSC / DVS.

Replaces the binning + fc_in input stage of DCLSNet with a
power-law-spaced ring buffer front-end:

    Baseline DCLS input path:
        700 → bin(n_bins=5) → 140 → fc_in → 256

    Power-law ring buffer input path:
        700 → fc_proj(k=32) → ring buffer → fc_expand(k*L → 256)

Power-law tap spacing (max_lag=64, n_taps=7, gamma=2.0):
    taps = [1, 2, 7, 16, 28, 44, 64]  — non-uniform, denser at recent end

The DCLS delay layers (Dcls1d), LIF neurons, BN, dropout, and
fc_out are all unchanged.

Registry key: "dcls_power_law_ring_buffer"
"""

import math
import numpy as np
from configs.dcls_config import DCLSConfig


class DCLSPowerLawRingBufferConfig(DCLSConfig):

    # ── Identity ──────────────────────────────────────────────────────
    model_name = "DCLSPowerLawRingBuffer"

    # ── Power-law ring buffer hyperparameters ─────────────────────────
    buffer_size    = 32    # k — bottleneck projection dimension
    buffer_max_lag = 64    # max lag
    n_taps_target  = 7     # how many taps to generate
    gamma          = 2.0   # power-law exponent (higher = denser at recent end)
    theta_u        = 0.5   # current scaling into first DCLS layer

    # ── n_inputs override ─────────────────────────────────────────────
    n_inputs = 700         # buffer fc_proj input dimension

    def __init__(self):
        super().__init__()
        self.taps   = self._compute_power_law_taps()
        self.n_taps = len(self.taps)
        print(f"  [DCLSPowerLawRingBuffer] taps={self.taps}  "
              f"k={self.buffer_size}  gamma={self.gamma}  storage=ring")

    def _compute_power_law_taps(self) -> list:
        """
        Generate n_taps_target power-law spaced taps in [1, max_lag].

        Uses the same formula as PowerLawTapAADNetSNNTorch:
            positions = round(((i / (n-1)) ** gamma) * max_lag)
        ensuring first tap=1 and last tap=max_lag, all unique, all >= 1.

        Args:
            n     : number of taps
            gamma : exponent (gamma > 1 → denser recent, sparser old)
        """
        n       = self.n_taps_target
        max_lag = self.buffer_max_lag
        gamma   = self.gamma

        if n < 2:
            raise ValueError("n_taps_target must be >= 2")

        raw  = [((i / (n - 1)) ** gamma) * max_lag for i in range(n)]
        taps = sorted(set(max(1, round(v)) for v in raw))

        # force endpoints
        if taps[0]  != 1:       taps[0]  = 1
        if taps[-1] != max_lag: taps[-1] = max_lag

        taps = sorted(set(taps))
        return taps