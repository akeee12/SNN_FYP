# configs/power_law_tap_snntorch_config.py
"""
Power-law tap strategy for the temporal buffer — ring buffer storage.

    t_i = round( max_lag * (i / (n-1))^gamma )

    for i = 0, 1, ..., n-1

Endpoints pinned: t_0=1, t_{n-1}=max_lag.

With max_lag=64, n_taps=7, gamma=2:
    i/(n-1) = [0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
    squared  = [0, 0.028, 0.111, 0.25, 0.444, 0.694, 1.0]
    x64      = [0, 1.8,   7.1,  16,   28.4,  44.4,  64 ]
    rounded  = [1, 2,     7,    16,   28,    44,    64  ]

Compare to log:
    log      = [1, 2,     4,     8,   16,    32,    64  ]

Power law gamma=2 is genuinely distinct from log — gaps grow
quadratically rather than doubling, giving denser mid-range coverage.

gamma sweep guide:
    gamma=1   : linear spacing (same as linear_tap_config)
    gamma=2   : quadratic — default, distinct from log
    gamma=3   : cubic — even denser at very short lags

Storage: ring buffer — O(B*k) per timestep vs O(B*k*max_lag)
for rolling window. Write pointer advances with modulo arithmetic.
No data is ever physically shifted in memory.
"""

import math
from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class PowerLawTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name   = "PowerLawTapAADSNNTorch"
    tap_strategy = "power_law"
    n_taps       = 7
    gamma        = 2.0
    use_ring     = True

    def __init__(self):
        super().__init__()
        self.taps = self._compute_power_law_taps()
        print(f"  [PowerLawTap] gamma={self.gamma}  "
              f"taps={self.taps}  storage=ring")

    def _compute_power_law_taps(self) -> list:
        
        max_lag = self.buffer_max_lag
        n       = self.n_taps
        gamma   = self.gamma

        if n == 1:
            return [max_lag]

        taps = []
        for i in range(n):
            t = max_lag * ((i / (n - 1)) ** gamma)
            taps.append(max(1, round(t)))

        # pin endpoints exactly
        taps[0]  = 1
        taps[-1] = max_lag

        if len(set(taps)) != n:
            raise ValueError(
                f"Power-law taps contain duplicates: {taps}\n"
                f"max_lag={max_lag}, n_taps={n}, gamma={gamma}\n"
                f"Fix: decrease n_taps, increase max_lag, or adjust gamma."
            )
        if max(taps) > max_lag:
            raise ValueError(
                f"Tap {max(taps)} exceeds max_lag={max_lag}"
            )
        if min(taps) < 1:
            raise ValueError(
                f"Tap {min(taps)} is less than 1"
            )

        return taps