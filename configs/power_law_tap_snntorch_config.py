# configs/power_law_tap_snntorch_config.py
"""
Power-law tap strategy with tunable gamma exponent.

    t_i = round( max_lag * (i / (n-1))^gamma )

    for i = 0, 1, ..., n-1

Endpoints are pinned: t_0=1, t_{n-1}=max_lag.

This is deliberately and mathematically distinct from both log
and true exponential strategies:

    Log spacing    : geometric sequence, t_i = max_lag^(i/(n-1))
                     ratio between consecutive taps is CONSTANT
                     → equal ratio spacing in log domain

    True exp (e^i) : grows faster than any polynomial, hits max_lag
                     quickly and zero-pads remaining taps

    Power law γ=2  : t_i = max_lag * (i/(n-1))^2
                     gaps INCREASE quadratically
                     denser coverage of recent lags
                     → equal spacing in sqrt domain

With max_lag=64, n_taps=7, gamma=2:
    i/(n-1) = [0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
    squared  = [0, 0.028, 0.111, 0.25, 0.444, 0.694, 1.0]
    × 64     = [0, 1.8,   7.1,  16,   28.4,  44.4,  64 ]
    rounded  = [1, 2,     7,    16,   28,    44,    64  ]

Compare directly:
    Log      (ratio=2) : [1,  2,  4,  8, 16, 32, 64]
    Power γ=2          : [1,  2,  7, 16, 28, 44, 64]
    Linear             : [9, 18, 27, 36, 45, 54, 64]

Power law with gamma=2 sits between log and linear — it keeps
dense short-lag coverage but reaches mid-range lags faster than
log. Supported by temporal coding literature (Perez-Nieves et al.
2021, NeurIPS) which shows SNN temporal features benefit from
non-uniform lag spacing biased toward recent history.

gamma sweep guide:
    gamma < 1  : sub-linear → denser at LONG lags (unusual)
    gamma = 1  : linear spacing (same as linear_tap_config)
    gamma = 2  : quadratic → denser at SHORT lags (default)
    gamma = 3  : cubic → even denser at very short lags
"""

import math
from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class PowerLawTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name   = "PowerLawTapAADSNNTorch"
    tap_strategy = "power_law"
    n_taps       = 7
    gamma        = 2.0     # exponent — sweep this as a hyperparameter
                           # gamma=2 is default, distinct from log

    def __init__(self):
        super().__init__()
        self.taps = self._compute_power_law_taps()
        print(f"  [PowerLawTap] gamma={self.gamma}  taps={self.taps}")

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

        # pin endpoints exactly — rounding can shift by 1
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