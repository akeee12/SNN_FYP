# configs/exponential_tap_snntorch_config.py
"""
Exponential tap strategy for the temporal buffer — ring buffer storage.

Taps computed as round(e^i) for i = 1, 2, ..., n_taps.
When a tap exceeds buffer_max_lag it is set to 0 — the buffer
returns a zero vector for that position.

With buffer_max_lag=64, n_taps=7:
    e^1=2.7  → 3
    e^2=7.4  → 7
    e^3=20.1 → 20
    e^4=54.6 → 55
    e^5=148  → EXCEEDS 64 → 0
    e^6=403  → EXCEEDS 64 → 0
    e^7=1097 → EXCEEDS 64 → 0
    taps = [3, 7, 20, 55, 0, 0, 0]

Storage: ring buffer — O(B*k) per timestep vs O(B*k*max_lag) for
rolling window. Write pointer advances with modulo arithmetic.
Zero taps return zeros — fc_expand input shape k*L unchanged.
"""

import math
from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class ExponentialTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name   = "ExponentialTapAADSNNTorch"
    tap_strategy = "exponential"
    n_taps       = 7
    use_ring     = True

    def __init__(self):
        super().__init__()
        self.taps, self.zero_mask = self._compute_exponential_taps()
        n_valid = sum(1 for t in self.taps if t > 0)
        n_zero  = sum(1 for t in self.taps if t == 0)
        print(f"  [ExponentialTap] taps={self.taps}  "
              f"valid={n_valid}  zero-padded={n_zero}  storage=ring")

    def _compute_exponential_taps(self):
        max_lag   = self.buffer_max_lag
        n         = self.n_taps
        taps      = []
        zero_mask = []

        for i in range(1, n + 1):
            t = round(math.exp(i))
            if t <= max_lag:
                taps.append(t)
                zero_mask.append(True)
            else:
                taps.append(0)
                zero_mask.append(False)

        valid_taps = [t for t in taps if t > 0]
        if len(set(valid_taps)) != len(valid_taps):
            raise ValueError(
                f"Exponential taps contain duplicates among valid "
                f"taps: {valid_taps}\n"
                f"max_lag={max_lag}, n_taps={n}\n"
                f"Fix: reduce n_taps."
            )
        if valid_taps and max(valid_taps) > max_lag:
            raise ValueError(
                f"Valid tap {max(valid_taps)} exceeds max_lag={max_lag}"
            )
        if valid_taps and min(valid_taps) < 1:
            raise ValueError(
                f"Valid tap {min(valid_taps)} is less than 1"
            )

        return taps, zero_mask