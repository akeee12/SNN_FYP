# configs/linear_tap_snntorch_config.py
"""
Linear tap strategy for the temporal buffer — ring buffer storage.

Taps are evenly spaced integers from 1 to buffer_max_lag inclusive.
With n_taps=7, buffer_max_lag=64:

    step = 64 / 7 ≈ 9.14
    taps = [9, 18, 27, 36, 45, 54, 64]

The last tap is always pinned to buffer_max_lag exactly.
fc_expand input dim = buffer_size * n_taps = 32 * 7 = 224  (same as log)

Storage: ring buffer — O(B*k) per timestep vs O(B*k*max_lag) for
rolling window. Write pointer advances with modulo arithmetic.
No data is ever physically shifted in memory.
"""

from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class LinearTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name    = "LinearTapAADSNNTorch"
    tap_strategy  = "linear"
    n_taps        = 7          # must match log baseline for fair comparison
    use_ring      = True       # ring buffer storage — O(B*k) per timestep

    def __init__(self):
        super().__init__()
        self.taps = self._compute_linear_taps()
        print(f"  [LinearTap] taps={self.taps}  storage=ring")

    def _compute_linear_taps(self) -> list:
        max_lag = self.buffer_max_lag
        n       = self.n_taps
        step    = max_lag / n
        taps    = [max(1, round(step * (i + 1))) for i in range(n)]
        taps[-1] = max_lag

        if len(taps) != n:
            raise ValueError(
                f"Linear taps produced {len(taps)} taps, expected {n}"
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