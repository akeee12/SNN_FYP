# configs/linear_tap_snntorch_config.py
"""
Linear tap strategy for the temporal buffer.

Taps are evenly spaced integers from 1 to buffer_max_lag inclusive.
With n_taps=7, buffer_max_lag=64:

    step = 64 / 7 ≈ 9.14
    taps = [9, 18, 27, 36, 45, 54, 64]

The last tap is always pinned to buffer_max_lag exactly.
fc_expand input dim = buffer_size * n_taps = 32 * 7 = 224  (same as log)
"""

from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class LinearTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name   = "LinearTapAADSNNTorch"
    tap_strategy = "linear"
    n_taps       = 7          # must match log baseline for fair comparison

    def __init__(self):
        super().__init__()
        self.taps = self._compute_linear_taps()
        print(f"  [LinearTap] taps = {self.taps}")

    def _compute_linear_taps(self) -> list:
        max_lag = self.buffer_max_lag
        n       = self.n_taps
        step    = max_lag / n
        taps    = [max(1, round(step * (i + 1))) for i in range(n)]
        taps[-1] = max_lag    # pin last tap exactly to max_lag
        assert len(taps) == n
        assert max(taps) <= max_lag
        assert min(taps) >= 1
        return taps