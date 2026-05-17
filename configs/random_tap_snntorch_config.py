# configs/random_tap_snntorch_config.py
"""
Random tap strategy for the temporal buffer.

n_taps-1 lags are sampled uniformly without replacement from
[1, buffer_max_lag-1], then buffer_max_lag is always appended
as the final tap. The result is sorted ascending.

random_tap_seed controls the sample so every run with the same
config is reproducible. Change the seed to get a different tap set
without changing anything else.

With buffer_max_lag=64, n_taps=7, seed=0:
    pool = [1..63], sample 6 → sort → append 64
    taps = (depends on seed — printed at init)
"""

import random
from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class RandomTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name      = "RandomTapAADSNNTorch"
    tap_strategy    = "random"
    n_taps          = 7
    random_tap_seed = 0       # change for a different random tap set

    def __init__(self):
        super().__init__()
        self.taps = self._compute_random_taps()
        print(f"  [RandomTap] seed={self.random_tap_seed}  taps={self.taps}")

    def _compute_random_taps(self) -> list:
        max_lag = self.buffer_max_lag
        n       = self.n_taps

        assert n <= max_lag, (
            f"n_taps={n} > buffer_max_lag={max_lag}, "
            f"cannot sample {n} unique lags."
        )

        rng  = random.Random(self.random_tap_seed)
        # pool excludes max_lag — it is always pinned as the last tap
        pool = list(range(1, max_lag))
        taps = sorted(rng.sample(pool, n - 1))
        taps.append(max_lag)

        assert len(taps) == n
        assert len(set(taps)) == n   # all unique by construction
        assert max(taps) <= max_lag
        assert min(taps) >= 1
        return taps