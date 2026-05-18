# configs/random_tap_snntorch_config.py
"""
Random tap strategy for the temporal buffer — ring buffer storage.

n_taps-1 lags sampled uniformly without replacement from
[1, buffer_max_lag-1], then buffer_max_lag always appended
as the final tap. Result is sorted ascending.

With buffer_max_lag=64, n_taps=7, seed=0:
    pool = [1..63], sample 6 → sort → append 64

random_tap_seed controls the sample so every run with the
same config is reproducible. Change seed for a different tap set.

Storage: ring buffer — O(B*k) per timestep vs O(B*k*max_lag)
for rolling window. Write pointer advances with modulo arithmetic.
No data is ever physically shifted in memory.
"""

import random
from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class RandomTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name      = "RandomTapAADSNNTorch"
    tap_strategy    = "random"
    n_taps          = 7
    random_tap_seed = 0
    use_ring        = True

    def __init__(self):
        super().__init__()
        self.taps = self._compute_random_taps()
        print(f"  [RandomTap] seed={self.random_tap_seed}  "
              f"taps={self.taps}  storage=ring")

    def _compute_random_taps(self) -> list:
        max_lag = self.buffer_max_lag
        n       = self.n_taps

        if n > max_lag:
            raise ValueError(
                f"n_taps={n} > buffer_max_lag={max_lag}, "
                f"cannot sample {n} unique lags from [1, {max_lag}]"
            )
        if n < 1:
            raise ValueError(
                f"n_taps must be >= 1, got {n}"
            )

        rng  = random.Random(self.random_tap_seed)
        pool = list(range(1, max_lag))   # excludes max_lag
        taps = sorted(rng.sample(pool, n - 1))
        taps.append(max_lag)             # always pin max_lag as last tap

        if len(taps) != n:
            raise ValueError(
                f"Random taps produced {len(taps)} taps, expected {n}"
            )
        if len(set(taps)) != n:
            raise ValueError(
                f"Random taps contain duplicates: {taps}"
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