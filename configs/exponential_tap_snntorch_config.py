# configs/exponential_tap_snntorch_config.py
"""
True exponential tap strategy using e^i spacing.

    t_i = round(e^i)  for i = 1, 2, ..., n_taps

This grows MUCH faster than log (ratio=2) or power-law.
When a tap exceeds buffer_max_lag it is set to 0 — the buffer
returns a zero vector for that tap, representing no signal at
that lag. This is a deliberate architectural choice: the network
learns that beyond a certain lag, no useful information exists.

With buffer_max_lag=64, n_taps=7:
    e^1=2.7  → 3
    e^2=7.4  → 7
    e^3=20.1 → 20
    e^4=54.6 → 55
    e^5=148  → EXCEEDS 64 → 0
    e^6=403  → EXCEEDS 64 → 0
    e^7=1097 → EXCEEDS 64 → 0
    taps = [3, 7, 20, 55, 0, 0, 0]

The zero taps are NOT read from the buffer — the buffer returns
torch.zeros for those positions. fc_expand still receives k*L
features so the weight matrix shape is unchanged. The zero taps
effectively give fc_expand dead input channels for those positions,
which the network can learn to ignore or use as a bias-like signal.

Why keep n_taps fixed at 7 with zeros?
    fc_expand input dim = k * n_taps must be identical across all
    strategies for a fair parameter count comparison. Zeros preserve
    this without changing any weight shapes.

Literature basis:
    True exponential spacing (e^i) reflects the hypothesis that
    temporal context decays exponentially in relevance — a common
    assumption in echo state networks (Jaeger 2001) and liquid state
    machines (Maass et al. 2002). The hard cutoff at max_lag with
    zero-padding is analogous to a causal filter with finite support.
"""

import math
from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig


class ExponentialTapSNNTorchConfig(LogBufferSNNTorchConfig):

    model_name   = "ExponentialTapAADSNNTorch"
    tap_strategy = "exponential"
    n_taps       = 7      # fixed — keeps fc_expand shape identical to log

    def __init__(self):
        super().__init__()
        self.taps, self.zero_mask = self._compute_exponential_taps()
        n_valid = sum(1 for t in self.taps if t > 0)
        n_zero  = sum(1 for t in self.taps if t == 0)
        print(f"  [ExponentialTap] taps={self.taps}")
        print(f"  valid={n_valid}  zero-padded={n_zero}")

    def _compute_exponential_taps(self):
        max_lag = self.buffer_max_lag
        n       = self.n_taps

        taps      = []
        zero_mask = []   # True where tap is valid, False where zeroed

        for i in range(1, n + 1):
            t = round(math.exp(i))
            if t <= max_lag:
                taps.append(t)
                zero_mask.append(True)
            else:
                taps.append(0)      # 0 signals "return zeros for this tap"
                zero_mask.append(False)

        # validate valid taps are unique and in range
        valid_taps = [t for t in taps if t > 0]
        if len(set(valid_taps)) != len(valid_taps):
            raise ValueError(
                f"Exponential taps contain duplicates among valid taps: "
                f"{valid_taps}\n"
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