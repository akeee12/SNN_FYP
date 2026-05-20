# configs/log_ring_buffer_snntorch_config.py
"""
Log-spaced tap strategy with ring buffer storage.

Identical to log_buffer_snntorch_config but uses ring buffer
storage mechanism — O(B*k) per timestep vs O(B*k*max_lag).

Registry key: "log_ring_buffer_snntorch"
"""

import math
from configs.adaptiveAxonalDelayV2SNNtorch_config import AdaptiveAxonalDelayV2ConfigSNNTorch


class LogRingBufferSNNTorchConfig(AdaptiveAxonalDelayV2ConfigSNNTorch):

    model_name   = "LogRingBufferAADSNNTorch"
    tap_strategy = "log"
    
    use_ring     = True

    buffer_size    = 32
    buffer_max_lag = 64
    target_true    = 0.8
    target_false   = 0.0
    theta_u        = 0.5

    def __init__(self):
        super().__init__()
        self.taps   = self._compute_log_taps()
        self.n_taps = len(self.taps)
        print(f"  [LogRingBuffer] taps={self.taps}  storage=ring")

    def _compute_log_taps(self) -> list:
        max_lag = self.buffer_max_lag

        if math.log2(max_lag) != int(math.log2(max_lag)):
            raise ValueError(
                f"buffer_max_lag={max_lag} must be a power of two "
                f"for log tap strategy."
            )

        n_levels = int(math.log2(max_lag))
        taps     = [2 ** i for i in range(n_levels + 1)]

        if max(taps) > max_lag:
            raise ValueError(
                f"Max tap {max(taps)} exceeds max_lag={max_lag}"
            )
        if min(taps) < 1:
            raise ValueError(
                f"Min tap {min(taps)} is less than 1"
            )

        return taps