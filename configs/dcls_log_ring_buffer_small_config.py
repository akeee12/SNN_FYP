# configs/dcls_log_ring_buffer_small_config.py
"""
DCLS + Log Ring Buffer — small variant (n_hidden=128).

Hypothesis: buffer temporal context compensates for reduced
hidden size. Comparable total params to baseline at n_hidden=256.

Parameter comparison:
    Baseline n_hidden=256 : fc_in=35,840  dcls=131,072  total~175k
    Buffer   n_hidden=128 : fc_proj+expand=40,960  dcls=32,768  total~82k
    Buffer   n_hidden=64  : fc_proj+expand=22,912  dcls=8,192   total~38k

Registry key: "dcls_log_ring_buffer_small"
"""

import math
from configs.dcls_log_ring_buffer_config import DCLSLogRingBufferConfig


class DCLSLogRingBufferSmallConfig(DCLSLogRingBufferConfig):

    model_name       = "DCLSLogRingBufferSmall"
    n_hidden_neurons = 128

    def __init__(self):
        super().__init__()
        print(f"  [DCLSLogRingBufferSmall] n_hidden=128")