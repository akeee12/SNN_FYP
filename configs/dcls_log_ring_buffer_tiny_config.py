# configs/dcls_log_ring_buffer_tiny_config.py
"""
DCLS + Log Ring Buffer — tiny variant (n_hidden=64).

Smallest viable hidden size. Tests whether buffer temporal
context alone is sufficient for competitive accuracy.

Registry key: "dcls_log_ring_buffer_tiny"
"""

import math
from configs.dcls_log_ring_buffer_config import DCLSLogRingBufferConfig


class DCLSLogRingBufferTinyConfig(DCLSLogRingBufferConfig):

    model_name       = "DCLSLogRingBufferTiny"
    n_hidden_neurons = 64

    def __init__(self):
        super().__init__()
        print(f"  [DCLSLogRingBufferTiny] n_hidden=64")