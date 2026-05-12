# configs/log_buffer_snntorch_config.py

from configs.log_buffer_aad_configv1 import LogBufferAADConfigv1


class LogBufferSNNTorchConfig(LogBufferAADConfigv1):

    model_name = "LogBufferAADSNNTorch"

    # ── SNNTorch specific ─────────────────────────────────────────────
    # tau_s and tau_r map to alpha/beta in snn.Synaptic
    # These are the same values that worked for the SRM version
    tau_s          = 10.0
    tau_r          = 10.0

    # Threshold — confirmed from activation sweep
    # snn.Synaptic accumulates differently from SRM
    # start with same value and probe if needed
    threshold      = 0.5
    threshold_out  = 0.5

    # Surrogate gradient slope
    surrogate_slope = 25

    # Log buffer — inherited from LogBufferAADConfigv1
    # buffer_size    = 32
    # buffer_max_lag = 64