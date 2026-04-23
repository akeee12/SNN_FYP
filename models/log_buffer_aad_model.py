# models/log_buffer_aad_model.py
"""
Adaptive Axonal Delay SNN with Logarithmic Spike Buffer.

Extension of Sun et al. ICASSP 2023 with a biologically-motivated
logarithmic temporal buffer at the input stage.

Architecture:
    Input (B, T, 700)
        ↓
    LogSpikeBuffer → (B, T, n_taps * 700)
        ↓
    FC1 → SRM1 → Delay1
        ↓
    FC2 → SRM2 → Delay2
        ↓
    FC3 → SRM3
        ↓
    sum(T) → (B, 20)

Key difference from baseline:
    FC1 input size = n_taps * 700 (e.g. 5600 for n_taps=8)
    vs baseline FC1 input size = 700
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from axonal_delay_modelPYTorch import (
    SLAYERSurrogate,
    DelayGradient,
    AxonalDelay,
    PSPKernel,
    RefractoryKernel,
    SRMLayer,
)
from log_buffer import LogarithmicSpikeBuffer


class AdaptiveAxonalDelayNetLogBuffer(nn.Module):
    """
    AAD-SNN with Logarithmic Spike Buffer extension.

    Inherits all SRM, delay, and surrogate gradient components
    from the baseline AAD model. Only the input stage changes.

    Args:
        cfg: LogBufferAADConfig instance
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── Logarithmic Spike Buffer ───────────────────────────────────────
        self.log_buffer = LogarithmicSpikeBuffer(
            n_inputs  = cfg.n_inputs,
            max_delay = cfg.log_buffer_max_delay,
            n_taps    = cfg.log_buffer_n_taps,
            learnable = cfg.log_buffer_learnable,
        )
        # Input to FC1 is now n_taps × n_inputs
        buffered_size = self.log_buffer.output_size()

        # ── SRM layers (identical to baseline) ────────────────────────────
        srm_kwargs = dict(
            tau_s      = cfg.tau_s,
            tau_r      = cfg.tau_r,
            theta_u    = cfg.theta_u,
            alpha_r    = cfg.alpha_r,
            tau_rho    = cfg.tau_rho,
            scale_rho  = cfg.scale_rho,
            kernel_size= cfg.kernel_size,
        )
        self.srm1 = SRMLayer(**srm_kwargs)
        self.srm2 = SRMLayer(**srm_kwargs)
        self.srm3 = SRMLayer(**srm_kwargs)

        # ── FC layers — FC1 takes buffered input ──────────────────────────
        self.fc1 = nn.utils.weight_norm(
            nn.Linear(buffered_size,  cfg.n_hidden,  bias=False))
        self.fc2 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,   cfg.n_hidden,  bias=False))
        self.fc3 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,   cfg.n_outputs, bias=False))

        # ── Learnable axonal delays (identical to baseline) ───────────────
        self.delay1 = AxonalDelay(
            cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(
            cfg.n_hidden, max_delay=cfg.delay_init_cap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 700) — raw SHD spikes from shared loader
        Returns:
            out: (B, 20) — summed spike counts
        """
        # ── Log buffer: (B, T, 700) → (B, T, n_taps*700) ─────────────────
        x = self.log_buffer(x)                               # (B, T, K*700)

        # ── Permute to (B, K*700, T) for FC layers ────────────────────────
        x = x.permute(0, 2, 1)                               # (B, K*700, T)

        # ── Layer 1 ───────────────────────────────────────────────────────
        x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 128, T)
        x = self.srm1(x)
        x = self.delay1(x)

        # ── Layer 2 ───────────────────────────────────────────────────────
        x = self.fc2(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 128, T)
        x = self.srm2(x)
        x = self.delay2(x)

        # ── Output layer ──────────────────────────────────────────────────
        x = self.fc3(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 20, T)
        x = self.srm3(x)

        return x.sum(dim=-1)                                  # (B, 20)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }

    def get_buffer_info(self) -> dict:
        """Returns log buffer tap positions for logging."""
        return {
            "tap_positions": self.log_buffer.get_tap_positions(),
            "n_taps"       : self.cfg.log_buffer_n_taps,
            "max_delay"    : self.cfg.log_buffer_max_delay,
            "input_size"   : self.log_buffer.output_size(),
        }