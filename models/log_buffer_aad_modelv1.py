# models/log_buffer_aad_modelv1.py
"""
LogBufferAADNetV1 — AAD network with log-buffer input projection.

Replaces FC1 [700→128] with:
    FC_proj  [700→k]     spatial compression
    LogBuffer [k→k*L]    temporal history — zero parameters
    FC_expand [k*L→128]  project back to hidden dim

Everything from SRM1 onwards is identical to AdaptiveAxonalDelayNetV2.
Imports all components from adaptiveAxonalDelayV2.py directly.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptiveAxonalDelayV2 import (
    SLAYERSurrogate,
    DelayGradient,
    AxonalDelay,
    PSPKernel,
    RefractoryKernel,
    SRMLayer,
    VanRossumLoss,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOG BUFFER — zero parameters, pure temporal structure
# ─────────────────────────────────────────────────────────────────────────────

class LogBufferAADNetV1(nn.Module):
    """
    Temporal log-spaced history buffer.

    Takes (B, k, T) and returns (B, k*L, T) where:
        L    = floor(log2(max_lag)) + 1
        lags = {1, 2, 4, 8, ..., max_lag}

    Zero learnable parameters — purely structural.
    Out-of-bounds (t < lag) filled with zeros — causal.
    """

    def __init__(self, n_neurons: int, max_lag: int = 64):
        super().__init__()
        self.n_neurons = n_neurons

        L         = int(math.floor(math.log2(max_lag))) + 1
        self.lags = [2**i for i in range(L)]
        self.L    = len(self.lags)

        print(f"    LogBuffer: n={n_neurons}, "
              f"max_lag={max_lag}, "
              f"lags={self.lags}, "
              f"L={self.L}, "
              f"output_dim={n_neurons * self.L}")

    @property
    def output_dim(self) -> int:
        return self.n_neurons * self.L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, k, T)
        Returns:
            out: (B, k*L, T)
        """
        B, N, T = x.shape
        lagged  = []

        for lag in self.lags:
            if lag == 0:
                lagged.append(x)
            else:
                shifted = F.pad(x, (lag, 0))[:, :, :T]
                lagged.append(shifted)

        return torch.cat(lagged, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class LogBufferAADNetV1(nn.Module):
    """
    AAD network with log-buffer replacing FC1.

    Input : (B, T, 700)
    Output: (B, 20, T)
    """

    def __init__(self, cfg):
        super().__init__()

        k       = cfg.buffer_size
        max_lag = cfg.buffer_max_lag

        # ── SRM kwargs — matches adaptiveAxonalDelayV2.SRMLayer ───────
        # SRMLayer takes: tau_s, tau_r, theta_u, alpha_r,
        #                 tau_rho, scale_rho, kernel_size
        srm_kwargs = dict(
            tau_s       = cfg.tau_s,
            tau_r       = cfg.tau_r,
            theta_u     = cfg.theta_u,
            alpha_r     = cfg.alpha_r,
            tau_rho     = cfg.tau_rho,
            scale_rho   = cfg.scale_rho,
            kernel_size = cfg.kernel_size,
        )

        # ── Log-buffer block (replaces FC1) ───────────────────────────
        self.fc_proj   = nn.Linear(cfg.n_inputs, k, bias=False)
        self.log_buf   = LogBufferAADNetV1(k, max_lag)
        self.fc_expand = nn.Linear(self.log_buf.output_dim,
                                   cfg.n_hidden, bias=False)

        # ── Rest of network — identical to V2 ─────────────────────────
        self.srm1   = SRMLayer(**srm_kwargs)
        self.delay1 = AxonalDelay(cfg.n_hidden, cfg.delay_init_cap)

        self.fc2    = nn.Linear(cfg.n_hidden, cfg.n_hidden,  bias=False)
        self.srm2   = SRMLayer(**srm_kwargs)
        self.delay2 = AxonalDelay(cfg.n_hidden, cfg.delay_init_cap)

        self.fc3    = nn.Linear(cfg.n_hidden, cfg.n_outputs, bias=False)
        self.srm3   = SRMLayer(**srm_kwargs)

        # ── Weight init ×10 — matches SLAYER weightScale=10 ───────────
        with torch.no_grad():
            self.fc_proj.weight.data   *= cfg.weight_scale
            self.fc_expand.weight.data *= cfg.weight_scale
            self.fc2.weight.data       *= cfg.weight_scale
            self.fc3.weight.data       *= cfg.weight_scale

        # ── Loss — VanRossumLoss matching V2 exactly ──────────────────
        self.loss_fn    = VanRossumLoss(
            tau_s       = cfg.tau_s,
            kernel_size = cfg.kernel_size,
            Ts          = cfg.Ts,
        )
        self._tgt_true  = cfg.tgt_count_true
        self._tgt_false = cfg.tgt_count_false

        # ── Parameter summary ─────────────────────────────────────────
        self._print_param_summary(k, max_lag)

    def _print_param_summary(self, k: int, max_lag: int):
        L         = self.log_buf.L
        fc_proj_p = self.fc_proj.weight.numel()
        fc_exp_p  = self.fc_expand.weight.numel()
        fc2_p     = self.fc2.weight.numel()
        fc3_p     = self.fc3.weight.numel()
        delay_p   = (self.delay1.delay.numel() +
                     self.delay2.delay.numel())
        total     = fc_proj_p + fc_exp_p + fc2_p + fc3_p + delay_p
        baseline  = 700*128 + 128*128 + 128*20 + 256

        print(f"\n  LogBufferAADNet parameter summary:")
        print(f"    k={k}, max_lag={max_lag}, L={L}")
        print(f"    fc_proj:   {fc_proj_p:>8,}  (700×{k})")
        print(f"    fc_expand: {fc_exp_p:>8,}  ({k*L}×128)")
        print(f"    fc2:       {fc2_p:>8,}  (128×128)")
        print(f"    fc3:       {fc3_p:>8,}  (128×20)")
        print(f"    delays:    {delay_p:>8,}")
        print(f"    ─────────────────────")
        print(f"    total:     {total:>8,}")
        print(f"    baseline:  {baseline:>8,}")
        print(f"    saving:    {baseline-total:>8,} "
              f"({(baseline-total)/baseline*100:.1f}%)\n")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 700)
        Returns:
            out: (B, 20, T)
        """
        x = x.permute(0, 2, 1)                              # (B, 700, T)

        # ── Log-buffer block ──────────────────────────────────────────
        x = self.fc_proj(
            x.permute(0, 2, 1)
        ).permute(0, 2, 1)                                   # (B, k, T)

        x = self.log_buf(x)                                  # (B, k*L, T)

        x = self.fc_expand(
            x.permute(0, 2, 1)
        ).permute(0, 2, 1)                                   # (B, 128, T)

        # ── Hidden layer 1 ────────────────────────────────────────────
        x = self.srm1(x)
        x = self.delay1(x)

        # ── Hidden layer 2 ────────────────────────────────────────────
        x = self.fc2(
            x.permute(0, 2, 1)
        ).permute(0, 2, 1)                                   # (B, 128, T)
        x = self.srm2(x)
        x = self.delay2(x)

        # ── Output layer ──────────────────────────────────────────────
        x = self.fc3(
            x.permute(0, 2, 1)
        ).permute(0, 2, 1)                                   # (B, 20, T)
        x = self.srm3(x)

        return x                                             # (B, 20, T)

    def compute_loss(self, spike_out: torch.Tensor,
                     labels: torch.Tensor,
                     cfg=None) -> torch.Tensor:
        """Matches V2 compute_loss signature exactly."""
        return self.loss_fn(
            spike_out, labels,
            self._tgt_true,
            self._tgt_false,
        )

    def predict(self, spike_out: torch.Tensor) -> torch.Tensor:
        """(B, 20, T) → (B,) predicted class."""
        return spike_out.sum(dim=-1).argmax(dim=1)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }