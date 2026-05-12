# models/log_buffer_aad_snntorch.py
"""
LogBuffer + SNNTorch AAD model.

Replaces FC1 with:
    FC_proj  [700 → k]     spatial compression
    LogBuffer [k → k*L]    temporal history (0 params)
    FC_expand [k*L → 128]  project to hidden dim

Everything from lif1 onwards is identical to SNNTorch AAD.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import functional as SF

from adaptiveAxonalDelayV2SNNTorch import AxonalDelay, DelayGradient


# ─────────────────────────────────────────────────────────────────────────────
# LOG BUFFER — identical to working version, zero parameters
# ─────────────────────────────────────────────────────────────────────────────

class LogBuffer(nn.Module):
    """
    Temporal log-spaced history buffer.
    Takes (B, k, T) → (B, k*L, T)
    lags = {1, 2, 4, 8, ..., max_lag}
    Zero learnable parameters.
    """

    def __init__(self, n_neurons: int, max_lag: int = 64):
        super().__init__()
        self.n_neurons = n_neurons
        L              = int(math.floor(math.log2(max_lag))) + 1
        self.lags      = [2**i for i in range(L)]
        self.L         = len(self.lags)

    @property
    def output_dim(self) -> int:
        return self.n_neurons * self.L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, k, T) → (B, k*L, T)"""
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

class LogBufferAADSNNTorch(nn.Module):
    """
    Log Buffer + SNNTorch AAD.

    Architecture:
        Input (B, T, 700)
            ↓
        FC_proj  [700 → k]          spatial compression
        LogBuffer [k → k*L]         temporal history
        FC_expand [k*L → 128]       project to hidden dim
            ↓
        lif1 [128] — snn.Synaptic   unchanged from SNNTorch AAD
        delay1 [128]
            ↓
        FC2 [128 → 128]
        lif2 [128] — snn.Synaptic
        delay2 [128]
            ↓
        FC3 [128 → 20]
        lif3 [20]  — snn.Synaptic
            ↓
        Output: (T, B, 20)          snnTorch convention

    Key difference from log_buffer_aad_modelv1:
        Neuron dynamics: snn.Synaptic instead of custom SRMLayer
        Output format:  (T, B, C) instead of (B, C, T)
        Loss:           SF.mse_count_loss instead of VanRossumLoss
    """

    def __init__(self, cfg):
        super().__init__()

        k       = cfg.buffer_size
        max_lag = cfg.buffer_max_lag

        # ── Synaptic time constants ───────────────────────────────────
        alpha     = float(torch.exp(torch.tensor(-1.0 / cfg.tau_s)))
        beta      = float(torch.exp(torch.tensor(-1.0 / cfg.tau_r)))
        surrogate = snn.surrogate.fast_sigmoid(slope=cfg.surrogate_slope)

        # ── Log-buffer block (replaces FC1) ───────────────────────────
        self.fc_proj   = nn.Linear(cfg.n_inputs, k, bias=False)
        self.log_buf   = LogBuffer(k, max_lag)
        self.fc_expand = nn.Linear(self.log_buf.output_dim,
                                   cfg.n_hidden, bias=False)

        # ── Rest of network — identical to SNNTorch AAD ───────────────
        self.fc2 = nn.Linear(cfg.n_hidden, cfg.n_hidden,  bias=False)
        self.fc3 = nn.Linear(cfg.n_hidden, cfg.n_outputs, bias=False)

        # ── Weight init — confirmed from sweep: 1.5x for proj/expand ──
        # fc2/fc3 use Xavier × weight_scale same as SNNTorch AAD
        nn.init.xavier_uniform_(self.fc_proj.weight)
        nn.init.xavier_uniform_(self.fc_expand.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        with torch.no_grad():
            # 1.5x for two-stage projection — confirmed from activation sweep
            self.fc_proj.weight.mul_(1.5)
            self.fc_expand.weight.mul_(1.5)
            # Standard weight_scale for hidden/output layers
            self.fc2.weight.mul_(cfg.weight_scale)
            self.fc3.weight.mul_(cfg.weight_scale)

        # ── snnTorch neurons ──────────────────────────────────────────
        self.lif1 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )
        self.lif2 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )
        self.lif3 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold_out,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )

        # ── Delays ────────────────────────────────────────────────────
        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

        # ── Loss config ───────────────────────────────────────────────
        self._tgt_true  = cfg.tgt_count_true
        self._tgt_false = cfg.tgt_count_false
        self._T         = cfg.time_steps

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

        print(f"\n  LogBufferAADSNNTorch parameter summary:")
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
        x   : (B, T, 700)
        out : (T, B, 20)   ← snnTorch convention
        """
        B, T, _ = x.shape

        # ── Log-buffer block ──────────────────────────────────────────
        # FC_proj: (B, T, 700) → (B, T, k) → permute → (B, k, T)
        z_proj = self.fc_proj(x).permute(0, 2, 1)      # (B, k, T)

        # LogBuffer: (B, k, T) → (B, k*L, T)
        z_buf  = self.log_buf(z_proj)                   # (B, k*L, T)

        # FC_expand: (B, T, k*L) → (B, T, 128) → permute → (B, 128, T)
        z_exp  = self.fc_expand(
            z_buf.permute(0, 2, 1)
        ).permute(0, 2, 1)                              # (B, 128, T)

        # ── Initialise hidden states ──────────────────────────────────
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        # ── Layer 1 — input is pre-computed z_exp ────────────────────
        spk1_buf = []
        for t in range(T):
            # z_exp[:, :, t] is already the log-buffer projected input
            spk1, syn1, mem1 = self.lif1(z_exp[:, :, t], syn1, mem1)
            spk1_buf.append(spk1)

        spk1_t = torch.stack(spk1_buf, dim=2)           # (B, 128, T)
        spk1_t = self.delay1(spk1_t)                    # (B, 128, T)

        # ── Layer 2 ───────────────────────────────────────────────────
        spk2_buf = []
        for t in range(T):
            cur2          = self.fc2(spk1_t[:, :, t])
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk2_buf.append(spk2)

        spk2_t = torch.stack(spk2_buf, dim=2)           # (B, 128, T)
        spk2_t = self.delay2(spk2_t)                    # (B, 128, T)

        # ── Output layer ──────────────────────────────────────────────
        spk3_buf = []
        for t in range(T):
            cur3          = self.fc3(spk2_t[:, :, t])
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            spk3_buf.append(spk3)

        # (T, B, 20) — snnTorch convention
        return torch.stack(spk3_buf, dim=0)

    def compute_loss(self, spike_out: torch.Tensor,
                     labels: torch.Tensor, cfg=None):
        """spike_out: (T, B, C), labels: (B,)"""
        loss_fn = SF.mse_count_loss(
            correct_rate   = self._tgt_true  / self._T,
            incorrect_rate = self._tgt_false / self._T,
        )
        return loss_fn(spike_out, labels)

    def predict(self, spike_out: torch.Tensor) -> torch.Tensor:
        """(T, B, C) → (B,)"""
        return spike_out.sum(dim=0).argmax(dim=-1)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }