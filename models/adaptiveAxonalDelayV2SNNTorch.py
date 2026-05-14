# models/adaptiveAxonalDelayV2SNNTorch.py
"""
snnTorch reimplementation of Adaptive Axonal Delay SNN.
Paper : Sun et al. (Pengfei), ICASSP 2023

Confirmed working configuration (12-05 run: 69.7% test @ epoch 20):
  - snn.Synaptic neurons with tau_s=2, tau_r=10, threshold=0.5
  - weight_norm on all FC layers, weight_v scaled by 0.5 (NOT weight_g)
  - Dropout exposed as config knob, default 0.0

DO NOT switch to snn.Alpha without re-tuning tau values and threshold.
The 14-05 attempt with Alpha + weight_g=10 + tau_r=1 collapsed completely.

Active fixes vs the broken 12-05 run:
  FIX 1/2 — DelayScheduler and its call site fixed in train_aadv2.py
  FIX 3   — Dropout knob (dropout=0.0 by default, matches Pengfei)
  FIX 4   — weight_norm on all FC layers, init via weight_v not weight_g
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import functional as SF


# ─────────────────────────────────────────────────────────────────────────────
# DELAY GRADIENT — unchanged from Sun et al.
# ─────────────────────────────────────────────────────────────────────────────

class DelayGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, spikes_orig, delay):
        ctx.save_for_backward(spikes_orig, delay)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        spikes_orig, delay = ctx.saved_tensors
        s_pad     = F.pad(spikes_orig, (1, 1))
        ds_dt     = (s_pad[:, :, 2:] - s_pad[:, :, :-2]) / 2.0
        grad_delay = (grad_output * ds_dt).sum(dim=(0, 2))
        return grad_output.clone(), None, grad_delay


# ─────────────────────────────────────────────────────────────────────────────
# AXONAL DELAY — unchanged from Sun et al.
# ─────────────────────────────────────────────────────────────────────────────

class AxonalDelay(nn.Module):
    def __init__(self, n_neurons: int, max_delay: int = 64):
        super().__init__()
        self.max_delay = max_delay
        self.delay = nn.Parameter(
            torch.zeros(n_neurons).uniform_(0, max_delay / 8)
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (B, N, T) → (B, N, T)"""
        d       = self.delay.clamp(0, self.max_delay)
        d_floor = d.floor().long()
        d_ceil  = d.ceil().long()
        alpha   = (d - d.floor()).view(1, -1, 1)

        B, N, T   = spikes.shape
        out_floor = torch.zeros_like(spikes)
        out_ceil  = torch.zeros_like(spikes)

        for i in range(N):
            sf = d_floor[i].item()
            sc = d_ceil[i].item()
            if sf > 0:
                out_floor[:, i, sf:] = spikes[:, i, :T - sf]
            else:
                out_floor[:, i, :]   = spikes[:, i, :]
            if sc > 0:
                out_ceil[:, i, sc:]  = spikes[:, i, :T - sc]
            else:
                out_ceil[:, i, :]    = spikes[:, i, :]

        out = (1.0 - alpha) * out_floor + alpha * out_ceil
        return DelayGradient.apply(out, spikes, d)

    def clamp_delays(self, max_val: float):
        with torch.no_grad():
            self.delay.data.clamp_(0.0, max_val)

    def get_stats(self) -> dict:
        d = self.delay.detach().cpu()
        return {
            "max" : d.max().item(),
            "mean": d.mean().item(),
            "p95" : torch.quantile(d, 0.95).item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAxonalDelayNetV2SNNTorch(nn.Module):
    """
    Adaptive Axonal Delay SNN — Sun et al. ICASSP 2023.

    Neuron dynamics : snn.Synaptic (double-exponential PSP + LIF)
    Delays          : AxonalDelay (learnable, unchanged from paper)
    Weight norm     : applied to all FC layers (matches Pengfei)
    Dropout         : optional, controlled by cfg.dropout (default 0.0)

    Input  : (B, T, 700)
    Output : (T, B, 20)
    """

    def __init__(self, cfg):
        super().__init__()

        alpha = float(torch.exp(torch.tensor(-1.0 / cfg.tau_s)))  # ≈ 0.607
        beta  = float(torch.exp(torch.tensor(-1.0 / cfg.tau_r)))  # ≈ 0.905

        # ── FC layers with weight_norm ────────────────────────────────
        # FIX 4: weight_norm matches Pengfei. Init weight_v with Xavier
        # then scale the DIRECTION, not weight_g. Setting weight_g=10
        # (as in the failed 14-05 run) produces weights 10x too large.
        self.fc1 = nn.utils.weight_norm(
            nn.Linear(cfg.n_inputs,  cfg.n_hidden,  bias=False), name="weight")
        self.fc2 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_hidden,  bias=False), name="weight")
        self.fc3 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_outputs, bias=False), name="weight")

        for fc in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(fc.weight_v)
            with torch.no_grad():
                fc.weight_v.mul_(cfg.weight_scale)

        # ── snn.Synaptic neurons ──────────────────────────────────────
        surrogate = snn.surrogate.fast_sigmoid(slope=cfg.surrogate_slope)

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

        # ── Dropout — off by default, matches Pengfei ─────────────────
        # FIX 3: exposed as config knob. Set cfg.dropout=0.1 if
        # overfitting returns (train >> test after epoch 30).
        dropout_p = getattr(cfg, "dropout", 0.0)
        self.drop1 = nn.Dropout(p=dropout_p)
        self.drop2 = nn.Dropout(p=dropout_p)

        # ── Axonal delays ─────────────────────────────────────────────
        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

        self._tgt_true  = cfg.tgt_count_true
        self._tgt_false = cfg.tgt_count_false
        self._T         = cfg.time_steps

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, T, n_inputs)
        out : (T, B, n_outputs)
        """
        B, T, _ = x.shape

        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        # Layer 1: input → hidden
        spk1_buf = []
        for t in range(T):
            cur1             = self.fc1(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            spk1_buf.append(spk1)

        spk1_t = torch.stack(spk1_buf, dim=2)   # (B, n_hidden, T)
        spk1_t = self.drop1(spk1_t)             # no-op when dropout=0
        spk1_t = self.delay1(spk1_t)

        # Layer 2: hidden → hidden
        spk2_buf = []
        for t in range(T):
            cur2             = self.fc2(spk1_t[:, :, t])
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk2_buf.append(spk2)

        spk2_t = torch.stack(spk2_buf, dim=2)   # (B, n_hidden, T)
        spk2_t = self.drop2(spk2_t)
        spk2_t = self.delay2(spk2_t)

        # Layer 3: hidden → output
        spk3_buf = []
        for t in range(T):
            cur3             = self.fc3(spk2_t[:, :, t])
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            spk3_buf.append(spk3)

        return torch.stack(spk3_buf, dim=0)   # (T, B, n_outputs)

    # ─────────────────────────────────────────────────────────────────
    def compute_loss(self, spike_out: torch.Tensor,
                     labels: torch.Tensor, cfg=None):
        loss_fn = SF.mse_count_loss(
            correct_rate   = self._tgt_true  / self._T,
            incorrect_rate = self._tgt_false / self._T,
        )
        return loss_fn(spike_out, labels)

    def predict(self, spike_out: torch.Tensor) -> torch.Tensor:
        return spike_out.sum(dim=0).argmax(dim=-1)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }