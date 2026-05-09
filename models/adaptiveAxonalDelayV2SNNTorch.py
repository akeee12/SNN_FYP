# models/adaptiveAxonalDelayV2.py
"""
snnTorch reimplementation of Adaptive Axonal Delay SNN.
Paper : Sun et al., ICASSP 2023

Neuron dynamics replaced with snnTorch Synaptic neurons.
AxonalDelay and DelayGradient kept identical to original.
Loss replaced with snnTorch mse_count_loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import functional as SF


# ─────────────────────────────────────────────────────────────────────────────
# DELAY GRADIENT — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class DelayGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, spikes_orig, delay):
        ctx.save_for_backward(spikes_orig, delay)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        spikes_orig, delay = ctx.saved_tensors
        s_pad      = F.pad(spikes_orig, (1, 1))
        ds_dt      = (s_pad[:, :, 2:] - s_pad[:, :, :-2]) / 2.0
        grad_delay = (grad_output * ds_dt).sum(dim=(0, 2))
        return grad_output.clone(), None, grad_delay


# ─────────────────────────────────────────────────────────────────────────────
# AXONAL DELAY — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────

class AxonalDelay(nn.Module):
    def __init__(self, n_neurons: int, max_delay: int = 64):
        super().__init__()
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
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
# FULL NETWORK — snnTorch Synaptic neurons
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAxonalDelayNetV2SNNTorch(nn.Module):
    """
    Adaptive Axonal Delay SNN — Sun et al. ICASSP 2023.
    Neuron dynamics: snnTorch Synaptic (alpha-beta LIF)
    Delay mechanism: custom AxonalDelay (unchanged)

    Input  : (B, T, 700)
    Output : (B, 20, T)
    """

    def __init__(self, cfg):
        super().__init__()

        # alpha = exp(-1/tau_s), beta = exp(-1/tau_r)
        # matches SLAYER SRM kernel decay rates
        alpha = float(torch.exp(torch.tensor(-1.0 / cfg.tau_s)))
        beta  = float(torch.exp(torch.tensor(-1.0 / cfg.tau_r)))

        # ── FC layers with weight norm — matches Pengfei exactly ──────
        self.fc1 = nn.utils.weight_norm(
            nn.Linear(cfg.n_inputs,  cfg.n_hidden,  bias=False), name='weight')
        self.fc2 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_hidden,  bias=False), name='weight')
        self.fc3 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_outputs, bias=False), name='weight')

        # Scale weight norm magnitude to match Pengfei's weight_scale
        with torch.no_grad():
            self.fc1.weight_g.fill_(cfg.weight_scale / cfg.n_inputs  ** 0.5)
            self.fc2.weight_g.fill_(cfg.weight_scale / cfg.n_hidden  ** 0.5)
            self.fc3.weight_g.fill_(cfg.weight_scale / cfg.n_hidden  ** 0.5)

        # ── snnTorch Synaptic neurons ─────────────────────────────────
        # Synaptic = double exponential PSP + LIF
        # alpha controls synaptic decay (tau_s)
        # beta  controls membrane decay  (tau_r)
        self.lif1 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.theta_u,
            learn_threshold=False,
            spike_grad=snn.surrogate.fast_sigmoid(slope=25),
        )
        self.lif2 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.theta_u,
            learn_threshold=False,
            spike_grad=snn.surrogate.fast_sigmoid(slope=25),
        )
        self.lif3 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.theta_u,
            learn_threshold=False,
            spike_grad=snn.surrogate.fast_sigmoid(slope=25),
        )

        # ── Learnable delays ──────────────────────────────────────────
        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

        # ── Loss targets ──────────────────────────────────────────────
        self._tgt_true  = cfg.tgt_count_true
        self._tgt_false = cfg.tgt_count_false
        self._T         = cfg.time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, T, 700)
        out : (B, 20, T)
        """
        B, T, _ = x.shape

        # Initialise hidden states
        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        spk1_list = []

        # ── Layer 1 ───────────────────────────────────────────────────
        for t in range(T):
            cur1       = self.fc1(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            spk1_list.append(spk1)

        spk1_t = torch.stack(spk1_list, dim=2)   # (B, 128, T)
        spk1_t = self.delay1(spk1_t)              # apply axonal delay

        # ── Layer 2 ───────────────────────────────────────────────────
        spk2_list = []
        for t in range(T):
            cur2       = self.fc2(spk1_t[:, :, t])
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk2_list.append(spk2)

        spk2_t = torch.stack(spk2_list, dim=2)   # (B, 128, T)
        spk2_t = self.delay2(spk2_t)

        # ── Output layer ──────────────────────────────────────────────
        spk3_list = []
        for t in range(T):
            cur3       = self.fc3(spk2_t[:, :, t])
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            spk3_list.append(spk3)

        return torch.stack(spk3_list, dim=2)      # (B, 20, T)

    def compute_loss(
        self,
        spike_out: torch.Tensor,   # (B, 20, T)
        labels:    torch.Tensor,   # (B,)
        cfg=None,
    ) -> torch.Tensor:
        # mse_count_loss matches Pengfei's spikeRate loss intent
        # correct_rate  = tgt_true  / T
        # incorrect_rate = tgt_false / T
        return SF.mse_count_loss(
            spike_out,
            labels,
            correct_rate   = self._tgt_true  / self._T,
            incorrect_rate = self._tgt_false / self._T,
        )

    def predict(self, spike_out: torch.Tensor) -> torch.Tensor:
        """(B, 20, T) → (B,)"""
        return spike_out.sum(dim=-1).argmax(dim=-1)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }