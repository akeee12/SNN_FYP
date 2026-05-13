# models/adaptiveAxonalDelayV2SNNTorch.py
"""
snnTorch reimplementation of Adaptive Axonal Delay SNN.
Paper : Sun et al. (Pengfei), ICASSP 2023
 
Fixes vs previous run (12-05_run):
  FIX 3 — Dropout(p=0.2) inserted after lif1 and lif2 to regularise the
           severe overfitting (train 100%, test regressing 69% → 58%).
  FIX 4 — weight_norm re-added to fc1/fc2/fc3, matching Pengfei exactly.
           Previously removed due to instability at lr=0.1; stable at
           lr=0.001 with NAdam + weight_decay=1e-4.
"""
 

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import functional as SF


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


class AxonalDelay(nn.Module):
    def __init__(self, n_neurons: int, max_delay: int = 64):
        super().__init__()
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
            torch.zeros(n_neurons).uniform_(0, max_delay / 8)
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        d       = self.delay.clamp(0, self.max_delay)
        d_floor = d.floor().long()
        d_ceil  = d.ceil().long()
        alpha   = (d - d.floor()).view(1, -1, 1)
        B, N, T = spikes.shape
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


class AdaptiveAxonalDelayNetV2SNNTorch(nn.Module):
    """
    Adaptive Axonal Delay SNN — Sun et al. ICASSP 2023.
    Uses snnTorch Alpha neuron to match SLAYER's alpha kernel PSP.
    """

    def __init__(self, cfg):
        super().__init__()

        # Alpha kernel: exp(-t/tau_r) - exp(-t/tau_s)
        # matches SLAYER getEpsilon() exactly
        alpha = float(torch.exp(torch.tensor(-1.0 / cfg.tau_s)))
        beta  = float(torch.exp(torch.tensor(-1.0 / cfg.tau_r)))

        # FC layers with weight_norm — matches Pengfei exactly
        self.fc1 = nn.utils.weight_norm(
            nn.Linear(cfg.n_inputs,  cfg.n_hidden,  bias=False), name='weight')
        self.fc2 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_hidden,  bias=False), name='weight')
        self.fc3 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_outputs, bias=False), name='weight')

        # Scale weight_g to match SLAYER dense weight_scale=10
        with torch.no_grad():
            self.fc1.weight_g.fill_(cfg.weight_scale)
            self.fc2.weight_g.fill_(cfg.weight_scale)
            self.fc3.weight_g.fill_(cfg.weight_scale)

        # snnTorch Alpha neurons — matches SLAYER alpha kernel PSP
        surrogate = snn.surrogate.fast_sigmoid(slope=cfg.surrogate_slope)

        self.lif1 = snn.Alpha(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )
        self.lif2 = snn.Alpha(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )
        self.lif3 = snn.Alpha(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold_out,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )

        # Optional dropout
        dropout_p = getattr(cfg, "dropout", 0.0)
        self.drop1 = nn.Dropout(p=dropout_p)
        self.drop2 = nn.Dropout(p=dropout_p)

        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

        self._tgt_true  = cfg.tgt_count_true
        self._tgt_false = cfg.tgt_count_false
        self._T         = cfg.time_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Alpha neuron has three hidden states
        syn_exc1, syn_inh1, mem1 = self.lif1.init_alpha()
        syn_exc2, syn_inh2, mem2 = self.lif2.init_alpha()
        syn_exc3, syn_inh3, mem3 = self.lif3.init_alpha()

        spk1_buf = []
        for t in range(T):
            cur1 = self.fc1(x[:, t, :])
            spk1, syn_exc1, syn_inh1, mem1 = self.lif1(
                cur1, syn_exc1, syn_inh1, mem1)
            spk1_buf.append(spk1)

        spk1_t = torch.stack(spk1_buf, dim=2)
        spk1_t = self.drop1(spk1_t)
        spk1_t = self.delay1(spk1_t)

        spk2_buf = []
        for t in range(T):
            cur2 = self.fc2(spk1_t[:, :, t])
            spk2, syn_exc2, syn_inh2, mem2 = self.lif2(
                cur2, syn_exc2, syn_inh2, mem2)
            spk2_buf.append(spk2)

        spk2_t = torch.stack(spk2_buf, dim=2)
        spk2_t = self.drop2(spk2_t)
        spk2_t = self.delay2(spk2_t)

        spk3_buf = []
        for t in range(T):
            cur3 = self.fc3(spk2_t[:, :, t])
            spk3, syn_exc3, syn_inh3, mem3 = self.lif3(
                cur3, syn_exc3, syn_inh3, mem3)
            spk3_buf.append(spk3)

        return torch.stack(spk3_buf, dim=0)  # (T, B, 20)

    def compute_loss(self, spike_out, labels, cfg=None):
        loss_fn = SF.mse_count_loss(
            correct_rate   = self._tgt_true  / self._T,
            incorrect_rate = self._tgt_false / self._T,
        )
        return loss_fn(spike_out, labels)

    def predict(self, spike_out):
        return spike_out.sum(dim=0).argmax(dim=-1)

    def clamp_delays(self, thea1, thea2):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self):
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }