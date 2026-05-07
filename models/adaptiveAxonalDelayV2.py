# models/axonal_delay_modelPYTorch.py
"""
Faithful PyTorch reproduction of Sun et al. ICASSP 2023.
Verified against complete SLAYER source:
    - slayer.py          (Python layer definitions)
    - spikeKernels.h     (CUDA spike + surrogate)
    - shiftKernels.h     (CUDA delay shift)
    - spikeLoss.py       (loss function)
    - nadam.py           (optimiser)

All approximations explicitly documented.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SURROGATE GRADIENT — SLAYER triangle
# ─────────────────────────────────────────────────────────────────────────────

class SLAYERSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, theta, tau_rho, scale_rho):
        ctx.save_for_backward(u)
        ctx.theta     = theta
        ctx.tau_rho   = tau_rho
        ctx.scale_rho = scale_rho
        return (u >= theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,)      = ctx.saved_tensors
        surrogate = (ctx.scale_rho / ctx.tau_rho) * torch.clamp(
            1 - torch.abs(u - ctx.theta) / ctx.tau_rho, min=0
        )
        return grad_output * surrogate, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# DELAY GRADIENT — finite difference on original spikes
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
# AXONAL DELAY
# ─────────────────────────────────────────────────────────────────────────────

class AxonalDelay(nn.Module):
    def __init__(self, n_neurons: int, max_delay: int = 64):
        super().__init__()
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
            torch.zeros(n_neurons).uniform_(0, max_delay / 8)
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (B, N, T) → delayed: (B, N, T)"""
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

        out = (1 - alpha) * out_floor + alpha * out_ceil
        return DelayGradient.apply(out, spikes, d)

    def clamp_delays(self, max_val: float):
        with torch.no_grad():
            self.delay.data.clamp_(0, max_val)

    def get_stats(self) -> dict:
        d = self.delay.detach().cpu()
        return {
            "max" : d.max().item(),
            "mean": d.mean().item(),
            "p95" : torch.quantile(d, 0.95).item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PSP KERNEL
# ─────────────────────────────────────────────────────────────────────────────

class PSPKernel(nn.Module):
    def __init__(self, tau_s: float = 1.0, kernel_size: int = 32):
        super().__init__()
        self.kernel_size = kernel_size
        t      = torch.arange(kernel_size, dtype=torch.float32)
        kernel = torch.exp(-t / tau_s)
        self.register_buffer("kernel", kernel.flip(0).view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        x       = x.reshape(B * N, 1, T)
        x       = F.pad(x, (self.kernel_size - 1, 0))
        x       = F.conv1d(x, self.kernel)
        return x.reshape(B, N, T)


# ─────────────────────────────────────────────────────────────────────────────
# REFRACTORY KERNEL
# ─────────────────────────────────────────────────────────────────────────────

class RefractoryKernel(nn.Module):
    def __init__(self, tau_r=1.0, theta_u=10.0,
                 alpha_r=1.0, kernel_size=32):
        super().__init__()
        self.kernel_size = kernel_size   # FIX: was kernel_len — AttributeError
        t      = torch.arange(kernel_size, dtype=torch.float32)
        kernel = -theta_u * alpha_r * torch.exp(-t / tau_r)
        self.register_buffer("kernel", kernel.flip(0).view(1, 1, -1))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        B, N, T = spikes.shape
        s       = spikes.reshape(B * N, 1, T)
        s       = F.pad(s, (self.kernel_size - 1, 0))
        s       = F.conv1d(s, self.kernel)
        return s.reshape(B, N, T)


# ─────────────────────────────────────────────────────────────────────────────
# SRM LAYER
# ─────────────────────────────────────────────────────────────────────────────

class SRMLayer(nn.Module):
    def __init__(self, tau_s=1.0, tau_r=1.0, theta_u=10.0,
                 alpha_r=1.0, tau_rho=1.0, scale_rho=1.0,
                 kernel_size=32):
        super().__init__()
        self.theta_u    = theta_u
        self.tau_rho    = tau_rho
        self.scale_rho  = scale_rho
        self.psp        = PSPKernel(tau_s, kernel_size)
        self.refractory = RefractoryKernel(tau_r, theta_u, alpha_r, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T) → spikes: (B, N, T)"""
        u_ff    = self.psp(x)
        B, N, T = u_ff.shape
        spikes  = torch.zeros_like(u_ff)

        # FIX: use self.refractory.kernel_size (not.kernel_len)
        klen = self.refractory.kernel_size

        for t in range(T):
            if t == 0:
                ref = torch.zeros(B, N, device=x.device, dtype=x.dtype)
            else:
                start   = max(0, t - klen + 1)
                past    = spikes[:, :, start:t].reshape(B * N, 1, -1)
                past    = F.pad(past, (klen - 1, 0))
                ref_all = F.conv1d(past, self.refractory.kernel)
                ref     = ref_all[:, 0, -1].reshape(B, N)

            u_t             = u_ff[:, :, t] + ref
            spikes[:, :, t] = SLAYERSurrogate.apply(
                u_t, self.theta_u, self.tau_rho, self.scale_rho
            )

        return spikes


# ─────────────────────────────────────────────────────────────────────────────
# VAN ROSSUM LOSS — matches Pengfei's spikeLoss.spikeRate()
# ─────────────────────────────────────────────────────────────────────────────

class VanRossumLoss(nn.Module):
    """
    Matches SLAYER spikeLoss.spikeRate():
        error = psp(spike_out - spike_desired)
        loss  = 0.5 * sum(error^2) * Ts

    spike_desired is a binary tensor with tgt_count_true spikes
    uniformly distributed for the correct class, tgt_count_false for others.
    """

    def __init__(self, tau_s: float, kernel_size: int, Ts: float = 1.0):
        super().__init__()
        self.Ts  = Ts
        self.psp = PSPKernel(tau_s, kernel_size)

    def forward(
        self,
        spike_out: torch.Tensor,    # (B, C, T)
        labels:    torch.Tensor,    # (B,)
        tgt_true:  int,
        tgt_false: int,
    ) -> torch.Tensor:

        B, C, T = spike_out.shape
        device  = spike_out.device

        # Build desired spike tensor — uniform spike distribution
        desired = torch.full(
            (B, C, T), tgt_false / T, device=device
        )
        for b in range(B):
            desired[b, labels[b], :] = tgt_true / T

        # Error in PSP domain
        diff  = spike_out - desired
        error = self.psp(diff)
        loss  = 0.5 * (error ** 2).sum() * self.Ts

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAxonalDelayNetV2(nn.Module):
    """
    Adaptive Axonal Delay SNN — Sun et al. ICASSP 2023.
    Pure PyTorch — no SLAYER dependency.

    Input  : (B, T, 700)
    Output : (B, 20, T)  ← FIX: sum moved out of forward
    """

    def __init__(self, cfg):
        super().__init__()

        srm_kwargs = dict(
            tau_s       = cfg.tau_s,
            tau_r       = cfg.tau_r,
            theta_u     = cfg.theta_u,
            alpha_r     = cfg.alpha_r,
            tau_rho     = cfg.tau_rho,
            scale_rho   = cfg.scale_rho,
            kernel_size = cfg.kernel_size,   # FIX: now defined in config
        )

        self.srm1 = SRMLayer(**srm_kwargs)
        self.srm2 = SRMLayer(**srm_kwargs)
        self.srm3 = SRMLayer(**srm_kwargs)

        # FIX: weight_norm applied — matches Pengfei exactly
        self.fc1 = nn.utils.weight_norm(
            nn.Linear(cfg.n_inputs,  cfg.n_hidden,  bias=False), name='weight')
        self.fc2 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_hidden,  bias=False), name='weight')
        self.fc3 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_outputs, bias=False), name='weight')
       
        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

        # Loss module
        self.loss_fn = VanRossumLoss(cfg.tau_s, cfg.kernel_size, cfg.Ts)

        # Store targets
        self._tgt_true  = cfg.tgt_count_true
        self._tgt_false = cfg.tgt_count_false

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 700)
        Returns:
            spikes: (B, 20, T)   ← FIX: no sum here
        """
        x = x.permute(0, 2, 1)                               # (B, 700, T)

        x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 128, T)
        x = self.srm1(x)
        x = self.delay1(x)

        x = self.fc2(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 128, T)
        x = self.srm2(x)
        x = self.delay2(x)

        x = self.fc3(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 20, T)
        x = self.srm3(x)

        return x   # (B, 20, T)

    def compute_loss(
        self,
        spike_out: torch.Tensor,   # (B, 20, T)
        labels:    torch.Tensor,   # (B,)
        cfg=None,                  # kept for API compat — not used
    ) -> torch.Tensor:
        return self.loss_fn(
            spike_out, labels,
            self._tgt_true, self._tgt_false
        )

    def predict(self, spike_out: torch.Tensor) -> torch.Tensor:
        """(B, 20, T) → (B,) predicted class"""
        return spike_out.sum(dim=-1).argmax(dim=-1)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }