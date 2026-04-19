# models/axonal_delay_model.py
"""
Pure PyTorch reimplementation of Adaptive Axonal Delay SNN.
Paper : Sun et al., ICASSP 2023
Repo  : https://github.com/sunpengfei1122/Adaptive_axonal_delay

Architecture:
    Input (B, T, 700)
        → FC1 → SRM1 → Delay1
        → FC2 → SRM2 → Delay2
        → FC3 → SRM3
        → sum(T) → (B, 20)

All hyperparameters read from cfg (AdaptiveAxonalDelayPyTorchConfig).
No SLAYER dependency — compatible with PyTorch 2.1.2 + CUDA 12.1.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SURROGATE GRADIENT — SLAYER triangle function
# ─────────────────────────────────────────────────────────────────────────────

class SLAYERSurrogate(torch.autograd.Function):
    """
    Forward : Heaviside spike function — fires if u >= theta
    Backward: SLAYER triangle surrogate gradient
              d_spike/d_u ≈ (scale_rho/tau_rho) * max(1-|u-theta|/tau_rho, 0)
    """

    @staticmethod
    def forward(ctx, u, theta, tau_rho, scale_rho):
        ctx.save_for_backward(u)
        ctx.theta     = theta
        ctx.tau_rho   = tau_rho
        ctx.scale_rho = scale_rho
        return (u >= theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        u,        = ctx.saved_tensors
        surrogate = (ctx.scale_rho / ctx.tau_rho) * \
                    torch.clamp(
                        1 - torch.abs(u - ctx.theta) / ctx.tau_rho,
                        min=0)
        return grad_output * surrogate, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# DIFFERENTIABLE DELAY GRADIENT
# dL/dd_i ≈ sum_t (dL/d_out(t)) * (s_orig(t+1) - s_orig(t-1)) / 2
# ─────────────────────────────────────────────────────────────────────────────

class DelayGradient(torch.autograd.Function):
    """
    Forward : identity — interpolated shift already applied in AxonalDelay
    Backward: finite difference on original unshifted spikes
              ensures forward and backward are consistent
    """

    @staticmethod
    def forward(ctx, out, spikes_orig, delay):
        ctx.save_for_backward(spikes_orig, delay)
        return out                              # identity pass-through

    @staticmethod
    def backward(ctx, grad_output):
        spikes_orig, delay = ctx.saved_tensors

        # Gradient w.r.t output — identity
        grad_out   = grad_output.clone()

        # Gradient w.r.t delay — finite difference
        s_pad      = F.pad(spikes_orig, (1, 1))
        ds_dt      = (s_pad[:, :, 2:] - s_pad[:, :, :-2]) / 2.0
        grad_delay = (grad_output * ds_dt).sum(dim=(0, 2))  # (N,)

        # Grads for: out, spikes_orig, delay
        return grad_out, None, grad_delay


# ─────────────────────────────────────────────────────────────────────────────
# AXONAL DELAY — learnable per-neuron delay with linear interpolation
# ─────────────────────────────────────────────────────────────────────────────

class AxonalDelay(nn.Module):
    """
    Learnable per-neuron axonal delay.
    - Continuous delay parameter (not discrete)
    - Linear interpolation between floor/ceil shifts
    - Gradient via finite difference (paper equation)
    - Hard clamping for adaptive delay scheduling
    """

    def __init__(self, n_neurons: int, max_delay: int = 64):
        super().__init__()
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
            torch.zeros(n_neurons).uniform_(0, max_delay / 8)
        )

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (B, N, T)
        Returns:
            delayed: (B, N, T)
        """
        d       = self.delay.clamp(0, self.max_delay)
        d_floor = d.floor().long()                      # (N,)
        d_ceil  = d.ceil().long()                       # (N,)
        alpha   = (d - d.floor()).view(1, -1, 1)        # (1, N, 1)

        B, N, T   = spikes.shape
        out_floor = torch.zeros_like(spikes)
        out_ceil  = torch.zeros_like(spikes)

        for i in range(N):
            sf = d_floor[i].item()
            sc = d_ceil[i].item()

            if sf > 0:
                out_floor[:, i, sf:] = spikes[:, i, :T - sf]
            else:
                out_floor[:, i, :] = spikes[:, i, :]

            if sc > 0:
                out_ceil[:, i, sc:] = spikes[:, i, :T - sc]
            else:
                out_ceil[:, i, :] = spikes[:, i, :]

        # Interpolated output — quasi-continuous shift
        out = (1 - alpha) * out_floor + alpha * out_ceil

        # Apply differentiable gradient — passes original spikes for backward
        return DelayGradient.apply(out, spikes, d)

    def clamp_delays(self, max_val: float):
        """Hard cap for adaptive delay scheduling."""
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
# PSP KERNEL — exponential synaptic response, causal convolution
# ─────────────────────────────────────────────────────────────────────────────

class PSPKernel(nn.Module):
    """
    Exponential PSP kernel: epsilon(t) = exp(-t/tau_s)
    Applied as causal convolution over time axis.
    kernel_size capped at 32 — not full time window.
    """

    def __init__(self, tau_s: float = 1.0, kernel_size: int = 32):
        super().__init__()
        self.kernel_size = kernel_size
        t      = torch.arange(kernel_size, dtype=torch.float32)
        kernel = torch.exp(-t / tau_s)
        kernel = kernel / (kernel.sum() + 1e-8)
        self.register_buffer("kernel", kernel.flip(0).view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T = x.shape
        x       = x.reshape(B * N, 1, T)
        x       = F.pad(x, (self.kernel_size - 1, 0))
        x       = F.conv1d(x, self.kernel)
        return x.reshape(B, N, T)


# ─────────────────────────────────────────────────────────────────────────────
# REFRACTORY KERNEL — spike history feedback
# ─────────────────────────────────────────────────────────────────────────────

class RefractoryKernel(nn.Module):
    """
    Refractory kernel: eta(t) = -theta_u * alpha_r * exp(-t/tau_r)
    Subtracts from membrane potential after each spike.
    """

    def __init__(self, tau_r=1.0, theta_u=10.0,
                 alpha_r=2.0, kernel_size=32):
        super().__init__()
        self.kernel_size = kernel_size
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
# SRM LAYER — PSP + refractory + SLAYER surrogate
# ─────────────────────────────────────────────────────────────────────────────

class SRMLayer(nn.Module):
    """
    Full SRM neuron layer:
        u(t) = PSP(weighted_input)(t) + Refractory(spike_history)(t)
        spike(t) = Heaviside(u(t) - theta_u)  [SLAYER surrogate in backward]
    """

    def __init__(self, tau_s=1.0, tau_r=1.0, theta_u=10.0,
                 alpha_r=2.0, tau_rho=1.0, scale_rho=1.0,
                 kernel_size=32):
        super().__init__()
        self.theta_u   = theta_u
        self.tau_rho   = tau_rho
        self.scale_rho = scale_rho
        self.psp        = PSPKernel(tau_s, kernel_size)
        self.refractory = RefractoryKernel(
            tau_r, theta_u, alpha_r, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, T) — weighted input current
        Returns:
            spikes: (B, N, T)
        """
        u       = self.psp(x)
        B, N, T = u.shape
        spikes  = torch.zeros_like(u)

        for t in range(T):
            # Refractory contribution from spike history
            if t > 0:
                ref = self.refractory(
                    F.pad(spikes[:, :, :t], (0, T - t))
                )[:, :, t]
            else:
                ref = torch.zeros(B, N, device=u.device)

            u_t = u[:, :, t] + ref
            spikes[:, :, t] = SLAYERSurrogate.apply(
                u_t, self.theta_u, self.tau_rho, self.scale_rho
            )

        return spikes


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAxonalDelayNet(nn.Module):
    """
    Adaptive Axonal Delay SNN — Sun et al. ICASSP 2023.
    Pure PyTorch — no SLAYER dependency.

    Input  : (B, T, 700) — from shared SHDDataset
    Output : (B, 20)     — summed spike counts per class
    """

    def __init__(self, cfg):
        super().__init__()

        srm_kwargs = dict(
            tau_s      = cfg.tau_s,
            tau_r      = cfg.tau_r,
            theta_u    = cfg.theta_u,
            alpha_r    = cfg.alpha_r,
            tau_rho    = cfg.tau_rho,
            scale_rho  = cfg.scale_rho,
            kernel_size= cfg.kernel_size,
        )

        # ── SRM layers ────────────────────────────────────────────────────────
        self.srm1 = SRMLayer(**srm_kwargs)
        self.srm2 = SRMLayer(**srm_kwargs)
        self.srm3 = SRMLayer(**srm_kwargs)

        # ── FC layers with weight norm ────────────────────────────────────────
        self.fc1 = nn.utils.weight_norm(
            nn.Linear(cfg.n_inputs,  cfg.n_hidden,  bias=False))
        self.fc2 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_hidden,  bias=False))
        self.fc3 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden,  cfg.n_outputs, bias=False))

        # ── Learnable delays ──────────────────────────────────────────────────
        self.delay1 = AxonalDelay(
            cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(
            cfg.n_hidden, max_delay=cfg.delay_init_cap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 700) — loader output
        Returns:
            out: (B, 20)
        """
        x = x.permute(0, 2, 1)                              # (B, 700, T)

        # Layer 1
        x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 128, T)
        x = self.srm1(x)
        x = self.delay1(x)

        # Layer 2
        x = self.fc2(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 128, T)
        x = self.srm2(x)
        x = self.delay2(x)

        # Output layer
        x = self.fc3(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 20, T)
        x = self.srm3(x)

        return x.sum(dim=-1)                                 # (B, 20)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }