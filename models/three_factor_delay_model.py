# models/three_factor_delay_model.py
"""
Three-Factor Delay Learning Rules for SNNs.
Paper : Vassallo & Taherinejad, arXiv:2601.00668, Jan 2026

Key architectural features:
    1. LIF neurons with both synaptic AND axonal delays
    2. Gaussian surrogate gradient (not triangular like SLAYER)
    3. Synaptic delays: delay the input current before integration
    4. Axonal delays: delay the output spike before transmission
    5. Both delay types are learnable via backprop (offline mode)

Architecture:
    Input (B, T, 700)
        → SynapticDelay1 → LIF1 → AxonalDelay1
        → SynapticDelay2 → LIF2 → AxonalDelay2
        → FC_out
        → sum(T) → (B, 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GAUSSIAN SURROGATE GRADIENT
# Paper uses Gaussian not triangular surrogate for eligibility traces
# ─────────────────────────────────────────────────────────────────────────────

class GaussianSurrogate(torch.autograd.Function):
    """
    Forward : Heaviside spike function
    Backward: Gaussian surrogate gradient
              d_spike/d_u ≈ scale * exp(-(u-theta)^2 / (2*sigma^2))

    Paper motivation:
        Gaussian is smoother than triangular
        Better for eligibility trace computation
        More biologically plausible
    """

    @staticmethod
    def forward(ctx, u, threshold, sigma, scale):
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        ctx.sigma     = sigma
        ctx.scale     = scale
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        u,        = ctx.saved_tensors
        threshold = ctx.threshold
        sigma     = ctx.sigma
        scale     = ctx.scale
        # Gaussian centred at threshold
        surrogate = scale * torch.exp(
            -(u - threshold) ** 2 / (2 * sigma ** 2 + 1e-8)
        )
        return grad_output * surrogate, None, None, None


def gaussian_spike(u, threshold=1.0, sigma=0.5, scale=1.0):
    return GaussianSurrogate.apply(u, threshold, sigma, scale)


# ─────────────────────────────────────────────────────────────────────────────
# SYNAPTIC DELAY
# Delays the INPUT CURRENT before it reaches the neuron membrane
# Different from axonal delay which delays the OUTPUT SPIKE
# ─────────────────────────────────────────────────────────────────────────────

class SynapticDelay(nn.Module):
    def __init__(self, n_pre: int, n_post: int, max_delay: int = 20):
        super().__init__()
        self.n_pre     = n_pre
        self.n_post    = n_post
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
            torch.zeros(n_post, n_pre).uniform_(0, max_delay / 4)
        )

    # ── REPLACE THIS ENTIRE METHOD ────────────────────────────────────────
    # OLD (slow — Python loop over N_pre neurons):
    #
    #   def forward(self, x):
    #       B, T, N = x.shape
    #       d       = self.delay.mean(dim=0).clamp(0, self.max_delay)
    #       d_floor = d.floor().long()
    #       d_ceil  = d.ceil().long()
    #       alpha   = (d - d.floor()).view(1, 1, -1)
    #       out_floor = torch.zeros_like(x)
    #       out_ceil  = torch.zeros_like(x)
    #       for i in range(N):                    # <-- THE SLOW LOOP
    #           sf = d_floor[i].item()
    #           ...
    #       return (1 - alpha) * out_floor + alpha * out_ceil
    #
    # ── NEW (vectorised — identical strategy to AxonalDelay) ─────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorised synaptic delay — no Python loop over neurons.

        Note on per-synapse vs per-pre-neuron:
            self.delay shape is (n_post, n_pre).
            We reduce to per-pre-neuron by averaging over n_post
            (same as the original implementation).
            This keeps the output shape (B, T, n_pre) which is what
            the downstream FC layer expects.
            True per-synapse delay would need (B, T, n_post, n_pre)
            — 128x more memory for layer 1.

        Args:
            x: (B, T, n_pre) — input current after FC projection
        Returns:
            delayed: (B, T, n_pre)
        """
        B, T, N = x.shape

        # Reduce (n_post, n_pre) → (n_pre,) by averaging over post
        d       = self.delay.mean(dim=0).clamp(0, self.max_delay)  # (N,)
        d_floor = d.floor().long()                                   # (N,)
        d_ceil  = d.ceil().long()                                    # (N,)
        alpha   = (d - d.floor()).view(1, 1, N)                      # (1,1,N)

        # ── Pad left on time axis ─────────────────────────────────────────
        pad    = torch.zeros(B, self.max_delay, N,
                             device=x.device, dtype=x.dtype)
        padded = torch.cat([pad, x], dim=1)                # (B, max_d+T, N)

        # ── Build gather indices ──────────────────────────────────────────
        t_base  = torch.arange(T, device=x.device).view(1, T, 1)
        start_f = (self.max_delay - d_floor).clamp(0).view(1, 1, N)
        start_c = (self.max_delay - d_ceil ).clamp(0).view(1, 1, N)

        idx_f   = (t_base + start_f).expand(B, T, N)
        idx_c   = (t_base + start_c).expand(B, T, N)

        # ── Gather ────────────────────────────────────────────────────────
        out_floor = torch.gather(padded, dim=1, index=idx_f)
        out_ceil  = torch.gather(padded, dim=1, index=idx_c)

        return (1 - alpha) * out_floor + alpha * out_ceil

    def get_stats(self) -> dict:
        d = self.delay.detach().cpu()
        return {
            "max" : d.max().item(),
            "mean": d.mean().item(),
            "p95" : torch.quantile(d.flatten(), 0.95).item(),
        }

# ─────────────────────────────────────────────────────────────────────────────
# AXONAL DELAY
# Delays the OUTPUT SPIKE after it leaves the neuron
# Same concept as AAD but used alongside synaptic delays
# ─────────────────────────────────────────────────────────────────────────────

class AxonalDelay(nn.Module):
    def __init__(self, n_neurons: int, max_delay: int = 20):
        super().__init__()
        self.max_delay = max_delay
        self.delay     = nn.Parameter(
            torch.zeros(n_neurons).uniform_(0, max_delay / 4)
        )

    # ── REPLACE THIS ENTIRE METHOD ────────────────────────────────────────
    # OLD (slow — Python loop over N neurons):
    #
    #   def forward(self, spikes):
    #       B, T, N = spikes.shape
    #       d       = self.delay.clamp(0, self.max_delay)
    #       d_floor = d.floor().long()
    #       d_ceil  = d.ceil().long()
    #       alpha   = (d - d.floor()).view(1, 1, -1)
    #       out_floor = torch.zeros_like(spikes)
    #       out_ceil  = torch.zeros_like(spikes)
    #       for i in range(N):                    # <-- THE SLOW LOOP
    #           sf = d_floor[i].item()
    #           sc = d_ceil[i].item()
    #           if sf > 0 and sf < T:
    #               out_floor[:, sf:, i] = spikes[:, :T-sf, i]
    #           elif sf == 0:
    #               out_floor[:, :, i] = spikes[:, :, i]
    #           if sc > 0 and sc < T:
    #               out_ceil[:, sc:, i] = spikes[:, :T-sc, i]
    #           elif sc == 0:
    #               out_ceil[:, :, i] = spikes[:, :, i]
    #       return (1 - alpha) * out_floor + alpha * out_ceil
    #
    # ── NEW (vectorised — no Python loop) ────────────────────────────────

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Vectorised axonal delay — no Python loop over neurons.

        Strategy:
            1. Pad the TIME dimension on the LEFT with max_delay zeros
               so that any shift stays in-bounds.
            2. For each neuron i, the delayed signal is:
                   padded[:, (max_delay - d[i]) : (max_delay - d[i] + T), i]
               i.e. we just change WHERE we start slicing, not the data.
            3. Use torch.gather on the time axis to do all neurons at once.

        Args:
            spikes: (B, T, N)
        Returns:
            delayed: (B, T, N)
        """
        B, T, N = spikes.shape

        d       = self.delay.clamp(0, self.max_delay)   # (N,)
        d_floor = d.floor().long()                       # (N,)
        d_ceil  = d.ceil().long()                        # (N,)
        alpha   = (d - d.floor()).view(1, 1, N)          # (1, 1, N)

        # ── Pad left on time axis ─────────────────────────────────────────
        # (B, T, N) → (B, max_delay + T, N)
        pad     = torch.zeros(B, self.max_delay, N,
                              device=spikes.device, dtype=spikes.dtype)
        padded  = torch.cat([pad, spikes], dim=1)        # (B, max_d+T, N)

        # ── Build gather index for floor and ceil delays ──────────────────
        # For neuron i with floor delay sf:
        #   we want timesteps  [max_delay-sf, max_delay-sf+1, ..., max_delay-sf+T-1]
        # Shape of index: (B, T, N)

        # t_base: (1, T, 1) — base time indices 0..T-1
        t_base  = torch.arange(T, device=spikes.device).view(1, T, 1)

        # start_f[i] = max_delay - d_floor[i]  →  shape (1, 1, N)
        start_f = (self.max_delay - d_floor).clamp(0).view(1, 1, N)
        start_c = (self.max_delay - d_ceil ).clamp(0).view(1, 1, N)

        # idx shape: (1, T, N) → broadcast to (B, T, N) inside gather
        idx_f   = (t_base + start_f).expand(B, T, N)    # (B, T, N)
        idx_c   = (t_base + start_c).expand(B, T, N)    # (B, T, N)

        # ── Gather ────────────────────────────────────────────────────────
        out_floor = torch.gather(padded, dim=1, index=idx_f)  # (B, T, N)
        out_ceil  = torch.gather(padded, dim=1, index=idx_c)  # (B, T, N)

        return (1 - alpha) * out_floor + alpha * out_ceil

    def get_stats(self) -> dict:
        d = self.delay.detach().cpu()
        return {
            "max" : d.max().item(),
            "mean": d.mean().item(),
            "p95" : torch.quantile(d, 0.95).item(),
        }

# ─────────────────────────────────────────────────────────────────────────────
# LIF NEURON WITH SYNAPTIC CURRENT
# Standard LIF with exponential synaptic current
# ─────────────────────────────────────────────────────────────────────────────

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with exponential synaptic current.

    Dynamics:
        I_syn(t) = I_syn(t-1) * exp(-dt/tau_syn) + input(t)
        u(t)     = u(t-1) * exp(-dt/tau_mem) + I_syn(t)
        spike(t) = Heaviside(u(t) - threshold)
        u(t)     = u(t) * (1 - spike(t))  [reset]

    Uses Gaussian surrogate gradient (paper choice).
    """

    def __init__(self, tau_mem=20.0, tau_syn=5.0,
                 v_threshold=1.0, v_reset=0.0,
                 surrogate_sigma=0.5, surrogate_scale=1.0,
                 dt=1.0):
        super().__init__()
        self.tau_mem         = tau_mem
        self.tau_syn         = tau_syn
        self.v_threshold     = v_threshold
        self.v_reset         = v_reset
        self.surrogate_sigma = surrogate_sigma
        self.surrogate_scale = surrogate_scale

        # Decay factors
        self.decay_mem = np.exp(-dt / tau_mem)
        self.decay_syn = np.exp(-dt / tau_syn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N) — input current over time
        Returns:
            spikes: (B, T, N)
        """
        B, T, N = x.shape
        u       = torch.zeros(B, N, device=x.device)
        i_syn   = torch.zeros(B, N, device=x.device)
        spikes  = torch.zeros(B, T, N, device=x.device)

        for t in range(T):
            # Synaptic current update
            i_syn = i_syn * self.decay_syn + x[:, t, :]

            # Membrane potential update
            u = u * self.decay_mem + i_syn

            # Spike generation with Gaussian surrogate
            s = gaussian_spike(
                u, self.v_threshold,
                self.surrogate_sigma, self.surrogate_scale
            )
            spikes[:, t, :] = s

            # Reset membrane potential after spike
            u = u * (1 - s.detach()) + self.v_reset * s.detach()

        return spikes


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class ThreeFactorDelayNet(nn.Module):
    """
    SNN with synaptic + axonal delays and Gaussian surrogate.
    Based on Vassallo & Taherinejad, arXiv:2601.00668, 2026.

    Architecture:
        Input  (B, T, 700)
            → FC1 (weight matrix)
            → SynapticDelay1 (per-synapse delay on input current)
            → LIF1 (Gaussian surrogate)
            → AxonalDelay1 (per-neuron delay on output spikes)
            → FC2
            → SynapticDelay2
            → LIF2
            → AxonalDelay2
            → FC_out
            → sum(T) → (B, 20)

    Key difference from AAD:
        AAD: axonal delays only, SRM neurons
        This: BOTH synaptic + axonal delays, LIF neurons

    Key difference from DCLS:
        DCLS: dilated conv delays, no explicit synaptic delays
        This: explicit per-synapse + per-neuron delays, simpler arch
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── LIF neuron kwargs ─────────────────────────────────────────────
        lif_kwargs = dict(
            tau_mem         = cfg.tau_mem,
            tau_syn         = cfg.tau_syn,
            v_threshold     = cfg.v_threshold,
            v_reset         = cfg.v_reset,
            surrogate_sigma = cfg.surrogate_sigma,
            surrogate_scale = cfg.surrogate_scale,
        )

        # ── Layer 1 ───────────────────────────────────────────────────────
        self.fc1    = nn.Linear(cfg.n_inputs,  cfg.n_hidden, bias=False)
        self.sdel1  = SynapticDelay(
            cfg.n_inputs, cfg.n_hidden,
            max_delay=cfg.max_synaptic_delay)
        self.lif1   = LIFNeuron(**lif_kwargs)
        self.adel1  = AxonalDelay(
            cfg.n_hidden, max_delay=cfg.max_axonal_delay)

        # ── Layer 2 ───────────────────────────────────────────────────────
        self.fc2    = nn.Linear(cfg.n_hidden,  cfg.n_hidden, bias=False)
        self.sdel2  = SynapticDelay(
            cfg.n_hidden, cfg.n_hidden,
            max_delay=cfg.max_synaptic_delay)
        self.lif2   = LIFNeuron(**lif_kwargs)
        self.adel2  = AxonalDelay(
            cfg.n_hidden, max_delay=cfg.max_axonal_delay)

        # ── Output ────────────────────────────────────────────────────────
        self.fc_out = nn.Linear(cfg.n_hidden, cfg.n_outputs, bias=False)

        # ── Optional dropout ──────────────────────────────────────────────
        self.drop = nn.Dropout(cfg.dropout_p) \
                    if cfg.dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 700)
        Returns:
            out: (B, 20) summed spike counts
        """
        # ── Layer 1 ───────────────────────────────────────────────────────
        # FC: (B, T, 700) → (B, T, 128)
        x = self.fc1(x)

        # Synaptic delay on input current
        x = self.sdel1(x)                    # (B, T, 128)

        # LIF neurons
        x = self.lif1(x)                     # (B, T, 128) spikes

        # Axonal delay on output spikes
        x = self.adel1(x)                    # (B, T, 128)
        x = self.drop(x)

        # ── Layer 2 ───────────────────────────────────────────────────────
        x = self.fc2(x)                      # (B, T, 128)
        x = self.sdel2(x)                    # (B, T, 128)
        x = self.lif2(x)                     # (B, T, 128) spikes
        x = self.adel2(x)                    # (B, T, 128)
        x = self.drop(x)

        # ── Readout ───────────────────────────────────────────────────────
        x = self.fc_out(x)                   # (B, T, 20)
        return x.sum(dim=1)                  # (B, 20) sum over T

    def get_delay_stats(self) -> dict:
        return {
            "synaptic_delay1": self.sdel1.get_stats(),
            "axonal_delay1"  : self.adel1.get_stats(),
            "synaptic_delay2": self.sdel2.get_stats(),
            "axonal_delay2"  : self.adel2.get_stats(),
        }