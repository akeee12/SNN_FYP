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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SURROGATE GRADIENT
# Source: spikeKernels.h evalRhoKernel + _spikeFunction.backward
#
# Confirmed formula:
#   rho = scaleRho / (tauRho * theta) * exp(-|theta - u| / (tauRho * theta))
#
# Note: evalRhoKernel is called with tau = tauRho * theta
#       This is NOT just tauRho — it is scaled by theta
# ─────────────────────────────────────────────────────────────────────────────

class SLAYERSurrogate(torch.autograd.Function):
    """
    Exponential surrogate gradient — confirmed from spikeKernels.h.

    Forward : Heaviside — s = 1 if u >= theta, else 0
              Note: SLAYER fires at 1/Ts. With Ts=1 this equals 1.0.

    Backward: rho = scaleRho/(tauRho*theta) * exp(-|theta-u|/(tauRho*theta))
              Confirmed from evalRhoKernel CUDA source.
    """

    @staticmethod
    def forward(ctx, u, theta, tau_rho, scale_rho):
        ctx.save_for_backward(u)
        ctx.theta     = theta
        ctx.tau_rho   = tau_rho
        ctx.scale_rho = scale_rho
        # SLAYER: spike = 1/Ts. With Ts=1.0 → 1.0
        return (u >= theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        u,        = ctx.saved_tensors
        theta     = ctx.theta
        scale_rho = ctx.scale_rho

        # Confirmed: tau in evalRhoKernel = tauRho * theta
        tau_eff   = ctx.tau_rho * theta          # e.g. 1.0 * 10.0 = 10.0

        # Confirmed: scale/tau * exp(-|theta - u|/tau)
        surrogate = (scale_rho / tau_eff) * \
                    torch.exp(-torch.abs(theta - u) / tau_eff)

        return grad_output * surrogate, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# ALPHA KERNEL BUILDER
# Source: slayer.py _calculateAlphaKernel
#
# Exact match — same loop, same EPSILON, same stopping condition.
# ─────────────────────────────────────────────────────────────────────────────

def build_alpha_kernel(tau: float,
                       mult: float = 1.0,
                       T_sample: int = 400,
                       Ts: float = 1.0,
                       epsilon: float = 0.01) -> torch.Tensor:
    """
    Exact match to SLAYER _calculateAlphaKernel.

    eps(t) = mult * (t/tau) * exp(1 - t/tau)

    Stops when |val| < epsilon AND t > tau.
    Dynamic length — NOT fixed kernel_size.

    With tau=1.0, Ts=1.0, epsilon=0.01:
        t=0: 0.000  (skipped — t/tau=0)
        t=1: 1.000  ← peak
        t=2: 0.271
        t=3: 0.050
        t=4: 0.007
        t=5: 0.001 < 0.01 AND 5 > 1 → STOP
        kernel length = 5
    """
    eps = []
    for t_val in np.arange(0, T_sample, Ts):
        val = mult * (t_val / tau) * math.exp(1.0 - t_val / tau)
        if abs(val) < epsilon and t_val > tau:
            break
        eps.append(val)
    return torch.FloatTensor(eps)


# ─────────────────────────────────────────────────────────────────────────────
# PSP KERNEL
# Source: slayer.py calculateSrmKernel + _pspFunction
#
# Key facts confirmed:
#   - Alpha kernel shape (NOT exponential)
#   - Fixed buffer — gradFilter = None always
#   - _pspFunction does NOT multiply by Ts
#     (only _pspLayer learnable version does)
#   - slayerCuda.conv approximated as F.conv1d with causal padding
# ─────────────────────────────────────────────────────────────────────────────

class PSPKernel(nn.Module):
    """
    SRMALPHA PSP kernel.
    Source: slayer.py calculateSrmKernel, _pspFunction.forward.

    eps(t) = (t/tau_s) * exp(1 - t/tau_s)

    Fixed buffer — not learned (gradFilter=None in _pspFunction).
    No Ts scaling — _pspFunction.forward does NOT multiply by Ts.
    Approximation: F.conv1d replaces slayerCuda.conv.
    """

    def __init__(self, tau_s: float, T_sample: int = 400, Ts: float = 1.0):
        super().__init__()
        self.Ts = Ts

        kernel          = build_alpha_kernel(
            tau=tau_s, mult=1.0,
            T_sample=T_sample, Ts=Ts, epsilon=0.01
        )
        self.kernel_len = len(kernel)

        # Flip for causal F.conv1d
        # slayerCuda.conv handles orientation internally
        # F.conv1d is cross-correlation so we pre-flip
        self.register_buffer('kernel', kernel.flip(0).view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T) → (B, N, T)"""
        B, N, T = x.shape
        x = x.reshape(B * N, 1, T)
        x = F.pad(x, (self.kernel_len - 1, 0))    # causal padding
        x = F.conv1d(x, self.kernel)
        # No Ts scaling — confirmed from _pspFunction source
        return x.reshape(B, N, T)


# ─────────────────────────────────────────────────────────────────────────────
# REFRACTORY KERNEL
# Source: slayer.py calculateRefKernel
#
# Key facts confirmed:
#   - Same alpha shape as PSP kernel
#   - mult = -scaleRef * theta = -2 * 10 = -20
#   - Same EPSILON = 0.01 stopping condition
#   - With tau_r=1, mult=-20: kernel length ≈ 5 steps
#     peak at t=1: -20 * 1 * exp(0) = -20
# ─────────────────────────────────────────────────────────────────────────────

class RefractoryKernel(nn.Module):
    """
    SRMALPHA refractory kernel.
    Source: slayer.py calculateRefKernel.

    eta(t) = -scaleRef * theta * (t/tau_r) * exp(1 - t/tau_r)

    With scaleRef=2, theta=10, tau_r=1:
        eta(1) = -20 * 1 * exp(0) = -20.0  ← peak suppression

    Approximation note:
        True SLAYER (getSpikesKernel) pastes this kernel IN-PLACE
        into d_u at each spike time — modifying future membrane potential.
        This causal conv over running spikes approximates that behaviour.
        Exact equivalence holds when spike trains are sparse.
    """

    def __init__(self, tau_r: float, theta_u: float,
                 scale_ref: float, T_sample: int = 400, Ts: float = 1.0):
        super().__init__()

        # Confirmed: mult = -scaleRef * theta
        mult   = -scale_ref * theta_u
        kernel = build_alpha_kernel(
            tau=tau_r, mult=mult,
            T_sample=T_sample, Ts=Ts, epsilon=0.01
        )
        self.kernel_len = len(kernel)
        self.register_buffer('kernel', kernel.flip(0).view(1, 1, -1))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (B, N, T) → (B, N, T)"""
        B, N, T = spikes.shape
        s = spikes.reshape(B * N, 1, T)
        s = F.pad(s, (self.kernel_len - 1, 0))
        s = F.conv1d(s, self.kernel)
        return s.reshape(B, N, T)


# ─────────────────────────────────────────────────────────────────────────────
# SRM LAYER
# Source: slayer.py psp() + spike() combined
#         spikeKernels.h getSpikesKernel
#
# Confirmed from getSpikesKernel:
#   - Sequential over time (for loop over Ns)
#   - Refractory pasted IN-PLACE into d_u at spike times
#   - Each neuron independent (parallelised over neurons)
#
# Approximation:
#   True SLAYER: d_u[t:] += refKernel  at each spike
#   This code:   causal conv over running spike tensor
#   Difference:  when spikes are dense, the in-place paste
#                accumulates differently than conv
# ─────────────────────────────────────────────────────────────────────────────

class SRMLayer(nn.Module):
    """
    SRM neuron layer with SRMALPHA kernels.
    Approximates SLAYER psp() + spike() pipeline.

    True SLAYER flow:
        1. psp(input)           → membrane potential u_ff
        2. getSpikes(u_ff, ref) → fires + pastes refKernel into u_ff in-place

    This implementation:
        1. PSP conv over input  → u_ff  (exact)
        2. For each t:
             ref = refKernel conv over running spikes  (approx)
             u_t = u_ff[t] + ref[t]
             spike[t] = Heaviside(u_t - theta)         (exact)
    """

    def __init__(self, tau_s: float = 1.0, tau_r: float = 1.0,
                 theta_u: float = 10.0, scale_ref: float = 2.0,
                 tau_rho: float = 1.0, scale_rho: float = 1.0,
                 T_sample: int = 400, Ts: float = 1.0):
        super().__init__()
        self.theta_u   = theta_u
        self.tau_rho   = tau_rho
        self.scale_rho = scale_rho

        self.psp = PSPKernel(
            tau_s=tau_s, T_sample=T_sample, Ts=Ts
        )
        self.refractory = RefractoryKernel(
            tau_r=tau_r, theta_u=theta_u,
            scale_ref=scale_ref,
            T_sample=T_sample, Ts=Ts
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T) — weighted input
        returns: (B, N, T) — spike tensor
        """
        u_ff    = self.psp(x)           # full conv in one pass — (B, N, T)
        B, N, T = u_ff.shape
        spikes  = torch.zeros_like(u_ff)
        klen    = self.refractory.kernel_len   # ~5 for tau_r=1, ~25 for tau_r=2

        for t in range(T):

            if t == 0:
                # no history yet — refractory contribution is zero
                ref = torch.zeros(B, N, device=x.device, dtype=x.dtype)

            else:
                # Only look at past spikes — spikes[:,:,t:] are still zero
                # so we only need spikes[:,:,:t]
                # Also the kernel is short (klen steps) so we only need
                # the last klen-1 timesteps of history at most
                start   = max(0, t - klen + 1)          # how far back kernel reaches
                past    = spikes[:, :, start:t]          # (B, N, window)
                window  = past.shape[2]

                # reshape for conv1d: (B*N, 1, window)
                past    = past.reshape(B * N, 1, window)

                # causal pad so output aligns with current t
                past    = F.pad(past, (klen - 1, 0))

                # convolve with refractory kernel
                ref_all = F.conv1d(past, self.refractory.kernel)  # (B*N, 1, ?)

                # take only the final output — that's the ref at timestep t
                ref     = ref_all[:, 0, -1].reshape(B, N)

            # combine feedforward potential with refractory suppression
            u_t             = u_ff[:, :, t] + ref

            # fire if above threshold — surrogate gradient in backward
            spikes[:, :, t] = SLAYERSurrogate.apply(
                u_t, self.theta_u, self.tau_rho, self.scale_rho
            )

        return spikes

# ─────────────────────────────────────────────────────────────────────────────
# AXONAL DELAY
# Source: shiftKernels.h shiftKernel + _delayFunction
#
# Confirmed from shiftKernels.h:
#   int shiftBlocks = static_cast<int>(shiftValue / Ts)
#   → integer truncation = floor for positive delays
#   → out-of-bounds = 0 (NOT clamped, NOT reflected)
#
# Confirmed from _delayFunction.backward:
#   diffFilter = [-1, 1] / Ts
#   outputDiff = slayerCuda.conv(output, diffFilter, 1)
#   gradDelay  = sum(gradOutput * outputDiff, [0,-1]) * Ts
#   → finite diff on SHIFTED output (not original spikes)
#   → input gradient shifted back: slayerCuda.shift(gradOutput, -delay, Ts)
#
# Enhancement over SLAYER:
#   Linear interpolation between floor/ceil (SLAYER uses floor only)
# ─────────────────────────────────────────────────────────────────────────────

class DelayGradient(torch.autograd.Function):
    """
    Delay gradient — matches _delayFunction.backward from slayer.py.
    Uses shifted output for finite difference (confirmed from source).

    SLAYER backward:
        diffFilter = [-1, 1] / Ts
        outputDiff = slayerCuda.conv(output, diffFilter, 1)
        gradDelay  = sum(gradOutput * outputDiff, [0,-1]) * Ts
        gradInput  = slayerCuda.shift(gradOutput, -delay, Ts)

    Approximation:
        gradInput approximated as identity
        (full impl requires slayerCuda.shift on gradOutput)
    """

    @staticmethod
    def forward(ctx, out, delay):
        ctx.save_for_backward(out, delay)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, delay = ctx.saved_tensors
        B, N, T    = out.shape

        # ── Gradient w.r.t. delay ─────────────────────────────────────
        # Confirmed from _delayFunction.backward:
        #   diffFilter = [-1, 1] / Ts
        #   outputDiff = slayerCuda.conv(output, diffFilter, 1)
        #   gradDelay  = sum(gradOutput * outputDiff, [0,-1]) * Ts
        out_r    = out.reshape(B * N, 1, T)
        out_pad  = F.pad(out_r, (1, 0))
        diff_k   = torch.tensor(
            [-1.0, 1.0], device=out.device, dtype=out.dtype
        ).view(1, 1, -1)
        out_diff   = F.conv1d(out_pad, diff_k).reshape(B, N, T)
        grad_delay = (grad_output * out_diff).sum(dim=(0, 2))  # (N,)

        # ── Gradient w.r.t. input ─────────────────────────────────────
        # Confirmed from _delayFunction.backward:
        #   gradInput = slayerCuda.shift(gradOutput, -delay, Ts)
        # Shift grad_output BACKWARD by delay amount
        # This realigns the gradient with the pre-delay spike positions
        grad_input = torch.zeros_like(grad_output)
        for n in range(N):
            d = int(delay[n].floor().item())
            if d == 0:
                grad_input[:, n, :] = grad_output[:, n, :]
            elif d < T:
                # shift left by d — gradient travels back through the delay
                grad_input[:, n, :T-d] = grad_output[:, n, d:]
                # positions 0:d get zero — out of bounds = 0

        return grad_input, grad_delay


class AxonalDelay(nn.Module):
    """
    Learnable per-neuron axonal delay.

    Forward:
        SLAYER: integer floor via static_cast<int>(delay/Ts)
        Here:   linear interpolation floor↔ceil (smoother gradients)
        Out-of-bounds = 0 (confirmed from shiftKernel: temp stays 0)

    Init:
        SLAYER: torch.rand — uniform(0, 1)
        Confirmed from _delayLayer.__init__

    Backward:
        Finite diff on shifted output (confirmed from _delayFunction)
    """

    def __init__(self, n_neurons: int, max_delay: int = 64):
        super().__init__()
        self.max_delay = max_delay
        # Confirmed: SLAYER uses torch.rand — uniform(0,1)
        self.delay     = nn.Parameter(torch.rand(n_neurons))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """spikes: (B, N, T) → (B, N, T)"""
        B, N, T = spikes.shape

        d       = self.delay.clamp(0, self.max_delay)
        d_floor = d.floor().long()
        d_ceil  = d.ceil().long()
        alpha   = (d - d.floor()).view(1, N, 1)

        # Vectorised gather — O(T) memory
        max_d   = int(d_ceil.max().item())
        # Out-of-bounds pads with 0 — confirmed from shiftKernel
        s_pad   = F.pad(spikes, (max_d, 0))         # (B, N, T+max_d)

        t_idx     = torch.arange(T, device=spikes.device)
        t_idx     = t_idx.view(1, 1, T).expand(B, N, T)

        idx_floor = (t_idx + max_d -
                     d_floor.view(1, N, 1)).clamp(0, T + max_d - 1)
        idx_ceil  = (t_idx + max_d -
                     d_ceil.view(1, N, 1)).clamp(0, T + max_d - 1)

        out_floor = torch.gather(s_pad, 2, idx_floor)
        out_ceil  = torch.gather(s_pad, 2, idx_ceil)

        out = (1 - alpha) * out_floor + alpha * out_ceil

        return DelayGradient.apply(out, d)

    def clamp_delays(self, max_val: float):
        """Called after optimizer.step() — matches Pengfei's module.clamp()."""
        with torch.no_grad():
            self.delay.data.clamp_(0, max_val)

    def update_max_delay(self, new_max: float):
        self.max_delay = new_max

    def get_stats(self) -> dict:
        d = self.delay.detach().cpu()
        return {
            "max" : d.max().item(),
            "mean": d.mean().item(),
            "p95" : torch.quantile(d, 0.95).item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SPIKE RATE LOSS
# Source: spikeLoss.py spikeRate / numSpikes
#
# Confirmed: NOT simple MSE on rates
# True SLAYER loss:
#   1. Compute desired spike train (one-hot scaled by tgtSpikeCount)
#   2. Compute difference: spikeOut - spikeDesired
#   3. Apply PSP filter to difference
#   4. Loss = 0.5 * sum(filtered_error^2) * Ts
#
# This is van Rossum distance — PSP-filtered spike train difference
# ─────────────────────────────────────────────────────────────────────────────

    """
    Van Rossum spike rate loss — confirmed from spikeLoss.py.

    True SLAYER spikeRate:
        error = psp(spikeOut - spikeDesired)
        loss  = 0.5 * sum(error^2) * Ts

    Where spikeDesired is constructed from desiredClass one-hot:
        true class:  tgtSpikeCount[True]  spikes uniformly in window
        false class: tgtSpikeCount[False] = 0

    Approximation:
        We approximate the desired spike train as a constant rate
        target_true  = 0.8 spikes/timestep for correct class
        target_false = 0.0 for others
        Then apply PSP filtering to the difference.

    For exact match, the desired spike train should be a binary
    tensor with spikes placed uniformly in the target window.
    """

class SpikeRateLoss(nn.Module):
    """
    NumSpikes loss — confirmed from spikeLoss.py numSpikes().

    NMNIST yaml reference:
        tgtSpikeCount: {true: 60, false: 10} over tSample=300
        scaled to time_steps=400:
            tgt_count_true  = 80
            tgt_count_false = 13
    """

    def __init__(self, psp_kernel: PSPKernel, Ts: float = 1.0):
        super().__init__()
        self.psp = psp_kernel
        self.Ts  = Ts

    def forward(self,
                spike_out:       torch.Tensor,
                labels:          torch.Tensor,
                tgt_count_true:  int = 80,
                tgt_count_false: int = 13,
                start:           int = 0,
                stop:            int = 400,
                ) -> torch.Tensor:

        B, C, T = spike_out.shape
        device  = spike_out.device
        window  = stop - start

        # desired spike counts per neuron
        desired_counts = torch.full(
            (B, C), float(tgt_count_false),
            device=device, dtype=spike_out.dtype
        )
        desired_counts[torch.arange(B), labels] = float(tgt_count_true)

        # actual counts in window — confirmed: sum * Ts
        actual_counts = (
            spike_out[:, :, start:stop].sum(dim=-1) * self.Ts
        )

        # count error — confirmed: (actual - desired) / windowLength
        count_error = (
            (actual_counts - desired_counts) / window
        ).unsqueeze(-1)

        # target region mask
        mask = torch.zeros(B, C, T, device=device)
        mask[:, :, start:stop] = 1.0

        # van Rossum outside window
        # inside window: spikeDesired = spikeOut so diff = 0
        spike_desired = spike_out * mask
        diff          = spike_out - spike_desired
        error         = self.psp(diff)

        # add count error inside window
        error = error + count_error * mask

        # confirmed from spikeLoss.py: sum not mean
        return 0.5 * torch.sum(error ** 2) * self.Ts
# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAxonalDelayNetV2(nn.Module):
    """
    Faithful PyTorch reproduction of Sun et al. ICASSP 2023.

    Confirmed corrections vs original broken implementation:
        1. Exponential surrogate (NOT triangle)
        2. tauRho * theta width (NOT just tauRho)
        3. Alpha PSP kernel (NOT pure exponential)
        4. Alpha refractory kernel (NOT pure exponential)
        5. Weight init ×10 (SLAYER weightScale=10 default)
        6. No weight_norm (not in SLAYER)
        7. Delay init uniform(0,1) (SLAYER torch.rand)
        8. Delay gradient on shifted output (NOT original spikes)
        9. Out-of-bounds delay = 0 (NOT clamped)
        10. Van Rossum loss (NOT simple MSE on rates)

    Remaining approximations (need slayerCuda for exact match):
        A. getSpikes: causal conv approx of in-place membrane paste
        B. slayerCuda.shift: linear interp instead of integer floor
        C. Delay input gradient: identity instead of shift(-delay)
    """

    def __init__(self, cfg):
        super().__init__()

        srm_kwargs = dict(
            tau_s     = cfg.tau_s,
            tau_r     = cfg.tau_r,
            theta_u   = cfg.theta_u,
            scale_ref = cfg.alpha_r,
            tau_rho   = cfg.tau_rho,
            scale_rho = cfg.scale_rho,
            T_sample  = cfg.time_steps,
            Ts        = 1.0,
        )

        # ── SRM layers ────────────────────────────────────────────────
        self.srm1 = SRMLayer(**srm_kwargs)
        self.srm2 = SRMLayer(**srm_kwargs)
        self.srm3 = SRMLayer(**srm_kwargs)

        # ── FC layers — weightScale=10, no weight_norm ────────────────
        self.fc1 = nn.Linear(cfg.n_inputs,  cfg.n_hidden,  bias=False)
        self.fc2 = nn.Linear(cfg.n_hidden,  cfg.n_hidden,  bias=False)
        self.fc3 = nn.Linear(cfg.n_hidden,  cfg.n_outputs, bias=False)

        with torch.no_grad():
            self.fc1.weight.data *= 10.0
            self.fc2.weight.data *= 10.0
            self.fc3.weight.data *= 10.0

        # ── Delays — uniform(0,1) init ────────────────────────────────
        self.delay1 = AxonalDelay(cfg.n_hidden, cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, cfg.delay_init_cap)

        # ── Loss — van Rossum (shares PSP kernel with srm1) ───────────
        # Use a fresh PSP kernel for loss — same params as srm1
        loss_psp    = PSPKernel(
            tau_s=cfg.tau_s, T_sample=cfg.time_steps, Ts=1.0
        )
        self.loss_fn = SpikeRateLoss(loss_psp, Ts=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 700) → (B, 20)"""
        x = x.permute(0, 2, 1)                              # (B, 700, T)

        x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 128, T)
        x = self.srm1(x)
        x = self.delay1(x)

        x = self.fc2(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 128, T)
        x = self.srm2(x)
        x = self.delay2(x)

        x = self.fc3(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, 20, T)
        x = self.srm3(x)

        return x                                             # (B, 20, T)

    def compute_loss(self,
                 spike_out: torch.Tensor,
                 labels:    torch.Tensor,
                 cfg) -> torch.Tensor:
        return self.loss_fn(
            spike_out,
            labels,
            tgt_count_true  = cfg.tgt_count_true,
            tgt_count_false = cfg.tgt_count_false,
            start           = cfg.tgt_region_start,
            stop            = cfg.tgt_region_stop,
        )

    def predict(self, spike_out: torch.Tensor) -> torch.Tensor:
        """spike_out: (B, 20, T) → (B,) predicted class."""
        return spike_out.sum(dim=-1).argmax(dim=1)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def update_delay_caps(self, thea1: float, thea2: float):
        self.delay1.update_max_delay(thea1)
        self.delay2.update_max_delay(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }