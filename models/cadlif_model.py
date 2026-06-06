# models/cadlif_model.py
"""
Constrained Adaptive LIF (cAdLIF) SNN with Synaptic Delays.
Paper: Deckers et al. 2024
"Advancing Spatio-Temporal Processing in Spiking Neural Networks"
Frontiers in Neuroscience.

Neuron dynamics (Equation 1):
    u[t] = α·u[t-1] + (1-α)·(I[t] - w[t-1]) - θ·s[t-1]
    w[t] = β·w[t-1] + (1-β)·a·u[t-1] + b·s[t-1]
    s[t] = u[t] >= θ

Constraints (cAdLIF vs AdLIF):
    a > 0  — prevents chaotic positive feedback
    b > 0  — prevents chaotic positive feedback

Delay gradient (Equation 9):
    Approximated implicitly via linear interpolation between
    floor/ceil integer delays using torch.gather.

Architecture (Figure 2):
    Input  (B, T, 140)
        → SynapticDelayLayer1 + FC1 → cAdLIFLayer1 → Dropout
        → SynapticDelayLayer2 + FC2 → cAdLIFLayer2 → Dropout
        → FC_out (readout, infinite threshold)
        → sum softmax over T → log_softmax over classes → NLLLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# BOXCAR SURROGATE GRADIENT
# Equation 6: d_spike/d_u = 0.5 if |u - theta| <= 0.5 else 0
# ─────────────────────────────────────────────────────────────────────────────

class BoxcarSurrogate(torch.autograd.Function):
    """
    Forward : Heaviside spike  s = (u >= theta)
    Backward: Boxcar surrogate
              d_spike/d_u = 0.5 if |u - theta| <= width else 0
    Width fixed at 0.5 per paper Equation 6.
    """

    @staticmethod
    def forward(ctx, u, threshold, width):
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        ctx.width     = width
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        u,        = ctx.saved_tensors
        threshold = ctx.threshold
        width     = ctx.width
        surrogate = ((u - threshold).abs() <= width).float() * 0.5
        return grad_output * surrogate, None, None


def boxcar_spike(u, threshold=1.0, width=0.5):
    return BoxcarSurrogate.apply(u, threshold, width)


# ─────────────────────────────────────────────────────────────────────────────
# SYNAPTIC DELAY LAYER
# Per-synapse delays: shape (n_post, n_pre)
# Gradient via linear interpolation (floor/ceil gather)
# Internal left-pad only — dataset does NOT right-pad
# ─────────────────────────────────────────────────────────────────────────────

class SynapticDelayLayer(nn.Module):
    """
    Learnable per-synapse synaptic delays.

    Each synapse (i,j) has its own delay d_ij in [0, max_delay].
    Delays initialised to zero (Section 4.2).

    Delay gradient:
        Approximated via linear interpolation between floor and ceil
        delays using torch.gather. This implicitly computes a valid
        gradient w.r.t. the real-valued delay parameter d_ij,
        approximating Equation 9.

    Padding:
        Internal left-pad of max_delay zeros on the time axis only.
        Dataset does NOT right-pad.

    Args:
        n_pre     : number of presynaptic neurons
        n_post    : number of postsynaptic neurons
        max_delay : maximum delay in timesteps (paper: 25)
    """

    def __init__(self, n_pre: int, n_post: int, max_delay: int = 25):
        super().__init__()
        self.n_pre     = n_pre
        self.n_post    = n_post
        self.max_delay = max_delay

        # Per-synapse delay — initialised to 0 (Section 4.2)
        self.delay = nn.Parameter(torch.zeros(n_post, n_pre))

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (B, T, n_pre)
        Returns:
            delayed: (B, T, n_post, n_pre)
        """
        B, T, N_pre = spikes.shape

        d       = self.delay.clamp(0, self.max_delay)
        d_floor = d.floor().long()
        d_ceil  = d.ceil().long()
        alpha   = (d - d.floor())                        # (n_post, n_pre)

        # Left-pad time axis
        pad    = torch.zeros(B, self.max_delay, N_pre,
                             device=spikes.device, dtype=spikes.dtype)
        padded = torch.cat([pad, spikes], dim=1)         # (B, max_d+T, n_pre)

        # Expand for per-synapse gather
        padded_exp = padded.unsqueeze(2).expand(
            B, self.max_delay + T, self.n_post, N_pre
        )

        # Build gather indices (B, T, n_post, n_pre)
        t_base  = torch.arange(T, device=spikes.device).view(1, T, 1, 1)
        start_f = (self.max_delay - d_floor).clamp(0).unsqueeze(0).unsqueeze(0)
        start_c = (self.max_delay - d_ceil ).clamp(0).unsqueeze(0).unsqueeze(0)

        idx_f = (t_base + start_f).expand(B, T, self.n_post, N_pre)
        idx_c = (t_base + start_c).expand(B, T, self.n_post, N_pre)

        out_f = torch.gather(padded_exp, dim=1, index=idx_f)
        out_c = torch.gather(padded_exp, dim=1, index=idx_c)

        alpha_exp = alpha.unsqueeze(0).unsqueeze(0)      # (1,1,n_post,n_pre)

        return (1 - alpha_exp) * out_f + alpha_exp * out_c

    def get_stats(self) -> dict:
        d = self.delay.detach().cpu()
        return {
            "max" : d.max().item(),
            "mean": d.mean().item(),
            "p95" : torch.quantile(d.flatten(), 0.95).item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# cAdLIF NEURON LAYER
# ─────────────────────────────────────────────────────────────────────────────

class cAdLIFLayer(nn.Module):
    """
    Constrained Adaptive LIF neuron layer.

    Dynamics (Equation 1):
        u[t] = α·u[t-1] + (1-α)·(I[t] - w[t-1]) - θ·s[t-1]
        w[t] = β·w[t-1] + (1-β)·a·u[t-1] + b·s[t-1]
        s[t] = Heaviside(u[t] - θ)

    Trainable per-neuron parameters:
        α ∈ [0.36, 0.96]
        β ∈ [0.96, 0.99]
        a ∈ [0,    1   ]  (MUST stay > 0 — cAdLIF constraint)
        b ∈ [0,    2   ]  (MUST stay > 0 — cAdLIF constraint)

    Initial states (Section 4.2):
        u[0] ~ Uniform(0, 1)
        w[0] ~ Uniform(0, 1)

    BPTT note:
        s_prev is NOT detached — gradients must flow through the
        reset term -θ·s[t-1] and adaptation term b·s[t-1] for
        correct temporal credit assignment via BPTT.
        The boxcar surrogate handles the non-differentiability
        of the Heaviside in the backward pass.
    """

    def __init__(self, n_neurons: int, cfg):
        super().__init__()
        self.n_neurons   = n_neurons
        self.v_threshold = cfg.v_threshold
        self.width       = cfg.surrogate_width

        self.alpha = nn.Parameter(
            torch.empty(n_neurons).uniform_(cfg.alpha_min, cfg.alpha_max)
        )
        self.beta  = nn.Parameter(
            torch.empty(n_neurons).uniform_(cfg.beta_min, cfg.beta_max)
        )
        self.a     = nn.Parameter(
            torch.empty(n_neurons).uniform_(cfg.a_min, cfg.a_max)
        )
        self.b     = nn.Parameter(
            torch.empty(n_neurons).uniform_(cfg.b_min, cfg.b_max)
        )

        self.alpha_min = cfg.alpha_min
        self.alpha_max = cfg.alpha_max
        self.beta_min  = cfg.beta_min
        self.beta_max  = cfg.beta_max
        self.a_min     = cfg.a_min
        self.a_max     = cfg.a_max
        self.b_min     = cfg.b_min
        self.b_max     = cfg.b_max

    def clamp_parameters(self):
        """Hard clamp after every optimiser step."""
        with torch.no_grad():
            self.alpha.clamp_(self.alpha_min, self.alpha_max)
            self.beta.clamp_(self.beta_min,   self.beta_max)
            self.a.clamp_(self.a_min, self.a_max)
            self.b.clamp_(self.b_min, self.b_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_neurons)
        Returns:
            spikes: (B, T, n_neurons)
        """
        B, T, N = x.shape
        device  = x.device

        # ── Initial states — Uniform(0,1) per Section 4.2 ─────────────────
        u      = torch.empty(B, N, device=device).uniform_(0, 1)
        w      = torch.empty(B, N, device=device).uniform_(0, 1)
        s_prev = torch.zeros(B, N, device=device)
        spikes = torch.zeros(B, T, N, device=device)

        # Soft clamp during forward
        alpha = self.alpha.clamp(self.alpha_min, self.alpha_max)
        beta  = self.beta.clamp(self.beta_min,   self.beta_max)
        a     = self.a.clamp(self.a_min, self.a_max)
        b     = self.b.clamp(self.b_min, self.b_max)

        for t in range(T):
            # Equation 1 — membrane potential
            u_new = (alpha * u
                     + (1 - alpha) * (x[:, t, :] - w)
                     - self.v_threshold * s_prev)

            # Equation 1 — adaptation current
            w_new = (beta * w
                     + (1 - beta) * a * u
                     + b * s_prev)

            # Spike — boxcar surrogate in backward (Equation 6)
            s = boxcar_spike(u_new, self.v_threshold, self.width)

            spikes[:, t, :] = s

            u      = u_new
            w      = w_new

            # NO detach — BPTT must flow through s_prev
            s_prev = s

        return spikes

    def get_stats(self) -> dict:
        return {
            "alpha": {"mean": self.alpha.mean().item(),
                      "std" : self.alpha.std().item()},
            "beta" : {"mean": self.beta.mean().item(),
                      "std" : self.beta.std().item()},
            "a"    : {"mean": self.a.mean().item(),
                      "std" : self.a.std().item()},
            "b"    : {"mean": self.b.mean().item(),
                      "std" : self.b.std().item()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK — d-cAdLIF
# ─────────────────────────────────────────────────────────────────────────────

class cAdLIFNet(nn.Module):
    """
    d-cAdLIF: 2-hidden-layer feedforward SNN.
    Deckers et al. 2024, Figure 2 architecture.

    Readout (Equation 3 + 4):
        u_out[t]  = FC_out(s2[t])              (B, T, n_outputs)
        probs     = sum_t softmax(u_out[t])    (B, n_outputs) unnormalised
        logits    = log_softmax(probs)         renormalise over classes
        loss      = NLLLoss(logits, labels)    equivalent to Eq. 4
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── Layer 1 ───────────────────────────────────────────────────────
        self.delay1 = SynapticDelayLayer(
            cfg.n_inputs, cfg.n_hidden,
            max_delay=cfg.max_delay,
        )
        self.fc1    = nn.Linear(cfg.n_inputs, cfg.n_hidden, bias=True)
        self.lif1   = cAdLIFLayer(cfg.n_hidden, cfg)
        self.drop1  = nn.Dropout(cfg.dropout_p)

        # ── Layer 2 ───────────────────────────────────────────────────────
        self.delay2 = SynapticDelayLayer(
            cfg.n_hidden, cfg.n_hidden,
            max_delay=cfg.max_delay,
        )
        self.fc2    = nn.Linear(cfg.n_hidden, cfg.n_hidden, bias=True)
        self.lif2   = cAdLIFLayer(cfg.n_hidden, cfg)
        self.drop2  = nn.Dropout(cfg.dropout_p)

        # ── Readout ───────────────────────────────────────────────────────
        self.fc_out = nn.Linear(cfg.n_hidden, cfg.n_outputs, bias=True)

        # Xavier uniform weights, zero biases (Section 4.2)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_inputs) — no right-padding needed
        Returns:
            logits: (B, n_outputs) — log_softmax, ready for NLLLoss
        """
        # ── Layer 1 ───────────────────────────────────────────────────────
        delayed1 = self.delay1(x)                        # (B,T,n_hid,n_in)

        # Equation 7: I_i = sum_j W_ij * s_j[t-d_ij] + b_i
        w1 = self.fc1.weight                             # (n_hidden, n_inputs)
        I1 = (delayed1 * w1.unsqueeze(0).unsqueeze(0)
              ).sum(-1) \
             + self.fc1.bias.unsqueeze(0).unsqueeze(0)   # (B, T, n_hidden)

        s1 = self.drop1(self.lif1(I1))                  # (B, T, n_hidden)

        # ── Layer 2 ───────────────────────────────────────────────────────
        delayed2 = self.delay2(s1)                       # (B,T,n_hid,n_hid)

        w2 = self.fc2.weight                             # (n_hidden, n_hidden)
        I2 = (delayed2 * w2.unsqueeze(0).unsqueeze(0)
              ).sum(-1) \
             + self.fc2.bias.unsqueeze(0).unsqueeze(0)   # (B, T, n_hidden)

        s2 = self.drop2(self.lif2(I2))                  # (B, T, n_hidden)

        # ── Readout — Equations 3 + 4 ─────────────────────────────────────
        u_out = self.fc_out(s2)                          # (B, T, n_outputs)

        # Step 1: sum softmax probabilities over T (Equation 3)
        probs = F.softmax(u_out, dim=-1).sum(dim=1)      # (B, n_outputs)

        # Step 2: renormalise over classes for valid log-probability
        logits = F.log_softmax(probs, dim=-1)            # (B, n_outputs)

        return logits

    def clamp_all(self):
        """Call after every optimiser step."""
        self.lif1.clamp_parameters()
        self.lif2.clamp_parameters()
        with torch.no_grad():
            self.delay1.delay.clamp_(0, self.cfg.max_delay)
            self.delay2.delay.clamp_(0, self.cfg.max_delay)

    def get_neuron_stats(self) -> dict:
        return {
            "layer1": self.lif1.get_stats(),
            "layer2": self.lif2.get_stats(),
        }

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }