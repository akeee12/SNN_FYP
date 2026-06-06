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

    Returns the weighted delayed sum I directly — shape (B, T, n_post) —
    rather than the full (B, T, n_post, n_pre) intermediate, which OOMs
    for large (n_post, n_pre) on SSC / DVS.

    The delay gradient still flows via linear interpolation between
    floor/ceil delays through the alpha interpolation weight.
    """

    def __init__(self, n_pre: int, n_post: int, max_delay: int = 25):
        super().__init__()
        self.n_pre     = n_pre
        self.n_post    = n_post
        self.max_delay = max_delay
        self.delay     = nn.Parameter(torch.zeros(n_post, n_pre))

    def forward(self, spikes: torch.Tensor,
                weight: torch.Tensor,
                bias:   torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            spikes : (B, T, n_pre)
            weight : (n_post, n_pre)   — fc.weight passed in from cAdLIFNet
            bias   : (n_post,)         — fc.bias  passed in from cAdLIFNet
        Returns:
            I : (B, T, n_post)         — weighted delayed synaptic input
        """
        B, T, N_pre = spikes.shape
        device      = spikes.device

        d       = self.delay.clamp(0, self.max_delay)
        d_floor = d.floor().long()          # (n_post, n_pre)
        d_ceil  = d.ceil().long()           # (n_post, n_pre)
        alpha   = d - d.floor()             # (n_post, n_pre) — grad flows here

        # Effective weights for floor / ceil interpolation
        Wf = weight * (1 - alpha)           # (n_post, n_pre)
        Wc = weight * alpha                 # (n_post, n_pre)

        # Left-pad time axis — (B, max_delay+T, n_pre)
        pad    = torch.zeros(B, self.max_delay, N_pre,
                             device=device, dtype=spikes.dtype)
        padded = torch.cat([pad, spikes], dim=1)

        # Accumulate weighted delayed inputs without ever building
        # the (B, T, n_post, n_pre) tensor
        # Peak memory: (B, T, n_pre) slice + (B, T, n_post) accumulator
        I = torch.zeros(B, T, self.n_post,
                        device=device, dtype=spikes.dtype)

        for k in range(self.max_delay + 1):
            # x_k[b, t, j] = spikes[b, t-k, j]  (zero-padded for t<k)
            x_k = padded[:, self.max_delay - k:
                             self.max_delay - k + T, :]   # (B, T, n_pre)
            x_k_2d = x_k.reshape(B * T, N_pre)            # (B*T, n_pre)

            # Floor contribution — synapses whose floor delay == k
            Wf_k = Wf * (d_floor == k).float()            # (n_post, n_pre)
            I = I + (x_k_2d @ Wf_k.T).view(B, T, self.n_post)

            # Ceil contribution — synapses whose ceil delay == k
            Wc_k = Wc * (d_ceil == k).float()
            I = I + (x_k_2d @ Wc_k.T).view(B, T, self.n_post)

        if bias is not None:
            I = I + bias.unsqueeze(0).unsqueeze(0)

        return I

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
                x: (B, T, n_inputs)
            Returns:
                logits: (B, n_outputs) — log_softmax for NLLLoss
            """
            # ── Layer 1 ───────────────────────────────────────────────────────
            # Pass fc1 weight+bias into delay layer — returns (B, T, n_hidden)
            I1 = self.delay1(x,  self.fc1.weight, self.fc1.bias)
            s1 = self.drop1(self.lif1(I1))

            # ── Layer 2 ───────────────────────────────────────────────────────
            I2 = self.delay2(s1, self.fc2.weight, self.fc2.bias)
            s2 = self.drop2(self.lif2(I2))

            # ── Readout — Equations 3 + 4 ─────────────────────────────────────
            u_out  = self.fc_out(s2)
            probs  = F.softmax(u_out, dim=-1).sum(dim=1)
            return F.log_softmax(probs, dim=-1)

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