# models/dhlif_model.py
"""
DH-LIF: Dendritic Heterogeneity LIF SNN.
Paper: Zheng et al. 2024, Nature Communications 15:277
"Temporal dendritic heterogeneity incorporated with spiking
neural networks for learning multi-timescale dynamics"
GitHub: https://github.com/eva1801/DH-SNN

Neuron dynamics (Equation 5):
    u[t+1]   = beta * u[t] + (1-beta) * sum_d(R * i_d[t+1]) - o[t]*u_th
    i_d[t+1] = alpha_d * i_d[t] + (1-alpha_d) * I_d[t+1]
    o[t+1]   = H(u[t+1] - u_th)

Timing factor constraints (Equation 6):
    alpha_d = sigmoid(alpha_d_hat)
    beta    = sigmoid(beta_hat)
    — both constrained to (0,1) automatically

Sparse connection (Equation 10):
    Each branch d connects to a non-overlapping subset of M/D inputs
    Total connections = M (same as vanilla SNN)

Surrogate gradient (Equation 13):
    H' = gamma*(1+h)*N(u|u_th,sigma^2)
         - gamma*h*N(u|sigma,(s*sigma)^2)
         - gamma*h*N(u|-sigma,(s*sigma)^2)

Architecture (DH-SFNN, feedforward only):
    Input (B, T, 700)
        → DHLIFLayer1 (sparse D-branch dendrites) → spikes
        → DHLIFLayer2 (sparse D-branch dendrites) → spikes
        → FC_readout (non-spiking, leaky membrane)
        → sum softmax over T → (B, 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-GAUSSIAN SURROGATE GRADIENT (Equation 13)
# ─────────────────────────────────────────────────────────────────────────────

class MultiGaussianSurrogate(torch.autograd.Function):
    """
    Forward : Heaviside spike  o = (u >= u_th)
    Backward: Multi-Gaussian surrogate (Equation 13)

    H' = gamma*(1+h)*N(u|u_th, sigma^2)
         - gamma*h*N(u|u_th+sigma, (s*sigma)^2)
         - gamma*h*N(u|u_th-sigma, (s*sigma)^2)

    Parameters from paper Methods:
        gamma=0.5, h=0.15, sigma=0.5, s=6.0
    """

    @staticmethod
    def forward(ctx, u, threshold, gamma, h, sigma, s):
        ctx.save_for_backward(u)
        ctx.threshold = threshold
        ctx.gamma     = gamma
        ctx.h         = h
        ctx.sigma     = sigma
        ctx.s         = s
        return (u >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        u,        = ctx.saved_tensors
        threshold = ctx.threshold
        gamma     = ctx.gamma
        h         = ctx.h
        sigma     = ctx.sigma
        s         = ctx.s

        def gaussian(x, mu, sig):
            return torch.exp(
                -0.5 * ((x - mu) / (sig + 1e-8)) ** 2
            ) / (sig * (2 * np.pi) ** 0.5 + 1e-8)

        # Three Gaussian components (Equation 13)
        g1 = gamma * (1 + h) * gaussian(u, threshold, sigma)
        g2 = gamma * h       * gaussian(u, threshold + sigma, s * sigma)
        g3 = gamma * h       * gaussian(u, threshold - sigma, s * sigma)

        surrogate = g1 - g2 - g3
        return grad_output * surrogate, None, None, None, None, None


def multi_gaussian_spike(u, threshold=0.5,
                          gamma=0.5, h=0.15, sigma=0.5, s=6.0):
    return MultiGaussianSurrogate.apply(u, threshold, gamma, h, sigma, s)


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE DENDRITIC CONNECTION MASK
# Equation 10: non-overlapping partition of M inputs across D branches
# ─────────────────────────────────────────────────────────────────────────────

def make_sparse_mask(n_pre: int, n_post: int,
                     n_branches: int) -> torch.Tensor:
    """
    Create sparse connection mask for DH-LIF dendritic branches.

    Each neuron's M inputs are partitioned into D non-overlapping
    subsets of size M/D. Different neurons get different random
    permutations (Equation 10, 19).

    Returns:
        mask: (n_post, n_branches, n_pre) binary mask
              mask[n, d, :] = 1 for inputs connected to branch d of neuron n
    """
    mask = torch.zeros(n_post, n_branches, n_pre)
    chunk = n_pre // n_branches

    for n in range(n_post):
        # Random permutation of inputs for this neuron
        perm = torch.randperm(n_pre)
        for d in range(n_branches):
            start = d * chunk
            end   = start + chunk if d < n_branches - 1 else n_pre
            mask[n, d, perm[start:end]] = 1.0

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# DH-LIF LAYER
# Multi-compartment LIF with D dendritic branches
# ─────────────────────────────────────────────────────────────────────────────

class DHLIFLayer(nn.Module):
    """
    DH-LIF hidden layer with D dendritic branches per neuron.

    Each neuron has:
        - D dendritic branches, each with its own timing factor alpha_d
        - 1 soma with timing factor beta
        - Sparse connection: branch d sees M/D non-overlapping inputs

    Parameters (all learnable):
        alpha_hat: (n_neurons, n_branches) — raw logits, sigmoid → alpha_d
        beta_hat : (n_neurons,)            — raw logits, sigmoid → beta
        W_d      : (n_neurons, n_branches, n_pre//n_branches) — branch weights

    Dynamics (Equation 5):
        i_d[t+1] = alpha_d * i_d[t] + (1-alpha_d) * I_d[t+1]
        u[t+1]   = beta * u[t] + (1-beta) * sum_d(i_d[t+1]) - o[t]*u_th
        o[t+1]   = H(u[t+1] - u_th)

    Note R=1 (paper simplification).
    Dendritic currents are NEVER reset (key feature for long-term memory).
    Only soma membrane potential resets via soft-reset: -o[t]*u_th
    """

    def __init__(self, n_pre: int, n_neurons: int,
                 n_branches: int, cfg):
        super().__init__()
        self.n_pre      = n_pre
        self.n_neurons  = n_neurons
        self.n_branches = n_branches
        self.v_threshold = cfg.v_threshold
        self.chunk      = n_pre // n_branches  # inputs per branch

        # Surrogate params
        self.gamma = cfg.surrogate_gamma
        self.h     = cfg.surrogate_h
        self.sigma = cfg.surrogate_sigma
        self.s     = cfg.surrogate_s

        # ── Timing factors (raw logits, sigmoid applied in forward) ───────
        # alpha_hat: (n_neurons, n_branches) — dendritic timing factors
        self.alpha_hat = nn.Parameter(
            torch.empty(n_neurons, n_branches).normal_(
                cfg.alpha_init_mean, cfg.alpha_init_std
            )
        )
        # beta_hat: (n_neurons,) — soma timing factor
        self.beta_hat  = nn.Parameter(
            torch.empty(n_neurons).normal_(
                cfg.beta_init_mean, cfg.beta_init_std
            )
        )

        # ── Sparse branch weights ─────────────────────────────────────────
        # W: (n_neurons, n_branches, chunk) — weight for each branch
        # Each branch only sees chunk = n_pre//n_branches inputs
        self.W = nn.Parameter(
            torch.empty(n_neurons, n_branches, self.chunk)
        )
        nn.init.xavier_uniform_(
            self.W.view(n_neurons, -1).unsqueeze(0)
        ).squeeze(0)
        # Simpler Xavier-like init
        nn.init.normal_(self.W, 0,
                        (2.0 / (n_pre + n_neurons)) ** 0.5)

        # ── Sparse connection index ───────────────────────────────────────
        # For each neuron n and branch d, which n_pre inputs are connected
        # indices: (n_neurons, n_branches, chunk)
        indices = torch.zeros(n_neurons, n_branches,
                              self.chunk, dtype=torch.long)
        for n in range(n_neurons):
            perm = torch.randperm(n_pre)
            for d in range(n_branches):
                start = d * self.chunk
                end   = start + self.chunk \
                        if d < n_branches - 1 else n_pre
                seg   = perm[start:end]
                # Pad or trim to chunk size
                seg   = seg[:self.chunk]
                indices[n, d, :len(seg)] = seg

        # Register as buffer (not a parameter — fixed after init)
        self.register_buffer("indices", indices)

        # ── Dropout ───────────────────────────────────────────────────────
        self.drop = nn.Dropout(cfg.dropout_p) \
                    if cfg.dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N_pre = x.shape
        device = x.device

        alpha = torch.sigmoid(self.alpha_hat)   # (n_neurons, n_branches)
        beta  = torch.sigmoid(self.beta_hat)    # (n_neurons,)

        # ── Build dense weight matrix via scatter (grad flows through self.W) ──
        # W_full: (n_neurons, n_branches, n_pre)  — sparse, only chunk entries filled
        W_full = torch.zeros(self.n_neurons, self.n_branches, self.n_pre,
                            device=device, dtype=self.W.dtype)
        W_full = W_full.scatter(2, self.indices, self.W)

        # ── Precompute ALL branch inputs for ALL timesteps in one matmul ────────
        # (B*T, n_pre) @ (n_pre, n_neurons*n_branches) → (B, T, n_neurons, n_branches)
        W_mat  = W_full.view(self.n_neurons * self.n_branches, self.n_pre)
        I_pre  = (x.reshape(B * T, N_pre) @ W_mat.T).view(B, T, self.n_neurons, self.n_branches)
        # SSC: (128,400,256,4) = ~200 MB  ✓   DVS: (32,300,256,4) = ~37 MB  ✓

        # ── Temporal loop — only small state tensors live here ──────────────────
        u      = torch.zeros(B, self.n_neurons,                  device=device)
        i_d    = torch.zeros(B, self.n_neurons, self.n_branches, device=device)
        o_prev = torch.zeros(B, self.n_neurons,                  device=device)
        spikes = torch.zeros(B, T, self.n_neurons,               device=device)

        alpha_exp = alpha.unsqueeze(0)   # (1, n_neurons, n_branches)
        beta_exp  = beta.unsqueeze(0)    # (1, n_neurons)

        for t in range(T):
            I_d_t  = I_pre[:, t]                          # (B, n_neurons, n_branches) — view, no alloc
            i_d    = alpha_exp * i_d + (1 - alpha_exp) * I_d_t
            i_soma = i_d.sum(dim=2)
            u      = beta_exp * u + (1 - beta_exp) * i_soma - o_prev * self.v_threshold
            o      = multi_gaussian_spike(u, self.v_threshold,
                                        self.gamma, self.h, self.sigma, self.s)
            spikes[:, t] = o
            o_prev = o

        return self.drop(spikes)

    def get_timing_stats(self) -> dict:
        alpha = torch.sigmoid(self.alpha_hat).detach().cpu()
        beta  = torch.sigmoid(self.beta_hat).detach().cpu()
        return {
            "alpha_mean": alpha.mean().item(),
            "alpha_std" : alpha.std().item(),
            "alpha_min" : alpha.min().item(),
            "alpha_max" : alpha.max().item(),
            "beta_mean" : beta.mean().item(),
            "beta_std"  : beta.std().item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# READOUT LAYER
# Non-spiking layer with leaky membrane potential
# SHD readout: y_i = sum_t softmax(u_out[t])
# ─────────────────────────────────────────────────────────────────────────────

class LeakyReadout(nn.Module):
    def __init__(self, n_pre: int, n_outputs: int,
                 beta_readout: float = 0.9):
        super().__init__()
        self.fc           = nn.Linear(n_pre, n_outputs, bias=True)
        self.beta_readout = beta_readout
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: (B, T, n_pre)
        Returns:
            logits: (B, n_outputs) — log_softmax for NLLLoss
        """
        B, T, _ = spikes.shape
        device  = spikes.device

        u         = torch.zeros(B, self.fc.out_features, device=device)
        probs_sum = torch.zeros(B, self.fc.out_features, device=device)

        for t in range(T):
            u         = self.beta_readout * u + self.fc(spikes[:, t, :])
            probs_sum = probs_sum + F.softmax(u, dim=-1)

        # ── FIX: log_softmax over classes on the SUM ──────────────────────
        # probs_sum is in range [0, T] — unnormalised over classes
        # log_softmax renormalises correctly for NLLLoss
        # Do NOT use log(probs_sum + eps) — that gives wrong scale
        return F.log_softmax(probs_sum, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK — DH-SFNN
# ─────────────────────────────────────────────────────────────────────────────

class DHLIFNet(nn.Module):
    """
    DH-SFNN: Feedforward SNN with DH-LIF neurons.
    Zheng et al. 2024, best SHD result: 2-layer, 4-branch = 92.1%

    Architecture:
        Input  (B, T, 700)
            → DHLIFLayer1 (n_hidden, n_branches) → spikes (B,T,n_hidden)
            → DHLIFLayer2 (n_hidden, n_branches) → spikes (B,T,n_hidden)
            → LeakyReadout → sum softmax over T → (B, 20)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Build stacked DH-LIF layers
        self.layers = nn.ModuleList()
        in_size = cfg.n_inputs
        for _ in range(cfg.n_layers):
            self.layers.append(
                DHLIFLayer(
                    n_pre      = in_size,
                    n_neurons  = cfg.n_hidden,
                    n_branches = cfg.n_branches,
                    cfg        = cfg,
                )
            )
            in_size = cfg.n_hidden

        # Readout
        self.readout = LeakyReadout(cfg.n_hidden, cfg.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_inputs)
        Returns:
            logits: (B, n_outputs) — log-prob for NLLLoss
        """
        for layer in self.layers:
            x = layer(x)
        return self.readout(x)

    def get_timing_stats(self) -> dict:
        return {
            f"layer{i+1}": layer.get_timing_stats()
            for i, layer in enumerate(self.layers)
        }