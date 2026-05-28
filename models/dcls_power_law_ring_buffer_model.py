# models/dcls_power_law_ring_buffer_model.py
"""
DCLSPowerLawRingBufferNet — DCLS SNN with power-law-spaced ring buffer
replacing the binning + fc_in input stage.

Architecture
────────────
Input (B, 700, T)   ← channels-first, as in original DCLS
    │
    ▼  [PowerLawRingBufferInput]
    fc_proj   : Linear(700, k, bias=False)   — spatial bottleneck
    ring buf  : circular tensor (B, k, max_lag+1)
    lags      : power-law spaced  — L taps
    fc_expand : Linear(k*L, n_hidden, bias=False)
    × theta_u — current scaling
    │  output: (B, n_hidden, T)
    │
    ▼  dcls1 (Dcls1d) → lif1 → bn1 → drop1
    ▼  dcls2 (Dcls1d) → lif2 → bn2 → drop2
    ▼  mean(T) → fc_out
    │
Output (B, n_outputs)   ← raw logits for CrossEntropy

Identical to DCLSLogRingBufferNet except the tap spacing is
power-law (gamma-spaced) rather than log (powers-of-two).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate
from DCLS.construct.modules import Dcls1d


# ─────────────────────────────────────────────────────────────────────────────
# POWER-LAW RING BUFFER INPUT STAGE
# ─────────────────────────────────────────────────────────────────────────────

class PowerLawRingBufferInput(nn.Module):
    """
    Replaces the binning + fc_in input stage of DCLSNet.

    Processes (B, n_inputs, T) by stepping through the time
    dimension and maintaining a history of k-dim projections.

    Gradient-safe implementation:
        - Projections are stored in a Python list so autograd
          can trace through fc_proj and fc_expand.
        - Zero-padding used when history is not deep enough.

    Power-law tap spacing:
        taps are pre-computed by DCLSPowerLawRingBufferConfig and
        passed in as a list, e.g. [1, 2, 7, 16, 28, 44, 64]
        for max_lag=64, n=7, gamma=2.0.

    Args:
        n_inputs    : raw input dimension (700 for SHD/SSC)
        n_hidden    : output dimension = n_hidden_neurons in DCLS config
        buffer_size : k — bottleneck width
        max_lag     : maximum lag (used for validation only in list mode)
        taps        : power-law spaced lag list
        theta_u     : current scaling
    """

    def __init__(
        self,
        n_inputs    : int,
        n_hidden    : int,
        buffer_size : int,
        max_lag     : int,
        taps        : list,
        theta_u     : float = 0.5,
    ):
        super().__init__()

        if len(taps) == 0:
            raise ValueError("taps list must not be empty")
        if max(taps) > max_lag:
            raise ValueError(
                f"max tap {max(taps)} exceeds max_lag {max_lag}"
            )
        if min(taps) < 1:
            raise ValueError(
                f"all taps must be >= 1, got min={min(taps)}"
            )

        self.k       = buffer_size
        self.max_lag = max_lag
        self.taps    = taps
        self.L       = len(taps)
        self.theta_u = theta_u

        # Plain Linear — consistent with DCLS style, no weight_norm
        self.fc_proj   = nn.Linear(n_inputs,        self.k,   bias=False)
        self.fc_expand = nn.Linear(self.k * self.L, n_hidden, bias=False)

        nn.init.xavier_uniform_(self.fc_proj.weight)
        nn.init.xavier_uniform_(self.fc_expand.weight)

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process full sequence in one call.

        Args:
            x : (B, n_inputs, T) — channels-first DCLS format

        Returns:
            out : (B, n_hidden, T) — ready for dcls1
        """
        B, C, T = x.shape

        history   = []   # history[t] = fc_proj(x[:, :, t]) — (B, k)
        out_steps = []
        zero      = torch.zeros(B, self.k, device=x.device, dtype=x.dtype)

        for t in range(T):
            # 1. Project current timestep → (B, k)
            p_t = self.fc_proj(x[:, :, t])
            history.append(p_t)

            # 2. Read power-law taps — zero-pad if history not deep enough
            tap_reads = []
            for lag in self.taps:
                idx = t - lag
                tap_reads.append(history[idx] if idx >= 0 else zero)

            # 3. Concatenate → (B, k*L), expand → (B, n_hidden), scale
            feat = torch.cat(tap_reads, dim=-1)   # (B, k*L)
            cur  = self.fc_expand(feat)           # (B, n_hidden)
            out_steps.append(cur * self.theta_u)

        return torch.stack(out_steps, dim=-1)     # (B, n_hidden, T)

    def get_info(self) -> dict:
        return {
            "k"      : self.k,
            "L"      : self.L,
            "taps"   : self.taps,
            "storage": "list",
        }


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class DCLSPowerLawRingBufferNet(nn.Module):
    """
    DCLS SNN with power-law-spaced ring buffer input stage.

    Input  : (B, n_inputs, T) — channels-first
    Output : (B, n_outputs)   — raw logits for CrossEntropy
    """

    def __init__(self, cfg):
        super().__init__()

        self.time_steps = cfg.time_steps
        self._current_epoch = 0
        self._sigma_max     = getattr(cfg, "sigma_max", 2.0)
        self._sigma_min     = getattr(cfg, "sigma_min", 0.23)

        # ── Power-law ring buffer input stage ─────────────────────────
        self.buffer_input = PowerLawRingBufferInput(
            n_inputs    = cfg.n_inputs,
            n_hidden    = cfg.n_hidden_neurons,
            buffer_size = cfg.buffer_size,
            max_lag     = cfg.buffer_max_lag,
            taps        = cfg.taps,
            theta_u     = cfg.theta_u,
        )

        # ── DCLS convolutional layers ──────────────────────────────────
        self.dcls1 = Dcls1d(
            in_channels      = cfg.n_hidden_neurons,
            out_channels     = cfg.n_hidden_neurons,
            kernel_count     = cfg.kernel_count,
            dilated_kernel_size = cfg.max_delay,
            stride           = 1,
            padding          = cfg.max_delay // 2,
            groups           = 1,
            bias             = False,
            version          = "gauss",
        )
        self.dcls2 = Dcls1d(
            in_channels      = cfg.n_hidden_neurons,
            out_channels     = cfg.n_hidden_neurons,
            kernel_count     = cfg.kernel_count,
            dilated_kernel_size = cfg.max_delay,
            stride           = 1,
            padding          = cfg.max_delay // 2,
            groups           = 1,
            bias             = False,
            version          = "gauss",
        )

        # ── LIF neurons ───────────────────────────────────────────────
        surr = surrogate.ATan()

        self.lif1 = neuron.LIFNode(
            tau             = cfg.tau,
            v_threshold     = cfg.v_threshold,
            surrogate_function = surr,
            detach_reset    = True,
        )
        self.lif2 = neuron.LIFNode(
            tau             = cfg.tau,
            v_threshold     = cfg.v_threshold,
            surrogate_function = surr,
            detach_reset    = True,
        )

        # ── Batch norm + dropout ──────────────────────────────────────
        self.bn1   = nn.BatchNorm1d(cfg.n_hidden_neurons)
        self.bn2   = nn.BatchNorm1d(cfg.n_hidden_neurons)
        self.drop1 = nn.Dropout(p=cfg.dropout_p)
        self.drop2 = nn.Dropout(p=cfg.dropout_p)

        # ── Output ────────────────────────────────────────────────────
        self.fc_out = nn.Linear(cfg.n_hidden_neurons, cfg.n_outputs)

    # ─────────────────────────────────────────────────────────────────
    def reset(self):
        """Reset LIF membrane potentials between sequences."""
        self.lif1.reset()
        self.lif2.reset()

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, n_inputs, T)

        Returns:
            logits : (B, n_outputs)
        """
        # 1. Power-law ring buffer → (B, n_hidden, T)
        h = self.buffer_input(x)

        # 2. dcls1 → lif1 → bn1 → drop1
        h = self.dcls1(h)                          # (B, n_hidden, T)
        # LIF expects (T, B, n_hidden) — permute, apply, permute back
        h = h.permute(2, 0, 1)                     # (T, B, n_hidden)
        h = self.lif1(h)
        h = h.permute(1, 2, 0)                     # (B, n_hidden, T)
        h = self.bn1(h)
        h = self.drop1(h)

        # 3. dcls2 → lif2 → bn2 → drop2
        h = self.dcls2(h)
        h = h.permute(2, 0, 1)
        h = self.lif2(h)
        h = h.permute(1, 2, 0)
        h = self.bn2(h)
        h = self.drop2(h)

        # 4. Temporal mean → fc_out
        h = h.mean(dim=-1)                         # (B, n_hidden)
        return self.fc_out(h)                      # (B, n_outputs)

    # ─────────────────────────────────────────────────────────────────
    def update_sigma(self) -> float:
        """
        Anneal DCLS Gaussian sigma. Call once per epoch after training.
        Mirrors DCLSNet.update_sigma().
        """
        self._current_epoch += 1
        ratio = self._current_epoch / max(1, self.time_steps)
        sigma = max(
            self._sigma_min,
            self._sigma_max * (1.0 - ratio),
        )
        for m in self.modules():
            if isinstance(m, Dcls1d):
                m.SIG = sigma
        return sigma

    # ─────────────────────────────────────────────────────────────────
    # Dual-optimiser interface — identical to DCLSNet
    # ─────────────────────────────────────────────────────────────────

    def get_delay_params(self):
        """DCLS position parameters only (optimizer_p)."""
        return [
            p for n, p in self.named_parameters()
            if "P" in n and any(
                isinstance(m, Dcls1d)
                for m in self.modules()
            )
        ]

    def get_weight_params(self):
        """All non-delay parameters including buffer (optimizer_w)."""
        delay_ids = {id(p) for p in self.get_delay_params()}
        return [p for p in self.parameters() if id(p) not in delay_ids]

    # ─────────────────────────────────────────────────────────────────
    def get_buffer_info(self) -> dict:
        return self.buffer_input.get_info()

    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, v):
        self._current_epoch = v