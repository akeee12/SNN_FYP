# models/dcls_log_ring_buffer_model.py
"""
DCLSLogRingBufferNet — DCLS SNN with log-spaced ring buffer
replacing the binning + fc_in input stage.

Architecture
────────────
Input (B, 700, T)   ← channels-first, as in original DCLS
    │
    ▼  [LogRingBufferInput]
    fc_proj   : Linear(700, k, bias=False)   — spatial bottleneck
    ring buf  : circular tensor (B, k, max_lag+1)
    lags      : {1, 2, 4, …, max_lag}  — L taps, log-spaced
    fc_expand : Linear(k*L, n_hidden, bias=False)
    × theta_u — current scaling
    │  output: (B, n_hidden, T)
    │
    ▼  dcls1 (Dcls1d) → lif1 → bn1 → drop1
    ▼  dcls2 (Dcls1d) → lif2 → bn2 → drop2
    ▼  mean(T) → fc_out
    │
Output (B, n_outputs)   ← raw logits for CrossEntropy

Key differences from DCLSNet:
    1. Binning (700→140) and fc_in (140→256) are REMOVED.
    2. LogRingBufferInput processes (B, 700, T) timestep-by-timestep
       using a ring buffer, producing (B, n_hidden, T).
    3. Everything from dcls1 onwards is identical to DCLSNet.
    4. Dual optimiser interface (get_delay_params / get_weight_params)
       is preserved — buffer params go into weight_params.

Ring buffer complexity:
    Write per timestep : O(B * k)           — one slot written
    Rolling alternative: O(B * k * max_lag) — entire buffer shifted
    Saving             : 64x for max_lag=64
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate
from DCLS.construct.modules import Dcls1d


# ─────────────────────────────────────────────────────────────────────────────
# LOG RING BUFFER INPUT STAGE
# ─────────────────────────────────────────────────────────────────────────────

class LogRingBufferInput(nn.Module):
    """
    Replaces the binning + fc_in input stage of DCLSNet.

    Processes (B, n_inputs, T) by stepping through the time
    dimension and maintaining a ring buffer of k-dim projections.

    At each timestep t:
      1. Project x_t ∈ R^{n_inputs} → p_t ∈ R^k  via fc_proj.
      2. Write p_t into ring buffer at write pointer.
      3. Read L log-spaced taps via modulo index arithmetic.
      4. Concatenate → (B, k*L), expand → (B, n_hidden) via fc_expand.
      5. Scale by theta_u.

    Args:
        n_inputs    : raw input dimension (700 for SHD/SSC, 2048 for DVS)
        n_hidden    : output dimension = n_hidden_neurons in DCLS config
        buffer_size : k — bottleneck width
        max_lag     : ring depth = max_lag + 1 slots
        taps        : log-spaced lag list (powers of two)
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

        # ── Learnable projections ──────────────────────────────────────
        # No weight_norm here — DCLS uses plain Linear layers
        self.fc_proj   = nn.Linear(n_inputs,        self.k,          bias=False)
        self.fc_expand = nn.Linear(self.k * self.L, n_hidden,        bias=False)

        nn.init.xavier_uniform_(self.fc_proj.weight)
        nn.init.xavier_uniform_(self.fc_expand.weight)

        # ── Ring buffer ────────────────────────────────────────────────
        # depth = max_lag + 1 slots so ptr never overwrites unread value
        # shape: (1, k, max_lag+1) — expanded to batch at reset()
        self.register_buffer(
            "_ring",
            torch.zeros(1, self.k, max_lag + 1),
        )
        self._ptr = 0

    # ─────────────────────────────────────────────────────────────────
    def reset(self, batch_size: int):
        """Zero ring and reset pointer. Call once per sequence."""
        device     = self._ring.device
        self._ring = torch.zeros(
            batch_size, self.k, self.max_lag + 1,
            device=device, dtype=self._ring.dtype,
        )
        self._ptr = 0

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
        self.reset(B)

        out_steps = []
        for t in range(T):
            x_t = x[:, :, t]                          # (B, n_inputs)

            # 1. Project → (B, k)
            p_t = self.fc_proj(x_t)

            # 2. Write at pointer
            self._ring[:, :, self._ptr] = p_t

            # 3. Read log-spaced taps
            tap_reads = [
                self._ring[:, :, (self._ptr - lag) % (self.max_lag + 1)]
                for lag in self.taps
            ]

            # 4. Advance pointer
            self._ptr = (self._ptr + 1) % (self.max_lag + 1)

            # 5. Concatenate → expand → scale
            feat    = torch.cat(tap_reads, dim=-1)     # (B, k*L)
            cur     = self.fc_expand(feat)             # (B, n_hidden)
            out_steps.append(cur * self.theta_u)

        return torch.stack(out_steps, dim=-1)          # (B, n_hidden, T)

    def get_info(self) -> dict:
        return {
            "k"      : self.k,
            "L"      : self.L,
            "taps"   : self.taps,
            "storage": "ring",
        }


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class DCLSLogRingBufferNet(nn.Module):
    """
    DCLS SNN with log ring buffer replacing binning + fc_in.

    Args:
        x: (B, 700, T)   ← permuted in training script as in original DCLS
    Returns:
        out: (B, n_outputs)  ← raw logits for CrossEntropy
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── Log ring buffer input (replaces bin + fc_in) ──────────────
        self.buffer_input = LogRingBufferInput(
            n_inputs    = cfg.n_inputs,        # 700
            n_hidden    = cfg.n_hidden_neurons, # 256
            buffer_size = cfg.buffer_size,     # 32
            max_lag     = cfg.buffer_max_lag,  # 64
            taps        = cfg.taps,
            theta_u     = cfg.theta_u,
        )

        # ── DCLS delay layers — identical to DCLSNet ──────────────────
        self.dcls1 = Dcls1d(
            in_channels         = cfg.n_hidden_neurons,
            out_channels        = cfg.n_hidden_neurons,
            kernel_count        = cfg.kernel_count,
            dilated_kernel_size = cfg.max_delay,
            padding             = cfg.left_padding,
            groups              = 1,
            bias                = cfg.bias,
            version             = cfg.DCLSversion,
        )
        self.dcls2 = Dcls1d(
            in_channels         = cfg.n_hidden_neurons,
            out_channels        = cfg.n_hidden_neurons,
            kernel_count        = cfg.kernel_count,
            dilated_kernel_size = cfg.max_delay,
            padding             = cfg.left_padding,
            groups              = 1,
            bias                = cfg.bias,
            version             = cfg.DCLSversion,
        )

        nn.init.uniform_(self.dcls1.P, cfg.init_pos_a, cfg.init_pos_b)
        nn.init.uniform_(self.dcls2.P, cfg.init_pos_a, cfg.init_pos_b)
        self.dcls1.SIG.data.fill_(cfg.sigInit)
        self.dcls2.SIG.data.fill_(cfg.sigInit)

        # ── LIF neurons — identical to DCLSNet ────────────────────────
        self.lif1 = neuron.LIFNode(
            tau                = cfg.init_tau,
            v_threshold        = cfg.v_threshold,
            surrogate_function = surrogate.ATan(alpha=cfg.alpha),
            detach_reset       = cfg.detach_reset,
            step_mode          = 'm',
        )
        self.lif2 = neuron.LIFNode(
            tau                = cfg.init_tau,
            v_threshold        = cfg.v_threshold,
            surrogate_function = surrogate.ATan(alpha=cfg.alpha),
            detach_reset       = cfg.detach_reset,
            step_mode          = 'm',
        )

        # ── Batch norm — identical to DCLSNet ─────────────────────────
        self.bn1 = nn.BatchNorm1d(cfg.n_hidden_neurons) \
                   if cfg.use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(cfg.n_hidden_neurons) \
                   if cfg.use_batchnorm else nn.Identity()

        # ── Dropout — identical to DCLSNet ────────────────────────────
        self.drop1 = nn.Dropout(cfg.dropout_p)
        self.drop2 = nn.Dropout(cfg.dropout_p)

        # ── Output FC — identical to DCLSNet ──────────────────────────
        self.fc_out = nn.Linear(
            cfg.n_hidden_neurons,
            cfg.n_outputs,
            bias=cfg.bias,
        )

        # ── Sigma annealing state — identical to DCLSNet ──────────────
        self.current_epoch       = 0
        self.final_epoch         = cfg.final_epoch
        self.sigInit             = cfg.sigInit
        self.decrease_sig_method = cfg.decrease_sig_method

    # ─────────────────────────────────────────────────────────────────
    def _compute_sigma(self) -> float:
        if self.current_epoch >= self.final_epoch:
            return 0.5
        if self.decrease_sig_method == "exp":
            ratio = self.current_epoch / max(self.final_epoch, 1)
            return max(
                self.sigInit * (0.5 / self.sigInit) ** ratio,
                0.5
            )
        else:
            return max(
                self.sigInit - (self.sigInit - 0.5) *
                self.current_epoch / max(self.final_epoch, 1),
                0.5
            )

    def update_sigma(self) -> float:
        """Update sigma — call once per epoch after training loop."""
        sigma = self._compute_sigma()
        self.dcls1.SIG.data.fill_(sigma)
        self.dcls2.SIG.data.fill_(sigma)
        self.current_epoch += 1
        return sigma

    # ─────────────────────────────────────────────────────────────────
    def get_delay_params(self):
        """Delay position parameters for separate optimiser."""
        return [self.dcls1.P, self.dcls2.P]

    def get_weight_params(self):
        """
        All non-delay parameters for weight optimiser.
        Buffer params (fc_proj, fc_expand) are included here.
        """
        delay_ids = {id(self.dcls1.P), id(self.dcls2.P)}
        return [p for p in self.parameters()
                if id(p) not in delay_ids]

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 700, T) — channels-first, permuted in training script
        Returns:
            out: (B, n_outputs) — raw logits
        """
        # ── Log ring buffer: (B, 700, T) → (B, n_hidden, T) ──────────
        x = self.buffer_input(x)               # (B, 256, T)

        # ── DCLS layer 1 — identical to DCLSNet ───────────────────────
        x = self.dcls1(x)                      # (B, 256, T)
        x = self.lif1(x.permute(2, 0, 1))     # (T, B, 256)
        x = x.permute(1, 2, 0)                # (B, 256, T)
        x = self.bn1(x)
        x = self.drop1(x)

        # ── DCLS layer 2 — identical to DCLSNet ───────────────────────
        x = self.dcls2(x)                      # (B, 256, T)
        x = self.lif2(x.permute(2, 0, 1))     # (T, B, 256)
        x = x.permute(1, 2, 0)                # (B, 256, T)
        x = self.bn2(x)
        x = self.drop2(x)

        # ── Readout — identical to DCLSNet ────────────────────────────
        x = x.mean(dim=-1)                     # (B, 256)
        x = self.fc_out(x)                     # (B, n_outputs)
        return x

    def reset(self):
        """Reset LIF neuron states — called before each sequence."""
        self.lif1.reset()
        self.lif2.reset()

    def get_buffer_info(self) -> dict:
        return self.buffer_input.get_info()