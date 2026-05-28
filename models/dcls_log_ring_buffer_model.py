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
    dimension and maintaining a history of k-dim projections.

    Gradient-safe implementation:
        - Projections are stored in a Python list (not a pre-allocated
          tensor) so autograd can trace through fc_proj and fc_expand.
        - The ring buffer index arithmetic is preserved for reading
          past projections, but writes go into a list rather than
          an in-place tensor assignment.

    At each timestep t:
      1. Project x_t → p_t ∈ R^k via fc_proj.
      2. Append p_t to history list.
      3. Read L log-spaced taps from history (zero-pad if not enough
         history yet).
      4. Concatenate → (B, k*L), expand → (B, n_hidden) via fc_expand.
      5. Scale by theta_u.
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

        # Plain Linear — no weight_norm, consistent with DCLS style
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

        # history[t] = fc_proj(x[:, :, t]) — shape (B, k)
        # stored as a list so autograd traces through fc_proj
        history = []
        out_steps = []

        # zero tensor used for padding when history is not deep enough
        zero = torch.zeros(B, self.k, device=x.device, dtype=x.dtype)

        for t in range(T):
            # 1. Project current timestep → (B, k)
            p_t = self.fc_proj(x[:, :, t])   # (B, k)
            history.append(p_t)

            # 2. Read log-spaced taps from history
            #    lag l → history[t - l] if t >= l else zero
            tap_reads = []
            for lag in self.taps:
                idx = t - lag
                if idx >= 0:
                    tap_reads.append(history[idx])
                else:
                    tap_reads.append(zero)

            # 3. Concatenate → (B, k*L), expand → (B, n_hidden), scale
            feat = torch.cat(tap_reads, dim=-1)      # (B, k*L)
            cur  = self.fc_expand(feat)              # (B, n_hidden)
            out_steps.append(cur * self.theta_u)

        return torch.stack(out_steps, dim=-1)        # (B, n_hidden, T)

    def get_info(self) -> dict:
        return {
            "k"      : self.k,
            "L"      : self.L,
            "taps"   : self.taps,
            "storage": "ring",
        }