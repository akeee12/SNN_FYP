# models/event_ssm_model.py
"""
EventSSM — PyTorch port of Schöne et al. 2024.
Paper: Scalable Event-by-event Processing of Neuromorphic Sensory
       Signals With Deep State-Space Models (arXiv:2404.18508)

Official implementation: JAX/Flax
    https://github.com/Efficient-Scalable-Machine-Learning/event-ssm

Architecture:
    Input (B, T, 700)
        → Input projection [700 → d_model]
        → N × SSMBlock [d_model, d_state]
        → LayerNorm → Mean pool over T
        → Output head [d_model → 20]
        → (B, 20) raw logits

Fixes from original:
    - B_bar computed once per forward, not inside T loop
    - _print_param_summary removed from __init__
    - Explicit dtype handling for complex-free real diagonal SSM
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# S5 DIAGONAL SSM LAYER
# ─────────────────────────────────────────────────────────────────────────────

class S5SSMLayer(nn.Module):
    """
    Simplified S5 SSM with real diagonal state matrix.

    State equations:
        h(t) = A_bar * h(t-1) + B_bar * x(t)
        y(t) = C * h(t) + D * x(t)

    ZOH discretisation:
        A_bar = exp(A * dt)
        B_bar = (A_bar - 1) / A * B   [elementwise in A, then matmul B]

    Stability: A = -exp(log_A_real) ensures Re(A) < 0 always.

    Args:
        d_model : input/output feature dimension
        d_state : SSM hidden state dimension H
        dt_min  : minimum timescale for dt init
        dt_max  : maximum timescale for dt init
    """

    def __init__(self, d_model: int, d_state: int,
                 dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # ── SSM parameters ────────────────────────────────────────────
        # log_A_real: (H,) — log of |A|, A = -exp(log_A_real) < 0
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(d_state))
        )

        # B: (H, d_model) — input projection into state
        self.B = nn.Parameter(
            torch.randn(d_state, d_model) / math.sqrt(d_model)
        )

        # C: (d_model, H) — state to output projection
        self.C = nn.Parameter(
            torch.randn(d_model, d_state) / math.sqrt(d_state)
        )

        # D: (d_model,) — skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # log_dt: (H,) — learnable timescale per state dimension
        # Log-uniform init between dt_min and dt_max
        log_dt = torch.rand(d_state) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def _get_discretised(self):
        """
        Compute A_bar and B_bar from learnable continuous params.
        Called once per forward pass, not inside the T loop.

        Returns:
            A_bar : (H,)        — discretised diagonal state matrix
            B_bar : (H, d_model) — discretised input matrix
        """
        dt    = torch.exp(self.log_dt)          # (H,) positive
        A     = -torch.exp(self.log_A_real)     # (H,) negative — stable
        A_bar = torch.exp(A * dt)               # (H,) ZOH for A

        # ZOH for B: B_bar = diag((A_bar-1)/A) @ B
        # (A_bar - 1) / A : (H,) elementwise
        # unsqueeze(1)    : (H, 1) for broadcasting with B (H, d_model)
        B_scale = (A_bar - 1.0) / A             # (H,)
        B_bar   = B_scale.unsqueeze(1) * self.B # (H, d_model)

        return A_bar, B_bar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            y: (B, T, d_model)
        """
        B, T, _ = x.shape

        # Discretise once — not inside the loop
        A_bar, B_bar = self._get_discretised()  # (H,), (H, d_model)

        # Initial hidden state
        h       = torch.zeros(B, self.d_state,
                              device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(T):
            x_t = x[:, t, :]                    # (B, d_model)

            # h(t) = A_bar * h(t-1) + B_bar @ x(t)
            # A_bar : (H,)      broadcast over batch
            # x_t @ B_bar.T : (B, d_model) @ (d_model, H) = (B, H)
            h = A_bar * h + x_t @ B_bar.T       # (B, H)

            # y(t) = h(t) @ C.T + D * x(t)
            # h @ C.T : (B, H) @ (H, d_model) = (B, d_model)
            y_t = h @ self.C.T + x_t * self.D   # (B, d_model)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)       # (B, T, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# SSM BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class SSMBlock(nn.Module):
    """
    Full SSM block:
        x → LayerNorm → SSM → GLU gate → out_proj → dropout → residual

    GLU gating: y = SSM(x) * sigmoid(gate_proj(x))
    The gate is computed from the pre-SSM normalised input.
    SSM is linear — GLU provides the nonlinearity.
    """

    def __init__(self, d_model: int, d_state: int,
                 dt_min: float, dt_max: float,
                 dropout: float = 0.1):
        super().__init__()

        self.norm      = nn.LayerNorm(d_model)
        self.ssm       = S5SSMLayer(d_model, d_state, dt_min, dt_max)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj  = nn.Linear(d_model, d_model, bias=True)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            x: (B, T, d_model)
        """
        residual = x
        x_norm   = self.norm(x)

        # SSM pass on normalised input
        ssm_out  = self.ssm(x_norm)                      # (B, T, d_model)

        # GLU gate from normalised input (not SSM output)
        gate     = torch.sigmoid(self.gate_proj(x_norm)) # (B, T, d_model)
        gated    = ssm_out * gate                        # (B, T, d_model)

        # Output projection + dropout + residual
        out      = self.dropout(self.out_proj(gated))    # (B, T, d_model)

        return out + residual


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class EventSSMNet(nn.Module):
    """
    Deep SSM for SHD event streams.
    PyTorch port of Schöne et al. 2024 (arXiv:2404.18508).

    Input : (B, T, 700) — binned SHD spike tensor
    Output: (B, 20)     — raw logits for CrossEntropyLoss
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.d_model  = cfg.d_model
        self.n_layers = cfg.n_layers

        # ── Input projection ──────────────────────────────────────────
        self.input_proj = nn.Linear(cfg.n_inputs, cfg.d_model, bias=False)
        self.input_norm = nn.LayerNorm(cfg.d_model)

        # ── SSM layers ────────────────────────────────────────────────
        self.layers = nn.ModuleList([
            SSMBlock(
                d_model = cfg.d_model,
                d_state = cfg.d_state,
                dt_min  = cfg.dt_min,
                dt_max  = cfg.dt_max,
                dropout = cfg.dropout,
            )
            for _ in range(cfg.n_layers)
        ])

        # ── Output head ───────────────────────────────────────────────
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.n_outputs, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 700)
        Returns:
            logits: (B, 20) — raw logits for CrossEntropyLoss
        """
        x = self.input_norm(self.input_proj(x))  # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)                          # (B, T, d_model)

        x = self.out_norm(x)
        x = x.mean(dim=1)                         # (B, d_model) mean over T
        return self.out_head(x)                   # (B, 20)

    def compute_loss(self, logits: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)