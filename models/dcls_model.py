# models/dcls_model.py
"""
DCLS (Dilated Convolutions with Learnable Spacings) SNN for SHD.
Paper : Hammouamri et al., ICLR 2024
Repo  : https://github.com/Thvnvtos/SNN-delays

Uses official DCLS library.
Fixes:
    1. Double fc_out bug removed
    2. BN placed after LIF (matches repo)
    3. Dropout placed after BN
    4. Sigma updated via.data.fill_()
    5. Delay params correctly separated for dual optimiser
    6. lif_out removed — mean over T + FC readout
    7. v_threshold=0.1 — tuned for real SHD input scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate
from DCLS.construct.modules import Dcls1d


class DCLSNet(nn.Module):
    """
    DCLS SNN for SHD.

    Architecture:
        Input  (B, 700, T)
            → Bin        : 700 → 140 (n_bins=5)
            → FC_in      : 140 → 256
            → DCLS1      : Dcls1d learnable delay conv
            → LIF1       : spiking neuron (v_threshold=0.1)
            → BN1        : batch norm
            → Dropout1
            → DCLS2      : Dcls1d learnable delay conv
            → LIF2       : spiking neuron
            → BN2
            → Dropout2
            → mean(T)    : average spike counts over time
            → FC_out     : 256 → 20 raw logits
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg    = cfg
        self.n_bins = cfg.n_bins      # 5
        self.n_in   = cfg.n_inputs    # 140

        # ── Input projection ──────────────────────────────────────────────
        self.fc_in = nn.Linear(
            self.n_in,
            cfg.n_hidden_neurons,
            bias=cfg.bias,
        )

        # ── DCLS delay layers ─────────────────────────────────────────────
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

        # ── Initialise delay positions ────────────────────────────────────
        nn.init.uniform_(self.dcls1.P, cfg.init_pos_a, cfg.init_pos_b)
        nn.init.uniform_(self.dcls2.P, cfg.init_pos_a, cfg.init_pos_b)

        # ── Set initial sigma ─────────────────────────────────────────────
        self.dcls1.SIG.data.fill_(cfg.sigInit)
        self.dcls2.SIG.data.fill_(cfg.sigInit)

        # ── LIF neurons ───────────────────────────────────────────────────
        self.lif1 = neuron.LIFNode(
            tau                = cfg.init_tau,
            v_threshold        = cfg.v_threshold,   # 0.1
            surrogate_function = surrogate.ATan(alpha=cfg.alpha),
            detach_reset       = cfg.detach_reset,
            step_mode          = 'm',
        )
        self.lif2 = neuron.LIFNode(
            tau                = cfg.init_tau,
            v_threshold        = cfg.v_threshold,   # 0.1
            surrogate_function = surrogate.ATan(alpha=cfg.alpha),
            detach_reset       = cfg.detach_reset,
            step_mode          = 'm',
        )

        # ── Batch norm ────────────────────────────────────────────────────
        self.bn1 = nn.BatchNorm1d(cfg.n_hidden_neurons) \
                   if cfg.use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(cfg.n_hidden_neurons) \
                   if cfg.use_batchnorm else nn.Identity()

        # ── Dropout ───────────────────────────────────────────────────────
        self.drop1 = nn.Dropout(cfg.dropout_p)
        self.drop2 = nn.Dropout(cfg.dropout_p)

        # ── Output FC ─────────────────────────────────────────────────────
        self.fc_out = nn.Linear(
            cfg.n_hidden_neurons,
            cfg.n_outputs,
            bias=cfg.bias,
        )

        # ── Sigma annealing state ─────────────────────────────────────────
        self.current_epoch       = 0
        self.final_epoch         = cfg.final_epoch
        self.sigInit             = cfg.sigInit
        self.decrease_sig_method = cfg.decrease_sig_method

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

    def get_delay_params(self):
        """Delay position parameters for separate optimiser."""
        return [self.dcls1.P, self.dcls2.P]

    def get_weight_params(self):
        """All non-delay parameters for weight optimiser."""
        delay_ids = {id(self.dcls1.P), id(self.dcls2.P)}
        return [p for p in self.parameters()
                if id(p) not in delay_ids]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 700, T) — after permute in train.py
        Returns:
            out: (B, 20) — raw logits for CrossEntropy
        """
        B, C, T = x.shape

        # FIXED — only bin if input is raw 700 channels
        if x.shape[1] == self.n_bins * self.n_in:
            x = x.reshape(B, self.n_in, self.n_bins, T)
            x = x.sum(dim=2).float()
        else:
            x = x.float()

        # ── FC input projection ───────────────────────────────────────────
        x = self.fc_in(
            x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, 256, T)

        # ── DCLS layer 1 ──────────────────────────────────────────────────
        x = self.dcls1(x)                           # (B, 256, T)
        x = self.lif1(x.permute(2, 0, 1))          # (T, B, 256)
        x = x.permute(1, 2, 0)                     # (B, 256, T)
        x = self.bn1(x)
        x = self.drop1(x)

        # ── DCLS layer 2 ──────────────────────────────────────────────────
        x = self.dcls2(x)                           # (B, 256, T)
        x = self.lif2(x.permute(2, 0, 1))          # (T, B, 256)
        x = x.permute(1, 2, 0)                     # (B, 256, T)
        x = self.bn2(x)
        x = self.drop2(x)

        # ── Readout: mean over T then FC ──────────────────────────────────
        x = x.mean(dim=-1)                          # (B, 256)
        x = self.fc_out(x)                          # (B, 20) raw logits
        return x

    def reset(self):
        """Reset LIF neuron states."""
        self.lif1.reset()
        self.lif2.reset()


def build_dcls_model(cfg) -> DCLSNet:
    """Factory — builds DCLSNet from cfg."""
    return DCLSNet(cfg)