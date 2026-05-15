# models/logBufferAADNetSNNTorch.py
"""
LogBufferAADNetSNNTorch — Log-spaced temporal buffer front-end
grafted onto the AdaptiveAxonalDelayNetV2SNNTorch backbone.

Architecture
────────────
Input (B, T, 700)
    │
    ▼  [LogBuffer]
    fc_proj   : Linear(700, k,   bias=False)  — shared spatial bottleneck
    history   : rolling window of depth max_lag, stores last max_lag
                projections. index 0 = most recent, index max_lag-1 = oldest.
    lags      : {1, 2, 4, …, buffer_max_lag}  — L taps, log-spaced
    fc_expand : Linear(k*L, n_hidden, bias=False) — temporal expansion
    × theta_u — current scaling (prevents lif1 saturation)
    │
    ▼  lif1 (snn.Synaptic)
    AxonalDelay 1
    │
    ▼  fc2 → lif2 → AxonalDelay 2
    │
    ▼  fc3 → lif3
    │
Output (T, B, n_outputs)

Buffer implementation — simple rolling window:
    _history : (B, k, max_lag) tensor.
    Each step:
      1. torch.roll shifts all stored projections one slot older.
      2. newest projection written at index 0.
      3. lag l is read directly at index l-1.
    Semantically identical to a ring buffer but index arithmetic
    is trivial (lag-1) and the tensor size is minimal (no +1 slot).

Weight norm + Xavier init:
    Applied to fc_proj, fc_expand, fc2, fc3.
    weight_v scaled by cfg.weight_scale (default 0.5).

theta_u:
    Scalar multiplier on fc_expand output before lif1.
    Calibrated to keep lif1 out of saturation given the larger
    u_ff produced by the two-stage projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import functional as SF

from adaptiveAxonalDelayV2SNNTorch import (
    DelayGradient,
    AxonalDelay,
)


# ─────────────────────────────────────────────────────────────────────────────
# LOG-SPACED TEMPORAL BUFFER — simple rolling window
# ─────────────────────────────────────────────────────────────────────────────

class LogBuffer(nn.Module):
    """
    Log-spaced temporal buffer — simple rolling window implementation.

    At each timestep t:
      1. Project x_t ∈ R^{n_inputs} → p_t ∈ R^k  via fc_proj.
      2. Roll history one slot older, write p_t at index 0.
      3. Read L taps at log-spaced lags {1,2,4,…,max_lag}
         directly as history[:, :, lag-1].
      4. Concatenate taps → (B, k*L), project → n_hidden via fc_expand.
      5. Scale by theta_u → current for lif1.

    Args:
        n_inputs    : input dimension (700 for SHD)
        n_hidden    : output dimension (128)
        buffer_size : k — bottleneck width
        max_lag     : maximum lag; lags = {1,2,4,…,max_lag} → L=log2+1
        weight_scale: multiplier on Xavier-init weight_v (default 0.5)
        theta_u     : current scaling before lif1 (default 0.5)
    """

    def __init__(
        self,
        n_inputs    : int,
        n_hidden    : int,
        buffer_size : int,
        max_lag     : int,
        weight_scale: float = 0.5,
        theta_u     : float = 0.5,
    ):
        super().__init__()

        self.k       = buffer_size
        self.max_lag = max_lag
        self.theta_u = theta_u

        # log-spaced lags: {1, 2, 4, ..., max_lag}
        # n_levels = log2(max_lag), e.g. log2(64) = 6
        # lags     = [1, 2, 4, 8, 16, 32, 64] → L = 7
        n_levels   = int(math.log2(max_lag))
        self.lags  = [2 ** i for i in range(n_levels + 1)]
        self.L     = len(self.lags)

        # ── Learnable projections ──────────────────────────────────────
        self.fc_proj = nn.utils.weight_norm(
            nn.Linear(n_inputs, self.k, bias=False), name="weight"
        )
        self.fc_expand = nn.utils.weight_norm(
            nn.Linear(self.k * self.L, n_hidden, bias=False), name="weight"
        )
        for fc in (self.fc_proj, self.fc_expand):
            nn.init.xavier_uniform_(fc.weight_v)
            with torch.no_grad():
                fc.weight_v.mul_(weight_scale)

        # ── History buffer ─────────────────────────────────────────────
        # Shape: (1, k, max_lag) — expanded to batch size at reset().
        # index 0 = most recent projection
        # index max_lag-1 = oldest projection
        # Registered as a buffer so it moves with .to(device) but
        # is not a trainable parameter.
        self.register_buffer(
            "_history",
            torch.zeros(1, self.k, max_lag),
        )

    # ─────────────────────────────────────────────────────────────────
    def reset(self, batch_size: int):
        """
        Zero the history and prepare for a new sequence.
        Must be called once per forward pass before the time loop.
        """
        device = self._history.device
        self._history = torch.zeros(
            batch_size, self.k, self.max_lag,
            device=device, dtype=self._history.dtype,
        )

    # ─────────────────────────────────────────────────────────────────
    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Process one timestep.

        Args:
            x_t : (B, n_inputs)
        Returns:
            cur : (B, n_hidden) — scaled current for lif1
        """
        # 1. Spatial projection → (B, k)
        p_t = self.fc_proj(x_t)

        # 2. Shift all stored projections one slot older
        #    slot 0 (newest) ← slot 1, slot 1 ← slot 2, ...
        #    then overwrite slot 0 with the new projection.
        #    After this: history[:, :, lag-1] = projection from lag steps ago.
        self._history = torch.roll(self._history, shifts=1, dims=2)
        self._history[:, :, 0] = p_t

        # 3. Read log-spaced taps directly by index
        #    lag=1 → index 0  (projection from 1 step ago)
        #    lag=2 → index 1  (projection from 2 steps ago)
        #    lag=4 → index 3  (projection from 4 steps ago)
        #    lag=N → index N-1
        taps = [self._history[:, :, lag - 1] for lag in self.lags]

        # 4. Concatenate → (B, k*L), expand → (B, n_hidden)
        feat = torch.cat(taps, dim=-1)
        cur  = self.fc_expand(feat)

        # 5. Scale to prevent lif1 saturation
        return cur * self.theta_u


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class LogBufferAADNetSNNTorch(nn.Module):
    """
    Adaptive Axonal Delay SNN with log-spaced temporal buffer
    replacing the plain fc1 layer.

    Identical to AdaptiveAxonalDelayNetV2SNNTorch except:
      • fc1 replaced by LogBuffer (fc_proj + rolling history + fc_expand).
      • compute_loss uses target_true/target_false rates directly.
      • theta_u applied inside LogBuffer.step().

    Input  : (B, T, 700)
    Output : (T, B, 20)
    """

    def __init__(self, cfg):
        super().__init__()

        alpha = float(torch.exp(torch.tensor(-1.0 / cfg.tau_s)))
        beta  = float(torch.exp(torch.tensor(-1.0 / cfg.tau_r)))

        # ── Log buffer (replaces fc1) ─────────────────────────────────
        self.log_buffer = LogBuffer(
            n_inputs     = cfg.n_inputs,
            n_hidden     = cfg.n_hidden,
            buffer_size  = cfg.buffer_size,
            max_lag      = cfg.buffer_max_lag,
            weight_scale = cfg.weight_scale,
            theta_u      = cfg.theta_u,
        )

        # ── Remaining FC layers — same weight_norm policy as V2 ───────
        self.fc2 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden, cfg.n_hidden,  bias=False), name="weight"
        )
        self.fc3 = nn.utils.weight_norm(
            nn.Linear(cfg.n_hidden, cfg.n_outputs, bias=False), name="weight"
        )
        for fc in (self.fc2, self.fc3):
            nn.init.xavier_uniform_(fc.weight_v)
            with torch.no_grad():
                fc.weight_v.mul_(cfg.weight_scale)

        # ── snn.Synaptic neurons — identical policy to V2 ─────────────
        surrogate = snn.surrogate.fast_sigmoid(slope=cfg.surrogate_slope)

        self.lif1 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )
        self.lif2 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )
        self.lif3 = snn.Synaptic(
            alpha=alpha, beta=beta,
            threshold=cfg.threshold_out,
            learn_threshold=False,
            spike_grad=surrogate,
            reset_mechanism="subtract",
        )

        # ── Dropout — off by default ───────────────────────────────────
        dropout_p = getattr(cfg, "dropout", 0.0)
        self.drop1 = nn.Dropout(p=dropout_p)
        self.drop2 = nn.Dropout(p=dropout_p)

        # ── Axonal delays — unchanged from V2 ─────────────────────────
        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

        # ── Loss targets — rates, not counts ──────────────────────────
        if hasattr(cfg, "target_true"):
            self._rate_true  = float(cfg.target_true)
            self._rate_false = float(cfg.target_false)
        else:
            T = cfg.time_steps
            self._rate_true  = cfg.tgt_count_true  / T
            self._rate_false = cfg.tgt_count_false / T

        self._T = cfg.time_steps

    # ─────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, T, n_inputs)
        out : (T, B, n_outputs)
        """
        B, T, _ = x.shape

        # Reset history buffer for this new batch
        self.log_buffer.reset(B)

        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        # ── Layer 1: input → hidden via LogBuffer ─────────────────────
        spk1_buf = []
        for t in range(T):
            cur1             = self.log_buffer.step(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            spk1_buf.append(spk1)

        spk1_t = torch.stack(spk1_buf, dim=2)   # (B, n_hidden, T)
        spk1_t = self.drop1(spk1_t)
        spk1_t = self.delay1(spk1_t)

        # ── Layer 2: hidden → hidden ──────────────────────────────────
        spk2_buf = []
        for t in range(T):
            cur2             = self.fc2(spk1_t[:, :, t])
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk2_buf.append(spk2)

        spk2_t = torch.stack(spk2_buf, dim=2)   # (B, n_hidden, T)
        spk2_t = self.drop2(spk2_t)
        spk2_t = self.delay2(spk2_t)

        # ── Layer 3: hidden → output ──────────────────────────────────
        spk3_buf = []
        for t in range(T):
            cur3             = self.fc3(spk2_t[:, :, t])
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            spk3_buf.append(spk3)

        return torch.stack(spk3_buf, dim=0)   # (T, B, n_outputs)

    # ─────────────────────────────────────────────────────────────────
    def compute_loss(self, spike_out: torch.Tensor,
                     labels: torch.Tensor, cfg=None):
        loss_fn = SF.mse_count_loss(
            correct_rate   = self._rate_true,
            incorrect_rate = self._rate_false,
        )
        return loss_fn(spike_out, labels)

    def predict(self, spike_out: torch.Tensor) -> torch.Tensor:
        return spike_out.sum(dim=0).argmax(dim=-1)

    def clamp_delays(self, thea1: float, thea2: float):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self) -> dict:
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }

    def get_buffer_info(self) -> dict:
        return {
            "k"   : self.log_buffer.k,
            "L"   : self.log_buffer.L,
            "lags": self.log_buffer.lags,
        }