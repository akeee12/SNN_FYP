# models/power_law_tap_aad_snntorch.py
"""
PowerLawTapAADNetSNNTorch — power-law gamma taps with ring buffer storage.

Ring buffer complexity:
    Write per timestep : O(B * k)           — one slot written
    Rolling alternative: O(B * k * max_lag) — entire buffer shifted
    Saving             : 64x for max_lag=64

All taps always valid — no zero-padding needed for power-law strategy.
taps example (max_lag=64, n=7, gamma=2):
    [1, 2, 7, 16, 28, 44, 64]
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF

from adaptiveAxonalDelayV2SNNTorch import AxonalDelay


# ─────────────────────────────────────────────────────────────────────────────
# POWER LAW RING BUFFER
# ─────────────────────────────────────────────────────────────────────────────

class PowerLawRingBuffer(nn.Module):
    """
    Ring buffer with power-law spaced taps.

    Storage is a circular tensor of depth (max_lag + 1).
    Write pointer advances one slot per timestep using modulo
    arithmetic. No data is ever physically moved in memory.

    Read at lag l:
        slot = (ptr - l) % (max_lag + 1)

    All taps in [1, max_lag] — no zero-padding needed.

    Complexity per timestep:
        Write : O(B * k)           — one slot
        Read  : O(B * k * n_taps)  — n_taps index lookups
        Total : O(B * k)           dominated by write

    Args:
        n_inputs    : input dimension
        n_hidden    : output dimension
        buffer_size : k — bottleneck width
        max_lag     : ring depth = max_lag + 1 slots
        taps        : power-law spaced lag list
        weight_scale: Xavier init multiplier
        theta_u     : current scaling before lif1
    """

    def __init__(
        self,
        n_inputs    : int,
        n_hidden    : int,
        buffer_size : int,
        max_lag     : int,
        taps        : list,
        weight_scale: float = 0.5,
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

        # ── Ring buffer ────────────────────────────────────────────────
        # depth = max_lag + 1 slots so ptr never overwrites unread value
        # shape: (1, k, max_lag+1) — expanded to batch at reset()
        # values never move — only write pointer advances
        self.register_buffer(
            "_ring",
            torch.zeros(1, self.k, max_lag + 1),
        )
        self._ptr = 0

    # ─────────────────────────────────────────────────────────────────
    def reset(self, batch_size: int):
        """
        Zero the ring and reset write pointer.
        Call once per forward pass before the time loop.
        """
        device     = self._ring.device
        self._ring = torch.zeros(
            batch_size, self.k, self.max_lag + 1,
            device=device, dtype=self._ring.dtype,
        )
        self._ptr = 0

    # ─────────────────────────────────────────────────────────────────
    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Process one timestep.
        x_t : (B, n_inputs)
        returns cur : (B, n_hidden)

        Write cost : O(B * k)     — one slot at _ptr
        Read cost  : O(B * k * L) — L index lookups, no copies
        """
        # 1. Project → (B, k)
        p_t = self.fc_proj(x_t)

        # 2. Write at current pointer — nothing else moves
        self._ring[:, :, self._ptr] = p_t

        # 3. Read power-law taps via modulo index arithmetic
        #    lag l → slot (ptr - l) % (max_lag + 1)
        tap_reads = [
            self._ring[:, :, (self._ptr - t) % (self.max_lag + 1)]
            for t in self.taps
        ]

        # 4. Advance pointer
        self._ptr = (self._ptr + 1) % (self.max_lag + 1)

        # 5. Concatenate → (B, k*L), expand → (B, n_hidden), scale
        feat = torch.cat(tap_reads, dim=-1)
        return self.fc_expand(feat) * self.theta_u


# ─────────────────────────────────────────────────────────────────────────────
# FULL NETWORK
# ─────────────────────────────────────────────────────────────────────────────

class PowerLawTapAADNetSNNTorch(nn.Module):
    """
    AAD SNN with power-law gamma taps and ring buffer storage.

    Input  : (B, T, n_inputs)
    Output : (T, B, n_outputs)
    """

    def __init__(self, cfg):
        super().__init__()

        alpha = float(torch.exp(torch.tensor(-1.0 / cfg.tau_s)))
        beta  = float(torch.exp(torch.tensor(-1.0 / cfg.tau_r)))

        # ── Power law ring buffer (replaces fc1) ──────────────────────
        self.tap_buffer = PowerLawRingBuffer(
            n_inputs     = cfg.n_inputs,
            n_hidden     = cfg.n_hidden,
            buffer_size  = cfg.buffer_size,
            max_lag      = cfg.buffer_max_lag,
            taps         = cfg.taps,
            weight_scale = cfg.weight_scale,
            theta_u      = cfg.theta_u,
        )

        # ── Remaining FC layers ───────────────────────────────────────
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

        # ── snn.Synaptic neurons ──────────────────────────────────────
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

        # ── Dropout ───────────────────────────────────────────────────
        dropout_p  = getattr(cfg, "dropout", 0.0)
        self.drop1 = nn.Dropout(p=dropout_p)
        self.drop2 = nn.Dropout(p=dropout_p)

        # ── Axonal delays ─────────────────────────────────────────────
        self.delay1 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)
        self.delay2 = AxonalDelay(cfg.n_hidden, max_delay=cfg.delay_init_cap)

        # ── Loss ──────────────────────────────────────────────────────
        if hasattr(cfg, "target_true"):
            self._rate_true  = float(cfg.target_true)
            self._rate_false = float(cfg.target_false)
        else:
            self._rate_true  = cfg.tgt_count_true  / cfg.time_steps
            self._rate_false = cfg.tgt_count_false / cfg.time_steps

        self._T = cfg.time_steps

    def forward(self, x):
        B, T, _ = x.shape
        self.tap_buffer.reset(B)

        syn1, mem1 = self.lif1.init_synaptic()
        syn2, mem2 = self.lif2.init_synaptic()
        syn3, mem3 = self.lif3.init_synaptic()

        spk1_buf = []
        for t in range(T):
            cur1             = self.tap_buffer.step(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            spk1_buf.append(spk1)

        spk1_t = torch.stack(spk1_buf, dim=2)
        spk1_t = self.drop1(spk1_t)
        spk1_t = self.delay1(spk1_t)

        spk2_buf = []
        for t in range(T):
            cur2             = self.fc2(spk1_t[:, :, t])
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            spk2_buf.append(spk2)

        spk2_t = torch.stack(spk2_buf, dim=2)
        spk2_t = self.drop2(spk2_t)
        spk2_t = self.delay2(spk2_t)

        spk3_buf = []
        for t in range(T):
            cur3             = self.fc3(spk2_t[:, :, t])
            spk3, syn3, mem3 = self.lif3(cur3, syn3, mem3)
            spk3_buf.append(spk3)

        return torch.stack(spk3_buf, dim=0)

    def compute_loss(self, spike_out, labels, cfg=None):
        return SF.mse_count_loss(
            correct_rate   = self._rate_true,
            incorrect_rate = self._rate_false,
        )(spike_out, labels)

    def predict(self, spike_out):
        return spike_out.sum(dim=0).argmax(dim=-1)

    def clamp_delays(self, thea1, thea2):
        self.delay1.clamp_delays(thea1)
        self.delay2.clamp_delays(thea2)

    def get_delay_stats(self):
        return {
            "delay1": self.delay1.get_stats(),
            "delay2": self.delay2.get_stats(),
        }

    def get_buffer_info(self):
        return {
            "k"      : self.tap_buffer.k,
            "L"      : self.tap_buffer.L,
            "taps"   : self.tap_buffer.taps,
            "storage": "ring",
        }