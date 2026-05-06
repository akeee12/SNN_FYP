# configs/axonal_delay_modelPYTORCH_config.py
"""
Configuration for AdaptiveAxonalDelayNetV2 — faithful PyTorch
reproduction of Sun et al. ICASSP 2023.

Verified against complete SLAYER source:
    - slayer.py        : kernels, surrogate, dense layer
    - spikeKernels.h   : getSpikes CUDA, evalRho CUDA
    - shiftKernels.h   : shift CUDA (floor, OOB=0)
    - spikeLoss.py     : van Rossum loss
    - nadam.py         : Nadam optimiser

Neuron params confirmed from AdaptiveAxonalDelayConfig (SLAYER version):
    tauSr=1.0, tauRef=1.0, theta=10, scaleRef=1, tauRho=1, scaleRho=1

Loss params derived from NMNIST yaml (closest available reference):
    tgtSpikeCount {true: 60, false: 10} over tSample=300
    scaled to time_steps by rate preservation:
        true_rate  = 60/300 = 0.200 spikes/timestep
        false_rate = 10/300 = 0.033 spikes/timestep
    tgt_count_true  = round(0.200 * time_steps)
    tgt_count_false = round(0.033 * time_steps)

Theta recalibration:
    SLAYER tauSr=1.0 with Ts=1.0 gives kernel length ~5 timesteps.
    PyTorch conv approximation of PSP produces u_ff max ~7-8
    regardless of weight scale (confirmed from probe: 100x weights
    only moved u_ff from 7.78 to 7.43).
    theta=10 is unreachable in PyTorch approximation with tauSr=1.0.
    Recalibrated to theta=2.0 — sits at same relative percentile
    of u_ff distribution as theta=10 does in SLAYER CUDA.
"""

from configs.base_config import BenchmarkBaseConfig


class AdaptiveAxonalDelayV2Config(BenchmarkBaseConfig):

    model_name = "AdaptiveAxonalDelayNetV2"

    # ── Architecture ──────────────────────────────────────────────────
    n_inputs   = 700
    n_hidden   = 128
    n_outputs  = 20
    time_steps = 400
    Ts         = 1.0

    # ── Neuron parameters — confirmed from Pengfei SLAYER config ──────
    # tauSr=1.0, tauRef=1.0, theta=10, scaleRef=1, tauRho=1, scaleRho=1
    tau_s      = 1.0        # FIX: was 10.0 — SLAYER tauSr=1.0
    tau_r      = 1.0        # tauRef=1.0
    alpha_r    = 1.0        # scaleRef=1
    theta_u    = 10.0       # FIX: was 2.0 — SLAYER theta=10
                            # With tau_s=1.0 the PSP kernel decays in ~5
                            # timesteps. u_ff will reach ~10 with weight_scale=10
    tau_rho    = 1.0        # tauRho=1
    scale_rho  = 1.0        # scaleRho=1

    # ── PSP / refractory kernel length ────────────────────────────────
    # With tau_s=1.0: exp(-t/1.0) < 0.01 at t=5 → kernel_size=32 is safe
    kernel_size = 32

    # ── Adaptive delay scheduler ──────────────────────────────────────
    delay_init_cap      = 64
    delay_adapt_start   = 250
    delay_adapt_window  = 150
    delay_percentile    = 0.95
    delay_expand_margin = 5

    # ── Loss — NumSpikes, scaled from NMNIST yaml ─────────────────────
    # true_rate  = 60/300 = 0.200 → tgt_count_true  = 80
    # false_rate = 10/300 = 0.033 → tgt_count_false = 13
  # ── Loss rates — fixed, scale with time_steps dynamically ─────────
    _true_rate  = 60 / 300    # 0.200
    _false_rate = 10 / 300    # 0.033

    target_true  = 0.8
    target_false = 0.0

    tgt_region_start = 0

    @property
    def tgt_count_true(self):
        return max(1, round(self._true_rate * self.time_steps))

    @property
    def tgt_count_false(self):
        return max(0, round(self._false_rate * self.time_steps))

    @property
    def tgt_region_stop(self):
        return self.time_steps

    # ── Optimiser — Pengfei uses lr=0.1, we keep 0.001 for stability ──
    optimizer     = "nadam"
    lr            = 0.001
    lr_milestones = [300]
    lr_gamma      = 0.5

    # ── Dataloader ────────────────────────────────────────────────────
    batch_size  = 128
    num_workers = 4

    # ── Weight initialisation — confirmed from SLAYER _denseLayer ─────
    weight_scale    = 10.0
    use_weight_norm = True    # FIX: Pengfei uses weight_norm — match it

    # ── Training ──────────────────────────────────────────────────────
    epochs           = 500
    save_checkpoints = True
    checkpoint_every = 50
    