# configs/aad_v2_snntorch_config.py
"""
Configuration for AdaptiveAxonalDelayNetV2SNNTorch — 500-epoch production run.

WHY NOT snn.Alpha (the 14-05 100-epoch attempt):
  That run collapsed to train=49%, test=26% at epoch 100. Two root causes:
    1. weight_g.fill_(10.0) → effective weight magnitude ~10x too large,
       neurons over-excited and die immediately.
    2. tau_r=1.0 → Alpha kernel decays in ~1 timestep → no temporal
       integration → membrane never reaches threshold=1.0.
  snn.Synaptic with tau_s=2, tau_r=10, threshold=0.5 is confirmed working
  (reached 69.7% test acc at epoch 20 in the 12-05 run).

REGULARISATION — off to match Pengfei:
  dropout=0.0 and weight_decay=0.0. If train/test gap reappears after
  epoch 30, re-enable with dropout=0.1.

DELAY SCHEDULER:
  adapt_window=150 → scheduler checks for cap expansion every 150 epochs
  (called once per epoch in train_aadv2.py).
  adapt_start=250  → cap is frozen for the first 250 epochs.

LR = 0.01:
  Pengfei uses 0.1 tuned for SLAYER. NAdam + Synaptic is unstable at 0.1.
  0.01 gave clean convergence to 69.7% in the prior run.
  Single milestone at 400 gives one decay in the second half.
"""

from configs.base_config import BenchmarkBaseConfig


class AdaptiveAxonalDelayV2ConfigSNNTorch(BenchmarkBaseConfig):

    model_name = "AdaptiveAxonalDelayNetV2SNNTorch"

    # ── Architecture ──────────────────────────────────────────────────
    n_inputs   = 700
    n_hidden   = 128
    n_outputs  = 20
    time_steps = 400

    # ── Neuron — snn.Synaptic (confirmed working, do NOT change to Alpha)
    # alpha = exp(-1/tau_s) = exp(-0.5) ≈ 0.607  synaptic current
    # beta  = exp(-1/tau_r) = exp(-0.1) ≈ 0.905  membrane (~10-step window)
    tau_s = 2.0
    tau_r = 10.0

    threshold     = 0.5   # reachable with weight_scale=0.5 + Xavier
    threshold_out = 2.0   # output fires more selectively

    # ── Weight initialisation ─────────────────────────────────────────
    # Multiplier on Xavier-init weight_v direction (not weight_g).
    # Keep at 0.5 — setting this to 10.0 causes weight_g=10 and
    # massively over-excites neurons at init (broke the 14-05 run).
    weight_scale = 0.5

    # ── Surrogate gradient ────────────────────────────────────────────
    surrogate_slope = 25

    # ── Regularisation — off to match Pengfei ────────────────────────
    dropout      = 0.0   # enable 0.1 if train/test gap reappears
    weight_decay = 0.0   # enable 1e-4 if overfitting returns

    # ── Delay scheduler ───────────────────────────────────────────────
    delay_init_cap      = 64    # initial delay cap (timesteps)
    delay_adapt_start   = 250   # freeze cap for first N epochs
    delay_adapt_window  = 150   # check for expansion every N epochs
    delay_expand_margin = 5     # expand cap if p84 > (cap - margin)

    # ── Loss targets ──────────────────────────────────────────────────
    tgt_count_true  = 80   # recomputed in __init__
    tgt_count_false = 13

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer     = "nadam"
    lr            = 0.01
    lr_milestones = [400]   # single decay in the second half of training
    lr_gamma      = 0.5

    # ── Dataloader ────────────────────────────────────────────────────
    batch_size  = 128
    num_workers = 4

    # ── Training ──────────────────────────────────────────────────────
    epochs           = 500
    save_checkpoints = True
    checkpoint_every = 50

    def __init__(self):
        self.tgt_count_true  = max(1, round(0.200 * self.time_steps))  # 80
        self.tgt_count_false = max(0, round(0.033 * self.time_steps))  # 13