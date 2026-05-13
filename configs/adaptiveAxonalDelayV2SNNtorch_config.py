# configs/aad_v2_snntorch_config.py
"""
Configuration for AdaptiveAxonalDelayNetV2SNNTorch (snnTorch reimplementation).

Changes vs the 12-05_run config:
  - weight_decay=1e-4 added (passed to NAdam, works with weight_norm)
  - dropout=0.2 added (applied after each hidden layer spike tensor)
  - delay_adapt_window kept at 125 — now correctly interpreted as epochs
    because delay_sched.step() is called once per epoch, not per batch
  - delay_adapt_start kept at 250 epochs (unchanged)
  - lr_milestones extended to [20, 50, 100, 200] for a 500-epoch run
"""

from configs.base_config import BenchmarkBaseConfig


class AdaptiveAxonalDelayV2ConfigSNNTorch(BenchmarkBaseConfig):

    model_name = "AdaptiveAxonalDelayNetV2SNNTorch"

    # ── Architecture ──────────────────────────────────────────────────
    n_inputs   = 700
    n_hidden   = 128
    n_outputs  = 20
    time_steps = 400

    # ── Neuron time constants ─────────────────────────────────────────
    # tau_syn=2  → alpha ≈ 0.607
    # tau_mem=10 → beta  ≈ 0.905
    tau_s = 2.0
    tau_r = 10.0

    # ── Neuron threshold ──────────────────────────────────────────────
    threshold     = 0.5
    threshold_out = 2.0

    # ── Weight initialisation ─────────────────────────────────────────
    weight_scale = 0.5

    # ── Surrogate gradient ────────────────────────────────────────────
    surrogate_slope = 25

    # ── Regularisation ───────────────────────────────────────────────
    # FIX 3: dropout after each hidden layer to reduce overfitting.
    dropout = 0.2
    # FIX 4: weight_decay passed to NAdam alongside weight_norm.
    weight_decay = 1e-4

    # ── Delay scheduler ───────────────────────────────────────────────
    delay_init_cap      = 64    # initial max delay (timesteps)
    delay_adapt_start   = 250   # start adapting after this many epochs
    # FIX 2: adapt_window is now in epochs (step() called once/epoch).
    # 125 epochs ≈ Pengfei's 150-batch window in a ~64-batch-per-epoch run.
    delay_adapt_window  = 125
    delay_percentile    = 0.95
    delay_expand_margin = 5

    # ── Loss targets ──────────────────────────────────────────────────
    tgt_count_true  = 80   # recomputed in __init__
    tgt_count_false = 13

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer     = "nadam"
    lr            = 0.001
    lr_milestones = [20, 50, 100, 200]
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