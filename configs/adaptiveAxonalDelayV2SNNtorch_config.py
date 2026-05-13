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

    # ── Neuron — match Pengfei SLAYER config exactly ──────────────────
    tau_s = 2.0   # alpha = 0.607  ← must be > beta
    tau_r = 1.0   # beta  = 0.368  ← must be < alpha
    threshold     = 10.0
    threshold_out = 10.0
    weight_scale  = 10.0


    # ── Surrogate ─────────────────────────────────────────────────────
    surrogate_slope = 25

    # ── Regularisation ────────────────────────────────────────────────
    dropout      = 0.0     # none — Pengfei uses none
    weight_decay = 0.0     # none — Pengfei uses none

    # ── Delay scheduler — match Pengfei exactly ───────────────────────
    delay_init_cap      = 64
    delay_adapt_start   = 250
    delay_adapt_window  = 150   # Pengfei: update > 150 batches
    delay_expand_margin = 5

    # ── Loss ──────────────────────────────────────────────────────────
    tgt_count_true  = 80
    tgt_count_false = 13

    # ── Optimiser — match Pengfei exactly ────────────────────────────
    optimizer     = "nadam"
    lr            = 0.1        # Pengfei lr=0.1
    lr_milestones = [800]      # Pengfei milestone=800
    lr_gamma      = 0.5
    epochs        = 1000       # Pengfei trains 1000 epochs

    # ── Dataloader ────────────────────────────────────────────────────
    batch_size  = 128
    num_workers = 4

    # ── Training ──────────────────────────────────────────────────────
    save_checkpoints = True
    checkpoint_every = 50

    def __init__(self):
        self.tgt_count_true  = max(1, round(0.200 * self.time_steps))
        self.tgt_count_false = max(0, round(0.033 * self.time_steps))