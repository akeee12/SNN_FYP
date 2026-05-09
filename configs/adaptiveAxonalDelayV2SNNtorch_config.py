from configs.base_config import BenchmarkBaseConfig

class AdaptiveAxonalDelayV2ConfigSNNTorch(BenchmarkBaseConfig):

    model_name = "AdaptiveAxonalDelayNetV2SNNTorch"

    # ── Architecture ──────────────────────────────────────────────────
    n_inputs   = 700
    n_hidden   = 128
    n_outputs  = 20
    time_steps = 400
    Ts         = 1.0

    # ── Neuron parameters — confirmed from Pengfei SLAYER config ──────
    # Pengfei: tauSr=1.0, tauRef=1.0, theta=10, scaleRef=1
    # snnTorch Synaptic uses alpha=exp(-1/tau_s), beta=exp(-1/tau_r)
    tau_s      = 1.0
    tau_r      = 1.0
    theta_u    = 10.0
    alpha_r    = 1.0
    tau_rho    = 1.0
    scale_rho  = 1.0

    # ── Delay scheduler ───────────────────────────────────────────────
    delay_init_cap      = 64
    delay_adapt_start   = 250
    delay_adapt_window  = 150
    delay_percentile    = 0.95
    delay_expand_margin = 5

    # ── Loss ──────────────────────────────────────────────────────────
    tgt_count_true  = 80
    tgt_count_false = 13

    # ── Optimiser — Pengfei uses lr=0.1 ──────────────────────────────
    optimizer     = "nadam"
    lr            = 0.1
    lr_milestones = [800]
    lr_gamma      = 0.5

    # ── Dataloader ────────────────────────────────────────────────────
    batch_size  = 128
    num_workers = 4

    # ── Weight init ───────────────────────────────────────────────────
    weight_scale    = 10.0
    use_weight_norm = True

    # ── Training ──────────────────────────────────────────────────────
    epochs           = 500
    save_checkpoints = True
    checkpoint_every = 50

    def __init__(self):
        self.tgt_count_true  = max(1, round(0.200 * self.time_steps))
        self.tgt_count_false = max(0, round(0.033 * self.time_steps))