# configs/aad_v2_snntorch_config.py
"""
Configuration for AdaptiveAxonalDelayNetV2SNNTorch (snnTorch reimplementation).
 
Key differences from the broken attempt:
  - tau_syn / tau_mem renamed and set to biologically fast values
    (tau=20 → alpha=0.951 caused near-perfect integration = dead neurons)
  - threshold raised to 1.0 but weight_scale lowered to match
  - surrogate_slope exposed for tuning
  - delay_adapt_start lowered (250 was never reached in 100-epoch run)
  - lr lowered from 0.1 to 0.01 (0.1 was too aggressive for Synaptic neurons)
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
    # snnTorch Synaptic:
    #   alpha = exp(-1/tau_syn)   synaptic current
    #   beta  = exp(-1/tau_mem)   membrane potential
    #
    # Pengfei SLAYER tauSr=1ms, tauRef=1ms with dt=1ms means very fast decay.
    # alpha=exp(-1/1)=0.368, beta=exp(-1/1)=0.368.
    # We use slightly longer membrane tau for stability, but keep both fast.
    #
    # tau_syn=2  → alpha=0.607  (synaptic current, matches fast EPSP rise)
    # tau_mem=10 → beta =0.905  (membrane, gives ~10-step integration window)
    #
    # If neurons still don't fire, reduce tau_mem toward 5 or lower threshold.
    tau_s = 2.0    # alpha ≈ 0.607
    tau_r = 10.0   # beta  ≈ 0.905
 
    # ── Neuron threshold ──────────────────────────────────────────────
    # Must be reachable given weight_scale and input sparsity.
    # SHD inputs: ~700 channels, typically 5-20 active per step.
    # fc1 output ≈ weight_scale * sqrt(n_active/n_inputs) * n_active
    # With weight_scale=0.5, Xavier init, ~10 active inputs:
    #   cur ≈ 0.5 * (10/700)^0.5 * 700 * 10/700 ≈ 0.5 * 0.12 * 10 = 0.6
    # So threshold=0.5 is reachable; adjust together with weight_scale.
    threshold = 0.5
    threshold_out = 2.0    # output layer fires more selectively
 
    # ── Weight initialisation ─────────────────────────────────────────
    # Applied as a multiplier on top of Xavier uniform.
    # Keep modest — weight_norm + high lr caused the collapse.
    weight_scale = 0.5
 
    # ── Surrogate gradient ────────────────────────────────────────────
    surrogate_slope = 25    # fast_sigmoid slope; increase if learning is slow
 
    # ── Delay scheduler ───────────────────────────────────────────────
    delay_init_cap      = 64    # initial max delay (timesteps)
    delay_adapt_start   = 125   # start adapting caps after this many epochs
    delay_adapt_window  = 75    # check every N delay_sched.step() calls
    delay_percentile    = 0.95
    delay_expand_margin = 5     # expand cap if p95 > (cap - margin)
 
    # ── Loss targets ──────────────────────────────────────────────────
    # correct_rate   = tgt_count_true  / time_steps = 80/400  = 0.20
    # incorrect_rate = tgt_count_false / time_steps = 13/400  = 0.033
    # These match Pengfei's spike-count targets.
    tgt_count_true  = 80   # re-computed in __init__ but left explicit here
    tgt_count_false = 13
 
    # ── Optimiser ─────────────────────────────────────────────────────
    # NAdam with lr=0.01 is much more stable than 0.1 for Synaptic neurons.
    # If training is slow, try 0.02. Pengfei's lr=0.1 is tuned for SLAYER.
    optimizer     = "nadam"
    lr            = 0.001
    lr_milestones = [20, 50, 100]   # decay at these epochs (for 500-epoch run)
    lr_gamma      = 0.5
 
    # ── Dataloader ────────────────────────────────────────────────────
    batch_size  = 128
    num_workers = 4
 
    # ── Training ──────────────────────────────────────────────────────
    epochs           = 500
    save_checkpoints = True
    checkpoint_every = 50
 
    def __init__(self):
        # Recompute count targets from time_steps to keep them consistent
        # if time_steps is ever changed.
        self.tgt_count_true  = max(1,  round(0.200 * self.time_steps))  # 80
        self.tgt_count_false = max(0,  round(0.033 * self.time_steps))  # 1