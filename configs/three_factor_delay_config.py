# configs/three_factor_delay_config.py
"""
Three-Factor Delay Learning Rules for SNNs.
Paper : Vassallo & Taherinejad, arXiv:2601.00668, Jan 2026
"Three factor delay learning rules for spiking neural networks"

Key features:
    - LIF neurons with synaptic + axonal delays
    - Three-factor learning rule (eligibility trace + error signal)
    - Gaussian surrogate for spike derivatives
    - Online learning capable
    - Very small model size (6.6x smaller than SOTA)
"""

from configs.base_config import BenchmarkBaseConfig


class ThreeFactorDelayConfig(BenchmarkBaseConfig):

    # ── Identity ──────────────────────────────────────────────────────────
    model_name = "ThreeFactorDelay"

    # ── Architecture ──────────────────────────────────────────────────────
    n_inputs    = 700
    n_hidden    = 128       # small model — paper emphasis on efficiency
    n_outputs   = 20
    time_steps  = 400

    # ── LIF Neuron Parameters ─────────────────────────────────────────────
    tau_mem     = 20.0      # membrane time constant (ms)
    tau_syn     = 5.0       # synaptic time constant (ms)
    v_threshold = 1.0       # firing threshold
    v_reset     = 0.0       # reset potential

    # ── Delay Parameters ──────────────────────────────────────────────────
    # Both synaptic and axonal delays
    max_synaptic_delay = 20   # max synaptic delay (timesteps)
    max_axonal_delay   = 20   # max axonal delay (timesteps)
    delay_init_std     = 2.0  # std for delay initialisation

    # ── Three-Factor Learning Rule ────────────────────────────────────────
    # Factor 1: pre-synaptic eligibility trace
    # Factor 2: post-synaptic eligibility trace
    # Factor 3: top-down error signal (neuromodulatory)
    tau_eligibility = 20.0    # eligibility trace time constant
    learning_rate_w = 0.001   # weight learning rate
    learning_rate_d = 0.01    # delay learning rate (higher than weights)

    # ── Gaussian Surrogate (for eligibility trace) ────────────────────────
    # Paper uses Gaussian not triangular surrogate
    surrogate_sigma = 0.5     # Gaussian width
    surrogate_scale = 1.0     # Gaussian scale

    # ── Optimiser (standard backprop for offline benchmark) ───────────────
    # Note: paper proposes online 3-factor rule
    # We use backprop for fair comparison with other models
    optimizer      = "adam"
    lr             = 0.001
    lr_milestones  = [800]
    lr_gamma       = 0.5

    # ── DataLoader ────────────────────────────────────────────────────────
    batch_size  = 256
    num_workers = 4

    # ── Loss ──────────────────────────────────────────────────────────────
    # CrossEntropy on summed spike counts
    # Same as DCLS for fair comparison

    # ── Regularisation ────────────────────────────────────────────────────
    dropout_p = 0.0    # paper uses small model, no dropout needed