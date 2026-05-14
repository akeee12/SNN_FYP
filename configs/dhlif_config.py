# configs/dhlif_config.py
"""
DH-LIF (Dendritic Heterogeneity LIF) SNN for SHD.
Paper: Zheng et al. 2024, Nature Communications 15:277
"Temporal dendritic heterogeneity incorporated with spiking
neural networks for learning multi-timescale dynamics"
GitHub: https://github.com/eva1801/DH-SNN

Key features:
    - Multi-compartment LIF: soma + D dendritic branches
    - Each branch has its own learnable timing factor alpha_d
    - Soma has learnable timing factor beta
    - Sparse connection restriction: each branch sees M/D inputs
    - Timing factors constrained to (0,1) via sigmoid
    - Multi-Gaussian surrogate gradient (Eq. 13)
    - DH-SFNN variant (feedforward only) for SHD benchmark

SHD config from Table 1 / Supplementary Table S2:
    Best result: DH-SFNN (2-layer, 8-branch) = 92.1% with 0.05M params
"""

from configs.base_config import BenchmarkBaseConfig


class DHLIFConfig(BenchmarkBaseConfig):

    # ── Identity ──────────────────────────────────────────────────────────
    model_name  = "DHLIF"

    # ── Architecture ──────────────────────────────────────────────────────
    n_inputs    = 700       # SHD raw channels (paper uses 700, dt=1ms)
    n_hidden    = 128       # hidden layer size (0.05M params target)
    n_outputs   = 20        # SHD classes
    time_steps  = 250       # T = Tmax/dt, Tmax~0.25s, dt=1ms
    n_layers    = 2         # DH-SFNN 2-layer (best SHD result)
    n_branches  = 4         # D dendritic branches per neuron

    # ── DH-LIF Neuron Parameters ──────────────────────────────────────────
    # alpha_d: dendritic timing factors — one per branch per neuron
    # beta   : soma timing factor — one per neuron
    # Both constrained to (0,1) via sigmoid (Equation 6)
    # Initialised as raw logit values (before sigmoid)
    # Large init → sigmoid(large) ≈ 0.9+ → slow decay → long memory
    # Medium init → sigmoid(0) = 0.5
    # Paper uses large init for SHD dendritic factors

    # Raw logit init ranges (sigmoid applied in forward)
    alpha_init_mean = 2.0   # sigmoid(2.0) ≈ 0.88 — large init for SHD
    alpha_init_std  = 0.5
    beta_init_mean  = 0.0   # sigmoid(0.0) = 0.5  — medium init
    beta_init_std   = 0.5

    # Fixed firing threshold
    v_threshold = 0.5       # u_th from paper (dimensionless)

    # ── Multi-Gaussian Surrogate (Equation 13) ────────────────────────────
    # H' = gamma*(1+h)*N(u|u_th, sigma^2)
    #      - gamma*h*N(u|sigma, (s*sigma)^2)
    #      - gamma*h*N(u|-sigma, (s*sigma)^2)
    surrogate_gamma = 0.5
    surrogate_h     = 0.15
    surrogate_sigma = 0.5
    surrogate_s     = 6.0

    # ── Sparse Connection Restriction ─────────────────────────────────────
    # Each branch connects to M/D inputs (non-overlapping)
    # Equation 10: sum_d |P_d| = M, union = {0..M-1}, |P_d| ≈ M/D
    # This keeps total params same as vanilla SNN

    # ── Readout ───────────────────────────────────────────────────────────
    # SHD: y_i = sum_t softmax(u_out[t])  (Experimental setting section)
    readout_mode = "sum_softmax"   # sum softmax over T then NLLLoss

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer   = "adam"
    lr          = 0.01      # initial lr (Methods: Adam lr=1e-2)

    # Step LR scheduler (Methods: step learning rate scheduler)
    lr_scheduler  = "step"
    lr_step_size  = 20      # decay every 20 epochs
    lr_gamma      = 0.5     # decay factor

    # ── Regularisation ────────────────────────────────────────────────────
    dropout_p   = 0.0       # paper does not mention dropout for SHD

    # ── DataLoader ────────────────────────────────────────────────────────
    batch_size  = 256       # from Supplementary Table S2
    num_workers = 4

    # ── Training ──────────────────────────────────────────────────────────
    epochs      = 200       # paper trains until convergence