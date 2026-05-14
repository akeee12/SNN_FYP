# configs/event_ssm_config.py
"""
Configuration for EventSSM — PyTorch port of Schöne et al. 2024.
Paper: Scalable Event-by-event Processing of Neuromorphic Sensory
       Signals With Deep State-Space Models
arXiv: 2404.18508

SHD reported accuracy: 95.9% (JAX implementation)
This PyTorch port targets ~93-95% on SHD.

Key differences from AAD model:
    - No spiking neurons — purely recurrent SSM
    - Processes binned spike frames (B, T, 700)
    - S5-style diagonal SSM layers
    - LayerNorm + dropout regularisation
    - Much higher accuracy ceiling than SRM-based models
"""

from configs.base_config import BenchmarkBaseConfig


class EventSSMConfig(BenchmarkBaseConfig):

    model_name = "event_ssm_model"

    # ── Architecture ──────────────────────────────────────────────────
    n_inputs    = 700          # SHD cochlear channels
    n_outputs   = 20           # SHD classes
    time_steps  = 400          # T timesteps
    Ts          = 1.0

    # ── SSM hyperparameters ───────────────────────────────────────────
    d_model     = 128          # model dimension
    d_state     = 64           # SSM state dimension (H in paper)
    n_layers    = 4            # number of SSM layers
    dropout     = 0.1          # dropout rate

    # ── Discretisation ────────────────────────────────────────────────
    dt_min      = 0.001        # minimum delta t
    dt_max      = 0.1          # maximum delta t

    # ── Optimiser ─────────────────────────────────────────────────────
    optimizer     = "adam"
    lr            = 0.001
    lr_milestones = [200, 400]
    lr_gamma      = 0.5

    # ── Dataloader ────────────────────────────────────────────────────
    batch_size  = 64
    num_workers = 4

    # ── Training ──────────────────────────────────────────────────────
    epochs           = 500
    save_checkpoints = True
    checkpoint_every = 50