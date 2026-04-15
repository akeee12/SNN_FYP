# configs/base_config.py
"""
Base benchmark configuration.
All model-specific configs inherit from BenchmarkBaseConfig.
Ensures fair comparison: same dataset, same epochs, same seed, same device.
"""


class BenchmarkBaseConfig:
    """
    Shared benchmark settings inherited by all model configs.

    Any attribute can be overridden in a child config class.
    Example:
        class MyModelConfig(BenchmarkBaseConfig):
            batch_size = 64   # overrides the base value of 256
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    model_name: str = "base"           # overridden by each child config

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset       = "shd"
    datasets_path = "data/SHD"         # HPC: adjust to $EPHEMERAL or /vol/bitbucket
    n_outputs     = 20                 # SHD has 20 classes

    # ── Training (fair-comparison knobs — same for ALL models) ────────────────
    epochs        = 100
    batch_size    = 256
    seed          = 42

    # ── Hardware ──────────────────────────────────────────────────────────────
    device        = "cuda"
    num_workers   = 4                  # safe default for HPC nodes

    # ── Logging / Output ──────────────────────────────────────────────────────
    results_dir       = "results"
    save_checkpoints  = True
    checkpoint_every  = 10             # save every N epochs

    # ── Profiling Toggles ─────────────────────────────────────────────────────
    measure_memory    = True           # GPU memory measurement
    measure_time      = True           # wall-clock timing per epoch

    def __repr__(self) -> str:
        attrs = {
            k: v for k, v in self.__class__.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
        lines = [f"  {k}: {v}" for k, v in attrs.items()]
        return f"{self.__class__.__name__}(\n" + "\n".join(lines) + "\n)"