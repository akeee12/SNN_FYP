# configs/registry.py
"""
Model registry: maps string keys → config classes.
Add new models here without touching any other file.

Usage:
    from configs.registry import get_config, list_models

    cfg = get_config("dcls")
    cfg = get_config("adaptive_axonal_delay")
"""

from configs.dcls_config import DCLSConfig
from configs.adaptive_axonal_delay_config import AdaptiveAxonalDelayConfig
from configs.base_config import BenchmarkBaseConfig


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# Add a new model by inserting one line here — nothing else needs to change.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type[BenchmarkBaseConfig]] = {
    "dcls"                  : DCLSConfig,
    "adaptive_axonal_delay" : AdaptiveAxonalDelayConfig,
}


def get_config(model_name: str, **overrides) -> BenchmarkBaseConfig:
    """
    Factory: returns an instantiated config for the given model name.

    Args:
        model_name : Registry key (case-insensitive, hyphens → underscores).
        **overrides: Optional runtime overrides applied after instantiation.
                     e.g. get_config("dcls", epochs=50, batch_size=128)

    Returns:
        Instantiated config object.

    Raises:
        ValueError: If model_name is not in the registry.

    Example:
        cfg = get_config("dcls", epochs=50)
        print(cfg.epochs)   # 50
    """
    key = model_name.lower().replace("-", "_")

    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'.\n"
            f"Available models: {list_models()}"
        )

    cfg = MODEL_REGISTRY[key]()

    # Apply any runtime overrides
    for attr, value in overrides.items():
        if not hasattr(cfg, attr):
            raise AttributeError(
                f"Config '{cfg.model_name}' has no attribute '{attr}'."
            )
        setattr(cfg, attr, value)

    return cfg


def list_models() -> list[str]:
    """Returns all registered model keys."""
    return sorted(MODEL_REGISTRY.keys())