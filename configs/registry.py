# configs/registry.py
"""
Model registry: maps string keys → config classes.
Add new models here without touching any other file.

Usage:
    from configs.registry import get_config, list_models

    cfg = get_config("dcls")
    cfg = get_config("adaptive_axonal_delay")
"""
from typing import Dict, List, Type

from configs.dcls_config import DCLSConfig
from configs.adaptive_axonal_delay_config import AdaptiveAxonalDelayConfig
from configs.base_config import BenchmarkBaseConfig
from configs.axonal_delay_modelPYTORCH_config import AdaptiveAxonalDelayPyTorchConfig
from configs.log_buffer_aad_config import LogBufferAADConfig
from configs.three_factor_delay_config import ThreeFactorDelayConfig
from configs.adaptiveAxonalDelayV2_config import AdaptiveAxonalDelayV2Config
from configs.log_buffer_aad_configv1 import LogBufferAADConfigv1
from configs.adaptiveAxonalDelayV2SNNtorch_config import AdaptiveAxonalDelayV2ConfigSNNTorch
from configs.log_buffer_snntorch_config import LogBufferSNNTorchConfig
from configs.cadlif_config import cAdLIFConfig
from configs.dhlif_config import DHLIFConfig
from configs.ssmConfig import EventSSMConfig
from configs.linear_tap_snntorch_config import LinearTapSNNTorchConfig
from configs.exponential_tap_snntorch_config import ExponentialTapSNNTorchConfig
from configs.random_tap_snntorch_config import RandomTapSNNTorchConfig
from configs.power_law_tap_snntorch_config import PowerLawTapSNNTorchConfig
from configs.log_ring_buffer_snntorch_config import LogRingBufferSNNTorchConfig
from configs.dcls_log_ring_buffer_config import DCLSLogRingBufferConfig

# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "dcls"                  : DCLSConfig,
    "adaptive_axonal_delay" : AdaptiveAxonalDelayConfig,
    "aad_pytorch"           : AdaptiveAxonalDelayPyTorchConfig, # PyTorch version
    "aad_log_buffer"        : LogBufferAADConfig,
    "three_factor_delay"    : ThreeFactorDelayConfig,
    "aad_v2"     : AdaptiveAxonalDelayV2Config,
    "log_buffer" : LogBufferAADConfigv1,
    "aad_snntorch" : AdaptiveAxonalDelayV2ConfigSNNTorch,
    "log_buffer_snntorch"   : LogBufferSNNTorchConfig,
    "cadlif": cAdLIFConfig,
    "dhlif": DHLIFConfig,
    "event_ssm": EventSSMConfig,
    "linear_tap_snntorch": LinearTapSNNTorchConfig,
    "exponential_tap_snntorch": ExponentialTapSNNTorchConfig,
    "random_tap_snntorch": RandomTapSNNTorchConfig,
    "power_law_tap_snntorch": PowerLawTapSNNTorchConfig,
    "log_ring_buffer_snntorch": LogRingBufferSNNTorchConfig,
    "dclslogringbuffer":DCLSLogRingBufferConfig

}

#
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


def list_models():
    """Returns all registered model keys."""
    return sorted(MODEL_REGISTRY.keys())