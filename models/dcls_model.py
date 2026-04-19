# models/dcls_model.py
"""
DCLS model wrapper.
Imports SNNDelays directly from SNN-delays submodule.
Paper : Hammouamri et al., ICLR 2024
Repo  : https://github.com/Thvnvtos/SNN-delays

All hyperparameters read from cfg (DCLSConfig).
Requires: spikingjelly, DCLS library
"""

import os
import sys

# ── Add SNN-delays submodule to path ─────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SNN_DELAYS_PATH = os.path.join(_ROOT, "model_notebooks", "SNN-delays")
sys.path.insert(0, _SNN_DELAYS_PATH)

try:
    from model import SNN, SNNDelays
except ImportError as e:
    raise ImportError(
        f"Could not import DCLS model: {e}\n"
        f"  → pip install spikingjelly --user\n"
        f"  → Ensure submodule is populated: "
        f"git submodule update --init --recursive"
    )


def build_dcls_model(cfg):
    """
    Factory function — builds and returns SNNDelays model from cfg.

    Args:
        cfg: DCLSConfig instance
    Returns:
        model: SNNDelays nn.Module
    """
    return SNNDelays(cfg)