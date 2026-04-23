# models/log_buffer.py
"""
Logarithmic Spike Buffer — novel extension to Adaptive Axonal Delay SNN.

Biological motivation:
    Cochlear hair cells respond logarithmically in time and frequency.
    Recent spikes have fine temporal resolution.
    Older spikes have coarse temporal resolution.
    This mirrors the log-spaced frequency tuning of the basilar membrane.

Formal definition:
    τ_k = floor(D * log(1+k) / log(1+K))
    k = 0, 1,..., K-1
    D = max_delay, K = n_taps

Reference: Biological plausibility motivated by:
    - Patterson et al. (1992) cochlear modelling
    - Izhikevich (2006) polychronization
"""

import torch
import torch.nn as nn
import numpy as np


class LogarithmicSpikeBuffer(nn.Module):
    """
    Logarithmically-spaced temporal buffer for incoming spike trains.

    Creates K delayed copies of the input spike train at log-spaced
    intervals. Fine resolution for recent spikes, coarse for older ones.

    Input  : (B, T, N)  — batch, time, neurons (our convention)
    Output : (B, T, K*N) — K delayed copies concatenated on neuron dim

    Args:
        n_inputs  : number of input channels (700 for SHD)
        max_delay : maximum delay D in timesteps
        n_taps    : number of log-spaced taps K
        learnable : if True, tap positions are learnable parameters
    """

    def __init__(
        self,
        n_inputs  : int,
        max_delay : int  = 64,
        n_taps    : int  = 8,
        learnable : bool = False,
    ):
        super().__init__()
        self.n_inputs  = n_inputs
        self.max_delay = max_delay
        self.n_taps    = n_taps
        self.learnable = learnable

        # Compute log-spaced tap positions
        taps = self._compute_log_taps(max_delay, n_taps)

        if learnable:
            self.tap_positions = nn.Parameter(
                torch.tensor(taps, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "tap_positions",
                torch.tensor(taps, dtype=torch.float32)
            )

    def _compute_log_taps(self, max_delay: int, n_taps: int) -> np.ndarray:
        """
        Compute log-spaced tap positions.
        τ_k = floor(D * log(1+k) / log(1+K))

        Example with D=64, K=8:
            τ = [0, 9, 18, 27, 36, 45, 54, 64]  ← uniform (wrong)
            τ = [0, 1, 3,  6,  12, 22, 39, 64]  ← log-spaced (correct)
        """
        if n_taps == 1:
            return np.array([0])
        k    = np.arange(n_taps)
        taps = np.floor(
            max_delay * np.log1p(k) / np.log1p(n_taps - 1)
        ).astype(int)
        return taps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply log-spaced temporal shifts and concatenate.

        Args:
            x: (B, T, N) — spike tensor in our (batch, time, neuron) format
        Returns:
            out: (B, T, n_taps * N)
        """
        B, T, N = x.shape
        taps    = self.tap_positions.long()

        delayed_copies = []
        for tau in taps:
            tau_int = int(tau.item())

            if tau_int == 0:
                delayed_copies.append(x)
            else:
                # Causal shift: pad start with zeros, drop end
                # (B, T, N) → shift along T dimension
                pad     = torch.zeros(B, tau_int, N,
                                      device=x.device, dtype=x.dtype)
                shifted = torch.cat([pad, x[:, :-tau_int, :]], dim=1)
                delayed_copies.append(shifted)

        # Concatenate along neuron dimension → (B, T, n_taps * N)
        return torch.cat(delayed_copies, dim=-1)

    def output_size(self) -> int:
        """Returns output neuron dimension after buffering."""
        return self.n_taps * self.n_inputs

    def get_tap_positions(self) -> list:
        """Returns current tap positions as a list of ints."""
        return self.tap_positions.long().tolist()

    def extra_repr(self) -> str:
        taps = self._compute_log_taps(self.max_delay, self.n_taps)
        return (
            f"n_inputs={self.n_inputs}, max_delay={self.max_delay}, "
            f"n_taps={self.n_taps}, taps={taps.tolist()}, "
            f"learnable={self.learnable}"
        )