"""Exact Terminal Condition Neural Network (ETCNN) for option pricing.

Reference: Zhang, Guo, Lu (2026) — ETCNN, Section 3.

The trial solution is constructed as

    ũ_NN(s, t) = g1(s, t) * u_NN(s, t) + g2(s, t)

where
    g1(s, T) = 0           (vanishes at the terminal/exercise date)
    g2(s, T) = Φ(s)        (matches the payoff exactly)

so ũ_NN automatically satisfies the terminal condition regardless of u_NN.

g2 is additionally designed to capture the near-terminal singularities of the
true solution (√τ behaviour near t = T and non-differentiability at s = K).

Subclasses implement specific option types (American put, American call with
dividends, Bermuda, multi-asset, …).
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from learning_option_pricing.models.resnet import ResNet


# ---------------------------------------------------------------------------
# Input normalisation (Section 3.3)
# ---------------------------------------------------------------------------

class InputNormalization(nn.Module):
    """Normalise asset price by strike before feeding into the network.

    Transforms input (s, t) → (s/K, t).  All other columns (if any) pass
    through unchanged.

    Args:
        K:         Strike price used for normalisation.
        s_column:  Index of the asset-price column in the input tensor.
    """

    def __init__(self, K: float, s_column: int = 0) -> None:
        super().__init__()
        self.K = K
        self.s_column = s_column

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[:, self.s_column] = x[:, self.s_column] / self.K
        return x


# ---------------------------------------------------------------------------
# ETCNN — base class
# ---------------------------------------------------------------------------

class ETCNN(nn.Module):
    """Base ETCNN wrapper around a ResNet backbone.

    Trial solution (Eq. 10, Fig. 3):
        ũ_NN(s, t) = g1(s, t) · u_NN(s, t) + g2(s, t)

    Args:
        resnet:       ResNet backbone (or any nn.Module mapping (batch, d_in) → (batch, 1)).
        g1:           Callable(s, t) → Tensor, must vanish at terminal dates.
        g2:           Callable(s, t) → Tensor, must equal the payoff at terminal.
        normalizer:   Optional InputNormalization layer applied before the ResNet.
    """

    def __init__(
        self,
        resnet: ResNet,
        g1: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        g2: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        normalizer: InputNormalization | None = None,
    ) -> None:
        super().__init__()
        self.resnet = resnet
        self._g1 = g1
        self._g2 = g2
        self.normalizer = normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 2) with columns [s, t].

        Returns:
            ũ_NN(s, t) of shape (batch, 1).
        """
        s = x[:, 0:1]  # (batch, 1)
        t = x[:, 1:2]  # (batch, 1)

        # Raw network output
        net_input = self.normalizer(x) if self.normalizer is not None else x
        u_nn = self.resnet(net_input)  # (batch, 1)

        # Terminal-condition functions
        g1_val = self._g1(s, t)  # (batch, 1)
        g2_val = self._g2(s, t)  # (batch, 1)

        return g1_val * u_nn + g2_val


# ---------------------------------------------------------------------------
# American put specialisation
# ---------------------------------------------------------------------------

class AmericanPutETCNN(ETCNN):
    """ETCNN for a single-asset American put option (no dividends).

    g1(s, t) = T - t
    g2(s, t) = V1^e(s, t) + V2^e(s, t)

    See Zhang, Guo, Lu (2026), Section 4.1.1.

    Args:
        K:      Strike price.
        r:      Risk-free rate.
        sigma:  Volatility.
        T:      Expiration time.
        resnet: ResNet backbone (uses defaults if None).
        normalize_input: Whether to apply s/K normalisation.
    """

    def __init__(
        self,
        K: float = 100.0,
        r: float = 0.02,
        sigma: float = 0.25,
        T: float = 1.0,
        resnet: ResNet | None = None,
        normalize_input: bool = True,
    ) -> None:
        from learning_option_pricing.pricing.terminal import (
            g1_linear,
            g2_american_put,
        )

        self._K = K
        self._r = r
        self._sigma = sigma
        self._T = T

        def g1(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return g1_linear(T, t)

        def g2(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            tau = T - t
            return g2_american_put(s, K, r, sigma, tau)

        if resnet is None:
            resnet = ResNet()

        normalizer = InputNormalization(K) if normalize_input else None

        super().__init__(resnet=resnet, g1=g1, g2=g2, normalizer=normalizer)


# ---------------------------------------------------------------------------
# PINN baseline (no terminal-condition enforcement)
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Plain physics-informed neural network (no g1/g2 modification).

    Same ResNet architecture as ETCNN but with a standard output layer.
    Used as a baseline to demonstrate the advantage of exact terminal
    condition enforcement.

    Args:
        resnet:     ResNet backbone.
        normalizer: Optional InputNormalization layer.
    """

    def __init__(
        self,
        resnet: ResNet | None = None,
        normalizer: InputNormalization | None = None,
    ) -> None:
        super().__init__()
        if resnet is None:
            resnet = ResNet()
        self.resnet = resnet
        self.normalizer = normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net_input = self.normalizer(x) if self.normalizer is not None else x
        return self.resnet(net_input)


# ---------------------------------------------------------------------------
# Bermuda (placeholder)
# ---------------------------------------------------------------------------

class BermudaETCNN(ETCNN):
    """ETCNN for a single-asset Bermuda option with discrete exercise dates.

    TODO: define g1 and g2 for the piecewise-in-time setting.
    """
