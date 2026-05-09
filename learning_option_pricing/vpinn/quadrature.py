"""Gauss-Legendre numerical quadrature on a 1D spatial interval.

A GL rule with N points integrates polynomials of degree ≤ 2N-1 exactly.
For smooth non-polynomial functions the error decays exponentially with N.
"""
from __future__ import annotations

import numpy as np
import torch


class GaussLegendreQuadrature:
    """Gauss-Legendre quadrature on [x_min, x_max].

    Nodes and weights are precomputed once and stored as tensors so they
    can be moved to the right device alongside the rest of the model.

    Args:
        n_points: Number of quadrature nodes.
        x_min:    Left endpoint of the integration interval.
        x_max:    Right endpoint of the integration interval.
        dtype:    Floating-point dtype for nodes and weights.
    """

    def __init__(
        self,
        n_points: int,
        x_min: float,
        x_max: float,
        dtype: torch.dtype = torch.float64,
    ):
        xi, wi = np.polynomial.legendre.leggauss(n_points)
        # Affine map from [-1, 1] to [x_min, x_max]
        mid = (x_max + x_min) / 2.0
        half = (x_max - x_min) / 2.0
        self.nodes = torch.tensor(mid + half * xi, dtype=dtype)
        self.weights = torch.tensor(half * wi, dtype=dtype)
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points

    def integrate(self, f_values: torch.Tensor) -> torch.Tensor:
        """Dot precomputed function values with quadrature weights.

        Args:
            f_values: ``(..., N)`` tensor evaluated at ``self.nodes``.

        Returns:
            ``(...)`` tensor of integral approximations.
        """
        return (f_values * self.weights).sum(dim=-1)
