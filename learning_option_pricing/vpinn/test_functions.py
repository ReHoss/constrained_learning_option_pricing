r"""Sinusoidal test functions with compact support on $[-x_{\max}, x_{\max}]$.

Each function is defined as

$$\phi_k(x) = \sin\!\left(\frac{k \pi (x + x_{\max})}{2 x_{\max}}\right), \quad k = 1, \ldots, K.$$

Boundary annihilation is guaranteed by construction:
  * $\phi_k(-x_{\max}) = \sin(0) = 0$ for all $k$.
  * $\phi_k(x_{\max}) = \sin(k\pi) = 0$ for all integer $k$.

The analytical first derivative is

$$\phi'_k(x) = \frac{k \pi}{2 x_{\max}} \cos\!\left(\frac{k \pi (x + x_{\max})}{2 x_{\max}}\right).$$
"""
from __future__ import annotations

import torch


class SinusoidalTestFunctions:
    r"""K sinusoidal test functions that vanish at $\pm x_{\max}$.

    Args:
        K:     Number of test functions (modes $k = 1, \ldots, K$).
        x_max: Half-width of the spatial domain $[-x_{\max}, x_{\max}]$.
        dtype: Floating-point dtype.
    """

    def __init__(
        self,
        K: int,
        x_max: float,
        dtype: torch.dtype = torch.float64,
    ):
        self.K = K
        self.x_max = x_max
        # Shape (K, 1) for broadcasting against (1, N) spatial inputs
        self._k = torch.arange(1, K + 1, dtype=dtype).unsqueeze(1)
        self._freq = self._k * torch.pi / (2.0 * x_max)  # (K, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r"""Evaluate all K test functions at spatial points x.

        Args:
            x: ``(N,)`` spatial points.

        Returns:
            ``(K, N)`` tensor of $\phi_k(x_j)$ values.
        """
        phase = self._freq * (x.unsqueeze(0) + self.x_max)  # (K, N)
        return torch.sin(phase)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        r"""Evaluate analytical first derivatives $\phi'_k$ at spatial points x.

        Args:
            x: ``(N,)`` spatial points.

        Returns:
            ``(K, N)`` tensor of $\phi'_k(x_j)$ values.
        """
        phase = self._freq * (x.unsqueeze(0) + self.x_max)  # (K, N)
        return self._freq * torch.cos(phase)                  # (K, N)
