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
    r"""Base ETCNN wrapper around a ResNet backbone.

    Trial solution (Eq. 10, Fig. 3):

        ũ_NN(s, t) = g1(s, t) · u_NN(s, t) + h(t) · g2(s, t)

    where h(t) is an optional temporal truncation factor (default h ≡ 1):

        h(t) = exp(-γ (t_k - t)²),  γ > 0

    This attenuates the static g2(s) term away from the terminal date t_k,
    so that the residual field does not dominate the interior of [t_{k-1}, t_k].
    At t = t_k, h(t_k) = 1 and the exact terminal condition is preserved.

    Args:
        resnet:              ResNet backbone (or any nn.Module mapping (batch, d_in) → (batch, 1)).
        g1:                  Callable(s, t) → Tensor, must vanish at terminal dates.
        g2:                  Callable(s, t) → Tensor, must equal the payoff at terminal.
        normalizer:          Optional InputNormalization layer applied before the ResNet.
        g2_temporal_gamma:   γ ≥ 0 for the temporal truncation h(t) = exp(-γ(t_k-t)²).
                             ``None`` (default) means h ≡ 1 (standard ETCNN).
        t_terminal:          Terminal date t_k used to compute τ = t_k - t.
                             Required when *g2_temporal_gamma* is not None.
    """

    def __init__(
        self,
        resnet: ResNet,
        g1: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        g2: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        normalizer: InputNormalization | None = None,
        g2_temporal_gamma: float | None = None,
        t_terminal: float | None = None,
    ) -> None:
        if g2_temporal_gamma is not None and t_terminal is None:
            raise ValueError(
                "t_terminal must be provided when g2_temporal_gamma is set."
            )
        super().__init__()
        self.resnet = resnet
        self._g1 = g1
        self._g2 = g2
        self.normalizer = normalizer
        self._g2_temporal_gamma = g2_temporal_gamma
        self._t_terminal = t_terminal

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

        # Temporal truncation h(t) = exp(-γ (t_k - t)²)
        if self._g2_temporal_gamma is not None:
            assert self._t_terminal is not None  # enforced by __init__
            tau = self._t_terminal - t
            h_val = torch.exp(-self._g2_temporal_gamma * tau ** 2)
            g2_val = h_val * g2_val

        return g1_val * u_nn + g2_val

    def forward_neural_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Return strictly the neural manifold component g1(s,t) · u_θ(s,t).

        Drops g2 entirely so that the BSM operator sees only the smooth,
        network-learned part of the solution.  Used by BermudaETCNN.forward_pde
        to implement the Ultimate Operator Bypass.

        Args:
            x: Input tensor of shape (batch, 2) with columns [s, t].

        Returns:
            g1(s, t) · u_θ(s, t) of shape (batch, 1).
        """
        s = x[:, 0:1]
        t = x[:, 1:2]
        net_input = self.normalizer(x) if self.normalizer is not None else x
        u_nn = self.resnet(net_input)
        g1_val = self._g1(s, t)
        return g1_val * u_nn


# ---------------------------------------------------------------------------
# American put specialisation
# ---------------------------------------------------------------------------

class AmericanPutETCNN(ETCNN):
    r"""ETCNN for a single-asset American put option (no dividends).

    g1(s, t) = T - t
    g2(s, t) depends on *g2_type*:

    - ``"taylor"`` (default): $g_2 = V_1^e + V_2^e$, the first-order Taylor
      expansion of the European put around $\tilde{d}_0$.  Captures the
      $\sqrt{\tau}$ singularity near expiry (Zhang, Guo, Lu 2026, Sec. 4.1.1).
    - ``"bs"``: $g_2 = V^e(s, t)$, the exact Black–Scholes European put price.
      Smoother than the Taylor form but does not explicitly decompose the
      singular behaviour.
    - ``"bs2002"``: $g_2 = \bar{p}^{\mathrm{BS02}}(s, K, \tau, r, \sigma)$,
      the Bjerksund–Stensland (2002) one-step American put approximation.
      Unlike the European anchors, this contains a flat exercise boundary
      and a $C^0$ kink at $s^*$, absorbing the dominant singularity of the
      true American solution.  The BSM operator applied to $g_2$ is *not*
      zero ($\mathcal{F}(g_2) \neq 0$), producing a non-homogeneous source
      term that the network compensates for.

    Args:
        K:      Strike price.
        r:      Risk-free rate.
        sigma:  Volatility.
        T:      Expiration time.
        q:      Continuous dividend yield (default 0).
        resnet: ResNet backbone (uses defaults if None).
        normalize_input: Whether to apply s/K normalisation.
        g2_type: Terminal function type — ``"taylor"``, ``"bs"``, or ``"bs2002"``.
    """

    def __init__(
        self,
        K: float = 100.0,
        r: float = 0.02,
        sigma: float = 0.25,
        T: float = 1.0,
        q: float = 0.0,
        resnet: ResNet | None = None,
        normalize_input: bool = True,
        g2_type: str = "taylor",
    ) -> None:
        from learning_option_pricing.pricing.terminal import (
            black_scholes_put,
            g1_linear,
            g2_american_put,
        )

        self._K = K
        self._r = r
        self._sigma = sigma
        self._T = T
        self._q = q
        self._g2_type = g2_type

        def g1(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return g1_linear(T, t)

        if g2_type == "taylor":
            def g2(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                tau = T - t
                return g2_american_put(s, K, r, sigma, tau)
        elif g2_type == "bs":
            def g2(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                tau = T - t
                return black_scholes_put(s, K, r, sigma, tau)
        elif g2_type == "bs2002":
            from learning_option_pricing.pricing.bjerksund_stensland import bs2002_put

            def g2(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                tau = T - t
                return bs2002_put(s, K, r, sigma, tau, q=q)
        else:
            raise ValueError(
                f"Unknown g2_type: {g2_type!r}. "
                "Choose 'taylor', 'bs', or 'bs2002'."
            )

        if resnet is None:
            resnet = ResNet()

        normalizer = InputNormalization(K) if normalize_input else None

        super().__init__(resnet=resnet, g1=g1, g2=g2, normalizer=normalizer)


# ---------------------------------------------------------------------------
# Analytical European put — drop-in Stage A replacement
# ---------------------------------------------------------------------------

class AnalyticalEuropeanPut(nn.Module):
    r"""Black-Scholes European put price as a parameter-free Stage A surrogate.

    Implements the exact analytical hold value $V^e(s, t) = P^{\text{BS}}(s, K, r,
    \sigma, T-t)$ so that Stage B can be trained without any Stage A network.

    The Bermudan intermediate condition at $t_1$ becomes
    $\max(\Phi(s),\, V^e(s, t_1))$, which is analytically exact for a European
    option exercised at $t_1$.

    Args:
        K, r, sigma, T: BSM contract parameters.
    """

    def __init__(self, K: float, r: float, sigma: float, T: float) -> None:
        super().__init__()
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from learning_option_pricing.pricing.terminal import black_scholes_put
        s = x[:, 0:1]
        t = x[:, 1:2]
        return black_scholes_put(s, self.K, self.r, self.sigma, self.T - t)


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
# Bermuda — singularity extraction ansatz
# ---------------------------------------------------------------------------

class BermudaETCNN(nn.Module):
    r"""ETCNN with singularity extraction for Bermudan put options.

    Decomposes the solution on $[0, t_1]$ as

    $$U_B(s, t) = v(s, t) + \tilde{u}_\theta(s, t)$$

    where

    * $v(s, t) = c \cdot P^{\text{BS}}(s, s^*, r, \sigma, t_1 - t)$ is a
      fictitious European put that analytically absorbs the $C^0$ kink at
      the exercise boundary $s^*$.  It is an exact BSM solution
      ($\mathcal{L}v = 0$).
    * $\tilde{u}_\theta$ is a standard :class:`ETCNN` whose $g_2$ equals the
      $C^1$-smooth residual $V_{\text{target}} - v|_{t_1}$, optionally
      attenuated by the temporal truncation $h(t) = \exp(-\gamma(t_1-t)^2)$
      (see :class:`ETCNN` for details).

    Because $\mathcal{L}$ is linear and $\mathcal{L}v = 0$, the PDE residual
    $\mathcal{L}U_B = \mathcal{L}\tilde{u}_\theta$.  The training loop can
    therefore operate on the *full* output without any special handling.

    Args:
        etcnn:          Residual ETCNN learning $\tilde{u}_\theta$.
        fictitious_put: :class:`~learning_option_pricing.pricing.singularity.FictitiousEuropeanPut`.
    """

    def __init__(
        self,
        etcnn: ETCNN,
        fictitious_put: nn.Module,
        bypass_v: bool = False,
    ) -> None:
        super().__init__()
        self.etcnn = etcnn
        self.fictitious_put = fictitious_put
        self.bypass_v = bypass_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Full forward pass: $U_B = v + g_1 u_\theta + g_2$.

        Always includes every component of the ansatz.  Used for the terminal
        condition loss $\mathcal{L}_{tc}$ so that $g_2$ enforces the exact
        terminal boundary.

        Args:
            x: Input tensor ``(batch, 2)`` with columns ``[s, t]``.

        Returns:
            $U_B(s, t)$ of shape ``(batch, 1)``.
        """
        s = x[:, 0:1]
        t = x[:, 1:2]
        v = self.fictitious_put(s, t)
        u_tilde = self.etcnn(x)
        return v + u_tilde

    def forward_pde(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for PDE residual computation.

        When ``bypass_v=True`` returns the coupled manifold

            $U_{\mathrm{pde}} = g_1(s,t) \cdot u_\theta(s,t) + g_2(s)$

        dropping only the fictitious put $v(s,t)$.  This prevents catastrophic
        floating-point cancellation from the diverging derivatives
        $\partial_{ss}v \to +\infty$, $\partial_t v \to -\infty$ near $s^*$,
        while keeping $g_2$ in the computational graph so that the network
        correctly learns the interior diffusion correction $\mathcal{L}(g_1 u_\theta) = -\mathcal{L}(g_2)$.

        Note: $\mathcal{L}(g_2) \neq 0$ for the PCHIP interpolant; decoupling
        $g_2$ was found empirically to break the backward-time diffusion
        (Bermudan price fell below the European price).  The localised autograd
        spike at $s^*$ from $\partial_{ss} g_2$ is accepted as an unavoidable
        geometric artifact of differentiating a $C^1$ PCHIP knot.
        """
        if self.bypass_v:
            return self.etcnn(x)
        return self.forward(x)

    # convenience accessors
    @property
    def resnet(self) -> ResNet:
        """Access the underlying ResNet (for weight diagnostics etc.)."""
        return self.etcnn.resnet
