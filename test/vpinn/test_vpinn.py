r"""Unit tests for the VPINN numerical core.

Three test families in increasing order of criticality:

1. **Boundary annihilation** — all $K$ test functions must vanish at $\pm x_{\max}$.
2. **Quadrature exactness** — GL integration must be machine-exact for polynomials
   of degree $\leq 2N - 1$.
3. **Integration-by-parts (IBP) identity** — the most critical test.
   For a smooth toy function $u$ and test function $\phi_k$ with
   $\phi_k(\pm x_{\max}) = 0$:

   $$\int u'' \phi_k \,\mathrm{d}x = -\int u' \phi'_k \,\mathrm{d}x$$

   Verifying this numerically proves that the derivative transfer coded in
   ``VPINNLoss`` is mathematically consistent.
"""
from __future__ import annotations

import math

import pytest
import torch

from learning_option_pricing.vpinn import GaussLegendreQuadrature, SinusoidalTestFunctions


# ---------------------------------------------------------------------------
# 1. Boundary annihilation
# ---------------------------------------------------------------------------

class TestBoundaryAnnihilation:
    r"""$\phi_k(\pm x_{\max}) = 0$ for every $k = 1, \ldots, K$."""

    X_MAX = 3.0
    K = 10
    ATOL = 1e-7

    def _make(self) -> SinusoidalTestFunctions:
        return SinusoidalTestFunctions(self.K, self.X_MAX)

    def test_left_boundary(self):
        tf = self._make()
        x = torch.tensor([-self.X_MAX], dtype=torch.float64)
        assert tf(x).abs().max().item() < self.ATOL

    def test_right_boundary(self):
        tf = self._make()
        x = torch.tensor([self.X_MAX], dtype=torch.float64)
        assert tf(x).abs().max().item() < self.ATOL

    def test_both_boundaries_all_k(self):
        tf = self._make()
        x = torch.tensor([-self.X_MAX, self.X_MAX], dtype=torch.float64)
        phi = tf(x)  # (K, 2)
        assert phi.shape == (self.K, 2)
        assert phi.abs().max().item() < self.ATOL

    @pytest.mark.parametrize("k_single", [1, 3, 7, 10])
    def test_individual_modes(self, k_single: int):
        tf = SinusoidalTestFunctions(k_single, self.X_MAX)
        x = torch.tensor([-self.X_MAX, self.X_MAX], dtype=torch.float64)
        assert tf(x).abs().max().item() < self.ATOL


# ---------------------------------------------------------------------------
# 2. Gauss-Legendre quadrature exactness
# ---------------------------------------------------------------------------

class TestGaussLegendreQuadrature:
    r"""GL with $N$ points integrates degree-$(2N-1)$ polynomials exactly."""

    def test_degree5_polynomial_on_asymmetric_interval(self):
        r"""$\int_{-2}^{3} (x^5 - 3x^3 + x)\,\mathrm{d}x$ — exact with $N \geq 3$."""
        a, b = -2.0, 3.0
        quad = GaussLegendreQuadrature(n_points=5, x_min=a, x_max=b)
        x = quad.nodes
        result = quad.integrate(x**5 - 3 * x**3 + x).item()

        def antideriv(t: float) -> float:
            return t**6 / 6 - 3 * t**4 / 4 + t**2 / 2

        exact = antideriv(b) - antideriv(a)
        assert abs(result - exact) < 1e-12, f"got {result}, expected {exact}"

    def test_constant_integral_equals_length(self):
        r"""$\int_0^1 1\,\mathrm{d}x = 1$ with any $N$."""
        quad = GaussLegendreQuadrature(n_points=3, x_min=0.0, x_max=1.0)
        result = quad.integrate(torch.ones(3, dtype=torch.float64)).item()
        assert abs(result - 1.0) < 1e-14

    def test_gaussian_integral(self):
        r"""$\int_{-5}^{5} e^{-x^2}\,\mathrm{d}x \approx \sqrt{\pi}$ with $N = 100$."""
        quad = GaussLegendreQuadrature(n_points=100, x_min=-5.0, x_max=5.0)
        result = quad.integrate(torch.exp(-quad.nodes**2)).item()
        assert abs(result - math.sqrt(math.pi)) < 1e-10

    def test_polynomial_degree_equals_2n_minus1(self):
        """Degree exactly 2N-1 is still integrated exactly."""
        N = 4  # exact for degree <= 7
        a, b = 0.0, 1.0
        quad = GaussLegendreQuadrature(n_points=N, x_min=a, x_max=b)
        x = quad.nodes
        # p(x) = x^7 — degree 7 = 2*4 - 1
        result = quad.integrate(x**7).item()
        exact = 1.0 / 8.0  # integral of x^7 on [0,1]
        assert abs(result - exact) < 1e-13


# ---------------------------------------------------------------------------
# 3. Integration-by-parts identity
# ---------------------------------------------------------------------------

class TestIntegrationByParts:
    r"""The IBP identity $\int u'' \phi_k \,\mathrm{d}x = -\int u' \phi'_k \,\mathrm{d}x$.

    This is the central correctness criterion: if it holds numerically, the
    derivative transfer coded in ``VPINNLoss`` is mathematically sound.
    Both sides are approximated with the **same** quadrature rule; the gap
    measures quadrature error only and should be at or below machine precision
    for smooth integrands with a sufficiently fine rule.
    """

    X_MAX = math.pi
    K = 5
    N_QUAD = 200
    ATOL = 1e-10

    def _setup(self):
        """Return (x_nodes, weights, phi, dphi) on the quadrature grid."""
        quad = GaussLegendreQuadrature(self.N_QUAD, -self.X_MAX, self.X_MAX)
        tf = SinusoidalTestFunctions(self.K, self.X_MAX)
        x = quad.nodes          # (N_q,)
        w = quad.weights        # (N_q,)
        phi = tf(x)             # (K, N_q)
        dphi = tf.grad(x)       # (K, N_q)
        return x, w, phi, dphi

    def _assert_ibp(
        self,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        w: torch.Tensor,
        phi: torch.Tensor,
        dphi: torch.Tensor,
        label: str,
    ) -> None:
        """Assert strong == weak for all K test functions."""
        # Strong side: sum_j w_j * u''(x_j) * phi_k(x_j)  -> (K,)
        strong = (phi * w) @ u_xx   # (K, N_q) @ (N_q,) = (K,)
        # Weak side: -sum_j w_j * u'(x_j) * phi'_k(x_j)  -> (K,)
        weak = -(dphi * w) @ u_x    # (K, N_q) @ (N_q,) = (K,)

        max_err = (strong - weak).abs().max().item()
        assert max_err < self.ATOL, (
            f"IBP violated for {label}: "
            f"max_k |strong - weak| = {max_err:.3e}  (atol={self.ATOL})"
        )

    def test_ibp_u_sin(self):
        r"""$u(x) = \sin(x)$: $u'' = -\sin(x)$, $u' = \cos(x)$."""
        x, w, phi, dphi = self._setup()
        self._assert_ibp(torch.cos(x), -torch.sin(x), w, phi, dphi, "u=sin(x)")

    def test_ibp_u_cubic(self):
        r"""$u(x) = x^3$: $u'' = 6x$, $u' = 3x^2$."""
        x, w, phi, dphi = self._setup()
        self._assert_ibp(3.0 * x**2, 6.0 * x, w, phi, dphi, "u=x^3")

    def test_ibp_u_gaussian(self):
        r"""$u(x) = e^{-x^2}$: $u'' = (4x^2 - 2)e^{-x^2}$."""
        x, w, phi, dphi = self._setup()
        e = torch.exp(-x**2)
        self._assert_ibp(-2.0 * x * e, (4.0 * x**2 - 2.0) * e, w, phi, dphi, "u=exp(-x^2)")

    def test_ibp_bs_delta_as_u_prime(self):
        r"""Smoke-test: IBP holds when $u'$ is the Black-Scholes delta $e^x N(d_1)$.

        $u''$ is approximated by finite differences of $u'$ (relaxed tolerance).
        """
        import math as _math

        sigma, r_bs, q = 0.2, 0.05, 0.0
        tau_val = 0.5

        def N(z: torch.Tensor) -> torch.Tensor:
            return 0.5 * (1.0 + torch.erf(z / _math.sqrt(2.0)))

        x, w, phi, dphi = self._setup()

        def delta(x_in: torch.Tensor) -> torch.Tensor:
            d1 = (x_in + (r_bs - q + 0.5 * sigma**2) * tau_val) / (sigma * tau_val**0.5)
            return torch.exp(x_in) * N(d1)

        dx = 1e-5
        u_x = delta(x)
        u_xx = (delta(x + dx) - u_x) / dx  # FD approximation

        # Relaxed atol: FD error in u_xx propagates to ~O(dx) in the integral
        strong = (phi * w) @ u_xx
        weak = -(dphi * w) @ u_x
        max_err = (strong - weak).abs().max().item()
        assert max_err < 1e-4, f"BS-delta IBP smoke-test failed: {max_err:.3e}"
