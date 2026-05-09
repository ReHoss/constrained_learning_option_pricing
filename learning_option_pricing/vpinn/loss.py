r"""VPINN PDE residual loss for the Black-Scholes equation (European call).

The PDE in reverse-time / log-moneyness coordinates $(\tau, x) = (T-t,\, \ln S/K)$ is

$$\frac{\partial u}{\partial \tau}
  = \frac{\sigma^2}{2}\frac{\partial^2 u}{\partial x^2}
  + \mu \frac{\partial u}{\partial x}
  - r u, \qquad \mu = r - q - \tfrac{\sigma^2}{2}.$$

Multiplying by a test function $\phi_k$ and integrating by parts over
$[-x_{\max}, x_{\max}]$, and using $\phi_k(\pm x_{\max}) = 0$, the boundary
term from the second-order term vanishes and we obtain the weak residual

$$R_{i,k} = \int\!\left[
  \frac{\partial u}{\partial \tau}\,\phi_k
  + \frac{\sigma^2}{2}\,\frac{\partial u}{\partial x}\,\phi'_k
  - \mu\,\frac{\partial u}{\partial x}\,\phi_k
  + r\,u\,\phi_k
\right]\mathrm{d}x = 0.$$

The VPINN loss is $\mathcal{L}_{\text{pde}} = \operatorname{mean}_{i,k} R_{i,k}^2$.

Tensor strategy — never materialise a 3-way $(N_\tau, K, N_q)$ array:

1. Build a flat grid of $(N_\tau \cdot N_q)$ input points, shape ``(N_tau*N_q, 2)``.
2. Evaluate $u$, $\partial u/\partial\tau$, $\partial u/\partial x$ via a single
   ``torch.autograd.grad`` call — shapes ``(N_tau, N_q)``.
3. Contract against pre-weighted test-function matrices via ``@``
   — shapes ``(N_tau, N_q) @ (N_q, K) = (N_tau, K)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .quadrature import GaussLegendreQuadrature
from .test_functions import SinusoidalTestFunctions


class VPINNLoss(nn.Module):
    r"""VPINN PDE residual loss for Black-Scholes.

    Pre-computes and caches all quadrature/test-function quantities so that
    ``forward`` only needs to call the neural network and one ``autograd.grad``.

    Args:
        sigma:   Volatility $\sigma > 0$.
        r:       Risk-free rate.
        q:       Continuous dividend yield (0 for a non-dividend-paying asset).
        x_max:   Spatial domain half-width; domain is $[-x_{\max}, x_{\max}]$.
        K_test:  Number of sinusoidal test functions.
        n_quad:  Number of Gauss-Legendre quadrature nodes.
        dtype:   Floating-point dtype for all precomputed buffers.
    """

    # Declared so Pyright knows these are Tensors (register_buffer is untyped)
    phi_w: torch.Tensor
    dphi_w: torch.Tensor
    x_nodes: torch.Tensor

    def __init__(
        self,
        sigma: float,
        r: float,
        q: float,
        x_max: float,
        K_test: int = 20,
        n_quad: int = 50,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.sigma = sigma
        self.r = r
        self.q = q
        self.mu = r - q - 0.5 * sigma**2  # drift coefficient in log-price SDE

        quad = GaussLegendreQuadrature(n_quad, -x_max, x_max, dtype=dtype)
        tf = SinusoidalTestFunctions(K_test, x_max, dtype=dtype)

        x_nodes = quad.nodes          # (N_q,)
        weights = quad.weights        # (N_q,)
        phi = tf(x_nodes)             # (K, N_q)
        dphi = tf.grad(x_nodes)       # (K, N_q)

        # Pre-weight once: avoids repeated multiplication in forward()
        # phi_w[k, j]  = phi_k(x_j)  * w_j
        # dphi_w[k, j] = phi'_k(x_j) * w_j
        self.register_buffer("phi_w", phi * weights)    # (K, N_q)
        self.register_buffer("dphi_w", dphi * weights)  # (K, N_q)
        self.register_buffer("x_nodes", x_nodes)        # (N_q,)

    def forward(self, model: nn.Module, tau_batch: torch.Tensor) -> torch.Tensor:
        r"""Compute the VPINN PDE residual loss.

        Args:
            model:     Neural network $u_\theta(\tau, x)$.
                       Input shape ``(..., 2)`` — first column is $\tau$, second
                       is $x$.  Output shape ``(..., 1)``.
            tau_batch: ``(N_tau,)`` collocation times $\tau_i \in (0, T]$.

        Returns:
            Scalar $\mathcal{L}_{\text{pde}} = \operatorname{mean}_{i,k} R_{i,k}^2$.
        """
        N_tau = tau_batch.shape[0]
        N_q = self.x_nodes.shape[0]

        # Flat (N_tau * N_q, 2) input grid — no 3-D tensor needed
        tau_rep = tau_batch.unsqueeze(1).expand(N_tau, N_q).reshape(-1, 1)
        x_rep = self.x_nodes.unsqueeze(0).expand(N_tau, N_q).reshape(-1, 1)

        tau_rep = tau_rep.detach().requires_grad_(True)
        x_rep = x_rep.detach().requires_grad_(True)

        u = model(torch.cat([tau_rep, x_rep], dim=1))  # (N_tau*N_q, 1)

        du_dtau, du_dx = torch.autograd.grad(
            u,
            [tau_rep, x_rep],
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )

        u_vals = u.squeeze(1).reshape(N_tau, N_q)             # (N_tau, N_q)
        u_tau = du_dtau.squeeze(1).reshape(N_tau, N_q)        # (N_tau, N_q)
        u_x = du_dx.squeeze(1).reshape(N_tau, N_q)            # (N_tau, N_q)

        # Integrand coefficients (N_tau, N_q)
        # f_phi  multiplies phi_k  in the quadrature sum
        # f_dphi multiplies phi'_k (from integration by parts)
        f_phi = u_tau - self.mu * u_x + self.r * u_vals   # (N_tau, N_q)
        f_dphi = (self.sigma**2 / 2.0) * u_x               # (N_tau, N_q)

        # R_{i,k} = sum_j [f_phi_{i,j} * phi_w_{k,j} + f_dphi_{i,j} * dphi_w_{k,j}]
        # (N_tau, N_q) @ (N_q, K) = (N_tau, K)
        R = f_phi @ self.phi_w.T + f_dphi @ self.dphi_w.T  # (N_tau, K)

        return R.pow(2).mean()
