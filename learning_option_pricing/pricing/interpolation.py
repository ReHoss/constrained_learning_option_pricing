"""Differentiable 1-D interpolation for autograd-compatible PDE residuals.

When a trial solution uses :math:`g_2(s,t) = V_{\\text{interp}}(s)` built from
tabulated data, the BSM PDE operator :math:`\\mathcal{F}` computes
:math:`\\partial^2 g_2/\\partial s^2` via autograd.  A piecewise-linear
interpolant is only :math:`C^0`, so this second derivative is zero almost
everywhere, dropping the diffusion term :math:`\\tfrac12 \\sigma^2 s^2
\\partial^2 g_2/\\partial s^2` entirely.

A natural cubic spline restores :math:`C^2` regularity and gives a
non-trivial (piecewise-linear) second derivative, correctly propagating
the volatility information through :math:`g_2`.
"""
from __future__ import annotations

import torch


class CubicSplineInterpolator:
    """Natural cubic spline interpolation, :math:`C^2`-differentiable via autograd.

    Given *n* nodes :math:`(x_i, y_i)`, the spline :math:`S` satisfies:

    * :math:`S(x_i) = y_i` (interpolation),
    * :math:`S \\in C^2` (twice continuously differentiable),
    * :math:`S''(x_0) = S''(x_{n-1}) = 0` (natural boundary conditions).

    On each sub-interval :math:`[x_i, x_{i+1}]`:

    .. math::
        S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)^2 + d_i (x - x_i)^3

    The evaluation uses only polynomial torch ops so that
    ``torch.autograd.grad`` can compute :math:`S'` and :math:`S''`.

    Args:
        x_nodes: Strictly increasing 1-D tensor of abscissae, shape ``(n,)``.
        y_nodes: Corresponding ordinates, shape ``(n,)``.

    Example::

        interp = CubicSplineInterpolator(s_nodes, v_nodes)
        v_query = interp(s_query)          # forward pass
        dv_ds   = torch.autograd.grad(v_query.sum(), s_query, create_graph=True)[0]
        d2v_ds2 = torch.autograd.grad(dv_ds.sum(), s_query)[0]
    """

    def __init__(self, x_nodes: torch.Tensor, y_nodes: torch.Tensor) -> None:
        if x_nodes.ndim != 1 or y_nodes.ndim != 1:
            raise ValueError("x_nodes and y_nodes must be 1-D tensors")
        n = x_nodes.shape[0]
        if n < 3:
            raise ValueError("Need at least 3 nodes for cubic spline")
        if y_nodes.shape[0] != n:
            raise ValueError("x_nodes and y_nodes must have the same length")

        x = x_nodes.detach().double()
        y = y_nodes.detach().double()
        h = x[1:] - x[:-1]  # (n-1,)

        # ------------------------------------------------------------------
        # Solve for second derivatives m_i = S''(x_i)  via tridiagonal system
        # Natural BC:  m_0 = m_{n-1} = 0
        # Interior eqs (i = 1 … n-2):
        #   h_{i-1} m_{i-1} + 2(h_{i-1}+h_i) m_i + h_i m_{i+1}
        #       = 6 [ (y_{i+1}-y_i)/h_i  -  (y_i-y_{i-1})/h_{i-1} ]
        # ------------------------------------------------------------------
        k = n - 2  # number of interior unknowns
        rhs = torch.zeros(k, dtype=torch.float64)
        for j in range(k):
            i = j + 1
            rhs[j] = 6.0 * (
                (y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]
            )

        # Build tridiagonal matrix (k × k)
        diag_main = torch.zeros(k, dtype=torch.float64)
        diag_lower = torch.zeros(k - 1, dtype=torch.float64)
        diag_upper = torch.zeros(k - 1, dtype=torch.float64)
        for j in range(k):
            i = j + 1
            diag_main[j] = 2.0 * (h[i - 1] + h[i])
        for j in range(k - 1):
            i = j + 1
            diag_lower[j] = h[i]      # A[j+1, j]
            diag_upper[j] = h[i]      # A[j, j+1]

        # Thomas algorithm (forward sweep + back-substitution)
        m_interior = _thomas_solve(diag_lower, diag_main, diag_upper, rhs)

        m = torch.zeros(n, dtype=torch.float64)
        m[1:-1] = m_interior

        # ------------------------------------------------------------------
        # Compute polynomial coefficients for each interval
        # S_i(x) = a_i + b_i dx + c_i dx^2 + d_i dx^3,  dx = x - x_i
        # ------------------------------------------------------------------
        a = y[:-1]                                                  # (n-1,)
        c = m[:-1] / 2.0                                            # (n-1,)
        d = (m[1:] - m[:-1]) / (6.0 * h)                           # (n-1,)
        b = (y[1:] - y[:-1]) / h - h * (2.0 * m[:-1] + m[1:]) / 6.0  # (n-1,)

        # Store as float32 on CPU
        self._x_nodes = x.float().cpu()
        self._a = a.float().cpu()
        self._b = b.float().cpu()
        self._c = c.float().cpu()
        self._d = d.float().cpu()

    def __call__(self, x_query: torch.Tensor) -> torch.Tensor:
        """Evaluate the spline at *x_query* (autograd-compatible)."""
        dev = x_query.device
        shape = x_query.shape
        xq = x_query.reshape(-1)

        x_nodes = self._x_nodes.to(dev)
        a = self._a.to(dev)
        b = self._b.to(dev)
        c = self._c.to(dev)
        d = self._d.to(dev)

        idx = torch.searchsorted(x_nodes.contiguous(), xq.contiguous()) - 1
        idx = idx.clamp(0, len(x_nodes) - 2)

        dx = xq - x_nodes[idx]
        result = a[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))
        return result.reshape(shape)

    def __repr__(self) -> str:
        n = self._x_nodes.shape[0]
        return (
            f"CubicSplineInterpolator(n_nodes={n}, "
            f"x=[{self._x_nodes[0]:.4g}, {self._x_nodes[-1]:.4g}])"
        )


class PiecewiseLinearInterpolator:
    """Piecewise-linear interpolation, :math:`C^0` only.

    .. warning::
        The second derivative :math:`\\partial^2 S/\\partial x^2 = 0` almost
        everywhere.  When used as :math:`g_2` in a trial solution, this drops
        the diffusion term from the PDE residual.  Prefer
        :class:`CubicSplineInterpolator` unless benchmarking.

    Args:
        x_nodes: Strictly increasing 1-D tensor of abscissae.
        y_nodes: Corresponding ordinates.
    """

    def __init__(self, x_nodes: torch.Tensor, y_nodes: torch.Tensor) -> None:
        if x_nodes.ndim != 1 or y_nodes.ndim != 1:
            raise ValueError("x_nodes and y_nodes must be 1-D tensors")
        if x_nodes.shape[0] != y_nodes.shape[0]:
            raise ValueError("x_nodes and y_nodes must have the same length")
        self._x_nodes = x_nodes.detach().cpu()
        self._y_nodes = y_nodes.detach().cpu()

    def __call__(self, x_query: torch.Tensor) -> torch.Tensor:
        """Evaluate the piecewise-linear interpolant at *x_query*."""
        dev = x_query.device
        shape = x_query.shape
        xq = x_query.reshape(-1)

        x_nodes = self._x_nodes.to(dev)
        y_nodes = self._y_nodes.to(dev)

        idx = torch.searchsorted(x_nodes.contiguous(), xq.contiguous()) - 1
        idx = idx.clamp(0, len(x_nodes) - 2)

        x0, x1 = x_nodes[idx], x_nodes[idx + 1]
        y0, y1 = y_nodes[idx], y_nodes[idx + 1]
        alpha = (xq - x0) / (x1 - x0)
        return (y0 + alpha * (y1 - y0)).reshape(shape)

    def __repr__(self) -> str:
        n = self._x_nodes.shape[0]
        return (
            f"PiecewiseLinearInterpolator(n_nodes={n}, "
            f"x=[{self._x_nodes[0]:.4g}, {self._x_nodes[-1]:.4g}])"
        )


# ---------------------------------------------------------------------------
# Thomas algorithm for tridiagonal systems
# ---------------------------------------------------------------------------

def _thomas_solve(
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Solve a tridiagonal system A x = rhs via the Thomas algorithm.

    Args:
        lower: Sub-diagonal, shape ``(n-1,)``.
        diag:  Main diagonal, shape ``(n,)``.
        upper: Super-diagonal, shape ``(n-1,)``.
        rhs:   Right-hand side, shape ``(n,)``.
    """
    n = diag.shape[0]
    c_prime = torch.zeros(n - 1, dtype=diag.dtype)
    d_prime = torch.zeros(n, dtype=diag.dtype)

    # Forward sweep
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    # Back-substitution
    x = torch.zeros(n, dtype=diag.dtype)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x
