"""Closed-form terminal data and exact solutions for the backward heat equation.

These references back the three initial (terminal) conditions of the
ansatz-form study described in
``latex_documents/reports/2026_01_29_constrained_learning_pde_lehalle_hosseinkhan/``:

1. **Two-mode sine** (``cal_notes/example_heat.tex`` Section 1).
   Terminal datum :math:`g(x) = \\sin(\\pi x) + c\\,\\sin(f\\pi x)` on
   :math:`[0, 1]`; the exact backward-heat solution keeps the sine modes with
   time-decaying amplitudes and satisfies homogeneous Dirichlet conditions
   :math:`u(t, 0) = u(t, 1) = 0`.

2. **Jacobi theta_3** (``cal_notes/example_heat.tex`` Section 2).
   Terminal datum :math:`g(x) = \\vartheta_3(x/2, e^{-1}) = 1 + 2\\sum_n
   e^{-n^2}\\cos(\\pi n x)` on :math:`[-1, 1]`; the exact solution is a
   theta_3 with a shifted nome.  **Sign convention:** under the terminal-value
   (backward) operator :math:`\\partial_t u + (\\sigma^2/2)\\partial_{xx}u = 0`
   used throughout this repository — the same convention as the Black--Scholes
   operator and the sine reference above — the mode amplitudes *decay* as ``t``
   decreases (the problem is well posed, equivalent to a forward heat equation
   in :math:`\\tau = T - t`).  The source note writes a growing-amplitude
   solution; that form actually solves :math:`\\partial_t u - (\\sigma^2/2)
   \\partial_{xx}u = 0` and is therefore inconsistent with the stated operator.
   We implement the operator-consistent decaying solution.

3. **Smoothed call payoff** (already implemented in the singularity study,
   re-expressed here under the *pure* heat operator).  Terminal datum is a
   smooth approximation of :math:`(e^x - K)^+` in the log-price coordinate
   :math:`x = \\ln S`; the exact pure-heat solution is a Black--Scholes call
   with zero rate and zero dividend, i.e. the Gaussian convolution of the
   payoff.

All functions are vectorised over batches of ``x`` and return tensors on the
same device/dtype as their input.  The diffusion coefficient is ``sigma^2 / 2``
to match :func:`learning_option_pricing.pde.operators.heat_operator`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_PI = math.pi


# ---------------------------------------------------------------------------
# Standard normal CDF (autograd-compatible, mirrors pricing.terminal)
# ---------------------------------------------------------------------------

def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# IC1 — two-mode sine
# ---------------------------------------------------------------------------

def heat_sine_terminal(
    x: torch.Tensor,
    *,
    c: float = 1.0,
    f: float = 4.0,
) -> torch.Tensor:
    r"""Two-mode sine terminal datum ``g(x) = sin(pi x) + c sin(f pi x)``.

    Args:
        x: Spatial coordinate (any shape).
        c: Amplitude of the high-frequency mode.
        f: Frequency multiplier of the high-frequency mode (integer-valued
           keeps homogeneous Dirichlet conditions exact on ``[0, 1]``).
    """
    return torch.sin(_PI * x) + c * torch.sin(f * _PI * x)


def heat_sine_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    T: float,
    sigma: float = 1.0,
    c: float = 1.0,
    f: float = 4.0,
) -> torch.Tensor:
    r"""Exact backward-heat solution for the two-mode sine terminal datum.

    .. math::

        u(t, x) = e^{\frac{\sigma^2 \pi^2}{2}(t - T)} \sin(\pi x)
                + c\, e^{\frac{\sigma^2 f^2 \pi^2}{2}(t - T)} \sin(f \pi x).

    Each mode decays as ``t`` decreases from ``T``; the high-frequency mode
    decays faster, which is the well-posed (forward-stable) direction.

    Args:
        x:     Spatial coordinate.
        t:     Time coordinate (broadcast-compatible with ``x``).
        T:     Terminal time.
        sigma: Diffusion scale (coefficient ``sigma^2 / 2``).
        c:     High-frequency amplitude.
        f:     High-frequency multiplier.
    """
    decay_1 = torch.exp(0.5 * sigma**2 * _PI**2 * (t - T))
    decay_f = torch.exp(0.5 * sigma**2 * (f * _PI) ** 2 * (t - T))
    return decay_1 * torch.sin(_PI * x) + c * decay_f * torch.sin(f * _PI * x)


# ---------------------------------------------------------------------------
# IC2 — Jacobi theta_3
# ---------------------------------------------------------------------------

def heat_theta3_terminal(
    x: torch.Tensor,
    *,
    n_modes: int = 6,
) -> torch.Tensor:
    r"""Jacobi theta_3 terminal datum ``g(x) = 1 + 2 sum_{n>=1} e^{-n^2} cos(pi n x)``.

    The series is truncated at ``n_modes`` terms; ``e^{-n^2}`` decays
    super-exponentially, so ``n_modes = 6`` is already machine-accurate
    (``e^{-49} ~ 5e-22``).

    Args:
        x:       Spatial coordinate.
        n_modes: Number of cosine modes retained.
    """
    out = torch.ones_like(x)
    for n in range(1, n_modes + 1):
        out = out + 2.0 * math.exp(-(n**2)) * torch.cos(_PI * n * x)
    return out


def heat_theta3_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    T: float,
    sigma: float = 1.0,
    n_modes: int = 6,
) -> torch.Tensor:
    r"""Exact backward-heat solution for the theta_3 terminal datum.

    .. math::

        u(t, x) = 1 + 2 \sum_{n \ge 1}
            e^{-n^2 + \frac{\sigma^2 \pi^2 n^2}{2}(t - T)} \cos(\pi n x).

    Under the terminal-value (backward) convention the mode amplitudes decay as
    ``t`` decreases from ``T``, so the solution is bounded for every ``T`` and
    no well-posedness horizon constraint is needed (cf. the module docstring on
    the sign convention).  The constant mode (amplitude ``1``) is stationary.

    Args:
        x:       Spatial coordinate.
        t:       Time coordinate.
        T:       Terminal time.
        sigma:   Diffusion scale.
        n_modes: Number of cosine modes retained.
    """
    out = torch.ones_like(x)
    for n in range(1, n_modes + 1):
        exponent = -(n**2) + 0.5 * sigma**2 * _PI**2 * n**2 * (t - T)
        out = out + 2.0 * torch.exp(exponent) * torch.cos(_PI * n * x)
    return out


# ---------------------------------------------------------------------------
# IC3 — smoothed call payoff under the pure heat operator (x = ln S)
# ---------------------------------------------------------------------------

def heat_call_payoff(x: torch.Tensor, K: float) -> torch.Tensor:
    """Exact call payoff in log-price coordinates: ``(e^x - K)^+``."""
    return torch.clamp(torch.exp(x) - K, min=0.0)


def smooth_call_payoff(x: torch.Tensor, K: float, *, beta: float) -> torch.Tensor:
    r"""Softplus smoothing of the call payoff ``(e^x - K)^+`` in ``x = ln S``.

    .. math::

        g_\beta(x) = \frac{1}{\beta}\log\!\bigl(1 + e^{\beta (e^x - K)}\bigr)
                     - \frac{\log 2}{\beta},

    which is ``C^\infty`` and converges uniformly to ``(e^x - K)^+`` as
    ``beta -> infinity`` (the ``-log 2 / beta`` shift makes the smoothing pass
    through the origin of the kink).  This matches the construction used in
    ``exp_singularity_european_call``.

    Args:
        x:    Log-price coordinate.
        K:    Strike.
        beta: Sharpness; larger ``beta`` is a tighter approximation.
    """
    log2 = math.log(2.0)
    return F.softplus(torch.exp(x) - K, beta=beta) - log2 / beta


def heat_call_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    K: float,
    T: float,
    sigma: float,
) -> torch.Tensor:
    r"""Exact pure-heat solution with the call payoff ``(e^x - K)^+`` at ``t = T``.

    Under ``d_t u + (sigma^2 / 2) d_xx u = 0`` the solution is the Gaussian
    convolution of the terminal payoff, i.e. a zero-rate Black--Scholes call
    with the no-drift convexity correction ``e^{sigma^2 tau / 2}``:

    .. math::

        u(t, x) = e^{x + \sigma^2 \tau / 2}\, N(d_1) - K\, N(d_2),
        \qquad \tau = T - t,

    with :math:`d_1 = (x - \ln K + \sigma^2 \tau) / (\sigma \sqrt{\tau})` and
    :math:`d_2 = d_1 - \sigma \sqrt{\tau}`.  At ``tau = 0`` it reduces to the
    payoff.

    Args:
        x:     Log-price coordinate.
        t:     Time coordinate.
        K:     Strike.
        T:     Terminal time.
        sigma: Diffusion scale.
    """
    tau = torch.clamp(T - t, min=1e-12)
    sqrt_tau = sigma * torch.sqrt(tau)
    d1 = (x - math.log(K) + sigma**2 * tau) / sqrt_tau
    d2 = d1 - sqrt_tau
    return torch.exp(x + 0.5 * sigma**2 * tau) * _normal_cdf(d1) - K * _normal_cdf(d2)
