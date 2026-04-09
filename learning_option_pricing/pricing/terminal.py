"""Exact terminal / exercise-date functions (g1, g2) for BSM equations.

Reference: Zhang, Guo, Lu (2026) — ETCNN, Sections 3.4–3.5.

For a European call:
    g1(s, t) = T - t
    g2(s, t) = (1/2) * (s - K + sqrt(s^2 + K^2 - 2sK e^{-r(T-t)}))

For an American put (recommended g2):
    g2 = V1^e(s, t) + V2^e(s, t)
    where V1^e and V2^e are the zeroth- and first-order terms of the
    Taylor expansion of the European put price around d_tilde_0.

Bermuda options require a piecewise generalisation: see
:mod:`learning_option_pricing.pricing.interpolation` for C² cubic spline
and C⁰ piecewise-linear interpolators used to build g2 from tabulated
continuation values at intermediate exercise dates.
"""
from __future__ import annotations

import math

import torch


_TAU_EPS = 1e-8  # epsilon floor to avoid division by zero when tau -> 0
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Standard normal CDF (via torch.erfc for autograd compatibility)
# ---------------------------------------------------------------------------

def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Payoff functions  (vectorised over batches of s)
# ---------------------------------------------------------------------------

def payoff_put(s: torch.Tensor, K: float) -> torch.Tensor:
    """American/European put payoff: Phi(s) = max(K - s, 0)."""
    return torch.clamp(K - s, min=0.0)


def payoff_call(s: torch.Tensor, K: float) -> torch.Tensor:
    """American/European call payoff: Phi(s) = max(s - K, 0)."""
    return torch.clamp(s - K, min=0.0)


# ---------------------------------------------------------------------------
# BSM PDE operator  F(V) — single-asset (Eq. 2)
# ---------------------------------------------------------------------------

def bsm_operator(
    V: torch.Tensor,
    s: torch.Tensor,
    t: torch.Tensor,
    r: float,
    q: float,
    sigma: float,
) -> torch.Tensor:
    """BSM PDE operator F(V) for a single-asset option (Eq. 2).

    F(V) = dV/dt + 0.5 * sigma^2 * s^2 * d^2V/ds^2 + (r-q)*s * dV/ds - r*V

    V, s, t must be leaf tensors with requires_grad=True (or created via
    operations on such tensors) so that autograd can compute the derivatives.

    Args:
        V:     Option price, shape (N,).  Must be connected to s and t via the
               computational graph.
        s:     Underlying asset price, shape (N,).
        t:     Time, shape (N,).
        r:     Risk-free rate.
        q:     Continuous dividend yield.
        sigma: Volatility.

    Returns:
        F(V), shape (N,).
    """
    # First-order gradients
    (grad_V,) = torch.autograd.grad(
        V, (s,), grad_outputs=torch.ones_like(V), create_graph=True
    )
    (grad_Vt,) = torch.autograd.grad(
        V, (t,), grad_outputs=torch.ones_like(V), create_graph=True
    )
    # Second-order gradient w.r.t. s
    (grad_Vss,) = torch.autograd.grad(
        grad_V, (s,), grad_outputs=torch.ones_like(grad_V), create_graph=True
    )

    dV_dt = grad_Vt
    dV_ds = grad_V
    d2V_ds2 = grad_Vss

    return dV_dt + 0.5 * sigma**2 * s**2 * d2V_ds2 + (r - q) * s * dV_ds - r * V


# ---------------------------------------------------------------------------
# Time value operator
# ---------------------------------------------------------------------------

def time_value(V: torch.Tensor, s: torch.Tensor, K: float, option_type: str = "put") -> torch.Tensor:
    """Time value TV(V) = V(s,t) - Phi(s).

    Args:
        V:           Option price.
        s:           Underlying asset price.
        K:           Strike price.
        option_type: "put" or "call".
    """
    if option_type == "put":
        phi = payoff_put(s, K)
    elif option_type == "call":
        phi = payoff_call(s, K)
    else:
        raise ValueError(f"Unknown option_type: {option_type!r}")
    return V - phi


# ---------------------------------------------------------------------------
# European put — Black-Scholes analytical solution (Eq. 16)
# ---------------------------------------------------------------------------

def _d_tilde_1(s: torch.Tensor, tau: torch.Tensor, K: float, r: float, sigma: float) -> torch.Tensor:
    """d_tilde_1 = -(1/(sigma*sqrt(tau))) * (ln(s/K) + (r + sigma^2/2)*tau)."""
    tau_safe = torch.clamp(tau, min=_TAU_EPS)
    return -(1.0 / (sigma * torch.sqrt(tau_safe))) * (
        torch.log(s / K) + (r + 0.5 * sigma**2) * tau_safe
    )


def _d_tilde_2(s: torch.Tensor, tau: torch.Tensor, K: float, r: float, sigma: float) -> torch.Tensor:
    """d_tilde_2 = -(1/(sigma*sqrt(tau))) * (ln(s/K) + (r - sigma^2/2)*tau)."""
    tau_safe = torch.clamp(tau, min=_TAU_EPS)
    return -(1.0 / (sigma * torch.sqrt(tau_safe))) * (
        torch.log(s / K) + (r - 0.5 * sigma**2) * tau_safe
    )


def _d_tilde_0(s: torch.Tensor, tau: torch.Tensor, K: float, r: float, sigma: float) -> torch.Tensor:
    """d_tilde_0 = 0.5*(d_tilde_1 + d_tilde_2) = -(1/(sigma*sqrt(tau)))*(ln(s/K) + r*tau)."""
    tau_safe = torch.clamp(tau, min=_TAU_EPS)
    return -(1.0 / (sigma * torch.sqrt(tau_safe))) * (
        torch.log(s / K) + r * tau_safe
    )


def black_scholes_put(
    s: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    """European put price V^e(s, t) via the Black-Scholes formula (Eq. 16).

    V^e = K * exp(-r*tau) * N(d_tilde_2) - s * N(d_tilde_1)

    Args:
        s:      Underlying asset price (tensor).
        K:      Strike price.
        r:      Risk-free rate.
        sigma:  Volatility.
        tau:    Time to maturity T - t (tensor, same shape as s or broadcastable).

    Returns:
        European put option price (same shape as s).
    """
    dt1 = _d_tilde_1(s, tau, K, r, sigma)
    dt2 = _d_tilde_2(s, tau, K, r, sigma)
    return K * torch.exp(-r * tau) * _normal_cdf(dt2) - s * _normal_cdf(dt1)


# ---------------------------------------------------------------------------
# Taylor expansion terms V1^e, V2^e  (Section 4.1.1)
# ---------------------------------------------------------------------------

def european_put_ve1(
    s: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Zeroth-order Taylor expansion term of the European put.

    V1^e(s, t) = N(d_tilde_0) * (K * exp(-r*tau) - s)
    """
    dt0 = _d_tilde_0(s, tau, K, r, sigma)
    return _normal_cdf(dt0) * (K * torch.exp(-r * tau) - s)


def european_put_ve2(
    s: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    """First-order Taylor expansion term capturing the sqrt(tau) singularity.

    V2^e(s, t) = (sigma*sqrt(tau)) / (2*sqrt(2*pi)) * exp(-d_tilde_0^2/2) * (K*exp(-r*tau) + s)
    """
    tau_safe = torch.clamp(tau, min=_TAU_EPS)
    dt0 = _d_tilde_0(s, tau, K, r, sigma)
    return (
        (sigma * torch.sqrt(tau_safe)) / (2.0 * _SQRT_2PI)
        * torch.exp(-0.5 * dt0**2)
        * (K * torch.exp(-r * tau_safe) + s)
    )


# ---------------------------------------------------------------------------
# Exact terminal functions g1, g2  (Section 4.1)
# ---------------------------------------------------------------------------

def g2_american_put(
    s: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Exact terminal function for the American put (Taylor-expanded form).

    g2 = V1^e + V2^e  (Zhang, Guo, Lu 2026, Section 4.1.1).
    Satisfies the terminal condition and captures the sqrt(tau) singularity.

    Args:
        s:      Underlying asset price.
        K:      Strike price.
        r:      Risk-free rate.
        sigma:  Volatility.
        tau:    Time to maturity T - t.
    """
    return european_put_ve1(s, K, r, sigma, tau) + european_put_ve2(s, K, r, sigma, tau)


def g1_linear(T: float, t: torch.Tensor) -> torch.Tensor:
    """Canonical choice g1(s, t) = T - t (vanishes at t = T).

    Args:
        T:  Expiration date.
        t:  Current time (scalar or tensor).
    """
    return T - t
