"""Exact terminal / exercise-date functions (g1, g2) for BSM equations.

Reference: Zhang, Guo, Lu (2026) — ETCNN, Sections 3.4–3.5.

For a European call:
    g1(s, t) = T - t
    g2(s, t) = (1/2) * (s - K + sqrt(s^2 + K^2 - 2sK e^{-r(T-t)}))

For an American put (recommended g2):
    g2 = V1^e(s, t) + V2^e(s, t)
    where V1^e and V2^e are the zeroth- and first-order terms of the
    Taylor expansion of the European put price around d̃_0.

Bermuda options require a piecewise generalisation — TODO.
"""
from __future__ import annotations

import math


def black_scholes_put(s, K: float, r: float, sigma: float, tau: float):
    """European put price V^e(s, t) via the Black-Scholes formula.

    Args:
        s:      Underlying asset price (scalar or array).
        K:      Strike price.
        r:      Risk-free rate.
        sigma:  Volatility.
        tau:    Time to maturity T - t.

    Returns:
        Option price (same type/shape as s).
    """
    # TODO
    raise NotImplementedError


def g2_american_put(s, K: float, r: float, sigma: float, tau: float):
    """Exact terminal function for the American put (Taylor-expanded form).

    g2 = V1^e + V2^e  (Zhang, Guo, Lu 2026, Section 4.1.1).
    Satisfies the terminal condition and captures the √τ singularity.

    Args:
        s:      Underlying asset price.
        K:      Strike price.
        r:      Risk-free rate.
        sigma:  Volatility.
        tau:    Time to maturity T - t.
    """
    # TODO
    raise NotImplementedError


def g1_linear(T: float, t):
    """Canonical choice g1(s, t) = T - t (vanishes at t = T).

    Args:
        T:  Expiration date.
        t:  Current time (scalar or array).
    """
    return T - t
