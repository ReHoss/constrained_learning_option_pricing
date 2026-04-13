"""Bjerksund--Stensland (2002) one-step American option approximation.

Implements the flat-boundary call approximation (Eq. 4) and obtains American
put prices via the put--call transformation (Eq. 19):

    P(S, K, T, r, b, sigma) = C(K, S, T, r-b, -b, sigma)

where b = r - q is the cost-of-carry.

All functions use differentiable PyTorch operations so the BSM PDE source
term F(g2) can be computed via autograd when g2 = BS-2002 put.

Reference
---------
P. Bjerksund and G. Stensland, *Closed form valuation of American options*,
Discussion paper 2002/09, Department of Finance, NHH (2002).
"""
from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

_TAU_EPS = 1e-8  # floor to prevent division by zero when tau -> 0


# ---------------------------------------------------------------------------
# Standard normal CDF  (duplicated from terminal.py to avoid circular import)
# ---------------------------------------------------------------------------

def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# phi function  (BS-2002 Eq. 7)
# ---------------------------------------------------------------------------

def _phi_bs2002(
    S: float,
    tau: torch.Tensor,
    sqrt_tau: torch.Tensor,
    gamma: float,
    H: torch.Tensor,
    I: torch.Tensor,
    r: float,
    b: float,
    sigma: float,
) -> torch.Tensor:
    r"""BS-2002 :math:`\varphi` function (Eq. 7).

    Following the Haug (2007) VBA sign convention:

    .. math::

        \varphi(S,T,\gamma,H,I)
          = e^{\lambda T}\,S^{\gamma}
            \bigl[N(d) - (I/S)^{\kappa}\,N(d_2)\bigr]

    where

    .. math::

        d = -\frac{\ln(S/H) + (b + (\gamma - \tfrac12)\sigma^2)T}
                   {\sigma\sqrt{T}}, \qquad
        d_2 = d - \frac{2\ln(I/S)}{\sigma\sqrt{T}}

    Args:
        S:        Scalar spot of the transformed call (= original strike *K*).
        tau:      Time-to-maturity tensor (clamped, > 0).
        sqrt_tau: Pre-computed :math:`\sqrt{\tau}` tensor.
        gamma:    Exponent parameter (0, 1, or :math:`\beta`).
        H:        Reference price tensor (threshold, H <= I).
        I:        Barrier trigger tensor.
        r:        Risk-free rate (transformed).
        b:        Cost-of-carry (transformed).
        sigma:    Volatility.

    Returns:
        :math:`\varphi` evaluated element-wise, same shape as *H*.
    """
    lam = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma ** 2) * tau
    kappa = 2.0 * b / sigma ** 2 + (2.0 * gamma - 1.0)

    S_t = torch.tensor(S, dtype=tau.dtype, device=tau.device)

    # d = -[ln(S/H) + (b + (gamma - 0.5)*sigma^2)*tau] / (sigma*sqrt_tau)
    d = -(torch.log(S_t / H) + (b + (gamma - 0.5) * sigma ** 2) * tau) / (
        sigma * sqrt_tau
    )

    # d2 = d - 2*ln(I/S) / (sigma*sqrt_tau)
    d2 = d - 2.0 * torch.log(I / S) / (sigma * sqrt_tau)

    ratio_kappa = (I / S) ** kappa
    S_gamma = S ** gamma if gamma != 0.0 else 1.0

    return torch.exp(lam) * S_gamma * (
        _normal_cdf(d) - ratio_kappa * _normal_cdf(d2)
    )


# ---------------------------------------------------------------------------
# American put — one-step BS-2002 via put--call transformation
# ---------------------------------------------------------------------------

def bs2002_put(
    s: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
    q: float = 0.0,
) -> torch.Tensor:
    r"""Bjerksund--Stensland (2002) one-step American put approximation.

    Applies the put--call transformation (Eq. 19)

    .. math::

        P(S,K,T,r,b,\sigma) = C(K,S,T,r{-}b,{-}b,\sigma)

    where :math:`b = r - q` is the cost-of-carry, then evaluates the
    one-step flat-boundary call formula (Eq. 4).

    The exercise trigger :math:`I_2` of the transformed call factorises as
    :math:`I_2 = s \cdot C_{I_2}` where :math:`C_{I_2}` is a *scalar*
    function of :math:`(r,q,\sigma,\tau)` only, giving exercise boundary
    :math:`s^* = K / C_{I_2}`.

    Args:
        s:     Spot price tensor (may have ``requires_grad``).
        K:     Strike price (scalar).
        r:     Risk-free rate (must be > 0 for non-trivial early exercise).
        sigma: Volatility.
        tau:   Time to maturity :math:`T - t` (tensor, same shape as *s*).
        q:     Continuous dividend yield (default 0).

    Returns:
        American put price approximation, same shape as *s*.
    """
    # --- Transformed call parameters (Eq. 19) ---
    r_t = q            # r_tilde = r - b = q
    b_t = q - r        # b_tilde = -b   = q - r

    # If b_t >= r_t (i.e. r <= 0), the transformed call has no early
    # exercise premium; fall back to the European put.
    if b_t >= r_t:
        from learning_option_pricing.pricing.terminal import black_scholes_put
        return black_scholes_put(s, K, r, sigma, tau)

    # --- Scalar quantities (independent of s) ---
    beta = (0.5 - b_t / sigma ** 2) + math.sqrt(
        (b_t / sigma ** 2 - 0.5) ** 2 + 2.0 * r_t / sigma ** 2
    )

    C_Binf = beta / (beta - 1.0)

    if r_t > 0.0 and (r_t - b_t) > 0.0:
        C_B0 = max(1.0, r_t / (r_t - b_t))
    else:
        C_B0 = 1.0

    # --- Time-dependent quantities ---
    tau_safe = torch.clamp(tau, min=_TAU_EPS)
    sqrt_tau = torch.sqrt(tau_safe)

    denom = C_Binf - C_B0

    # h(T) = -(b_t*T + 2*sigma*sqrt(T)) * K_call^2 / ((B1 - B0)*B0)
    # Under transformation K_call = s, so the s^2 cancels:
    # h = -(b_t*tau + 2*sigma*sqrt_tau) / (denom * C_B0)   [Eq. 11]
    h = -(b_t * tau_safe + 2.0 * sigma * sqrt_tau) / (denom * C_B0)

    C_I2 = C_B0 + denom * (1.0 - torch.exp(h))

    # I_2 = s * C_I2  (exercise trigger of the transformed call)
    I2 = s * C_I2

    # alpha = (I_2 - s) * I_2^(-beta)   [Eq. 5 under transformation]
    alpha = (I2 - s) * I2 ** (-beta)

    # --- Call price formula (Eq. 4 under transformation) ---
    # S_call = K (original strike, scalar); K_call = s (original spot, tensor)
    S_call = K

    call_value = (
        alpha * S_call ** beta
        - alpha * _phi_bs2002(S_call, tau_safe, sqrt_tau, beta, I2, I2, r_t, b_t, sigma)
        + _phi_bs2002(S_call, tau_safe, sqrt_tau, 1.0, I2, I2, r_t, b_t, sigma)
        - _phi_bs2002(S_call, tau_safe, sqrt_tau, 1.0, s, I2, r_t, b_t, sigma)
        - s * _phi_bs2002(S_call, tau_safe, sqrt_tau, 0.0, I2, I2, r_t, b_t, sigma)
        + s * _phi_bs2002(S_call, tau_safe, sqrt_tau, 0.0, s, I2, r_t, b_t, sigma)
    )

    # Where K >= I_2 (deep ITM for the put): exercise value = (K - s)+
    exercise_value = torch.clamp(K - s, min=0.0)
    deep_itm = (K >= I2)

    return torch.where(deep_itm, exercise_value, torch.clamp(call_value, min=0.0))


# ---------------------------------------------------------------------------
# Exercise boundary (diagnostic helper)
# ---------------------------------------------------------------------------

def bs2002_exercise_boundary(
    K: float,
    r: float,
    sigma: float,
    tau: float,
    q: float = 0.0,
) -> float:
    r"""Exercise boundary :math:`s^*` of the BS-2002 one-step American put.

    Under the put--call transformation the trigger factorises as
    :math:`I_2 = s \cdot C_{I_2}`, so the exercise boundary is

    .. math::
        s^* = K / C_{I_2}

    All spots :math:`s \le s^*` are in the immediate-exercise region.

    Args:
        K:     Strike price.
        r:     Risk-free rate (> 0).
        sigma: Volatility.
        tau:   Time to maturity.
        q:     Continuous dividend yield (default 0).

    Returns:
        Exercise boundary price :math:`s^*`.
    """
    b_t = q - r
    r_t = q

    if b_t >= r_t:
        return 0.0  # no early exercise

    beta = (0.5 - b_t / sigma ** 2) + math.sqrt(
        (b_t / sigma ** 2 - 0.5) ** 2 + 2.0 * r_t / sigma ** 2
    )
    C_Binf = beta / (beta - 1.0)

    if r_t > 0.0 and (r_t - b_t) > 0.0:
        C_B0 = max(1.0, r_t / (r_t - b_t))
    else:
        C_B0 = 1.0

    tau_safe = max(tau, _TAU_EPS)
    denom = C_Binf - C_B0
    h = -(b_t * tau_safe + 2.0 * sigma * math.sqrt(tau_safe)) / (denom * C_B0)
    C_I2 = C_B0 + denom * (1.0 - math.exp(h))

    return K / C_I2


# ---------------------------------------------------------------------------
# BSM source term F(g2) — numerical finite-difference evaluation
# ---------------------------------------------------------------------------

def bs2002_source_term(
    s: torch.Tensor,
    tau: torch.Tensor,
    K: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    h: float = 1e-3,
) -> torch.Tensor:
    r"""Compute the BSM PDE defect :math:`\mathcal{F}(g_2)` via finite differences.

    Since the BS-2002 approximation is *not* an exact BSM solution,
    :math:`\mathcal{F}(g_2) \neq 0` in general. This function evaluates
    the source term using central finite differences for spatial derivatives
    and a backward difference for the temporal derivative:

    .. math::

        \mathcal{F}(g_2) = \frac{\partial g_2}{\partial t}
          + \tfrac12 \sigma^2 s^2 \frac{\partial^2 g_2}{\partial s^2}
          + (r-q) s \frac{\partial g_2}{\partial s} - r\, g_2

    This can be pre-computed and cached since it does not depend on the
    network parameters :math:`\theta`.

    Args:
        s:     Spot price tensor.
        tau:   Time to maturity :math:`T - t` tensor.
        K:     Strike price.
        r:     Risk-free rate.
        sigma: Volatility.
        q:     Continuous dividend yield.
        h:     Finite-difference step size for spatial derivatives.

    Returns:
        Source term :math:`\mathcal{F}(g_2)`, same shape as *s*.
    """
    with torch.no_grad():
        g2 = bs2002_put(s, K, r, sigma, tau, q)
        g2_sp = bs2002_put(s + h, K, r, sigma, tau, q)
        g2_sm = bs2002_put(s - h, K, r, sigma, tau, q)

        dg2_ds = (g2_sp - g2_sm) / (2.0 * h)
        d2g2_ds2 = (g2_sp - 2.0 * g2 + g2_sm) / (h ** 2)

        h_tau = max(1e-6, float(tau.min()) * 0.01)
        g2_tminus = bs2002_put(s, K, r, sigma, tau + h_tau, q)
        # dg2/dt = -dg2/dtau
        dg2_dt = -(g2_tminus - g2) / h_tau

    return (
        dg2_dt
        + 0.5 * sigma ** 2 * s ** 2 * d2g2_ds2
        + (r - q) * s * dg2_ds
        - r * g2
    )
