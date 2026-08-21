r"""Down-and-out put: composite distance, corner-regularised extension, closed form.

Reference: working note "A rigorous statement of exact-constraint learning at a
conflicting constraint corner: the knock-out barrier option" (S. Ouaissi
internship, 2026-06-24), Sections 2 and 4.

The contract is a down-and-out put with strike ``K``, knock-out barrier ``B``,
and ``0 < B < K`` (Assumption 1 of the note: a *reverse* knock-out, in which the
intrinsic payoff at the barrier is strictly positive, ``g(B) = K - B > 0``).
The pricing boundary-value problem on ``Q = (B, +inf) x (0, T)`` is

.. math::

    \mathcal L^{BS} V_{DO} = 0 \text{ on } Q, \qquad
    V_{DO} = g \text{ on } \Sigma_T, \qquad
    V_{DO} = 0 \text{ on } \Sigma_B,

where :math:`\Sigma_T` is the terminal lid and :math:`\Sigma_B` the barrier
face.  The two data conflict at the corner :math:`\mathfrak c = (B, T)`
(Proposition 1 of the note): no continuous function carries both traces
exactly.  This module supplies the three code-level counterparts of the note's
Sections 2 and 4:

- :func:`barrier_composite_distance` -- the composite distance
  :math:`d_{\partial_p Q} = (T-t)(s-B)` of Definition 4, vanishing exactly on
  :math:`\Sigma_T \cup \Sigma_B`.
- :func:`make_corner_regularised_extension` -- a corner-regularised extension
  :math:`h_\varepsilon` in the sense of Definition 5, i.e. satisfying (11).
- :func:`reiner_rubinstein_down_and_out_put` -- the exact closed-form price
  :math:`V_{DO}` (method of images / Reiner-Rubinstein 1991), the reference
  of Remark 6.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from learning_option_pricing.pricing.terminal import payoff_put

_TAU_EPS = 1e-8  # epsilon floor to avoid division by zero when tau -> 0


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Composite distance d_{ΣT} d_{ΣB}  (Definition 4)
# ---------------------------------------------------------------------------

def barrier_composite_distance(
    s: torch.Tensor,
    t: torch.Tensor,
    B: float,
    T: float,
) -> torch.Tensor:
    r"""Composite distance :math:`d_{\partial_p Q}(s,t) = (T-t)(s-B)` (eq. 9).

    The canonical elementary choice of Definition 4: :math:`d_{\Sigma_T}(s,t) =
    T-t` and :math:`d_{\Sigma_B}(s,t) = s-B`.  Vanishes exactly on
    :math:`\Sigma_T` (every ``s``, at ``t=T``) and on :math:`\Sigma_B` (every
    ``t``, at ``s=B``), including at the corner itself; no regularisation is
    needed for this factor (only the extension :math:`h_\varepsilon` carries
    the corner nuisance -- see :func:`make_corner_regularised_extension`).

    Args:
        s: Underlying asset price, any shape.
        t: Time, broadcastable with ``s``.
        B: Knock-out barrier.
        T: Maturity.

    Returns:
        :math:`d_{\partial_p Q}(s,t)`, broadcast shape of ``s`` and ``t``.
    """
    return (T - t) * (s - B)


# ---------------------------------------------------------------------------
# Corner-regularised extension h_epsilon  (Definition 5)
# ---------------------------------------------------------------------------

def _bump(r: torch.Tensor) -> torch.Tensor:
    r"""The standard :math:`C^\infty` bump :math:`f(r) = e^{-1/r}` for :math:`r>0`, 0 otherwise.

    ``safe`` substitutes a placeholder (1.0) wherever ``r <= 0`` purely to keep
    ``torch.exp`` finite there; the substituted value is discarded by
    ``torch.where`` and never contributes to the returned value or its
    gradient (the usual safe-masking pattern for a piecewise analytic
    function).
    """
    safe = torch.where(r > 0, r, torch.ones_like(r))
    return torch.where(r > 0, torch.exp(-1.0 / safe), torch.zeros_like(r))


def _smoothstep01(r: torch.Tensor) -> torch.Tensor:
    r""":math:`C^\infty` transition :math:`\zeta:\mathbb R\to[0,1]`, :math:`\zeta(r)=0` for
    :math:`r\le 0`, :math:`\zeta(r)=1` for :math:`r\ge 1`.

    :math:`\zeta(r) = f(r) / (f(r) + f(1-r))` with :math:`f` the bump of
    :func:`_bump`.  The denominator is strictly positive for every real
    ``r`` (at least one of ``r>0`` or ``1-r>0`` always holds), so no epsilon
    floor is needed in the division.
    """
    a = _bump(r)
    b = _bump(1.0 - r)
    return a / (a + b)


def make_corner_regularised_extension(
    K: float,
    B: float,
    epsilon: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""Build an admissible :math:`\varepsilon`-extension :math:`h_\varepsilon` (Definition 5).

    .. math::

        h_\varepsilon(s,t) = \zeta\!\left(\frac{s-B}{\varepsilon}\right) g(s),
        \qquad g(s) = (K-s)^+,

    with :math:`\zeta` the smooth cutoff of :func:`_smoothstep01`.  This
    construction is **time-independent** -- simpler than the literal
    :math:`\ell^1`-ball corner layer :math:`\mathcal N_\varepsilon` of
    Definition 5, but rigorously admissible: it satisfies the three
    conditions (11) of the note.

    - On :math:`\Sigma_T` (``t=T``): :math:`\mathcal N_\varepsilon \cap
      \Sigma_T = \{s : s-B \le \varepsilon\}` exactly, and :math:`\zeta((s-B)/
      \varepsilon)=1` exactly for :math:`s-B>\varepsilon`, so
      :math:`h_\varepsilon(s,T) = g(s)` there -- matching the required trace
      exactly outside the corner layer.
    - On :math:`\Sigma_B` (``s=B``): :math:`\zeta(0)=0` identically, so
      :math:`h_\varepsilon(B,t) = 0` for *every* ``t``, not only outside
      :math:`\mathcal N_\varepsilon` -- strictly stronger than required,
      because the barrier datum is identically zero.
    - :math:`\|h_\varepsilon\|_{L^\infty(\mathcal N_\varepsilon)} \le K-B` is
      automatic: :math:`\zeta \in [0,1]` and, on the domain :math:`\Omega =
      (B,+\infty)`, :math:`g` is strictly decreasing, so :math:`g(s) < g(B) =
      K-B` for every :math:`s > B`.

    Args:
        K: Strike price.
        B: Knock-out barrier, :math:`0 < B < K`.
        epsilon: Bandwidth of the corner regularisation, :math:`\varepsilon > 0`.

    Returns:
        A callable ``h_eps(s, t) -> Tensor`` broadcasting over ``s`` and ``t``.

    Raises:
        ValueError: If ``epsilon <= 0`` or ``B >= K``.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}.")
    if not (0.0 < B < K):
        raise ValueError(f"the reverse knock-out regime requires 0 < B < K; got {B=}, {K=}.")

    def h_eps(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        weight = _smoothstep01((s - B) / epsilon)
        return weight * payoff_put(s, K)

    return h_eps


# ---------------------------------------------------------------------------
# Closed-form reference: Reiner-Rubinstein / method of images  (Remark 6)
# ---------------------------------------------------------------------------

def _put_style_price(
    s: torch.Tensor,
    strike: float,
    r: float,
    sigma: float,
    tau_safe: torch.Tensor,
) -> torch.Tensor:
    r"""European put price :math:`K'e^{-r\tau}N(-d_-) - sN(-d_+)` for an arbitrary strike.

    Internal building block, evaluated once with ``strike=K`` and once with
    ``strike=B`` by :func:`_truncated_put`; ``d_+``/``d_-`` are the usual
    Black-Scholes terms with ``strike`` in the log-moneyness.
    """
    sigma_sqrt_tau = sigma * torch.sqrt(tau_safe)
    d_plus = (
        torch.log(s / strike) + (r + 0.5 * sigma**2) * tau_safe
    ) / sigma_sqrt_tau
    d_minus = d_plus - sigma_sqrt_tau
    return (
        strike * torch.exp(-r * tau_safe) * _normal_cdf(-d_minus)
        - s * _normal_cdf(-d_plus)
    )


def _truncated_put(
    s: torch.Tensor,
    K: float,
    B: float,
    r: float,
    sigma: float,
    tau_safe: torch.Tensor,
) -> torch.Tensor:
    r"""Truncated expectation :math:`e^{-r\tau}\mathbb E[(K-S_\tau)^+ \mathbb 1_{S_\tau > B}]`.

    Since :math:`B < K`, this equals :math:`e^{-r\tau}\mathbb E[(K-S_\tau)
    \mathbb 1_{B<S_\tau<K}]` = (full put with strike ``K``) minus (the same
    put-style expression truncated below at ``B``, i.e. evaluated with
    ``strike=B`` in the log-moneyness but ``K`` kept as the payoff scale) --
    verified against direct numerical integration of the lognormal density
    over :math:`(B,K)` to machine precision during development.
    """
    return (
        _put_style_price(s, K, r, sigma, tau_safe)
        - K * torch.exp(-r * tau_safe) * _normal_cdf(
            -( (torch.log(s / B) + (r - 0.5 * sigma**2) * tau_safe)
               / (sigma * torch.sqrt(tau_safe)) )
        )
        + s * _normal_cdf(
            -( (torch.log(s / B) + (r + 0.5 * sigma**2) * tau_safe)
               / (sigma * torch.sqrt(tau_safe)) )
        )
    )


def reiner_rubinstein_down_and_out_put(
    s: torch.Tensor,
    K: float,
    B: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    r"""Exact down-and-out put price :math:`V_{DO}(s,t)` (method of images, Remark 6).

    Derived from the reflection principle for absorbed Brownian motion with
    drift, in the log-price coordinate :math:`x=\ln s`, drift :math:`\nu=r-
    \sigma^2/2`, absorbed at :math:`b=\ln B`.  For :math:`x_0=\ln s > b`, the
    transition density of the process killed at first hitting :math:`b` is

    .. math::

        p_{\mathrm{abs}}(\tau;x_0,x) = p(\tau;x_0,x)
            - e^{2\nu(b-x_0)/\sigma^2}\, p(\tau;2b-x_0,x), \qquad x>b,

    where :math:`p` is the ordinary (unrestricted) drifted transition
    density.  Because the payoff :math:`g` does **not** vanish at the barrier
    here (:math:`g(B)=K-B>0`, the reverse knock-out of Remark 4 -- unlike the
    textbook case where a plain reflected-vanilla-price identity would
    suffice), the price is the *truncated* expectation over
    :math:`S_\tau>B` in both the direct and the reflected term:

    .. math::

        V_{DO}(s,\tau) = \mathrm{TP}(s,K,B,\tau)
            - \left(\frac{B}{s}\right)^{2\nu/\sigma^2}
              \mathrm{TP}\!\left(\frac{B^2}{s},K,B,\tau\right),

    with :math:`\mathrm{TP}` the truncated put of :func:`_truncated_put` and
    exponent :math:`2\nu/\sigma^2 = 2r/\sigma^2-1`.  The reflected spot
    :math:`B^2/s` comes from :math:`e^{2b-x_0} = B^2/s`.

    Development note (not re-derived at import time): an initial
    implementation copied from memory of the Reiner-Rubinstein tabulated
    A-B-C-D form had the reflection prefactor inverted
    (:math:`(s/B)^{2\nu/\sigma^2}` instead of :math:`(B/s)^{2\nu/\sigma^2}`)
    and was off by 10-20% against an independent discretely-monitored
    Monte-Carlo simulation with a convergence sweep over the monitoring
    frequency. The formula above was re-derived directly from the reflection
    principle and validated: the truncated-put building block matches direct
    numerical integration of the lognormal density to machine precision, and
    the full formula's deviation from Monte-Carlo shrinks like :math:`O(1/
    \sqrt N)` in the number of monitoring steps :math:`N` (from -0.175 at
    N=252 to -0.042 at N=4000, on ``K=100,B=80,r=0.02,sigma=0.25,T=1,s=100``)
    -- the signature of a correct continuous-barrier formula compared against
    a discretely-monitored simulation, not a residual formula error.

    Args:
        s:     Underlying asset price tensor.  Values ``s <= B`` are already
               knocked out and price at exactly ``0.0``.
        K:     Strike price, with ``K > B`` (reverse knock-out regime).
        B:     Knock-out barrier, :math:`0 < B < K`.
        r:     Risk-free rate (also the cost-of-carry; no dividend).
        sigma: Volatility.
        tau:   Time to maturity :math:`T-t`, tensor broadcastable with ``s``.

    Returns:
        :math:`V_{DO}(s,t)`, same broadcast shape as ``s``/``tau``.

    Raises:
        ValueError: If ``B >= K`` (outside the regime this formula covers).
    """
    if not (0.0 < B < K):
        raise ValueError(
            f"reiner_rubinstein_down_and_out_put covers only the reverse "
            f"knock-out regime 0 < B < K; got {B=}, {K=}."
        )

    tau_safe = torch.clamp(tau, min=_TAU_EPS)

    # s is clamped away from B before taking logs; the formula is overridden
    # to 0.0 for s <= B by the final torch.where, so the clamped branch is
    # never actually used there.
    s_safe = torch.clamp(s, min=B * (1.0 + 1e-6))
    s_reflected = B**2 / s_safe

    exponent = 2.0 * r / sigma**2 - 1.0  # = 2*nu/sigma^2, nu = r - sigma^2/2

    price = (
        _truncated_put(s_safe, K, B, r, sigma, tau_safe)
        - (B / s_safe) ** exponent * _truncated_put(s_reflected, K, B, r, sigma, tau_safe)
    )
    return torch.where(s > B, price, torch.zeros_like(price))
