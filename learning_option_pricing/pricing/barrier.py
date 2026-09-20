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
- :func:`barrier_composite_distance_with_far_field` -- the same with the
  far segment ``s = s_inf`` of the truncated domain added (Dirichlet there)
  :math:`d_{\partial_p Q} = (T-t)(s-B)` of Definition 4, vanishing exactly on
  :math:`\Sigma_T \cup \Sigma_B`.
- :func:`make_corner_regularised_extension` -- a corner-regularised extension
  :math:`h_\varepsilon` in the sense of Definition 5, i.e. satisfying (11).
- :func:`mangasarian_smoothed_put_payoff` -- the Chen-Mangasarian smoothed put
  payoff, replacing the non-smooth :math:`(K-s)^+` by a :math:`C^\infty`
  approximation with a bandwidth :math:`\varepsilon_0(t)` that may itself
  depend on time.
- :func:`make_corner_regularised_extension_with_smoothed_payoff` -- the same
  construction as :func:`make_corner_regularised_extension`, with the raw
  payoff replaced by :func:`mangasarian_smoothed_put_payoff`.
- :func:`make_corner_regularised_extension_with_black_scholes_payoff` -- the
  same construction again, with the raw payoff replaced by the exact
  Black-Scholes European put price (paralleling the ``--g2 bs`` mode already
  documented for the American-put ETCNN in ``documents/methodology/\
architecture.md``).
- :func:`make_corner_regularised_extension_split` -- the same corner cutoff
  applied to the split-semigroup extension of Proposition 7 ("Split-generator
  extension removes the floor") of the working note "On boundary-constrained
  learning of partial differential equations", Example 7: the terminal payoff
  smoothed by the heat semigroup of the Black-Scholes diffusion's principal
  part, i.e. :func:`~learning_option_pricing.pde.real_line_extension_fields.\
GaussianSemigroupExtensionField` evaluated on the log-price line and
  substituted back to the price coordinate ``s``.
- :func:`reiner_rubinstein_down_and_out_put` -- the exact closed-form price
  :math:`V_{DO}` (method of images / Reiner-Rubinstein 1991), the reference
  of Remark 6.
- :func:`down_and_out_digital_price` and
  :func:`down_and_out_digital_price_and_derivatives` -- the closed-form
  down-and-out cash-or-nothing price :math:`V_{DOD}` of equation (15) of the
  note (Section 5.1, Method 1), with its price and time derivatives.
- :class:`SubtractedDigitalCornerExtension` (built by
  :func:`make_subtracted_digital_extension`) -- the extension
  :math:`g_2 = \Delta V_{DOD} + \pi - \pi(B,\cdot)` of the exact-subtraction
  ansatz (Definition 7), :math:`\Delta = K - B`, for a terminal profile
  :math:`\pi` among :class:`RawPutPayoffTerminalProfile`,
  :class:`BlackScholesPutTerminalProfile` and
  :class:`SplitSemigroupPutTerminalProfile`; no corner layer, no bandwidth
  :math:`\varepsilon`; interior residual and price derivatives in closed form.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from learning_option_pricing.pde.real_line_extension_fields import (
    GaussianSemigroupExtensionField,
    PutPayoffGaussianSemigroupExtensionField,
)
from learning_option_pricing.pricing.terminal import black_scholes_put, payoff_put, _report_tau_floor_activation

_TAU_EPS = 1e-8  # epsilon floor to avoid division by zero when tau -> 0


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Probability density function of the standard normal distribution."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


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


def barrier_composite_distance_with_far_field(
    s: torch.Tensor,
    t: torch.Tensor,
    B: float,
    T: float,
    s_inf: float,
) -> torch.Tensor:
    r"""Composite distance of the TRUNCATED domain, :math:`(T-t)(s-B)\,\frac{s_\infty-s}{s_\infty-B}`.

    :func:`barrier_composite_distance` is the composite distance of the
    natural domain :math:`(B,\infty)\times(0,T)`, on which no boundary value is
    required at infinity (uniqueness holds in the class of bounded solutions).
    Training, however, samples the truncated domain
    :math:`Q_{s_\infty}=(B,s_\infty)\times(0,T)`, whose far segment
    :math:`\{s=s_\infty\}` belongs to the parabolic boundary: without a
    condition there the truncated problem is not well posed -- any solution
    :math:`w` of :math:`\mathcal L^{BS}w=0` vanishing on :math:`s=B` and
    :math:`t=T` with arbitrary trace on :math:`s=s_\infty` has zero interior
    residual, so the interior loss cannot see it. This factor vanishes on the
    three faces :math:`s=B`, :math:`t=T` and :math:`s=s_\infty`, so the trial
    solution :math:`g_1u_\theta+g_2` takes the value :math:`g_2(s_\infty,t)`
    on the far segment: a Dirichlet condition with the datum supplied by the
    terminal-function extension. The normalisation :math:`1/(s_\infty-B)`
    keeps the factor of the same order as the untruncated one near the
    barrier (:math:`(s_\infty-s)/(s_\infty-B)\to1` as :math:`s\to B`).

    Truncation error (weak maximum principle for the Black-Scholes operator,
    zeroth-order coefficient :math:`-r\le0`): the difference between the
    solution of the truncated problem with far datum :math:`\varphi(t)` and
    the exact price is bounded on the whole domain by
    :math:`\sup_t|\varphi(t)-V_{DO}(s_\infty,t)|`. The pilot logs this
    bound with :math:`\varphi=g_2(s_\infty,\cdot)`.

    Args:
        s: Underlying asset price, any shape.
        t: Time, broadcastable with ``s``.
        B: Knock-out barrier.
        T: Maturity.
        s_inf: Truncation price :math:`s_\infty>B`.

    Returns:
        The factor, broadcast shape of ``s`` and ``t``.

    Raises:
        ValueError: If ``s_inf <= B``.
    """
    if not s_inf > B:
        raise ValueError(f"barrier_composite_distance_with_far_field needs s_inf > B; got {s_inf=}, {B=}.")
    return (T - t) * (s - B) * (s_inf - s) / (s_inf - B)


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


def _bump_first_derivative(r: torch.Tensor) -> torch.Tensor:
    r"""First derivative :math:`f'(r) = f(r)/r^2` of the bump :func:`_bump`, 0 for :math:`r\le 0`."""
    safe = torch.where(r > 0, r, torch.ones_like(r))
    return torch.where(r > 0, _bump(r) / safe**2, torch.zeros_like(r))


def _bump_second_derivative(r: torch.Tensor) -> torch.Tensor:
    r"""Second derivative :math:`f''(r) = f(r)(1-2r)/r^4` of the bump, 0 for :math:`r\le 0`.

    From :math:`f'(r) = f(r)/r^2` (:func:`_bump_first_derivative`): :math:`f''(r)
    = f'(r)/r^2 - 2f(r)/r^3 = f(r)/r^4 - 2f(r)/r^3 = f(r)(1-2r)/r^4`.
    """
    safe = torch.where(r > 0, r, torch.ones_like(r))
    return torch.where(r > 0, _bump(r) * (1.0 - 2.0 * safe) / safe**4, torch.zeros_like(r))


def _smoothstep01_value_and_derivatives(
    r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r""":math:`(\zeta(r), \zeta'(r), \zeta''(r))` of the cutoff :func:`_smoothstep01`, in closed form.

    With :math:`a=f(r)`, :math:`b=f(1-r)` (:math:`f` the bump of :func:`_bump`)
    and :math:`\zeta=a/(a+b)`, the quotient rule gives

    .. math::

        \zeta' = \frac{a'b - ab'}{(a+b)^2},

    and differentiating again -- the :math:`a'b'` cross-terms cancel
    identically, since :math:`(a'b-ab')' = a''b + a'b' - a'b' - ab'' =
    a''b - ab''` -- gives

    .. math::

        \zeta'' = \frac{(a''b-ab'')(a+b) - 2(a'b-ab')(a'+b')}{(a+b)^3}.

    :math:`b(r)=f(1-r)` so :math:`b'(r)=-f'(1-r)` and :math:`b''(r)=f''(1-r)`
    (two applications of the chain rule on the sign-flipped argument).
    Verified against ``torch.autograd`` (double backward through
    :func:`_smoothstep01`) to machine precision (:math:`<2\times10^{-15}`)
    during development.
    """
    a = _bump(r)
    b = _bump(1.0 - r)
    a_prime = _bump_first_derivative(r)
    b_prime = -_bump_first_derivative(1.0 - r)
    a_double_prime = _bump_second_derivative(r)
    b_double_prime = _bump_second_derivative(1.0 - r)

    denominator = a + b
    zeta = a / denominator

    numerator_first_derivative = a_prime * b - a * b_prime
    zeta_prime = numerator_first_derivative / denominator**2

    numerator_second_derivative = a_double_prime * b - a * b_double_prime
    zeta_double_prime = (
        numerator_second_derivative * denominator
        - 2.0 * numerator_first_derivative * (a_prime + b_prime)
    ) / denominator**3

    return zeta, zeta_prime, zeta_double_prime


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
# Chen-Mangasarian smoothed put payoff, with an optional time-dependent
# smoothing bandwidth
# ---------------------------------------------------------------------------

def mangasarian_smoothed_put_payoff(
    s: torch.Tensor,
    t: torch.Tensor,
    K: float,
    T: float,
    eps0: float,
    grading: str = "time_graded",
) -> torch.Tensor:
    r"""Chen-Mangasarian :math:`C^\infty` smoothed put payoff.

    .. math::

        g_{\varepsilon_0}(s,t) = \frac{1}{2}\left(K-s+
            \sqrt{(K-s)^2+\varepsilon_0(t)^2}\right),

    a smoothed replacement for the non-smooth put payoff :math:`g(s)=(K-s)^+`
    of :func:`~learning_option_pricing.pricing.terminal.payoff_put`, distinct
    from and not to be confused with the corner-layer bandwidth
    :math:`\varepsilon` of :func:`make_corner_regularised_extension` -- the
    two epsilons act on different singularities (the payoff kink at
    :math:`s=K` here, the corner :math:`\mathfrak c=(B,T)` there) and are
    independent parameters throughout this module.

    The bandwidth :math:`\varepsilon_0(t)` is selected by ``grading``:

    - ``"constant"``: :math:`\varepsilon_0(t) = \varepsilon_0` for every
      ``t`` -- a uniform smoothing scale, never exactly recovering the raw
      payoff.
    - ``"time_graded"``: :math:`\varepsilon_0(t) = \varepsilon_0 (T-t)/T` --
      the smoothing bandwidth decreases linearly from :math:`\varepsilon_0`
      at ``t=0`` to exactly ``0`` at ``t=T``, so :math:`g_{\varepsilon_0}(s,T)
      = g(s)` exactly (the terminal trace is not perturbed by the smoothing).

    Args:
        s: Underlying asset price, any shape.
        t: Time, broadcastable with ``s``.
        K: Strike price.
        T: Maturity.
        eps0: Smoothing bandwidth scale, :math:`\varepsilon_0 > 0`.
        grading: ``"constant"`` or ``"time_graded"`` (default), selecting
            :math:`\varepsilon_0(t)` as above.

    Returns:
        :math:`g_{\varepsilon_0}(s,t)`, broadcast shape of ``s`` and ``t``.

    Raises:
        ValueError: If ``eps0 <= 0``, ``T <= 0``, or ``grading`` is neither
            ``"constant"`` nor ``"time_graded"``.
    """
    if eps0 <= 0.0:
        raise ValueError(f"eps0 must be > 0; got {eps0}.")
    if T <= 0.0:
        raise ValueError(f"T must be > 0; got {T}.")

    if grading == "constant":
        eps_t = torch.full_like(t, eps0)
    elif grading == "time_graded":
        eps_t = eps0 * (T - t) / T
    else:
        raise ValueError(
            f'grading must be "constant" or "time_graded"; got {grading!r}.'
        )

    diff = K - s
    return 0.5 * (diff + torch.sqrt(diff**2 + eps_t**2))


def make_corner_regularised_extension_with_smoothed_payoff(
    K: float,
    B: float,
    epsilon: float,
    T: float,
    eps0: float,
    grading: str = "time_graded",
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""Corner-regularised extension using the Chen-Mangasarian smoothed payoff.

    Identical construction to :func:`make_corner_regularised_extension`,

    .. math::

        h_{\varepsilon,\varepsilon_0}(s,t) = \zeta\!\left(\frac{s-B}{
            \varepsilon}\right) g_{\varepsilon_0}(s,t),

    except that the raw payoff :math:`g(s)=(K-s)^+` is replaced by the smoothed
    payoff :math:`g_{\varepsilon_0}` of :func:`mangasarian_smoothed_put_payoff`.
    The original, exact-payoff extension is left untouched by this addition;
    use that one when the non-smoothness of :math:`g` at :math:`s=K` is not a
    concern.

    ``epsilon`` (corner-layer bandwidth, Definition 5) and ``eps0``
    (Chen-Mangasarian smoothing bandwidth) are two independent parameters --
    see :func:`mangasarian_smoothed_put_payoff`.

    Two of the three conditions (11) of the note transfer unchanged, because
    they act on the cutoff :math:`\zeta`, not on the payoff:

    - On :math:`\Sigma_B` (``s=B``): :math:`\zeta(0)=0` identically, so
      :math:`h_{\varepsilon,\varepsilon_0}(B,t) = 0` for every ``t``.
    - :math:`\|h_{\varepsilon,\varepsilon_0}\|_{L^\infty(\mathcal N_\varepsilon)}
      \le K-B` (up to :math:`O(\varepsilon_0)`; see below).

    The terminal-face condition, :math:`h_{\varepsilon,\varepsilon_0}(s,T) =
    g(s)` for :math:`s-B>\varepsilon`, holds **exactly** only with
    ``grading="time_graded"`` (where :math:`\varepsilon_0(T)=0` makes
    :math:`g_{\varepsilon_0}(\cdot,T)=g` exactly); with ``grading="constant"``
    it holds only up to :math:`O(\varepsilon_0)`.

    Args:
        K: Strike price.
        B: Knock-out barrier, :math:`0 < B < K`.
        epsilon: Bandwidth of the corner regularisation, :math:`\varepsilon > 0`.
        T: Maturity, passed through to :func:`mangasarian_smoothed_put_payoff`.
        eps0: Chen-Mangasarian smoothing bandwidth, :math:`\varepsilon_0 > 0`.
        grading: ``"constant"`` or ``"time_graded"`` (default), forwarded to
            :func:`mangasarian_smoothed_put_payoff`.

    Returns:
        A callable ``h_eps(s, t) -> Tensor`` broadcasting over ``s`` and ``t``.

    Raises:
        ValueError: If ``epsilon <= 0``, ``B >= K``, ``eps0 <= 0``, ``T <= 0``,
            or ``grading`` is neither ``"constant"`` nor ``"time_graded"``.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}.")
    if not (0.0 < B < K):
        raise ValueError(f"the reverse knock-out regime requires 0 < B < K; got {B=}, {K=}.")
    if eps0 <= 0.0:
        raise ValueError(f"eps0 must be > 0; got {eps0}.")
    if T <= 0.0:
        raise ValueError(f"T must be > 0; got {T}.")
    if grading not in ("constant", "time_graded"):
        raise ValueError(
            f'grading must be "constant" or "time_graded"; got {grading!r}.'
        )

    def h_eps(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        weight = _smoothstep01((s - B) / epsilon)
        smoothed_payoff = mangasarian_smoothed_put_payoff(s, t, K, T, eps0, grading=grading)
        return weight * smoothed_payoff

    return h_eps


def make_corner_regularised_extension_with_black_scholes_payoff(
    K: float,
    B: float,
    epsilon: float,
    r: float,
    sigma: float,
    T: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    r"""Corner-regularised extension using the exact Black-Scholes European put price.

    Identical construction to :func:`make_corner_regularised_extension`,

    .. math::

        h_\varepsilon^{BS}(s,t) = \zeta\!\left(\frac{s-B}{\varepsilon}\right)
            V^e(s,t),

    except that the raw payoff :math:`g(s)=(K-s)^+` is replaced by the exact
    European put price :math:`V^e(s,t)` of
    :func:`~learning_option_pricing.pricing.terminal.black_scholes_put`, the
    third choice of terminal function alongside the raw payoff
    (:func:`make_corner_regularised_extension`) and the Chen-Mangasarian
    smoothed payoff (:func:`make_corner_regularised_extension_with_smoothed_payoff`)
    -- paralleling the ``--g2 bs`` mode already used for the American-put
    ETCNN (``documents/methodology/architecture.md``). Unlike the raw payoff,
    :math:`V^e` already solves :math:`\mathcal L^{BS}V^e=0` everywhere it is
    smooth, so :math:`h_\varepsilon^{BS}` is :math:`C^\infty` in :math:`s` for
    :math:`t<T` and its own extension-forcing term :math:`\mathcal P
    h_\varepsilon^{BS}` should vanish away from the corner layer, unlike the
    two other modes.

    On :math:`\Sigma_T` (``t=T``), :math:`V^e(s,T) = (K-s)^+` only up to the
    :math:`\tau`-floor :math:`\varepsilon_{\tau}=10^{-8}` internal to
    :func:`black_scholes_put` (a numerical floor against division by zero,
    not a deliberate smoothing bandwidth): the terminal-face match is exact
    to machine precision but not bit-exact, unlike
    :func:`make_corner_regularised_extension`.

    Args:
        K: Strike price.
        B: Knock-out barrier, :math:`0 < B < K`.
        epsilon: Bandwidth of the corner regularisation, :math:`\varepsilon > 0`.
        r: Risk-free rate, forwarded to :func:`black_scholes_put`.
        sigma: Volatility, forwarded to :func:`black_scholes_put`.
        T: Maturity.

    Returns:
        A callable ``h_eps(s, t) -> Tensor`` broadcasting over ``s`` and ``t``.

    Raises:
        ValueError: If ``epsilon <= 0``, ``B >= K``, or ``T <= 0``.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}.")
    if not (0.0 < B < K):
        raise ValueError(f"the reverse knock-out regime requires 0 < B < K; got {B=}, {K=}.")
    if T <= 0.0:
        raise ValueError(f"T must be > 0; got {T}.")

    def h_eps(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        weight = _smoothstep01((s - B) / epsilon)
        european_put_price = black_scholes_put(s, K, r, sigma, T - t)
        return weight * european_put_price

    return h_eps


class BlackScholesCornerExtension:
    r"""Black-Scholes corner extension exposing its interior residual in closed form.

    Same field as :func:`make_corner_regularised_extension_with_black_scholes_payoff`,

    .. math::

        h_\varepsilon^{BS}(s,t) = \zeta\!\left(\frac{s-B}{\varepsilon}\right) V^e(s,t),

    but callable as an object that additionally exposes
    ``black_scholes_residual(s, t, r, sigma)``.  The presence of that method is
    what selects the two-term assembly of the interior residual in
    ``pilot_down_and_out_put.compute_loss`` (the route introduced for the
    split-semigroup extension): the loss is then built as
    :math:`\mathcal F(g_1u_\theta) + \mathcal F(h_\varepsilon^{BS})`, the second
    term analytic, instead of differentiating the full trial solution as one
    autograd graph.

    This exists to remove a confound in the comparison of terminal-function
    modes: the split extension was the only mode trained through the two-term
    route, so its measured advantage on the learned Greeks could not be
    attributed to the extension rather than to the residual assembly.  With
    this class the Black-Scholes mode can be trained through the same route.

    **The residual in closed form.**  :math:`\zeta` depends on :math:`s` only,
    and :math:`V^e` solves the operator exactly for :math:`\tau>0`, so
    :math:`\mathcal L^{BS}V^e = 0` and the product rule leaves only the terms
    carrying a derivative of the cutoff:

    .. math::

        \mathcal L^{BS}h_\varepsilon^{BS}
            = \tfrac12\sigma^2s^2\big(\zeta''V^e + 2\zeta'\partial_sV^e\big)
              + rs\,\zeta'V^e,

    with :math:`\zeta'=\zeta_r'/\varepsilon` and
    :math:`\zeta''=\zeta_r''/\varepsilon^2` (chain rule on
    :math:`r=(s-B)/\varepsilon`), and
    :math:`\partial_sV^e(s,t) = -N(-d_+)` the European put Delta.  It vanishes
    identically for :math:`s-B>\varepsilon`, where :math:`\zeta'=\zeta''=0` --
    exactly, not to a tolerance.

    Args:
        K: Strike.
        B: Knock-out barrier, ``0 < B < K``.
        epsilon: Corner-layer bandwidth, ``epsilon > 0``.
        r: Risk-free rate, used by the field itself.
        sigma: Volatility, used by the field itself.
        T: Maturity.

    Raises:
        ValueError: If ``epsilon <= 0``, ``B >= K``, or ``T <= 0``.

    Note:
        The argument order matches its sibling builder
        :func:`make_corner_regularised_extension_with_black_scholes_payoff`
        (``K, B, epsilon, r, sigma, T``) so the two are interchangeable at a
        call site.
    """

    def __init__(self, K: float, B: float, epsilon: float, r: float, sigma: float, T: float) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0; got {epsilon}.")
        if not (0.0 < B < K):
            raise ValueError(f"the reverse knock-out regime requires 0 < B < K; got {B=}, {K=}.")
        if T <= 0.0:
            raise ValueError(f"T must be > 0; got {T}.")
        self.K, self.B, self.epsilon, self.T = K, B, epsilon, T
        self.r, self.sigma = r, sigma

    def __call__(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        weight = _smoothstep01((s - self.B) / self.epsilon)
        return weight * black_scholes_put(s, self.K, self.r, self.sigma, self.T - t)

    def _european_put_price_and_delta(
        self, s: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r""":math:`(V^e, \partial_sV^e)` with :math:`\partial_sV^e=-N(-d_+)`.

        The same ``_TAU_EPS`` floor as the rest of this module guards
        :math:`\tau\to0`; it is inert away from the terminal slice.
        """
        tau_safe = torch.clamp(self.T - t, min=_TAU_EPS)
        sigma_sqrt_tau = self.sigma * torch.sqrt(tau_safe)
        d_plus = (
            torch.log(s / self.K) + (self.r + 0.5 * self.sigma**2) * tau_safe
        ) / sigma_sqrt_tau
        price = black_scholes_put(s, self.K, self.r, self.sigma, self.T - t)
        # At tau = 0 the price is the payoff exactly (see black_scholes_put), whose
        # derivative is -1_{s<K}; the closed-form Delta at tau = _TAU_EPS would
        # differ from it only in an O(sqrt(_TAU_EPS)) neighbourhood of s = K.
        delta = torch.where(torch.as_tensor(self.T - t) > 0, -_normal_cdf(-d_plus),
                            -(s < self.K).to(price.dtype))
        return price, delta

    def black_scholes_residual(
        self, s: torch.Tensor, t: torch.Tensor, r: float, sigma: float
    ) -> torch.Tensor:
        r""":math:`\mathcal L^{BS}h_\varepsilon^{BS}(s,t)`, closed form (see the class docstring).

        Never autograd, never a finite difference.  ``h_eps`` does not depend
        on any trainable parameter, so this is a parameter-independent forcing
        term to be added to the residual of :math:`g_1u_\theta` computed
        separately.

        Args:
            s: Underlying asset price tensor.
            t: Time tensor, broadcastable with ``s``.
            r: Risk-free rate (the contract's; matches this extension's own).
            sigma: Volatility (the contract's; matches this extension's own).

        Returns:
            The residual, broadcast shape of ``s``/``t``.
        """
        ratio = (s - self.B) / self.epsilon
        _, zeta_prime_ratio, zeta_double_prime_ratio = _smoothstep01_value_and_derivatives(ratio)
        zeta_prime = zeta_prime_ratio / self.epsilon
        zeta_double_prime = zeta_double_prime_ratio / self.epsilon**2

        price, delta = self._european_put_price_and_delta(s, t)
        return (
            0.5 * sigma**2 * s**2 * (zeta_double_prime * price + 2.0 * zeta_prime * delta)
            + r * s * zeta_prime * price
        )


# ---------------------------------------------------------------------------
# Split-semigroup extension  (Proposition 7 / Example 7 of the working note)
# ---------------------------------------------------------------------------

def _log_price_and_broadcast(
    s: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """``(ln s, t, broadcast shape)`` with ``s``/``t`` broadcast against each
    other and flattened, as the real-line extension fields expect."""
    broadcast_shape = torch.broadcast_shapes(s.shape, t.shape)
    s_broadcast = s.expand(broadcast_shape).reshape(-1)
    t_broadcast = t.expand(broadcast_shape).reshape(-1)
    # Explicit log-price substitution: GaussianSemigroupExtensionField
    # lives on the log-price line x = ln(s), not on the price line s
    # itself (see that class's docstring, part (c) of its own question).
    return torch.log(s_broadcast), t_broadcast, broadcast_shape


def split_profile_price_and_time_derivatives(
    split_field: GaussianSemigroupExtensionField | PutPayoffGaussianSemigroupExtensionField,
    s: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``(pi, d_s pi, d_ss pi, d_t pi)`` of the split-semigroup profile in the
    price coordinate, chain-ruled from the field's own analytic log-price/time
    derivatives -- never by autograd or a finite difference through the
    quadratured field.  ``d_t`` needs no chain rule: ``t`` does not transform
    under ``x = ln(s)``, so the log-price partial derivative in ``t`` at fixed
    ``x`` already equals the price-space partial derivative in ``t`` at fixed
    ``s``.

    Queries the field exactly **once**, through
    ``field_and_space_derivatives`` (one shared kernel evaluation for
    ``h``/``d_x h``/``d_xx h``), and obtains ``d_t h = -nu_c d_xx h`` from
    the returned second derivative by the field's own heat equation -- the
    identity ``time_derivative`` itself implements.  For the quadrature-backed
    field this is one ``O(batch_size x n_quad)`` exponential instead of four
    (``field``, ``"dt"``, ``"dx"``, ``"dxx"`` each re-evaluating the same
    Gaussian), with bitwise the same tensors; for the closed-form field the
    cost is ``O(batch_size)`` either way.

    Shared by :class:`SplitSemigroupCornerExtension` (corner-smoothing
    ansatz) and :class:`SplitSemigroupPutTerminalProfile` (exact-subtraction
    ansatz), so both differentiate the profile through one code path.
    """
    log_price, t_flat, broadcast_shape = _log_price_and_broadcast(s, t)
    pi_value, d_x_pi, d_xx_pi = split_field.field_and_space_derivatives(log_price, t_flat)
    pi_value = pi_value.reshape(broadcast_shape)
    d_x_pi = d_x_pi.reshape(broadcast_shape)
    d_xx_pi = d_xx_pi.reshape(broadcast_shape)
    d_t_pi = -split_field.comparison_diffusivity * d_xx_pi

    s_reshaped = s.expand(broadcast_shape)
    d_s_pi = d_x_pi / s_reshaped
    d_ss_pi = (d_xx_pi - d_x_pi) / s_reshaped**2
    return pi_value, d_s_pi, d_ss_pi, d_t_pi


class SplitSemigroupCornerExtension:
    r"""Callable corner-regularised split-semigroup extension, with price derivatives.

    ``instance(s, t)`` returns :math:`h_\varepsilon^{\mathrm{split}}(s,t)`,
    exactly like the plain callables returned by the other three
    ``make_corner_regularised_extension*`` variants -- this class exists
    only so that ``first_price_derivative``/``second_price_derivative`` are
    declared, typed attributes rather than attached dynamically to a
    ``def``-created function (which a static type checker rejects for the
    ``Callable[[Tensor, Tensor], Tensor]`` return type used elsewhere in
    this module). Built by :func:`make_corner_regularised_extension_split`,
    which documents the mathematical construction; not instantiated
    directly elsewhere.
    """

    def __init__(
        self,
        K: float,
        B: float,
        epsilon: float,
        split_field: GaussianSemigroupExtensionField | PutPayoffGaussianSemigroupExtensionField,
    ) -> None:
        self.K = K
        self.B = B
        self.epsilon = epsilon
        self.split_field = split_field

    def __call__(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        log_price, t_flat, broadcast_shape = _log_price_and_broadcast(s, t)
        value = self.split_field.field(log_price, t_flat).reshape(broadcast_shape)
        weight = _smoothstep01((s - self.B) / self.epsilon)
        return weight * value

    def _profile_price_and_time_derivatives(
        self, s: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""``(pi, d_s pi, d_ss pi, d_t pi)`` of the split profile in the price
        coordinate; see :func:`split_profile_price_and_time_derivatives`,
        which this delegates to (shared with the exact-subtraction ansatz)."""
        return split_profile_price_and_time_derivatives(self.split_field, s, t)

    def first_price_derivative(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\partial_s h_\varepsilon^{\mathrm{split}} = \zeta'\,\pi + \zeta\,\partial_s\pi`."""
        pi_value, d_s_pi, _, _ = self._profile_price_and_time_derivatives(s, t)
        r = (s - self.B) / self.epsilon
        weight, zeta_prime, _ = _smoothstep01_value_and_derivatives(r)
        d_s_zeta = zeta_prime / self.epsilon
        return d_s_zeta * pi_value + weight * d_s_pi

    def second_price_derivative(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\partial_{ss} h_\varepsilon^{\mathrm{split}} = \zeta''\pi + 2\zeta'\partial_s\pi + \zeta\,\partial_{ss}\pi`."""
        pi_value, d_s_pi, d_ss_pi, _ = self._profile_price_and_time_derivatives(s, t)
        r = (s - self.B) / self.epsilon
        weight, zeta_prime, zeta_double_prime = _smoothstep01_value_and_derivatives(r)
        d_s_zeta = zeta_prime / self.epsilon
        d_ss_zeta = zeta_double_prime / self.epsilon**2
        return d_ss_zeta * pi_value + 2.0 * d_s_zeta * d_s_pi + weight * d_ss_pi

    def first_time_derivative(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\partial_t h_\varepsilon^{\mathrm{split}} = \zeta\,\partial_t\pi`
        (:math:`\zeta` does not depend on ``t``, so no product-rule term here,
        unlike the price derivatives above)."""
        _, _, _, d_t_pi = self._profile_price_and_time_derivatives(s, t)
        weight = _smoothstep01((s - self.B) / self.epsilon)
        return weight * d_t_pi

    def black_scholes_residual(
        self, s: torch.Tensor, t: torch.Tensor, r: float, sigma: float
    ) -> torch.Tensor:
        r"""Full Black-Scholes PDE operator applied to :math:`h_\varepsilon^{\mathrm{split}}`:

        .. math::

            \mathcal L^{BS}h_\varepsilon^{\mathrm{split}}
                = \partial_t h_\varepsilon^{\mathrm{split}}
                  + \tfrac12\sigma^2 s^2\,\partial_{ss}h_\varepsilon^{\mathrm{split}}
                  + r s\,\partial_s h_\varepsilon^{\mathrm{split}}
                  - r\,h_\varepsilon^{\mathrm{split}},

        assembled entirely from :meth:`_profile_price_and_time_derivatives`
        and the corner-cutoff product rule -- never by autograd or a finite
        difference through the quadratured field (see the class docstring).
        ``h_eps`` does not depend on any trainable parameter, so this is a
        parameter-independent forcing term: by linearity of
        :math:`\mathcal L^{BS}`,
        :math:`\mathcal L^{BS}(g_1 u_\theta + h_\varepsilon^{\mathrm{split}})
        = \mathcal L^{BS}(g_1 u_\theta) + \mathcal L^{BS}(h_\varepsilon^{\mathrm{split}})`,
        so a caller assembling the interior PDE residual of the full trial
        solution should add this to the residual of :math:`g_1 u_\theta`
        computed separately (by ordinary autograd on the smooth network
        manifold alone -- see
        :meth:`~learning_option_pricing.models.etcnn.ETCNN.forward_neural_manifold`),
        rather than differentiate the full trial solution as one graph.

        Args:
            s: Underlying asset price tensor.
            t: Time tensor, broadcastable with ``s``.
            r: Risk-free rate.
            sigma: Volatility (the contract's own, distinct from this
                extension's ``comparison_volatility``; see
                :func:`make_corner_regularised_extension_split`).

        Returns:
            :math:`\mathcal L^{BS}h_\varepsilon^{\mathrm{split}}(s,t)`, same
            broadcast shape as ``s``/``t``.
        """
        pi_value, d_s_pi, d_ss_pi, d_t_pi = self._profile_price_and_time_derivatives(s, t)
        r_arg = (s - self.B) / self.epsilon
        weight, zeta_prime, zeta_double_prime = _smoothstep01_value_and_derivatives(r_arg)
        d_s_zeta = zeta_prime / self.epsilon
        d_ss_zeta = zeta_double_prime / self.epsilon**2

        value = weight * pi_value
        d_t = weight * d_t_pi
        d_s = d_s_zeta * pi_value + weight * d_s_pi
        d_ss = d_ss_zeta * pi_value + 2.0 * d_s_zeta * d_s_pi + weight * d_ss_pi

        return d_t + 0.5 * sigma**2 * s**2 * d_ss + r * s * d_s - r * value


#: The two evaluation routes of the split-semigroup profile accepted by
#: :func:`make_corner_regularised_extension_split`.
SPLIT_PROFILE_ROUTES = ("closed_form", "quadrature")


def make_corner_regularised_extension_split(
    K: float,
    B: float,
    epsilon: float,
    T: float,
    comparison_volatility: float,
    y_lo: float | None = None,
    y_hi: float | None = None,
    n_quad: int = 8000,
    profile: str = "closed_form",
) -> SplitSemigroupCornerExtension:
    r"""Corner-regularised extension using the split-semigroup terminal profile.

    Identical corner cutoff to :func:`make_corner_regularised_extension`,

    .. math::

        h_{\varepsilon}^{\mathrm{split}}(s,t)
            = \zeta\!\left(\frac{s-B}{\varepsilon}\right)\,
              \bigl(e^{(T-t)\mathcal A_c} g\bigr)(\ln s),
            \qquad g(x) = (K-e^x)^+,

    except that the raw payoff is replaced by the split-semigroup profile of
    Proposition 7 (Example 7 specialises it to Black-Scholes): the terminal
    payoff advanced backward from :math:`T` by the heat semigroup of the
    diffusion's principal part :math:`\mathcal A_c = \nu_c\,\partial_{xx}`
    alone, leaving the drift-and-discount remainder :math:`\mathcal B` to be
    supplied by the caller's own residual assembly -- exactly as the other
    three ``make_corner_regularised_extension*`` variants only supply a
    terminal-value function of ``(s, t)`` and leave the PDE operator to the
    caller. This function performs no computation of its own beyond the
    corner cutoff and the log-price substitution: the semigroup convolution,
    its far-field behaviour, and the near-maturity quadrature floor are all
    :class:`~learning_option_pricing.pde.real_line_extension_fields.\
GaussianSemigroupExtensionField`'s (see that class's docstring for the
    quadrature-floor caveat, exposed by its own ``quadrature_floor_report``
    on the returned field).

    ``comparison_volatility`` is :math:`\sigma_c` (equivalently
    :math:`\nu_c=\sigma_c^2/2`), independent of the model's own volatility;
    passing the model's own :math:`\sigma` gives the matched split of
    Example 7, whose remainder forcing is bounded uniformly up to the
    terminal slice (mis-matching it reinstates an unbounded second-order
    channel -- see the referenced class's tests).

    **Evaluation route** (``profile``).  The datum is the put payoff
    :math:`g(x) = (K - e^x)^+`, for which the Gaussian convolution is an
    explicit integral (with :math:`m = \sigma_c\sqrt{T-t}` and
    :math:`c = \ln(K/s)/m`):

    .. math::

        \pi(s, t) = K\,\Phi(c) - s\,e^{\sigma_c^2 (T-t)/2}\,\Phi(c - m),

    together with closed-form :math:`\partial_x\pi`, :math:`\partial_{xx}\pi`
    (see :class:`~learning_option_pricing.pde.real_line_extension_fields.\
PutPayoffGaussianSemigroupExtensionField`).  ``profile="closed_form"`` (the
    default) evaluates that formula: it is the exact value of the same
    mathematical object the quadrature approximates, with no support
    truncation, no resolution error, no near-maturity unresolved band, and an
    :math:`O(n)` cost per call instead of :math:`O(n \times n_{\rm quad})`.
    ``profile="quadrature"`` keeps the fixed-grid trapezoidal route of
    :class:`~learning_option_pricing.pde.real_line_extension_fields.\
GaussianSemigroupExtensionField`, which is the generic route for a datum
    with no closed form and is retained here as a cross-check of the closed
    form (``test/pricing/test_barrier.py`` pins their agreement) and for
    reproducing runs made before the closed form existed.

    ``y_lo``/``y_hi``/``n_quad`` concern the quadrature route only: they are
    the fixed quadrature support of the underlying field, in the
    **log-price** coordinate, and must cover the evaluation window in ``s``
    padded by several diffusion lengths :math:`\sigma_c\sqrt T`, per the
    referenced class's own requirement.  They are ignored (and may be left
    ``None``) for the closed form.

    Args:
        K: Strike price.
        B: Knock-out barrier, :math:`0 < B < K`.
        epsilon: Bandwidth of the corner regularisation, :math:`\varepsilon > 0`.
        T: Maturity.
        comparison_volatility: :math:`\sigma_c` of the comparison heat
            semigroup, forwarded to
            :class:`~learning_option_pricing.pde.real_line_extension_fields.\
GaussianSemigroupExtensionField`.
        y_lo: Lower end of the log-price quadrature support
            (``profile="quadrature"`` only).
        y_hi: Upper end of the log-price quadrature support
            (``profile="quadrature"`` only).
        n_quad: Number of quadrature nodes (``profile="quadrature"`` only).
        profile: ``"closed_form"`` (default) or ``"quadrature"``; see above.

    Returns:
        A :class:`SplitSemigroupCornerExtension`, callable as ``h_eps(s, t)
        -> Tensor`` broadcasting over ``s`` and ``t``, exactly like the
        plain callables returned by the other three
        ``make_corner_regularised_extension*`` variants. It additionally
        exposes, as declared methods/attributes rather than dynamically
        attached ones (absent from the other three because their payoffs
        are plain closed forms differentiable by autograd with no
        cancellation risk, whereas the split-semigroup profile is a
        quadratured convolution -- see the referenced class's module
        docstring on why its second-order channel must not be assembled by
        autograd or finite differences near the terminal slice):

        - ``h_eps.first_price_derivative(s, t)``: :math:`\partial_s
          h_\varepsilon^{\mathrm{split}}`, by the product rule on
          :math:`\zeta` and the chain rule :math:`\partial_s\pi =
          (1/s)\,\partial_x\pi` from
          :meth:`GaussianSemigroupExtensionField.derivative_callables`'s
          ``"dx"``.
        - ``h_eps.second_price_derivative(s, t)``: :math:`\partial_{ss}
          h_\varepsilon^{\mathrm{split}} = \zeta''\pi + 2\zeta'\partial_s\pi +
          \zeta\,\partial_{ss}\pi`, with :math:`\partial_{ss}\pi =
          (1/s^2)(\partial_{xx}\pi - \partial_x\pi)` from the same
          derivative callables' ``"dxx"``/``"dx"`` -- never by autograd or a
          finite difference through the quadratured field, which would
          amplify the quadrature's own discretisation error by
          :math:`1/h^2`.
        - ``h_eps.split_field``: the underlying profile field
          (:class:`PutPayoffGaussianSemigroupExtensionField` or
          :class:`GaussianSemigroupExtensionField`), for
          ``h_eps.split_field.quadrature_floor_report()``.

    Raises:
        ValueError: If ``epsilon <= 0``, ``B >= K`` or ``profile`` is not one
            of :data:`SPLIT_PROFILE_ROUTES` (this function's own checks); if
            ``profile="quadrature"`` and ``y_lo``/``y_hi`` are missing; or if
            ``comparison_volatility <= 0`` or ``y_hi <= y_lo`` (raised by the
            profile field itself).
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon}.")
    if not (0.0 < B < K):
        raise ValueError(f"the reverse knock-out regime requires 0 < B < K; got {B=}, {K=}.")
    if profile not in SPLIT_PROFILE_ROUTES:
        raise ValueError(
            f"profile must be one of {SPLIT_PROFILE_ROUTES}; got {profile!r}."
        )

    if profile == "closed_form":
        split_field = PutPayoffGaussianSemigroupExtensionField(
            K=K,
            terminal_time=T,
            comparison_volatility=comparison_volatility,
            name="barrier_split_semigroup_closed_form",
        )
        return SplitSemigroupCornerExtension(K, B, epsilon, split_field)

    if y_lo is None or y_hi is None:
        raise ValueError(
            "profile='quadrature' requires the log-price quadrature support y_lo/y_hi."
        )

    def terminal_datum_on_the_log_price_line(x: torch.Tensor) -> torch.Tensor:
        return payoff_put(torch.exp(x), K)

    split_field = GaussianSemigroupExtensionField(
        terminal_datum_on_the_log_price_line,
        terminal_time=T,
        comparison_volatility=comparison_volatility,
        y_lo=y_lo,
        y_hi=y_hi,
        n_quad=n_quad,
        name="barrier_split_semigroup",
    )

    return SplitSemigroupCornerExtension(K, B, epsilon, split_field)


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
    # tau = 0 exactly: the closed form is undefined there and its limit is the
    # knocked-out payoff (K-s)^+ 1_{s>B}; return it exactly rather than the
    # price at tau = _TAU_EPS (see black_scholes_put for the same fix and the
    # size of the defect the floor used to introduce, 1.2e-5 at the strike).
    _report_tau_floor_activation(torch.as_tensor(tau), "reiner_rubinstein_down_and_out_put")
    price = torch.where(torch.as_tensor(tau) > 0, price, torch.clamp(K - s, min=0.0))
    return torch.where(s > B, price, torch.zeros_like(price))


# ---------------------------------------------------------------------------
# Closed-form Gamma of the Reiner-Rubinstein price (d^2 V_DO / d s^2)
# ---------------------------------------------------------------------------
#
# Derivation (mirrored, term for term, in a sympy script during development;
# every closed form below was checked there against sympy's own symbolic
# derivative by simplify-to-zero, i.e. exactly, not merely numerically):
#
#   Writing TP(x) = _truncated_put(x,K,B,r,sigma,tau), a direct term-by-term
#   differentiation (put-style Delta identity dP(x,strike)/dx = N(d_+)-1 =
#   -N(-d_+), applied once to the K-strike and once to the B-strike part of
#   TP, plus the product rule on the remaining N(-d_-(x,B)) term) gives
#
#     TP'(x)  =  N(d_+(x,K)) - N(d_+(x,B))
#                + (K-B) e^{-r tau} N'(d_-(x,B)) / (x sigma sqrt(tau)),
#
#     TP''(x) =  Gamma(x,K) - Gamma(x,B)
#                - (K-B) e^{-r tau} N'(d_-(x,B)) / (x^2 sigma sqrt(tau))
#                  * ( d_-(x,B) / (sigma sqrt(tau)) + 1 ),
#
#   with the ordinary vanilla Gamma building block Gamma(x,strike) =
#   N'(d_+(x,strike)) / (x sigma sqrt(tau)) (the formula quoted in the task).
#
#   V_DO(s) = TP(s) - w(s) TP(x2(s)), w(s)=(B/s)^p, x2(s)=B^2/s,
#   p = 2r/sigma^2-1 (the reflection prefactor and reflected spot -- Remark 6).
#   Both w and x2 depend on s alone (not on x), so the reflected term is a
#   product of two univariate compositions of s; the ordinary product rule
#   and chain rule give
#
#     w'(s)  = -(p/s) w(s),         w''(s)  = (p(p+1)/s^2) w(s),
#     x2'(s) = -B^2/s^2 = -x2(s)/s, x2''(s) = 2B^2/s^3 = 2 x2(s)/s^2,
#
#     R''(s) = w''(s) TP(x2) + 2 w'(s) TP'(x2) x2'(s)
#              + w(s) TP''(x2) x2'(s)^2 + w(s) TP'(x2) x2''(s),
#
#   and the Gamma of the full price is V_DO''(s) = TP''(s) - R''(s).

def _truncated_put_first_derivative(
    x: torch.Tensor,
    K: float,
    B: float,
    r: float,
    sigma: float,
    tau_safe: torch.Tensor,
) -> torch.Tensor:
    r""":math:`\mathrm d\,\mathrm{TP}(x)/\mathrm dx`, see the module-level derivation note."""
    sigma_sqrt_tau = sigma * torch.sqrt(tau_safe)
    d_plus_K = (torch.log(x / K) + (r + 0.5 * sigma**2) * tau_safe) / sigma_sqrt_tau
    d_plus_B = (torch.log(x / B) + (r + 0.5 * sigma**2) * tau_safe) / sigma_sqrt_tau
    d_minus_B = d_plus_B - sigma_sqrt_tau
    return (
        _normal_cdf(d_plus_K) - _normal_cdf(d_plus_B)
        + (K - B) * torch.exp(-r * tau_safe) * _normal_pdf(d_minus_B) / (x * sigma_sqrt_tau)
    )


def _truncated_put_gamma(
    x: torch.Tensor,
    K: float,
    B: float,
    r: float,
    sigma: float,
    tau_safe: torch.Tensor,
) -> torch.Tensor:
    r""":math:`\mathrm d^2\,\mathrm{TP}(x)/\mathrm dx^2`, see the module-level derivation note."""
    sigma_sqrt_tau = sigma * torch.sqrt(tau_safe)
    d_plus_K = (torch.log(x / K) + (r + 0.5 * sigma**2) * tau_safe) / sigma_sqrt_tau
    d_plus_B = (torch.log(x / B) + (r + 0.5 * sigma**2) * tau_safe) / sigma_sqrt_tau
    d_minus_B = d_plus_B - sigma_sqrt_tau
    gamma_K = _normal_pdf(d_plus_K) / (x * sigma_sqrt_tau)
    gamma_B = _normal_pdf(d_plus_B) / (x * sigma_sqrt_tau)
    return (
        gamma_K - gamma_B
        - (K - B) * torch.exp(-r * tau_safe) * _normal_pdf(d_minus_B) / (x**2 * sigma_sqrt_tau)
        * (d_minus_B / sigma_sqrt_tau + 1.0)
    )


def reiner_rubinstein_down_and_out_put_gamma(
    s: torch.Tensor,
    K: float,
    B: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    r"""Closed-form Gamma :math:`\partial_{ss}V_{DO}(s,t)` of the Reiner-Rubinstein price.

    Since :math:`V_{DO}` is itself a combination of Black-Scholes-style put
    terms (method of images, :func:`reiner_rubinstein_down_and_out_put`), its
    Gamma is a combination of the corresponding vanilla Gammas
    :math:`\Gamma(x,\mathrm{strike})=N'(d_+(x,\mathrm{strike}))/(x\sigma
    \sqrt\tau)`, together with the extra terms generated by differentiating
    the reflection prefactor :math:`(B/s)^{2\nu/\sigma^2}` and the reflected
    spot :math:`B^2/s` through the chain and product rules (both are
    functions of :math:`s` alone). Full derivation in the comment above this
    function; every closed form used here was checked there against a
    sympy symbolic derivative of the exact same expression the Python
    :func:`reiner_rubinstein_down_and_out_put` code evaluates, and the
    checks simplify to exactly zero (not merely numerically small).

    Args:
        s:     Underlying asset price tensor.  Values ``s <= B`` price at
               exactly ``0.0`` (the price is identically zero there, hence
               so is its second derivative).
        K:     Strike price, with ``K > B`` (reverse knock-out regime).
        B:     Knock-out barrier, :math:`0 < B < K`.
        r:     Risk-free rate (also the cost-of-carry; no dividend).
        sigma: Volatility.
        tau:   Time to maturity :math:`T-t`, tensor broadcastable with ``s``.

    Returns:
        :math:`\partial_{ss}V_{DO}(s,t)`, same broadcast shape as ``s``/``tau``.

    Raises:
        ValueError: If ``B >= K`` (outside the regime this formula covers).
    """
    if not (0.0 < B < K):
        raise ValueError(
            f"reiner_rubinstein_down_and_out_put_gamma covers only the reverse "
            f"knock-out regime 0 < B < K; got {B=}, {K=}."
        )

    tau_safe = torch.clamp(tau, min=_TAU_EPS)
    s_safe = torch.clamp(s, min=B * (1.0 + 1e-6))
    x2 = B**2 / s_safe

    p = 2.0 * r / sigma**2 - 1.0  # exponent, matches reiner_rubinstein_down_and_out_put

    w = (B / s_safe) ** p
    w_prime = -(p / s_safe) * w
    w_double_prime = (p * (p + 1.0) / s_safe**2) * w
    x2_prime = -x2 / s_safe
    x2_double_prime = 2.0 * x2 / s_safe**2

    TP_at_x2 = _truncated_put(x2, K, B, r, sigma, tau_safe)
    TP_prime_at_x2 = _truncated_put_first_derivative(x2, K, B, r, sigma, tau_safe)
    TP_gamma_at_x2 = _truncated_put_gamma(x2, K, B, r, sigma, tau_safe)

    reflected_term_gamma = (
        w_double_prime * TP_at_x2
        + 2.0 * w_prime * TP_prime_at_x2 * x2_prime
        + w * TP_gamma_at_x2 * x2_prime**2
        + w * TP_prime_at_x2 * x2_double_prime
    )

    gamma = _truncated_put_gamma(s_safe, K, B, r, sigma, tau_safe) - reflected_term_gamma
    return torch.where(s > B, gamma, torch.zeros_like(gamma))


# ---------------------------------------------------------------------------
# Down-and-out cash-or-nothing (digital) claim: closed form and derivatives
# (Method 1 of Section 5.1 of the working note -- exact singular subtraction)
# ---------------------------------------------------------------------------
#
# Derivation of the price derivatives (same structure as the Gamma of the
# Reiner-Rubinstein put above, with the truncated put replaced by the
# cumulative normal G(x) = N(a(x)), a(x) = d_-(x) taken relative to the
# barrier):
#
#   V_DOD(s, tau) = e^{-r tau} [ G(s) - w(s) G(x2(s)) ],
#   G(x)  = N(a(x)),   a(x) = (ln(x/B) + (r - sigma^2/2) tau) / m,   m = sigma sqrt(tau),
#   G'(x)  =  N'(a(x)) / (x m),
#   G''(x) = -N'(a(x)) (1 + a(x)/m) / (x^2 m),
#   w(s)  = (B/s)^p,  p = 2r/sigma^2 - 1,  w' = -(p/s) w,  w'' = p(p+1) w / s^2,
#   x2(s) = B^2/s,    x2' = -x2/s,          x2'' = 2 x2 / s^2,
#
#   d/ds  [w G(x2)] = w' G(x2) + w G'(x2) x2',
#   d2/ds2[w G(x2)] = w'' G(x2) + 2 w' G'(x2) x2' + w G''(x2) x2'^2 + w G'(x2) x2''.
#
# The time derivative is obtained from the operator identity L^BS V_DOD = 0
# (Proposition 4 of the note; the closed form is a superposition of a
# lognormal-tail probability and its reflection, each annihilated by the
# constant-coefficient operator), d_t V_DOD = -(sigma^2 s^2 / 2) d_ss V_DOD
# - r s d_s V_DOD + r V_DOD. test/pricing/test_barrier.py checks the three
# closed-form derivatives against float64 autograd of the price, and the
# operator identity itself by autograd, so no derivative here is trusted on
# the derivation alone.

def _digital_barrier_d_minus(x: torch.Tensor, B: float, r: float, sigma: float,
                             tau_safe: torch.Tensor) -> torch.Tensor:
    r""":math:`d_-(x) = (\ln(x/B) + (r - \sigma^2/2)\tau)/(\sigma\sqrt\tau)`,
    the Black-Scholes :math:`d_-` taken relative to the barrier ``B`` rather
    than a strike (equation (16) of the note)."""
    return (torch.log(x / B) + (r - 0.5 * sigma**2) * tau_safe) / (sigma * torch.sqrt(tau_safe))


def _digital_pieces(
    s: torch.Tensor, B: float, r: float, sigma: float, tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``(price, delta, gamma, tau_as_tensor, s_safe)`` of the digital for
    ``tau > 0``, on the clamped branch; the callers apply the ``tau = 0`` and
    ``s <= B`` conventions with ``torch.where``."""
    tau_tensor = torch.as_tensor(tau)
    tau_safe = torch.clamp(tau_tensor, min=_TAU_EPS)
    s_safe = torch.clamp(s, min=B * (1.0 + 1e-6))
    m = sigma * torch.sqrt(tau_safe)
    discount = torch.exp(-r * tau_safe)

    p = 2.0 * r / sigma**2 - 1.0  # reflection exponent, as in reiner_rubinstein_down_and_out_put
    x2 = B**2 / s_safe
    w = (B / s_safe) ** p
    w_prime = -(p / s_safe) * w
    w_double_prime = (p * (p + 1.0) / s_safe**2) * w
    x2_prime = -x2 / s_safe
    x2_double_prime = 2.0 * x2 / s_safe**2

    def G(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = _digital_barrier_d_minus(x, B, r, sigma, tau_safe)
        density = _normal_pdf(a)
        value = _normal_cdf(a)
        first = density / (x * m)
        second = -density * (1.0 + a / m) / (x**2 * m)
        return value, first, second

    G_s, G_prime_s, G_double_prime_s = G(s_safe)
    G_x2, G_prime_x2, G_double_prime_x2 = G(x2)

    reflected = w * G_x2
    reflected_prime = w_prime * G_x2 + w * G_prime_x2 * x2_prime
    reflected_double_prime = (
        w_double_prime * G_x2
        + 2.0 * w_prime * G_prime_x2 * x2_prime
        + w * G_double_prime_x2 * x2_prime**2
        + w * G_prime_x2 * x2_double_prime
    )
    price = discount * (G_s - reflected)
    delta = discount * (G_prime_s - reflected_prime)
    gamma = discount * (G_double_prime_s - reflected_double_prime)
    return price, delta, gamma, tau_tensor, s_safe


def down_and_out_digital_price(
    s: torch.Tensor,
    B: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> torch.Tensor:
    r"""Price :math:`V_{DOD}(s,t)` of the down-and-out cash-or-nothing claim
    (unit payoff :math:`\mathbf 1_{s>B}` at maturity, knocked out at ``B``),
    equation (15) of the note (Reiner-Rubinstein 1991):

    .. math::

        V_{DOD}(s,t) = e^{-r(T-t)}\Big[N\big(d_-(s,t)\big)
            - (s/B)^{1-2r/\sigma^2}\,N\big(d_-(B^2/s,\,t)\big)\Big],
        \qquad
        d_-(s,t) = \frac{\ln(s/B) + (r-\tfrac12\sigma^2)(T-t)}{\sigma\sqrt{T-t}}.

    It is a discounted no-knock-out probability, hence with values in
    :math:`[0,1]`; it solves :math:`\mathcal L^{BS}V_{DOD}=0` on ``Q``,
    vanishes on the barrier face (the two terms coincide at ``s = B``) and
    reproduces the unit corner jump exactly on the terminal face,
    :math:`V_{DOD}(s,T) = \mathbf 1_{s>B}`.  Multiplied by the jump
    :math:`\Delta = K - B` it is the closed-form, operator-exact singular
    part subtracted by :class:`SubtractedDigitalCornerExtension`.

    Args:
        s:     Underlying asset price tensor.  Values ``s <= B`` are already
               knocked out and price at exactly ``0.0``.
        B:     Knock-out barrier, ``B > 0``.
        r:     Risk-free rate.
        sigma: Volatility.
        tau:   Time to maturity :math:`T-t`, tensor broadcastable with ``s``.
               At ``tau = 0`` exactly the indicator :math:`\mathbf 1_{s>B}` is
               returned (the closed form's limit), not the value at the
               ``_TAU_EPS`` floor.

    Returns:
        :math:`V_{DOD}(s,t)`, same broadcast shape as ``s``/``tau``.
    """
    if B <= 0.0:
        raise ValueError(f"the barrier must be positive; got {B=}.")
    price, _, _, tau_tensor, _ = _digital_pieces(s, B, r, sigma, tau)
    _report_tau_floor_activation(tau_tensor, "down_and_out_digital_price")
    indicator = (s > B).to(price.dtype)
    price = torch.where(tau_tensor > 0, price, indicator)
    return torch.where(s > B, price, torch.zeros_like(price))


def down_and_out_digital_price_and_derivatives(
    s: torch.Tensor,
    B: float,
    r: float,
    sigma: float,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""``(V_DOD, d_s V_DOD, d_ss V_DOD, d_t V_DOD)`` in closed form.

    The price derivatives follow the derivation in the comment above
    :func:`_digital_barrier_d_minus`; the time derivative is obtained from the
    operator identity :math:`\mathcal L^{BS}V_{DOD}=0`,
    :math:`\partial_tV_{DOD} = -\tfrac12\sigma^2s^2\partial_{ss}V_{DOD}
    - rs\,\partial_sV_{DOD} + rV_{DOD}`.  At ``tau = 0`` exactly, the price is
    the indicator :math:`\mathbf 1_{s>B}` and every derivative is returned as
    ``0`` (the indicator is locally constant for ``s != B``).  For ``s <= B``
    everything is ``0``.

    Args:
        s, B, r, sigma, tau: As in :func:`down_and_out_digital_price`.

    Returns:
        Four tensors of the broadcast shape of ``s``/``tau``.
    """
    if B <= 0.0:
        raise ValueError(f"the barrier must be positive; got {B=}.")
    price, delta, gamma, tau_tensor, s_safe = _digital_pieces(s, B, r, sigma, tau)
    _report_tau_floor_activation(tau_tensor, "down_and_out_digital_price_and_derivatives")
    theta = -0.5 * sigma**2 * s_safe**2 * gamma - r * s_safe * delta + r * price

    positive_tau = tau_tensor > 0
    zero = torch.zeros_like(price)
    price = torch.where(positive_tau, price, (s > B).to(price.dtype))
    delta = torch.where(positive_tau, delta, zero)
    gamma = torch.where(positive_tau, gamma, zero)
    theta = torch.where(positive_tau, theta, zero)
    alive = s > B
    return (
        torch.where(alive, price, zero),
        torch.where(alive, delta, zero),
        torch.where(alive, gamma, zero),
        torch.where(alive, theta, zero),
    )


# ---------------------------------------------------------------------------
# Terminal profiles for the exact-subtraction ansatz: a terminal function
# pi(s, t) with pi(s, T) = (K - s)^+, together with its first two price
# derivatives and its time derivative, all in closed form.
# ---------------------------------------------------------------------------

class RawPutPayoffTerminalProfile:
    r"""The raw put payoff :math:`\pi(s,t) = (K-s)^+` as a terminal profile.

    Time-independent; :math:`\partial_s\pi = -\mathbf 1_{s<K}` and
    :math:`\partial_{ss}\pi = 0` almost everywhere (the first-derivative
    discontinuity at the strike is not represented -- the same convention as
    autograd through :func:`~learning_option_pricing.pricing.terminal.payoff_put`).
    Baseline with no treatment of the strike singularity.
    """

    name = "raw"

    def __init__(self, K: float) -> None:
        self.K = K

    def value_and_derivatives(
        self, s: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        broadcast_shape = torch.broadcast_shapes(s.shape, torch.as_tensor(t).shape)
        s_b = s.expand(broadcast_shape)
        value = torch.clamp(self.K - s_b, min=0.0)
        first = -(s_b < self.K).to(value.dtype)
        zero = torch.zeros_like(value)
        return value, first, zero, zero


class BlackScholesPutTerminalProfile:
    r"""The exact Black-Scholes European put price :math:`\pi = V^e(s,t)` as a
    terminal profile, with its closed-form Delta :math:`-N(-d_+)`, Gamma
    :math:`N'(d_+)/(s\sigma\sqrt\tau)` and, from :math:`\mathcal L^{BS}V^e=0`,
    :math:`\partial_tV^e = -\tfrac12\sigma^2s^2\Gamma - rs\Delta + rV^e`.  At
    ``tau = 0`` exactly the payoff and its almost-everywhere derivatives are
    returned, as in :func:`~learning_option_pricing.pricing.terminal.black_scholes_put`.
    """

    name = "black_scholes"

    def __init__(self, K: float, r: float, sigma: float, T: float) -> None:
        self.K, self.r, self.sigma, self.T = K, r, sigma, T

    def value_and_derivatives(
        self, s: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tau = torch.as_tensor(self.T - t)
        broadcast_shape = torch.broadcast_shapes(s.shape, tau.shape)
        s_b = s.expand(broadcast_shape)
        tau_b = tau.expand(broadcast_shape)
        tau_safe = torch.clamp(tau_b, min=_TAU_EPS)
        sigma_sqrt_tau = self.sigma * torch.sqrt(tau_safe)
        d_plus = (torch.log(s_b / self.K) + (self.r + 0.5 * self.sigma**2) * tau_safe) / sigma_sqrt_tau
        price = black_scholes_put(s_b, self.K, self.r, self.sigma, tau_b)  # exact payoff at tau = 0
        delta = -_normal_cdf(-d_plus)
        gamma = _normal_pdf(d_plus) / (s_b * sigma_sqrt_tau)
        theta = -0.5 * self.sigma**2 * s_b**2 * gamma - self.r * s_b * delta + self.r * price
        positive_tau = tau_b > 0
        zero = torch.zeros_like(price)
        delta = torch.where(positive_tau, delta, -(s_b < self.K).to(price.dtype))
        gamma = torch.where(positive_tau, gamma, zero)
        theta = torch.where(positive_tau, theta, zero)
        return price, delta, gamma, theta


class SplitSemigroupPutTerminalProfile:
    r"""The split-semigroup profile :math:`\pi(\cdot,t) = e^{(T-t)\nu_c\partial_{xx}}(K-e^{(\cdot)})^+`
    at :math:`x=\ln s` (Proposition 7 / Example 7 of the note) as a terminal
    profile; derivatives chain-ruled from the field's own analytic ones by
    :func:`split_profile_price_and_time_derivatives`.
    """

    name = "split"

    def __init__(
        self,
        split_field: GaussianSemigroupExtensionField | PutPayoffGaussianSemigroupExtensionField,
    ) -> None:
        self.split_field = split_field

    def value_and_derivatives(
        self, s: torch.Tensor, t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return split_profile_price_and_time_derivatives(self.split_field, s, torch.as_tensor(t))


#: Terminal profiles accepted by :func:`make_subtracted_digital_extension`.
SUBTRACTION_TERMINAL_PROFILES = ("raw", "black_scholes", "split")


class SubtractedDigitalCornerExtension:
    r"""Terminal-and-barrier extension of the exact-subtraction ansatz (Method 1,
    Section 5.1 of the note, Definition 7), with its interior residual and
    price derivatives in closed form.

    With the corner jump :math:`\Delta = K - B` and a terminal profile
    :math:`\pi` (one of :class:`RawPutPayoffTerminalProfile`,
    :class:`BlackScholesPutTerminalProfile`,
    :class:`SplitSemigroupPutTerminalProfile`, each with
    :math:`\pi(s,T) = (K-s)^+`), the object evaluates

    .. math::

        g_2(s,t) = \Delta\,V_{DOD}(s,t) + h(s,t),
        \qquad
        h(s,t) = \pi(s,t) - \pi(B,t),

    so that the trial solution :math:`\Phi_\theta = g_2 + d_{\partial_pQ}\Psi_\theta`
    is the subtracted estimator (17) of the note.  The function :math:`h` is
    an extension of the data of the subtracted price
    :math:`\widetilde V_{DO} = V_{DO} - \Delta V_{DOD}` (Proposition 3):

    - on :math:`\Sigma_T`: :math:`h(s,T) = (K-s)^+ - (K-B) = g(s) - \Delta\mathbf 1_{s>B}`
      for :math:`s>B`, the subtracted terminal datum;
    - on :math:`\Sigma_B`: :math:`h(B,t) = 0` for every ``t``, the subtracted
      barrier datum;
    - at the corner the two traces coincide (both are ``0``), so :math:`h` is
      continuous there -- no corner layer, no bandwidth :math:`\varepsilon`.

    The subtraction of :math:`\pi(B,t)` rather than of the constant
    :math:`\Delta` is what makes the barrier trace hold exactly for the
    time-dependent profiles (:math:`V^e(B,t) \neq K - B` for ``t < T``); for
    the raw payoff :math:`\pi(B,t) = \Delta` and :math:`h = (K-s)^+ - \Delta`
    literally.  The price is :math:`\pi(B,t) = \pi(s,t)|_{s=B}`, a function
    of ``t`` alone, so :math:`h` is not annihilated by the operator even when
    :math:`\pi` is: :math:`\mathcal L^{BS}h = \mathcal L^{BS}\pi + \partial_t\pi(B,\cdot)
    - r\,\pi(B,\cdot)`, bounded and smooth up to the terminal face, which the
    free network absorbs.

    **Interior residual** (``black_scholes_residual``).  By Proposition 4
    :math:`\mathcal L^{BS}(\Delta V_{DOD}) = 0` exactly for constant
    coefficients, so

    .. math::

        \mathcal L^{BS}g_2 = \mathcal L^{BS}h
            = \partial_t\pi(s,t) - \partial_t\pi(B,t) + \tfrac12\sigma^2s^2\partial_{ss}\pi
              + rs\,\partial_s\pi - r\big(\pi(s,t) - \pi(B,t)\big),

    assembled from the profile's closed-form derivatives -- never autograd
    through :math:`V_{DOD}`, whose second price derivative is unbounded at the
    corner and would be evaluated by autograd as the difference of large
    cancelling terms.  The presence of this method selects the two-term
    residual assembly in ``pilot_down_and_out_put.compute_loss``.  The
    exactness of the omitted digital term is a property of the constant
    coefficients ``(r, sigma)`` the digital was built with (Remark 8 of the
    note); the method therefore refuses other coefficients rather than
    silently returning a residual missing the term (19).

    **Price derivatives** (``first_price_derivative``,
    ``second_price_derivative``): :math:`\Delta\,\partial_sV_{DOD} + \partial_s\pi`
    and :math:`\Delta\,\partial_{ss}V_{DOD} + \partial_{ss}\pi` (the term
    :math:`\pi(B,t)` does not depend on ``s``), consumed by the Greeks
    evaluation exactly as for :class:`SplitSemigroupCornerExtension`.

    Args:
        K: Strike.
        B: Knock-out barrier, ``0 < B < K``.
        r: Risk-free rate (constant coefficient the digital is exact for).
        sigma: Volatility (idem).
        T: Maturity.
        terminal_profile: The profile :math:`\pi`; see the class list above.

    Raises:
        ValueError: If ``B >= K``, ``B <= 0`` or ``T <= 0``.
    """

    def __init__(
        self, K: float, B: float, r: float, sigma: float, T: float, terminal_profile,
    ) -> None:
        if not (0.0 < B < K):
            raise ValueError(f"the reverse knock-out regime requires 0 < B < K; got {B=}, {K=}.")
        if T <= 0.0:
            raise ValueError(f"T must be > 0; got {T}.")
        self.K, self.B, self.r, self.sigma, self.T = K, B, r, sigma, T
        self.jump = K - B  # Delta = K - B, the corner jump of Proposition 1
        self.terminal_profile = terminal_profile
        self.profile_name = getattr(terminal_profile, "name", type(terminal_profile).__name__)

    # -- pieces ------------------------------------------------------------

    def _digital(self, s: torch.Tensor, t: torch.Tensor):
        return down_and_out_digital_price_and_derivatives(s, self.B, self.r, self.sigma, self.T - t)

    def _profile_at_barrier(self, s: torch.Tensor, t: torch.Tensor):
        r""":math:`(\pi(B,t), \partial_t\pi(B,t))`, broadcast to the shape of ``s``/``t``."""
        t_tensor = torch.as_tensor(t)
        broadcast_shape = torch.broadcast_shapes(s.shape, t_tensor.shape)
        s_barrier = torch.full(broadcast_shape, self.B, dtype=s.dtype, device=s.device)
        value, _, _, d_t = self.terminal_profile.value_and_derivatives(s_barrier, t_tensor.expand(broadcast_shape))
        return value, d_t

    def digital_price(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The subtracted singular part :math:`\Delta\,V_{DOD}(s,t)` alone."""
        return self.jump * self._digital(s, t)[0]

    def subtracted_data_extension(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The regular part :math:`h(s,t) = \pi(s,t) - \pi(B,t)` alone."""
        pi_value, _, _, _ = self.terminal_profile.value_and_derivatives(s, t)
        pi_at_barrier, _ = self._profile_at_barrier(s, t)
        return pi_value - pi_at_barrier

    # -- the extension and its derivatives ---------------------------------

    def __call__(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.digital_price(s, t) + self.subtracted_data_extension(s, t)

    def first_price_derivative(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\partial_s g_2 = \Delta\,\partial_sV_{DOD} + \partial_s\pi`."""
        _, digital_delta, _, _ = self._digital(s, t)
        _, d_s_pi, _, _ = self.terminal_profile.value_and_derivatives(s, t)
        return self.jump * digital_delta + d_s_pi

    def second_price_derivative(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\partial_{ss} g_2 = \Delta\,\partial_{ss}V_{DOD} + \partial_{ss}\pi`."""
        _, _, digital_gamma, _ = self._digital(s, t)
        _, _, d_ss_pi, _ = self.terminal_profile.value_and_derivatives(s, t)
        return self.jump * digital_gamma + d_ss_pi

    def first_time_derivative(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r""":math:`\partial_t g_2 = \Delta\,\partial_tV_{DOD} + \partial_t\pi(s,t) - \partial_t\pi(B,t)`."""
        _, _, _, digital_theta = self._digital(s, t)
        _, _, _, d_t_pi = self.terminal_profile.value_and_derivatives(s, t)
        _, d_t_pi_at_barrier = self._profile_at_barrier(s, t)
        return self.jump * digital_theta + d_t_pi - d_t_pi_at_barrier

    def black_scholes_residual(
        self, s: torch.Tensor, t: torch.Tensor, r: float, sigma: float,
    ) -> torch.Tensor:
        r""":math:`\mathcal L^{BS}g_2 = \mathcal L^{BS}h` in closed form (class docstring).

        Never autograd, never a finite difference.  The digital term is
        omitted because it is exactly annihilated (Proposition 4); this holds
        for the coefficients the digital was built with only, so differing
        ``r``/``sigma`` are refused.

        Args:
            s: Underlying asset price tensor.
            t: Time tensor, broadcastable with ``s``.
            r: Risk-free rate of the operator; must equal the contract's.
            sigma: Volatility of the operator; must equal the contract's.

        Returns:
            The residual, broadcast shape of ``s``/``t``.

        Raises:
            ValueError: If ``r`` or ``sigma`` differ from the ones the digital
                was built with (the omitted digital residual (19) would then
                be nonzero).
        """
        if r != self.r or sigma != self.sigma:
            raise ValueError(
                f"SubtractedDigitalCornerExtension.black_scholes_residual: the digital is "
                f"operator-exact for (r, sigma) = ({self.r}, {self.sigma}) only; got ({r}, {sigma}). "
                f"Under other coefficients the omitted term (19) of the note is nonzero."
            )
        pi_value, d_s_pi, d_ss_pi, d_t_pi = self.terminal_profile.value_and_derivatives(s, t)
        pi_at_barrier, d_t_pi_at_barrier = self._profile_at_barrier(s, t)
        h_value = pi_value - pi_at_barrier
        d_t_h = d_t_pi - d_t_pi_at_barrier
        return d_t_h + 0.5 * sigma**2 * s**2 * d_ss_pi + r * s * d_s_pi - r * h_value


def make_subtracted_digital_extension(
    K: float,
    B: float,
    r: float,
    sigma: float,
    T: float,
    terminal_profile: str = "black_scholes",
    comparison_volatility: float | None = None,
    y_lo: float | None = None,
    y_hi: float | None = None,
    n_quad: int = 8000,
    split_profile: str = "closed_form",
) -> SubtractedDigitalCornerExtension:
    r"""Build the exact-subtraction extension :math:`g_2 = \Delta V_{DOD} + \pi - \pi(B,\cdot)`
    of :class:`SubtractedDigitalCornerExtension` for a named terminal profile.

    Args:
        K: Strike.
        B: Knock-out barrier, ``0 < B < K``.
        r: Risk-free rate.
        sigma: Volatility.
        T: Maturity.
        terminal_profile: One of :data:`SUBTRACTION_TERMINAL_PROFILES` --
            ``"raw"`` (the payoff :math:`(K-s)^+`, no strike treatment),
            ``"black_scholes"`` (the exact European put :math:`V^e`),
            ``"split"`` (the split-semigroup profile of Proposition 7).
        comparison_volatility: :math:`\sigma_c` of the split profile
            (``"split"`` only); defaults to the contract's ``sigma`` (matched
            split of Example 7).
        y_lo, y_hi, n_quad: Quadrature support and resolution of the split
            profile when ``split_profile="quadrature"``; ignored by the
            closed-form route.
        split_profile: Evaluation route of the split profile, one of
            :data:`SPLIT_PROFILE_ROUTES` (``"split"`` only).

    Returns:
        The extension object (callable ``(s, t) -> Tensor``, with
        ``black_scholes_residual`` and the price derivatives).

    Raises:
        ValueError: If ``terminal_profile`` is not one of
            :data:`SUBTRACTION_TERMINAL_PROFILES`, or on the constraints of
            :class:`SubtractedDigitalCornerExtension` and of the profile
            fields.
    """
    if terminal_profile not in SUBTRACTION_TERMINAL_PROFILES:
        raise ValueError(
            f"terminal_profile must be one of {SUBTRACTION_TERMINAL_PROFILES}; got {terminal_profile!r}."
        )
    if terminal_profile == "raw":
        profile = RawPutPayoffTerminalProfile(K)
    elif terminal_profile == "black_scholes":
        profile = BlackScholesPutTerminalProfile(K, r, sigma, T)
    else:
        resolved_comparison_volatility = comparison_volatility if comparison_volatility is not None else sigma
        if split_profile not in SPLIT_PROFILE_ROUTES:
            raise ValueError(f"split_profile must be one of {SPLIT_PROFILE_ROUTES}; got {split_profile!r}.")
        if split_profile == "closed_form":
            split_field = PutPayoffGaussianSemigroupExtensionField(
                K=K, terminal_time=T, comparison_volatility=resolved_comparison_volatility,
                name="barrier_subtraction_split_semigroup_closed_form",
            )
        else:
            if y_lo is None or y_hi is None:
                raise ValueError("split_profile='quadrature' requires the log-price quadrature support y_lo/y_hi.")

            def terminal_datum_on_the_log_price_line(x: torch.Tensor) -> torch.Tensor:
                return payoff_put(torch.exp(x), K)

            split_field = GaussianSemigroupExtensionField(
                terminal_datum_on_the_log_price_line, terminal_time=T,
                comparison_volatility=resolved_comparison_volatility,
                y_lo=y_lo, y_hi=y_hi, n_quad=n_quad, name="barrier_subtraction_split_semigroup",
            )
        profile = SplitSemigroupPutTerminalProfile(split_field)
    return SubtractedDigitalCornerExtension(K, B, r, sigma, T, profile)
