r"""Black--Scholes references on the log-price line, built from the heat kernel.

Setting.  The spatial variable is the log-price :math:`x = \ln s`, in which the
Black--Scholes generator has constant coefficients,

.. math::

    \mathcal{L}^{X}
        = \nu\,\partial_{xx}
        + (r - \nu)\,\partial_{x}
        - r ,
    \qquad \nu = \tfrac{\sigma^{2}}{2} ,

and the backward parabolic operator is
:math:`\mathcal{L} = \partial_{t} + \mathcal{L}^{X}`.

Why this module exists.  The companion Bermudan experiments of
:mod:`learning_option_pricing.pde.heat_references` are posed on the **pure heat**
generator :math:`\mathcal{A} = \nu\,\partial_{xx}` (no drift, no discount).  For
the study of *split-generator* terminal-data extensions the pure-heat setting is
degenerate: the split
:math:`\mathcal{L}^{X} = \mathcal{A} + \mathcal{B}` has defect part
:math:`\mathcal{B} = 0`, so the split extension coincides with the exact solution
and carries no forcing.  The genuine Black--Scholes generator, with
:math:`r \in (0, +\infty)`, has

.. math::

    \mathcal{A} = \nu\,\partial_{xx},
    \qquad
    \mathcal{B} = (r - \nu)\,\partial_{x} - r \neq 0 ,

of differential order one, which is the setting in which the split extension has
a non-zero --- and bounded --- forcing.

The conjugacy, and how everything here is built.  The parts :math:`\mathcal{A}`
and :math:`\mathcal{B}` are both constant-coefficient, hence **commute**, so the
semigroup factorises,

.. math::

    e^{\varsigma \mathcal{L}^{X}}
        = e^{\varsigma \mathcal{A}}\, e^{\varsigma \mathcal{B}} ,
    \qquad
    \bigl(e^{\varsigma \mathcal{B}} \phi\bigr)(x)
        = e^{-r\varsigma}\, \phi\bigl(x + (r - \nu)\,\varsigma\bigr) ,

whence the Black--Scholes propagation of a terminal datum is an explicit **shift
and discount** of its heat propagation:

.. math::

    \bigl(e^{\varsigma \mathcal{L}^{X}} V\bigr)(x)
        = e^{-r\varsigma}\,
          \bigl(e^{\varsigma \mathcal{A}} V\bigr)
              \bigl(x + (r - \nu)\,\varsigma\bigr) .

Every reference in this module is obtained from that identity applied to the
already-tested heat primitives of
:mod:`learning_option_pricing.pde.heat_references`, so no new quadrature is
introduced and the Black--Scholes references inherit the accuracy of the heat
ones.  The identity itself is exercised by the regression tests: it is not merely
a derivation convenience but a checkable statement, and a check of it is the
cheapest available guard against a sign error in the drift.

Claim strength.  The conjugacy is *proven* (the two parts commute; the shift
semigroup is verified by differentiation).  It is also the reason the
constant-coefficient Black--Scholes model is a *diagnostic with a known answer*
rather than an application of the split-extension construction: anything able to
evaluate the split extension can evaluate the exact stage solution at the same
cost.  The construction becomes non-redundant only where the parts do not commute
(local volatility, stochastic rates, baskets).
"""

from __future__ import annotations

import math

import torch

from learning_option_pricing.pde.heat_references import (
    _normal_cdf,
    heat_propagate,
    heat_put_payoff,
)

__all__ = [
    "black_scholes_generator_coefficients",
    "black_scholes_defect_coefficients",
    "black_scholes_propagate",
    "black_scholes_put_exact",
    "bermudan_put_value_exact_black_scholes",
    "bermudan_exercise_boundary",
]


def black_scholes_generator_coefficients(
    *, volatility: float, risk_free_rate: float
) -> dict[int, float]:
    r"""Coefficients of the Black--Scholes generator in log-price coordinates.

    Returns the mapping from differential order to real coefficient expected by
    :func:`learning_option_pricing.models.terminal_ansatz.residual_decomposition`,
    for

    .. math::

        \mathcal{L}^{X}
            = \nu\,\partial_{xx} + (r - \nu)\,\partial_{x} - r ,
        \qquad \nu = \tfrac{\sigma^{2}}{2} .

    Args:
        volatility:      :math:`\sigma \in (0, +\infty)`.
        risk_free_rate:  :math:`r \in [0, +\infty)`.

    Returns:
        ``{2: nu, 1: r - nu, 0: -r}``.
    """
    diffusivity = 0.5 * volatility**2
    return {
        2: diffusivity,
        1: risk_free_rate - diffusivity,
        0: -risk_free_rate,
    }


def black_scholes_defect_coefficients(
    *, volatility: float, risk_free_rate: float
) -> dict[int, float]:
    r"""Coefficients of the defect part :math:`\mathcal{B}` of the split.

    With the principal part :math:`\mathcal{A} = \nu\,\partial_{xx}` taken out,
    the defect is :math:`\mathcal{B} = (r - \nu)\,\partial_{x} - r`, of
    differential order :math:`m_{\mathcal{B}} = 1`.  It is the order-one
    character of :math:`\mathcal{B}` that makes the split forcing
    :math:`\mathcal{B} h` **bounded** on a Lipschitz terminal datum, while the
    order-two principal part applied to the same datum would produce a Dirac
    mass at the free boundary.
    """
    diffusivity = 0.5 * volatility**2
    return {2: 0.0, 1: risk_free_rate - diffusivity, 0: -risk_free_rate}


def black_scholes_propagate(
    terminal_datum_fn,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    t_terminal: float,
    volatility: float,
    risk_free_rate: float,
    y_lo: float,
    y_hi: float,
    n_quad: int = 4000,
) -> torch.Tensor:
    r"""Exact Black--Scholes propagation of a terminal datum, by the conjugacy.

    Solves :math:`\partial_{t} u + \mathcal{L}^{X} u = 0` on
    :math:`(t, t_{\mathrm{terminal}})` with
    :math:`u(\cdot, t_{\mathrm{terminal}}) = V`, by the identity of the module
    docstring:

    .. math::

        u(x, t)
            = e^{-r\varsigma}\,
              \bigl(e^{\varsigma \mathcal{A}} V\bigr)
                  \bigl(x + (r - \nu)\,\varsigma\bigr) ,
        \qquad \varsigma = t_{\mathrm{terminal}} - t ,

    the inner heat propagation being delegated to
    :func:`~learning_option_pricing.pde.heat_references.heat_propagate` (fixed
    trapezoidal grid on ``[y_lo, y_hi]``).  No new quadrature is introduced.

    At :math:`\varsigma = 0` the identity returns ``terminal_datum_fn(x)``
    exactly, whatever the regularity of the datum.

    Args:
        terminal_datum_fn: Callable ``V(y) -> Tensor`` on the log-price line.
        x, t:              Query points (broadcast-compatible tensors).
        t_terminal:        Terminal time of the interval.
        volatility:        :math:`\sigma`.
        risk_free_rate:    :math:`r`.
        y_lo, y_hi:        Quadrature support (must cover the Gaussian mass).
        n_quad:            Number of quadrature nodes.
    """
    diffusivity = 0.5 * volatility**2
    time_to_terminal = t_terminal - t
    shifted_x = x + (risk_free_rate - diffusivity) * time_to_terminal
    heat_value = heat_propagate(
        terminal_datum_fn,
        shifted_x,
        t,
        t_terminal=t_terminal,
        sigma=volatility,
        y_lo=y_lo,
        y_hi=y_hi,
        n_quad=n_quad,
    )
    return torch.exp(-risk_free_rate * time_to_terminal) * heat_value


def black_scholes_put_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    K: float,
    T: float,
    volatility: float,
    risk_free_rate: float,
) -> torch.Tensor:
    r"""Closed-form Black--Scholes European put in log-price coordinates.

    .. math::

        u(x, t)
            = K e^{-r\varsigma}\, N(-d_{2}) - e^{x}\, N(-d_{1}) ,
        \qquad
        \varsigma = T - t ,

    with
    :math:`d_{1} = \bigl(x - \ln K + (r + \tfrac{\sigma^{2}}{2})\varsigma\bigr)
    / (\sigma\sqrt{\varsigma})` and
    :math:`d_{2} = d_{1} - \sigma\sqrt{\varsigma}`.

    This is the Black--Scholes propagation of the put payoff, and the regression
    tests check it against :func:`black_scholes_propagate` applied to
    :func:`~learning_option_pricing.pde.heat_references.heat_put_payoff` --- that
    is, they check the conjugacy identity of the module docstring on the one datum
    for which both sides are available in closed form.

    At :math:`\varsigma \to 0` it reduces to the payoff :math:`(K - e^{x})^{+}`.
    """
    time_to_maturity = torch.clamp(T - t, min=1e-12)
    volatility_root_time = volatility * torch.sqrt(time_to_maturity)
    d_one = (
        x
        - math.log(K)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / volatility_root_time
    d_two = d_one - volatility_root_time
    return K * torch.exp(-risk_free_rate * time_to_maturity) * _normal_cdf(
        -d_two
    ) - torch.exp(x) * _normal_cdf(-d_one)


def bermudan_put_value_exact_black_scholes(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    exercise_times,
    K: float,
    volatility: float,
    risk_free_rate: float,
    y_lo: float,
    y_hi: float,
    n_quad: int = 4000,
) -> torch.Tensor:
    r"""Exact Bermudan-put value under Black--Scholes, by backward induction.

    The dynamic program is

    .. math::

        V(\cdot, t_{M}) &= (K - e^{x})^{+} , \\
        C(\cdot, t_{k}) &= e^{\Delta_{k+1} \mathcal{L}^{X}}\,V(\cdot, t_{k+1}) , \\
        V(\cdot, t_{k}) &= \max\bigl((K - e^{x})^{+},\ C(\cdot, t_{k})\bigr) ,

    with the propagation performed **exactly** (one application of
    :func:`black_scholes_propagate` per inter-exercise interval, no
    time-discretisation error) and the gluing performed with the **exact**
    :func:`torch.maximum` --- the reference against which the learned chain and
    its smoothed gluing are measured.

    For ``t`` strictly inside an inter-exercise interval the value is the pure
    continuation toward the next exercise date; at an exercise date it includes
    the maximum.  Equal time values are grouped, so an arbitrary ``t`` tensor is
    handled.

    Args:
        x:               Log-price, shape ``(N,)``.
        t:               Time (scalar-valued or shape ``(N,)``).
        exercise_times:  Ascending exercise dates; the last is maturity.
        K:               Strike.
        volatility:      :math:`\sigma`.
        risk_free_rate:  :math:`r`.
        y_lo, y_hi:      Quadrature support.
        n_quad:          Quadrature nodes.
    """
    exercise_times = [float(v) for v in exercise_times]

    def value_at(query_x: torch.Tensor, query_time: float) -> torch.Tensor:
        # Walk the dynamic program downward from maturity to the first exercise
        # date at or after `query_time`, building a callable for the value there,
        # then propagate that callable back to `query_time`.
        def datum_at_maturity(y: torch.Tensor) -> torch.Tensor:
            return heat_put_payoff(y, K)

        value_fn = datum_at_maturity
        # Indices of the exercise dates strictly greater than query_time, in
        # descending order; the value is glued at each of them.
        for index in range(len(exercise_times) - 1, -1, -1):
            date = exercise_times[index]
            if date <= query_time + 1e-12:
                break
            previous_date = (
                exercise_times[index - 1] if index > 0 else 0.0
            )
            # Lower end of this propagation: either the previous exercise date, or
            # query_time if the query lies inside this interval.
            lower = max(previous_date, query_time)
            interval_value_fn = _make_propagated_value_fn(
                value_fn,
                t_terminal=date,
                t_lower=lower,
                K=K,
                volatility=volatility,
                risk_free_rate=risk_free_rate,
                y_lo=y_lo,
                y_hi=y_hi,
                n_quad=n_quad,
                glue=(lower > query_time + 1e-12),
            )
            value_fn = interval_value_fn
        return value_fn(query_x)

    if t.dim() == 0 or t.numel() == 1:
        return value_at(x, float(t.reshape(-1)[0]))

    out = torch.empty_like(x)
    for time_value in torch.unique(t):
        mask = t == time_value
        out[mask] = value_at(x[mask], float(time_value))
    return out


def _make_propagated_value_fn(
    value_fn_above,
    *,
    t_terminal: float,
    t_lower: float,
    K: float,
    volatility: float,
    risk_free_rate: float,
    y_lo: float,
    y_hi: float,
    n_quad: int,
    glue: bool,
):
    """Propagate ``value_fn_above`` from ``t_terminal`` down to ``t_lower``.

    The returned callable evaluates the value at ``t_lower``; when ``glue`` is
    true (``t_lower`` is itself an exercise date) the exercise maximum is applied
    there.  Bound as a closure over the *current* loop variables --- binding them
    by default argument is what the late-binding defect of the earlier heat
    reference got wrong, and the regression tests pin the tower property that
    detects it.
    """

    def propagated(query_x: torch.Tensor) -> torch.Tensor:
        time_tensor = torch.full_like(query_x, t_lower)
        continuation = black_scholes_propagate(
            value_fn_above,
            query_x,
            time_tensor,
            t_terminal=t_terminal,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            y_lo=y_lo,
            y_hi=y_hi,
            n_quad=n_quad,
        )
        if glue:
            return torch.maximum(heat_put_payoff(query_x, K), continuation)
        return continuation

    return propagated


def bermudan_exercise_boundary(
    continuation_fn,
    *,
    K: float,
    x_lo: float,
    x_hi: float,
    n_bisection: int = 80,
) -> float:
    r"""Locate the free boundary :math:`\Gamma = \{x : \payoff(x) = C(x)\}`.

    For the put the exercise region is the interval
    :math:`[x_{\mathrm{lo}}, x^{\star}]` and :math:`x^{\star}` is the unique root
    of :math:`x \mapsto (K - e^{x})^{+} - C(x)` on the window; it is located by
    bisection to a tolerance of order
    :math:`(x_{\mathrm{hi}} - x_{\mathrm{lo}})\,2^{-n_{\mathrm{bisection}}}`,
    i.e. to machine precision for the default node count.

    The sign of the difference is checked at both ends and a ``ValueError`` is
    raised when the crossing is not bracketed --- an *empty* exercise region (no
    corner in the stage datum, so the whole free-boundary analysis is vacuous)
    must be detected, not silently mis-located.

    Args:
        continuation_fn:  Callable ``C(x) -> Tensor`` on the log-price line.
        K:                Strike.
        x_lo, x_hi:       Bracket.
        n_bisection:      Bisection steps.

    Returns:
        The root :math:`x^{\star}`.
    """

    def difference(value: float) -> float:
        point = torch.tensor([value], dtype=torch.float64)
        return float(
            heat_put_payoff(point, K) - continuation_fn(point)
        )

    lower, upper = float(x_lo), float(x_hi)
    difference_lower, difference_upper = difference(lower), difference(upper)
    if difference_lower * difference_upper > 0.0:
        raise ValueError(
            "the payoff and the continuation value do not cross on "
            f"[{lower}, {upper}]: payoff - continuation is "
            f"{difference_lower:.6e} at the lower end and "
            f"{difference_upper:.6e} at the upper end. The exercise region is "
            "empty or fills the window; the stage datum then has no corner and "
            "the free-boundary analysis does not apply."
        )
    for _ in range(n_bisection):
        middle = 0.5 * (lower + upper)
        if difference(middle) * difference_lower > 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)
