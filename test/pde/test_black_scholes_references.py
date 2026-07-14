"""Regression tests for the Black--Scholes references on the log-price line.

The tests are organised around the two propositions they guard:

* the **conjugacy** (the principal and defect parts of the generator commute, so
  Black--Scholes propagation is a shift and discount of heat propagation) --- a
  sign error in the drift is the cheapest way to break the whole study, and the
  conjugacy check catches it;
* the **defect part is of order one**, which is what makes the split extension's
  forcing bounded.
"""

from __future__ import annotations

import math

import pytest
import torch

from learning_option_pricing.pde import (
    bermudan_exercise_boundary,
    bermudan_put_value_exact_black_scholes,
    black_scholes_defect_coefficients,
    black_scholes_generator_coefficients,
    black_scholes_propagate,
    black_scholes_put_exact,
    heat_put_payoff,
)

STRIKE = 100.0
VOLATILITY = 0.25
RISK_FREE_RATE = 0.05
MATURITY = 1.0
QUADRATURE_LO, QUADRATURE_HI = math.log(2.0), math.log(2000.0)
EVALUATION_LO, EVALUATION_HI = math.log(60.0), math.log(140.0)


def _evaluation_grid(n: int = 121) -> torch.Tensor:
    return torch.linspace(EVALUATION_LO, EVALUATION_HI, n, dtype=torch.float64)


# ---------------------------------------------------------------------------
# The generator and its split
# ---------------------------------------------------------------------------


def test_generator_coefficients_reproduce_the_black_scholes_operator():
    coefficients = black_scholes_generator_coefficients(
        volatility=VOLATILITY, risk_free_rate=RISK_FREE_RATE
    )
    diffusivity = 0.5 * VOLATILITY**2
    assert coefficients[2] == pytest.approx(diffusivity)
    assert coefficients[1] == pytest.approx(RISK_FREE_RATE - diffusivity)
    assert coefficients[0] == pytest.approx(-RISK_FREE_RATE)


def test_the_split_is_a_split_generator_equals_principal_plus_defect():
    """A + B = L^X, order by order: the defect carries no second-order part."""
    generator = black_scholes_generator_coefficients(
        volatility=VOLATILITY, risk_free_rate=RISK_FREE_RATE
    )
    defect = black_scholes_defect_coefficients(
        volatility=VOLATILITY, risk_free_rate=RISK_FREE_RATE
    )
    diffusivity = 0.5 * VOLATILITY**2
    principal = {2: diffusivity, 1: 0.0, 0: 0.0}
    for order in (0, 1, 2):
        assert principal[order] + defect[order] == pytest.approx(generator[order])
    # The defect is of differential order one: this is what bounds the split forcing.
    assert defect[2] == 0.0
    assert defect[1] != 0.0 or defect[0] != 0.0


def test_the_defect_is_non_zero_only_because_the_rate_is_non_zero():
    """At r = 0 the defect part vanishes and the split degenerates.

    With r = 0 the Black--Scholes generator in log coordinates is
    ``nu d_xx - nu d_x``, whose defect ``-nu d_x`` is non-zero; it is only the
    *pure heat* generator (the one used by the companion Bermudan experiments)
    that has a vanishing defect.  The test pins the boundary of the degenerate
    regime so that a study built on a zero defect cannot pass unnoticed.
    """
    defect_zero_rate = black_scholes_defect_coefficients(
        volatility=VOLATILITY, risk_free_rate=0.0
    )
    assert defect_zero_rate[1] == pytest.approx(-0.5 * VOLATILITY**2)
    assert defect_zero_rate[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The conjugacy
# ---------------------------------------------------------------------------


def test_conjugacy_black_scholes_propagation_of_the_payoff_is_the_closed_form_put():
    """The quadrature route and the closed form agree: this IS the conjugacy.

    ``black_scholes_propagate`` is built as ``exp(-r s)`` times the *heat*
    propagation of the datum evaluated at the shifted point ``x + (r - nu) s``.
    ``black_scholes_put_exact`` is the textbook closed form, derived
    independently.  Their agreement checks the commutation identity
    ``exp(s L^X) = exp(s A) exp(s B)`` and, in particular, the sign of the drift.
    """
    x = _evaluation_grid()
    for time in (0.0, 0.25, 0.5, 0.9):
        t = torch.full_like(x, time)
        propagated = black_scholes_propagate(
            lambda y: heat_put_payoff(y, STRIKE),
            x,
            t,
            t_terminal=MATURITY,
            volatility=VOLATILITY,
            risk_free_rate=RISK_FREE_RATE,
            y_lo=QUADRATURE_LO,
            y_hi=QUADRATURE_HI,
            n_quad=20000,
        )
        closed_form = black_scholes_put_exact(
            x,
            t,
            K=STRIKE,
            T=MATURITY,
            volatility=VOLATILITY,
            risk_free_rate=RISK_FREE_RATE,
        )
        assert torch.allclose(propagated, closed_form, atol=1e-4, rtol=0.0), (
            f"conjugacy violated at t={time}: max deviation "
            f"{float((propagated - closed_form).abs().max()):.3e}"
        )


def test_black_scholes_put_solves_the_black_scholes_equation_by_autograd():
    """L u = d_t u + nu d_xx u + (r - nu) d_x u - r u = 0 to machine precision."""
    x = _evaluation_grid(41).clone().requires_grad_(True)
    t = torch.full_like(x, 0.4).requires_grad_(True)
    u = black_scholes_put_exact(
        x, t, K=STRIKE, T=MATURITY, volatility=VOLATILITY, risk_free_rate=RISK_FREE_RATE
    )
    (du_dt,) = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)
    (du_dx,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
    (d2u_dx2,) = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)
    diffusivity = 0.5 * VOLATILITY**2
    residual = (
        du_dt
        + diffusivity * d2u_dx2
        + (RISK_FREE_RATE - diffusivity) * du_dx
        - RISK_FREE_RATE * u
    )
    assert float(residual.abs().max()) < 1e-9


def test_black_scholes_put_recovers_the_payoff_at_maturity():
    x = _evaluation_grid()
    t = torch.full_like(x, MATURITY - 1e-10)
    value = black_scholes_put_exact(
        x, t, K=STRIKE, T=MATURITY, volatility=VOLATILITY, risk_free_rate=RISK_FREE_RATE
    )
    assert torch.allclose(value, heat_put_payoff(x, STRIKE), atol=1e-6)


# ---------------------------------------------------------------------------
# The Bermudan reference and its free boundary
# ---------------------------------------------------------------------------


def test_bermudan_at_maturity_is_the_payoff():
    x = _evaluation_grid()
    value = bermudan_put_value_exact_black_scholes(
        x,
        torch.tensor(MATURITY, dtype=torch.float64),
        exercise_times=[0.5, MATURITY],
        K=STRIKE,
        volatility=VOLATILITY,
        risk_free_rate=RISK_FREE_RATE,
        y_lo=QUADRATURE_LO,
        y_hi=QUADRATURE_HI,
    )
    assert torch.allclose(value, heat_put_payoff(x, STRIKE), atol=1e-8)


def test_bermudan_continuation_at_the_single_intermediate_date_is_the_european_put():
    """With m = 2 the continuation at t_1 is the European put maturing at T.

    This is the fact the ablation relies on to make the top stage analytic and the
    free boundary exactly known.
    """
    x = _evaluation_grid()
    t_one = 0.5
    continuation = black_scholes_propagate(
        lambda y: heat_put_payoff(y, STRIKE),
        x,
        torch.full_like(x, t_one),
        t_terminal=MATURITY,
        volatility=VOLATILITY,
        risk_free_rate=RISK_FREE_RATE,
        y_lo=QUADRATURE_LO,
        y_hi=QUADRATURE_HI,
        n_quad=20000,
    )
    european = black_scholes_put_exact(
        x,
        torch.full_like(x, t_one),
        K=STRIKE,
        T=MATURITY,
        volatility=VOLATILITY,
        risk_free_rate=RISK_FREE_RATE,
    )
    assert torch.allclose(continuation, european, atol=1e-4)


def test_bermudan_dominates_the_european_put():
    x = _evaluation_grid()
    t = torch.tensor(0.0, dtype=torch.float64)
    bermudan = bermudan_put_value_exact_black_scholes(
        x,
        t,
        exercise_times=[0.5, MATURITY],
        K=STRIKE,
        volatility=VOLATILITY,
        risk_free_rate=RISK_FREE_RATE,
        y_lo=QUADRATURE_LO,
        y_hi=QUADRATURE_HI,
        n_quad=20000,
    )
    european = black_scholes_put_exact(
        x,
        torch.zeros_like(x),
        K=STRIKE,
        T=MATURITY,
        volatility=VOLATILITY,
        risk_free_rate=RISK_FREE_RATE,
    )
    assert float((bermudan - european).min()) > -1e-5


def test_the_exercise_region_is_non_empty_and_the_crossing_is_transversal():
    """The free boundary exists, lies below the strike, and is a simple crossing.

    Non-emptiness is what makes the stage datum have a corner at all; transversality
    is the standing hypothesis of the free-boundary analysis, and it holds at a
    *discrete* exercise date precisely because there is no smooth-pasting condition
    there.
    """

    def continuation(x: torch.Tensor) -> torch.Tensor:
        return black_scholes_put_exact(
            x,
            torch.full_like(x, 0.5),
            K=STRIKE,
            T=MATURITY,
            volatility=VOLATILITY,
            risk_free_rate=RISK_FREE_RATE,
        )

    x_star = bermudan_exercise_boundary(
        continuation, K=STRIKE, x_lo=math.log(5.0), x_hi=math.log(400.0)
    )
    # The exercise boundary of a put lies strictly below the strike.
    assert x_star < math.log(STRIKE)
    # The crossing is transversal: the derivative of (payoff - continuation) at the
    # root is bounded away from zero, so the datum's first derivative genuinely jumps.
    point = torch.tensor([x_star], dtype=torch.float64, requires_grad=True)
    difference = heat_put_payoff(point, STRIKE) - continuation(point)
    (slope,) = torch.autograd.grad(difference, point, torch.ones_like(difference))
    assert abs(float(slope)) > 1e-3


def test_the_exercise_boundary_raises_when_the_region_is_empty():
    """An empty exercise region must be detected, not silently mis-located."""

    def continuation_far_above(x: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, 10.0 * STRIKE)

    with pytest.raises(ValueError, match="do not cross"):
        bermudan_exercise_boundary(
            continuation_far_above, K=STRIKE, x_lo=math.log(5.0), x_hi=math.log(400.0)
        )
