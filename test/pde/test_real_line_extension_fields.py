"""Regression tests for the log-price-line terminal-data extensions.

Each test pins one assertion of the free-boundary theory, and is named after it:

* the extension reproduces the datum **exactly** at the slice (so the smoothing
  bias of the error recursion vanishes) --- and this holds for the mis-specified
  comparison diffusivity too, so exactness does *not* discriminate;
* the analytic derivatives agree with automatic differentiation away from the
  slice (so the analytic bypass is not merely faster, it is correct);
* the **matched** split's forcing is ``B h`` and is **bounded uniformly** as the
  slice is approached, whereas the mis-specified split's forcing **diverges** like
  ``(T - t)^(-1/2)``.  This is the property that singles out the split, and it is
  the one the ledger of the paper turns on;
* the quadrature floor is **reported**, never silent;
* the fused ``field_and_space_derivatives`` returns bitwise the tensors of the
  three single-quantity methods (it is a cost optimisation, not a new formula);
* the closed-form put-payoff field agrees with the quadrature field to the
  quadrature's own discretisation error, and has no floor.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from learning_option_pricing.pde import (
    GaussianSemigroupExtensionField,
    GradedChenMangasarianExtensionField,
    PutPayoffGaussianSemigroupExtensionField,
    black_scholes_put_exact,
    exact_maximum_datum,
    heat_put_payoff,
)

STRIKE = 100.0
VOLATILITY = 0.25
RISK_FREE_RATE = 0.05
DIFFUSIVITY = 0.5 * VOLATILITY**2
STAGE_TERMINAL_TIME = 0.5
MATURITY = 1.0
QUADRATURE_LO, QUADRATURE_HI = math.log(2.0), math.log(2000.0)
EVALUATION_LO, EVALUATION_HI = math.log(60.0), math.log(140.0)


def _exact_continuation(x: torch.Tensor) -> torch.Tensor:
    """The exact European put continuation at the intermediate date."""
    return black_scholes_put_exact(
        x,
        torch.full_like(x, STAGE_TERMINAL_TIME),
        K=STRIKE,
        T=MATURITY,
        volatility=VOLATILITY,
        risk_free_rate=RISK_FREE_RATE,
    )


def _stage_datum():
    return exact_maximum_datum(_exact_continuation, K=STRIKE)


def _field(comparison_volatility: float, n_quad: int = 8000):
    return GaussianSemigroupExtensionField(
        _stage_datum(),
        terminal_time=STAGE_TERMINAL_TIME,
        comparison_volatility=comparison_volatility,
        y_lo=QUADRATURE_LO,
        y_hi=QUADRATURE_HI,
        n_quad=n_quad,
        name=f"split_sigma_c={comparison_volatility}",
    )


def _grid(n: int = 201) -> torch.Tensor:
    return torch.linspace(EVALUATION_LO, EVALUATION_HI, n, dtype=torch.float64)


def _forcing(field, x, t, *, comparison_volatility):
    """The stage forcing ``L h`` assembled from the analytic derivatives."""
    derivative = field.derivative_callables()
    value = field.field(x, t)
    return (
        derivative["dt"](x, t)
        + DIFFUSIVITY * derivative["dxx"](x, t)
        + (RISK_FREE_RATE - DIFFUSIVITY) * derivative["dx"](x, t)
        - RISK_FREE_RATE * value
    )


# ---------------------------------------------------------------------------
# Exactness at the slice: true for every comparison diffusivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("comparison_volatility", [VOLATILITY, VOLATILITY / 2, 2 * VOLATILITY])
def test_the_extension_reproduces_the_datum_exactly_at_the_slice(comparison_volatility):
    """Exactness rests only on the semigroup at parameter zero being the identity.

    Every comparison diffusivity gives it, so exactness at the slice does NOT
    single out the matched split -- which is precisely why the paper's ledger has
    further columns.
    """
    field = _field(comparison_volatility)
    x = _grid()
    t = torch.full_like(x, STAGE_TERMINAL_TIME)
    datum = _stage_datum()(x).reshape(-1, 1)
    assert torch.allclose(field.field(x, t), datum, atol=1e-12, rtol=0.0)


def test_exactness_at_the_slice_holds_for_a_kinked_datum():
    """The datum's corner is reproduced exactly, not smoothed away, at t = T."""
    field = _field(VOLATILITY)
    x = _grid(401)
    t = torch.full_like(x, STAGE_TERMINAL_TIME)
    value = field.field(x, t).reshape(-1)
    payoff = heat_put_payoff(x, STRIKE)
    # At the slice the extension is >= the payoff everywhere (it IS the maximum).
    assert float((value - payoff).min()) > -1e-12


# ---------------------------------------------------------------------------
# The analytic derivatives are correct
# ---------------------------------------------------------------------------


def test_analytic_derivatives_match_autograd_away_from_the_slice():
    field = _field(VOLATILITY)
    x = _grid(61).clone().requires_grad_(True)
    t = torch.full_like(x, 0.25).requires_grad_(True)
    value = field.field(x, t)
    (autograd_dx,) = torch.autograd.grad(
        value, x, torch.ones_like(value), create_graph=True
    )
    (autograd_dxx,) = torch.autograd.grad(
        autograd_dx, x, torch.ones_like(autograd_dx), create_graph=True
    )
    (autograd_dt,) = torch.autograd.grad(value, t, torch.ones_like(value))

    derivative = field.derivative_callables()
    analytic_dx = derivative["dx"](x.detach(), t.detach()).reshape(-1)
    analytic_dxx = derivative["dxx"](x.detach(), t.detach()).reshape(-1)
    analytic_dt = derivative["dt"](x.detach(), t.detach()).reshape(-1)

    assert torch.allclose(analytic_dx, autograd_dx.detach(), atol=1e-6, rtol=1e-4)
    assert torch.allclose(analytic_dxx, autograd_dxx.detach(), atol=1e-4, rtol=1e-3)
    assert torch.allclose(analytic_dt, autograd_dt.detach(), atol=1e-4, rtol=1e-3)


def test_the_extension_solves_its_own_heat_equation():
    """d_t h + nu_c d_xx h = 0 identically -- the exact cancellation.

    This is the mechanism that distinguishes the split from every mollifier: the
    singular second-order channel is not made integrable, it is cancelled.
    """
    for comparison_volatility in (VOLATILITY, VOLATILITY / 2):
        field = _field(comparison_volatility)
        comparison_diffusivity = 0.5 * comparison_volatility**2
        x = _grid(61)
        t = torch.full_like(x, 0.2)
        derivative = field.derivative_callables()
        residual = derivative["dt"](x, t) + comparison_diffusivity * derivative["dxx"](
            x, t
        )
        assert float(residual.abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# THE DISCRIMINANT: bounded forcing for the matched split, unbounded otherwise
# ---------------------------------------------------------------------------


def test_the_matched_split_forcing_equals_the_defect_part_applied_to_the_extension():
    """L h = B h = (r - nu) d_x h - r h, exactly, for the matched split."""
    field = _field(VOLATILITY)
    x = _grid(61)
    t = torch.full_like(x, 0.2)
    derivative = field.derivative_callables()
    defect_forcing = (RISK_FREE_RATE - DIFFUSIVITY) * derivative["dx"](
        x, t
    ) - RISK_FREE_RATE * field.field(x, t)
    full_forcing = _forcing(field, x, t, comparison_volatility=VOLATILITY)
    assert torch.allclose(full_forcing, defect_forcing, atol=1e-12, rtol=0.0)


def test_the_matched_split_forcing_is_bounded_uniformly_up_to_the_slice():
    """The supremum of |L h| does not grow as the terminal slice is approached.

    This is Proposition (the split extension is exact at the slice, at bounded
    forcing), part (ii), measured: the defect part is of order one and the datum is
    Lipschitz, so the forcing is bounded by
    |r - nu| Lip(V) + r sup|V| -- independently of the distance to the slice.
    """
    field = _field(VOLATILITY, n_quad=16000)
    x = _grid(401)
    suprema = []
    for time_to_terminal in (0.2, 0.05, 0.01, 0.002):
        t = torch.full_like(x, STAGE_TERMINAL_TIME - time_to_terminal)
        forcing = _forcing(field, x, t, comparison_volatility=VOLATILITY)
        suprema.append(float(forcing.abs().max()))

    # Flat across two decades of time-to-terminal: the ratio of the largest to the
    # smallest supremum stays close to one.
    assert max(suprema) / min(suprema) < 1.5, f"suprema not flat: {suprema}"

    # And it respects the a priori bound: |r - nu| Lip(V) + r sup|V|.
    datum = _stage_datum()
    probe = torch.linspace(QUADRATURE_LO, QUADRATURE_HI, 20001, dtype=torch.float64)
    probe.requires_grad_(True)
    datum_values = datum(probe)
    (datum_derivative,) = torch.autograd.grad(
        datum_values, probe, torch.ones_like(datum_values)
    )
    a_priori_bound = abs(RISK_FREE_RATE - DIFFUSIVITY) * float(
        datum_derivative.abs().max()
    ) + RISK_FREE_RATE * float(datum_values.abs().max())
    assert max(suprema) <= a_priori_bound * 1.05


def test_the_mis_specified_split_forcing_diverges_at_the_slice():
    r"""The second-order channel survives and blows up like (T - t)^(-1/2).

    The mis-specified extension is still exact at the slice and still has a finite
    L^2 forcing; what it loses is BOUNDEDNESS -- hence the finite variance of the
    loss estimator, and the bounded target.

    On the *asymptotic* exponent.  The forcing is the sum of two channels,

        L h = (nu - nu_c) d_xx h  +  B h ,

    of which the first diverges like ``J / (sigma_c sqrt(2 pi tau))`` at the corner
    (the datum's Dirac mass smoothed by the kernel) and the second is *bounded*.  The
    asymptotic exponent is therefore 1/2, but at finite ``tau`` the two channels have
    opposite sign near the corner and partially cancel, and the cancellation is
    stronger at the larger ``tau`` where the singular channel is smaller.  That
    suppresses the supremum at large ``tau`` and biases the finite-``tau`` fitted
    exponent *above* 1/2.  The test therefore pins the divergence and a band around
    1/2 that admits the bias, rather than the asymptotic value to three digits, which
    is not what a measurement at ``tau >= 1.25e-3`` can deliver.
    """
    comparison_volatility = VOLATILITY / math.sqrt(2.0)  # nu_c = nu / 2
    field = _field(comparison_volatility, n_quad=16000)
    x = _grid(801)
    times_to_terminal = [0.02, 0.005, 0.00125]
    suprema = []
    for time_to_terminal in times_to_terminal:
        t = torch.full_like(x, STAGE_TERMINAL_TIME - time_to_terminal)
        forcing = _forcing(field, x, t, comparison_volatility=comparison_volatility)
        suprema.append(float(forcing.abs().max()))

    # It diverges, and monotonically.
    assert suprema == sorted(suprema), f"forcing not monotone in 1/tau: {suprema}"
    # Over a sixteen-fold decrease of tau the ideal sqrt law gives a factor of four;
    # the measured factor exceeds it (the cancellation described above).
    assert suprema[-1] > 3.5 * suprema[0], f"forcing did not diverge: {suprema}"

    log_slope = math.log(suprema[-1] / suprema[0]) / math.log(
        times_to_terminal[0] / times_to_terminal[-1]
    )
    assert 0.42 <= log_slope <= 0.72, (
        f"divergence exponent {log_slope:.3f} outside the band admitting the "
        f"finite-tau bias about the asymptotic 1/2 "
        f"(suprema {suprema} at tau {times_to_terminal})"
    )

    # The property that matters is not the value of the exponent but that the
    # mis-specified forcing is UNBOUNDED while the matched one is bounded.  The
    # observable form of that statement is that the SEPARATION between the two WIDENS
    # without limit as the slice is approached: the matched supremum is flat (previous
    # test) and the mis-specified one grows, so their ratio grows.  A separation by a
    # fixed factor at one distance would say nothing -- both forcings are finite at any
    # tau > 0, and at moderate tau the two are within an order of magnitude of each
    # other.  It is the trend, not the gap, that distinguishes boundedness from
    # unboundedness.
    matched = _field(VOLATILITY, n_quad=16000)
    ratios = []
    for time_to_terminal, mis_specified_supremum in zip(times_to_terminal, suprema):
        t = torch.full_like(x, STAGE_TERMINAL_TIME - time_to_terminal)
        matched_supremum = float(
            _forcing(matched, x, t, comparison_volatility=VOLATILITY).abs().max()
        )
        ratios.append(mis_specified_supremum / matched_supremum)

    assert ratios == sorted(ratios), (
        f"the separation does not widen as the slice is approached: {ratios} "
        f"at tau {times_to_terminal}"
    )
    assert ratios[-1] > 3.0 * ratios[0], (
        f"the separation widened by only a factor {ratios[-1] / ratios[0]:.2f} over a "
        f"sixteen-fold decrease of the time-to-terminal (ratios {ratios})"
    )


def test_the_graded_mollifier_forcing_diverges_faster_than_the_mis_specified_split():
    """For the linear grading the forcing peaks like 1/eps(t) ~ (T - t)^(-1).

    Exponent one, against one half for the mis-specified split: the graded mollifier
    is the worse of the two, and its strip forcing is logarithmically divergent.
    """
    field = GradedChenMangasarianExtensionField(
        _exact_continuation,
        K=STRIKE,
        terminal_time=STAGE_TERMINAL_TIME,
        smoothing_scale=2.0,
        grading_exponent=1.0,
    )
    x = _grid(2001).clone()
    times_to_terminal = [0.02, 0.005, 0.00125]
    suprema = []
    for time_to_terminal in times_to_terminal:
        xg = x.clone().requires_grad_(True)
        t = torch.full_like(xg, STAGE_TERMINAL_TIME - time_to_terminal).requires_grad_(
            True
        )
        value = field.field(xg, t)
        (dx,) = torch.autograd.grad(value, xg, torch.ones_like(value), create_graph=True)
        (dxx,) = torch.autograd.grad(dx, xg, torch.ones_like(dx), create_graph=True)
        (dt,) = torch.autograd.grad(value, t, torch.ones_like(value), create_graph=True)
        forcing = (
            dt
            + DIFFUSIVITY * dxx
            + (RISK_FREE_RATE - DIFFUSIVITY) * dx
            - RISK_FREE_RATE * value
        )
        suprema.append(float(forcing.abs().max()))

    log_slope = math.log(suprema[-1] / suprema[0]) / math.log(
        times_to_terminal[0] / times_to_terminal[-1]
    )
    # The asymptotic exponent is 1 (the curvature peaks like 1/eps(t) and the linear
    # grading gives eps(t) proportional to T - t); the band admits the same
    # finite-tau bias from the bounded additive channel as in the split case.
    assert 0.80 <= log_slope <= 1.30, (
        f"divergence exponent {log_slope:.3f} outside the band about the asymptotic 1 "
        f"(suprema {suprema} at tau {times_to_terminal})"
    )
    # And it is the WORSE of the two: strictly faster than the mis-specified split's
    # square-root divergence.  This ordering is what the ledger asserts.
    assert log_slope > 0.72, "the graded mollifier must diverge faster than sqrt"


def test_the_graded_mollifier_is_exact_at_the_slice():
    """eps(T) = 0, so the mollified max degenerates to the exact maximum."""
    field = GradedChenMangasarianExtensionField(
        _exact_continuation,
        K=STRIKE,
        terminal_time=STAGE_TERMINAL_TIME,
        smoothing_scale=2.0,
        grading_exponent=1.0,
    )
    x = _grid()
    t = torch.full_like(x, STAGE_TERMINAL_TIME)
    exact = _stage_datum()(x).reshape(-1, 1)
    assert torch.allclose(field.field(x, t), exact, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# The quadrature floor is reported
# ---------------------------------------------------------------------------


def test_the_quadrature_floor_is_counted_and_warned_about(caplog):
    """A silently-active floor would regularise the very corner under study."""
    field = _field(VOLATILITY, n_quad=2000)
    x = _grid(11)
    # A time strictly inside the unresolved band.
    unresolved_time = STAGE_TERMINAL_TIME - 0.5 * field.time_to_terminal_floor
    t = torch.full_like(x, unresolved_time)
    with caplog.at_level(logging.WARNING):
        field.field(x, t)
    report = field.quadrature_floor_report()
    assert report["activation_count"] == x.numel()
    assert report["activation_fraction"] > 0.0
    assert any("cannot resolve the Gaussian kernel" in r.message for r in caplog.records)


def test_the_quadrature_floor_does_not_fire_at_the_slice_itself():
    """At t = T the datum is returned by definition; that is exact, not truncated."""
    field = _field(VOLATILITY, n_quad=2000)
    x = _grid(11)
    t = torch.full_like(x, STAGE_TERMINAL_TIME)
    field.field(x, t)
    assert field.quadrature_floor_report()["activation_count"] == 0


def test_the_field_refuses_a_vanishing_comparison_volatility():
    with pytest.raises(ValueError, match="strictly positive"):
        _field(0.0)


def test_the_graded_field_refuses_a_vanishing_grading_exponent():
    with pytest.raises(ValueError, match="grading_exponent"):
        GradedChenMangasarianExtensionField(
            _exact_continuation,
            K=STRIKE,
            terminal_time=STAGE_TERMINAL_TIME,
            smoothing_scale=2.0,
            grading_exponent=0.0,
        )


# ---------------------------------------------------------------------------
# The fused evaluation is a cost optimisation only: bitwise the same tensors
# ---------------------------------------------------------------------------


def test_the_fused_evaluation_is_bitwise_the_single_quantity_methods():
    field = _field(VOLATILITY, n_quad=2000)
    x = _grid(101)
    for time_to_terminal in (0.3, 0.05, 0.5 * field.time_to_terminal_floor, 0.0):
        t = torch.full_like(x, STAGE_TERMINAL_TIME - time_to_terminal)
        value, first, second = field.field_and_space_derivatives(x, t)
        assert torch.equal(value, field.field(x, t))
        assert torch.equal(first, field.space_derivative(x, t))
        assert torch.equal(second, field.second_space_derivative(x, t))


# ---------------------------------------------------------------------------
# Closed-form put-payoff field: the same object as the quadrature, exactly
# ---------------------------------------------------------------------------


def _put_payoff_datum(x: torch.Tensor) -> torch.Tensor:
    return heat_put_payoff(x, STRIKE)


def _put_payoff_fields(comparison_volatility: float, n_quad: int = 200_000):
    quadrature = GaussianSemigroupExtensionField(
        _put_payoff_datum,
        terminal_time=STAGE_TERMINAL_TIME,
        comparison_volatility=comparison_volatility,
        y_lo=QUADRATURE_LO,
        y_hi=QUADRATURE_HI,
        n_quad=n_quad,
    )
    closed_form = PutPayoffGaussianSemigroupExtensionField(
        K=STRIKE,
        terminal_time=STAGE_TERMINAL_TIME,
        comparison_volatility=comparison_volatility,
    )
    return quadrature, closed_form


@pytest.mark.parametrize("comparison_volatility", [VOLATILITY, 2 * VOLATILITY])
def test_the_closed_form_put_field_matches_the_quadrature_field(comparison_volatility):
    """Value and both space derivatives agree to the quadrature's discretisation
    error (the closed form has none): a genuinely independent check, since the
    quadrature knows nothing of the normal distribution function."""
    quadrature, closed_form = _put_payoff_fields(comparison_volatility)
    x = _grid(101)
    for time_to_terminal in (0.4, 0.1, 0.01):
        t = torch.full_like(x, STAGE_TERMINAL_TIME - time_to_terminal)
        reference = quadrature.field_and_space_derivatives(x, t)
        candidate = closed_form.field_and_space_derivatives(x, t)
        for want, got, tolerance in zip(reference, candidate, (1e-8, 1e-7, 1e-5)):
            assert torch.allclose(got, want, atol=tolerance * STRIKE, rtol=1e-7), (
                f"time_to_terminal={time_to_terminal}: max |difference| "
                f"{(got - want).abs().max().item():.3e}"
            )


def test_the_closed_form_put_field_is_exact_at_the_slice_and_has_no_floor():
    _, closed_form = _put_payoff_fields(VOLATILITY)
    x = _grid(101)
    t = torch.full_like(x, STAGE_TERMINAL_TIME)
    value, first, second = closed_form.field_and_space_derivatives(x, t)
    assert torch.equal(value, _put_payoff_datum(x).reshape(-1, 1))
    expected_first = torch.where(x < math.log(STRIKE), -torch.exp(x), torch.zeros_like(x))
    assert torch.equal(first, expected_first.reshape(-1, 1))
    assert torch.equal(second, torch.zeros_like(second))
    # No quadrature, so no band in which the kernel is unresolved: a point
    # arbitrarily close to the slice is evaluated, and the report says so.
    t_near = torch.full_like(x, STAGE_TERMINAL_TIME - 1e-12)
    assert torch.isfinite(closed_form.second_space_derivative(x, t_near)).all()
    report = closed_form.quadrature_floor_report()
    assert closed_form.time_to_terminal_floor == 0.0
    assert report["activation_count"] == 0
    assert report["profile"] == "closed_form"


def test_the_closed_form_put_field_solves_its_own_heat_equation_by_autograd():
    """d_t h + nu_c d_xx h = 0 with d_t h taken by autograd on the closed-form
    value (the analytic route supplies it as -nu_c d_xx h; here that identity
    is checked rather than assumed)."""
    _, closed_form = _put_payoff_fields(VOLATILITY)
    x = _grid(51)
    t = torch.full_like(x, STAGE_TERMINAL_TIME - 0.2).requires_grad_(True)
    value = closed_form.field(x, t)
    (autograd_time_derivative,) = torch.autograd.grad(value.sum(), t)
    analytic_second = closed_form.second_space_derivative(x, t.detach()).reshape(-1)
    residual = autograd_time_derivative + 0.5 * VOLATILITY**2 * analytic_second
    assert residual.abs().max().item() < 1e-9 * STRIKE
    x_grad = _grid(51).requires_grad_(True)
    value = closed_form.field(x_grad, t.detach())
    (autograd_space_derivative,) = torch.autograd.grad(value.sum(), x_grad)
    analytic_first = closed_form.space_derivative(x_grad.detach(), t.detach()).reshape(-1)
    assert torch.allclose(autograd_space_derivative, analytic_first, atol=1e-9 * STRIKE)


def test_the_closed_form_put_field_refuses_a_non_positive_strike():
    with pytest.raises(ValueError):
        PutPayoffGaussianSemigroupExtensionField(
            K=0.0, terminal_time=STAGE_TERMINAL_TIME, comparison_volatility=VOLATILITY
        )
