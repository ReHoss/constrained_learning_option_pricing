"""Tests for the analytic terminal-data extensions and their strip forcing.

Covered here: the split-semigroup forcing identity to machine precision, the
order-2 convergence of a central-difference check of the analytic time
derivative, the identically-zero forcing of the exact-solution extension, the
closed-form time integrals against numpy trapezoid quadrature, the graded /
constant-in-time power ratio at large wavenumber, the growth-versus-
convergence contrast of the strip forcing, and the dissipativity validation
on the extension side.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from learning_option_pricing.pde import (
    ConstantCoefficientGenerator,
    ConstantInTimeExtension,
    ConvexRawExtension,
    ExactSolutionExtension,
    GradedGaussianExtension,
    PeriodisedBernoulliDatum,
    SplitSemigroupExtension,
    advection_diffusion_reaction,
    biharmonic_advection_reaction,
    exponential_time_integral_factor,
    symmetric_wavenumber_band,
    total_strip_forcing_squared,
)


TWO_PI = 2.0 * math.pi
TERMINAL_TIME = 1.0


def _extension_catalogue():
    """Extensions exercised by the quadrature test, keyed by description."""
    datum = PeriodisedBernoulliDatum(1)
    generator_g1 = advection_diffusion_reaction()
    generator_g3 = biharmonic_advection_reaction()
    return {
        "convex_raw_g1": ConvexRawExtension(datum, generator_g1, TERMINAL_TIME),
        "constant_in_time_g1": ConstantInTimeExtension(
            datum, generator_g1, TERMINAL_TIME
        ),
        "split_diffusion_g1": SplitSemigroupExtension(
            datum, generator_g1, [2], TERMINAL_TIME
        ),
        "split_biharmonic_reaction_g3": SplitSemigroupExtension(
            datum, generator_g3, [4, 0], TERMINAL_TIME
        ),
        "graded_mismatched_g1": GradedGaussianExtension(
            datum, generator_g1, 0.35, TERMINAL_TIME
        ),
        "graded_matched_g1": GradedGaussianExtension(
            datum, generator_g1, 0.7, TERMINAL_TIME
        ),
        "exact_g1": ExactSolutionExtension(datum, generator_g1, TERMINAL_TIME),
    }


# ---------------------------------------------------------------------------
# Convex raw extension: terminal identity and derivative
# ---------------------------------------------------------------------------


def test_convex_raw_extension_equals_datum_at_terminal_time():
    datum = PeriodisedBernoulliDatum(1)
    extension = ConvexRawExtension(
        datum, advection_diffusion_reaction(), TERMINAL_TIME
    )
    wavenumbers = symmetric_wavenumber_band(8)
    np.testing.assert_allclose(
        extension.extension_coefficient(wavenumbers, TERMINAL_TIME),
        datum.fourier_coefficients(wavenumbers),
        rtol=1e-15,
    )
    assert np.all(extension.extension_coefficient(wavenumbers, 0.0) == 0.0)
    np.testing.assert_allclose(
        extension.extension_coefficient_time_derivative(wavenumbers, 0.5),
        datum.fourier_coefficients(wavenumbers) / TERMINAL_TIME,
        rtol=1e-15,
    )


# ---------------------------------------------------------------------------
# Split identity (P1) to machine precision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generator_factory, subset_orders",
    [
        (advection_diffusion_reaction, [2]),
        (advection_diffusion_reaction, [2, 0]),
        (biharmonic_advection_reaction, [4, 0]),
    ],
)
def test_split_forcing_identity_to_machine_precision(
    generator_factory, subset_orders
):
    # (d) For the split extension, dt_hhat + a * hhat == b * hhat: the
    # relative deviation, normalised by the largest defect forcing over the
    # band, is at most 1e-13.
    datum = PeriodisedBernoulliDatum(1)
    generator = generator_factory()
    extension = SplitSemigroupExtension(
        datum, generator, subset_orders, TERMINAL_TIME
    )
    wavenumbers = symmetric_wavenumber_band(16)
    defect_symbol_values = extension.generator_split.defect_symbol(wavenumbers)
    for time in (0.0, 0.25, 0.5, 0.9, TERMINAL_TIME):
        assembled_forcing = extension.forcing_coefficient(wavenumbers, time)
        defect_forcing = defect_symbol_values * extension.extension_coefficient(
            wavenumbers, time
        )
        normalisation = np.max(np.abs(defect_forcing))
        assert normalisation > 0.0
        relative_deviation = (
            np.max(np.abs(assembled_forcing - defect_forcing)) / normalisation
        )
        assert relative_deviation <= 1e-13, (time, relative_deviation)


def test_split_time_derivative_central_difference_order_two():
    # (P1, finite-difference check) A central difference of the extension
    # coefficient converges to the analytic time derivative at order 2: the
    # fitted log-log slope lies within [1.9, 2.1].
    datum = PeriodisedBernoulliDatum(1)
    extension = SplitSemigroupExtension(
        datum, advection_diffusion_reaction(), [2], TERMINAL_TIME
    )
    wavenumbers = symmetric_wavenumber_band(3)
    central_time = 0.5
    step_sizes = np.array([4e-2, 2e-2, 1e-2, 5e-3])
    worst_errors = []
    analytic_derivative = extension.extension_coefficient_time_derivative(
        wavenumbers, central_time
    )
    for step in step_sizes:
        central_difference = (
            extension.extension_coefficient(wavenumbers, central_time + step)
            - extension.extension_coefficient(wavenumbers, central_time - step)
        ) / (2.0 * step)
        worst_errors.append(
            float(np.max(np.abs(central_difference - analytic_derivative)))
        )
    fitted_slope = np.polyfit(np.log(step_sizes), np.log(worst_errors), 1)[0]
    assert 1.9 <= fitted_slope <= 2.1, fitted_slope


# ---------------------------------------------------------------------------
# Exact-solution extension: identically zero forcing (P3)
# ---------------------------------------------------------------------------


def test_exact_extension_has_identically_zero_forcing():
    # (e) The forcing of the exact solution vanishes identically — exactly,
    # not merely to a tolerance: both terms of dt_hhat + a * hhat are the
    # same computed product with opposite signs.
    datum = PeriodisedBernoulliDatum(0)
    extension = ExactSolutionExtension(
        datum, advection_diffusion_reaction(), TERMINAL_TIME
    )
    wavenumbers = symmetric_wavenumber_band(32)
    for time in (0.0, 0.3, 0.7, TERMINAL_TIME):
        forcing_values = extension.forcing_coefficient(wavenumbers, time)
        assert float(np.max(np.abs(forcing_values))) == 0.0
    assert np.all(extension.squared_forcing_time_integral(wavenumbers) == 0.0)
    assert total_strip_forcing_squared(extension, wavenumbers) == 0.0


def test_exact_extension_recovers_datum_at_terminal_time():
    datum = PeriodisedBernoulliDatum(1)
    extension = ExactSolutionExtension(
        datum, advection_diffusion_reaction(), TERMINAL_TIME
    )
    wavenumbers = symmetric_wavenumber_band(8)
    np.testing.assert_allclose(
        extension.extension_coefficient(wavenumbers, TERMINAL_TIME),
        datum.fourier_coefficients(wavenumbers),
        rtol=1e-15,
    )


# ---------------------------------------------------------------------------
# Closed-form time integrals against numpy trapezoid quadrature
# ---------------------------------------------------------------------------


def test_exponential_time_integral_factor_zero_branch():
    factor_values = exponential_time_integral_factor(
        np.array([0.0, -2.0]), TERMINAL_TIME
    )
    assert factor_values[0] == TERMINAL_TIME
    assert factor_values[1] == pytest.approx(
        -np.expm1(-2.0 * TERMINAL_TIME) / 2.0, rel=1e-15
    )


@pytest.mark.parametrize("extension_name", sorted(_extension_catalogue()))
def test_closed_form_time_integral_matches_trapezoid(extension_name):
    # (f) The closed-form per-wavenumber time integral of the squared forcing
    # matches a dense numpy trapezoid quadrature to 1e-10 relative.  The
    # quadrature grid is fine enough (dt ~ 3e-6) that the O(dt^2) trapezoid
    # error is below 3e-11 relative for the fastest-decaying integrand in the
    # band |k| <= 2.
    extension = _extension_catalogue()[extension_name]
    wavenumbers = symmetric_wavenumber_band(2)
    time_grid = np.linspace(0.0, TERMINAL_TIME, 320_001)
    forcing_on_grid = extension.forcing_coefficient(
        wavenumbers, time_grid[:, None]
    )
    quadrature_integral = np.trapezoid(
        np.abs(forcing_on_grid) ** 2, time_grid, axis=0
    )
    closed_form_integral = extension.squared_forcing_time_integral(wavenumbers)
    if extension_name == "exact_g1":
        assert np.all(quadrature_integral == 0.0)
        assert np.all(closed_form_integral == 0.0)
    else:
        np.testing.assert_allclose(
            closed_form_integral, quadrature_integral, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Graded Gaussian extension: matched split equivalence and power ratio (P3)
# ---------------------------------------------------------------------------


def test_matched_graded_gaussian_equals_split_semigroup():
    # With the comparison diffusivity equal to the generator diffusivity, the
    # graded Gaussian extension coincides with the split extension whose
    # subset is the diffusion order.
    datum = PeriodisedBernoulliDatum(1)
    generator = advection_diffusion_reaction()
    graded_extension = GradedGaussianExtension(datum, generator, 0.7, TERMINAL_TIME)
    split_extension = SplitSemigroupExtension(datum, generator, [2], TERMINAL_TIME)
    wavenumbers = symmetric_wavenumber_band(8)
    for time in (0.2, 0.8):
        np.testing.assert_allclose(
            graded_extension.extension_coefficient(wavenumbers, time),
            split_extension.extension_coefficient(wavenumbers, time),
            rtol=1e-14,
        )
    np.testing.assert_allclose(
        graded_extension.squared_forcing_time_integral(wavenumbers),
        split_extension.squared_forcing_time_integral(wavenumbers),
        rtol=1e-14,
    )


def test_mismatched_graded_power_ratio_at_terminal_time():
    # (P3) At t = T the Gaussian factor equals 1, so the squared-forcing
    # ratio between the mismatched graded extension (nu_c = nu/2) and the
    # constant-in-time extension tends to ((nu - nu_c)/nu)^2 = 0.25 as
    # |k| -> infinity.
    datum = PeriodisedBernoulliDatum(1)
    generator = advection_diffusion_reaction()
    graded_extension = GradedGaussianExtension(datum, generator, 0.35, TERMINAL_TIME)
    constant_extension = ConstantInTimeExtension(datum, generator, TERMINAL_TIME)
    large_wavenumber = np.array([10_000], dtype=np.int64)
    graded_power = np.abs(
        graded_extension.forcing_coefficient(large_wavenumber, TERMINAL_TIME)
    ) ** 2
    constant_power = np.abs(
        constant_extension.forcing_coefficient(large_wavenumber, TERMINAL_TIME)
    ) ** 2
    ratio = float(graded_power[0] / constant_power[0])
    assert abs(ratio - 0.25) < 1e-3, ratio


# ---------------------------------------------------------------------------
# Strip forcing: growth versus convergence in the band edge (P4, light)
# ---------------------------------------------------------------------------


def test_strip_forcing_growth_versus_convergence():
    # (P4, light version) For rho = 1 and p = 1 the constant-in-time strip
    # forcing grows linearly in the band edge, while the split extension with
    # defect order m = 1 <= rho converges.
    datum = PeriodisedBernoulliDatum(1)
    generator = advection_diffusion_reaction()
    constant_extension = ConstantInTimeExtension(datum, generator, TERMINAL_TIME)
    split_extension = SplitSemigroupExtension(datum, generator, [2], TERMINAL_TIME)
    small_band = symmetric_wavenumber_band(1024)
    large_band = symmetric_wavenumber_band(4096)

    constant_growth_ratio = total_strip_forcing_squared(
        constant_extension, large_band
    ) / total_strip_forcing_squared(constant_extension, small_band)
    assert abs(constant_growth_ratio - 4.0) < 0.2, constant_growth_ratio

    split_growth_ratio = total_strip_forcing_squared(
        split_extension, large_band
    ) / total_strip_forcing_squared(split_extension, small_band)
    assert abs(split_growth_ratio - 1.0) < 1e-2, split_growth_ratio


# ---------------------------------------------------------------------------
# Dissipativity validation on the extension side (g)
# ---------------------------------------------------------------------------


def test_exact_extension_refuses_antidissipative_generator():
    antidissipative_generator = ConstantCoefficientGenerator(
        {2: -0.5, 1: 1.0}, name="antidissipative_diffusion"
    )
    datum = PeriodisedBernoulliDatum(0)
    extension = ExactSolutionExtension(
        datum, antidissipative_generator, TERMINAL_TIME
    )
    wavenumbers = symmetric_wavenumber_band(8)
    with pytest.raises(ValueError, match="dissipativity violated"):
        extension.extension_coefficient(wavenumbers, 0.5)
    with pytest.raises(ValueError, match="dissipativity violated"):
        extension.squared_forcing_time_integral(wavenumbers)


def test_split_extension_refuses_antidissipative_subset():
    antidissipative_generator = ConstantCoefficientGenerator(
        {2: -0.5, 1: 1.0}, name="antidissipative_diffusion"
    )
    datum = PeriodisedBernoulliDatum(0)
    extension = SplitSemigroupExtension(
        datum, antidissipative_generator, [2], TERMINAL_TIME
    )
    wavenumbers = symmetric_wavenumber_band(8)
    with pytest.raises(ValueError, match="dissipativity violated"):
        extension.extension_coefficient(wavenumbers, 0.5)


def test_graded_extension_refuses_negative_diffusivity():
    datum = PeriodisedBernoulliDatum(0)
    with pytest.raises(ValueError, match="non-negative"):
        GradedGaussianExtension(
            datum, advection_diffusion_reaction(), -0.1, TERMINAL_TIME
        )
