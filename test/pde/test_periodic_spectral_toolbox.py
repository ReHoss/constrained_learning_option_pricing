"""Tests for the exact Fourier toolbox on the circle.

Covered here: exact Bernoulli / square-wave Fourier coefficients against the
FFT of high-resolution synthesised samples, the jump of the rho-th derivative
against one-sided finite-difference measurements, Parseval consistency,
generator symbols and splits, the dissipativity validation, and the
operator-channel floor against its predicted power-law constant.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from learning_option_pricing.pde import (
    ConstantCoefficientGenerator,
    PeriodisedBernoulliDatum,
    SquareWaveDatum,
    advection_diffusion_reaction,
    biharmonic_advection_reaction,
    black_scholes_log_price,
    operator_channel_floor,
    predicted_floor_exponent,
    predicted_operator_channel_floor_constant,
    symmetric_wavenumber_band,
    synthesise_datum_on_grid,
)


TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Datum coefficients
# ---------------------------------------------------------------------------


def test_bernoulli_coefficient_closed_form_and_symmetry():
    wavenumbers = np.array([-3, -1, 0, 1, 2, 5], dtype=np.int64)
    for regularity_index in (0, 1, 2):
        datum = PeriodisedBernoulliDatum(regularity_index)
        coefficient_values = datum.fourier_coefficients(wavenumbers)
        degree = regularity_index + 1
        for wavenumber, value in zip(wavenumbers, coefficient_values):
            if wavenumber == 0:
                assert value == 0.0
            else:
                expected = -math.factorial(degree) / (TWO_PI * 1j * wavenumber) ** degree
                assert abs(value - expected) < 1e-15
                # |c_k| = (rho+1)! / (2 pi |k|)^(rho+1), exactly.
                assert abs(
                    abs(value) - math.factorial(degree) / (TWO_PI * abs(wavenumber)) ** degree
                ) < 1e-16
        # Real datum: conjugate symmetry c_{-k} = conj(c_k).
        assert abs(
            datum.fourier_coefficients(np.array([-4]))[0]
            - np.conjugate(datum.fourier_coefficients(np.array([4]))[0])
        ) < 1e-16


@pytest.mark.parametrize("regularity_index", [0, 1, 2])
def test_bernoulli_coefficients_match_fft_of_samples(regularity_index):
    # (a) FFT of a high-resolution grid of closed-form samples (midpoint
    # convention at the break point) reproduces the exact coefficients up to
    # the aliasing error, which is well below 1e-5 on a 4096-point grid for
    # the low band |k| <= 32.
    number_of_grid_points = 4096
    datum = PeriodisedBernoulliDatum(regularity_index)
    grid_points = np.linspace(0.0, TWO_PI, number_of_grid_points, endpoint=False)
    sampled_values = datum.pointwise_values(grid_points)
    fft_coefficients = np.fft.fft(sampled_values) / number_of_grid_points
    fft_wavenumbers = np.rint(
        np.fft.fftfreq(number_of_grid_points, d=1.0 / number_of_grid_points)
    ).astype(np.int64)
    low_band_mask = np.abs(fft_wavenumbers) <= 32
    exact_coefficients = datum.fourier_coefficients(fft_wavenumbers[low_band_mask])
    worst_deviation = np.max(
        np.abs(fft_coefficients[low_band_mask] - exact_coefficients)
    )
    assert worst_deviation < 1e-5, worst_deviation


def test_square_wave_coefficients_match_fft_of_samples():
    number_of_grid_points = 4096
    datum = SquareWaveDatum()
    grid_points = np.linspace(0.0, TWO_PI, number_of_grid_points, endpoint=False)
    sampled_values = datum.pointwise_values(grid_points)
    fft_coefficients = np.fft.fft(sampled_values) / number_of_grid_points
    fft_wavenumbers = np.rint(
        np.fft.fftfreq(number_of_grid_points, d=1.0 / number_of_grid_points)
    ).astype(np.int64)
    low_band_mask = np.abs(fft_wavenumbers) <= 32
    exact_coefficients = datum.fourier_coefficients(fft_wavenumbers[low_band_mask])
    worst_deviation = np.max(
        np.abs(fft_coefficients[low_band_mask] - exact_coefficients)
    )
    assert worst_deviation < 1e-4, worst_deviation


def test_square_wave_even_coefficients_vanish():
    datum = SquareWaveDatum()
    even_wavenumbers = np.array([-6, -2, 0, 2, 4, 100], dtype=np.int64)
    assert np.all(datum.fourier_coefficients(even_wavenumbers) == 0.0)
    odd_value = datum.fourier_coefficients(np.array([3]))[0]
    assert abs(odd_value - 2.0 / (1j * math.pi * 3.0)) < 1e-16


# ---------------------------------------------------------------------------
# Jump of the rho-th derivative from one-sided finite differences
# ---------------------------------------------------------------------------


def _one_sided_rho_derivative_limits(datum, derivative_order, step):
    """Measure g^(rho)(0+) and g^(rho)(0-) from closed-form samples.

    A forward (respectively backward) finite-difference stencil of order
    ``derivative_order`` is evaluated with every point strictly inside the
    right (respectively left) side of the break point at 0.
    """
    stencil_weights = np.array(
        [
            (-1.0) ** (derivative_order - j) * math.comb(derivative_order, j)
            for j in range(derivative_order + 1)
        ]
    )
    right_points = step + step * np.arange(derivative_order + 1)
    right_limit = float(
        np.dot(stencil_weights, datum.pointwise_values(right_points))
        / step**derivative_order
    )
    left_points = (TWO_PI - step) - step * np.arange(derivative_order + 1)[::-1]
    left_limit = float(
        np.dot(stencil_weights, datum.pointwise_values(left_points))
        / step**derivative_order
    )
    return right_limit, left_limit


@pytest.mark.parametrize("regularity_index", [0, 1, 2])
def test_bernoulli_jump_matches_finite_difference(regularity_index):
    # (b) The jump of the rho-th derivative measured from one-sided
    # finite-difference limits matches the analytic value
    # -(rho+1)! (2 pi)^(-rho) to within the O(step) stencil bias.
    datum = PeriodisedBernoulliDatum(regularity_index)
    right_limit, left_limit = _one_sided_rho_derivative_limits(
        datum, regularity_index, step=1e-3
    )
    measured_jump = right_limit - left_limit
    relative_deviation = abs(
        measured_jump - datum.jump_of_rho_derivative
    ) / abs(datum.jump_of_rho_derivative)
    assert relative_deviation < 1e-2, (measured_jump, datum.jump_of_rho_derivative)


def test_square_wave_jumps_match_samples():
    datum = SquareWaveDatum()
    offset = 1e-6
    jump_at_zero = float(
        datum.pointwise_values(np.array([offset]))[0]
        - datum.pointwise_values(np.array([TWO_PI - offset]))[0]
    )
    jump_at_pi = float(
        datum.pointwise_values(np.array([math.pi + offset]))[0]
        - datum.pointwise_values(np.array([math.pi - offset]))[0]
    )
    assert jump_at_zero == datum.break_point_jumps[0.0] == 2.0
    assert jump_at_pi == datum.break_point_jumps[math.pi] == -2.0


# ---------------------------------------------------------------------------
# Parseval consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("regularity_index", [0, 1, 2])
def test_bernoulli_parseval_consistency(regularity_index):
    # (c) 2 pi * sum_{0<|k|<=K} |c_k|^2 converges to the analytic squared
    # L^2 norm; the slowest tail (rho = 0) is O(1/K), below 1e-5 relative at
    # K = 1e5.
    datum = PeriodisedBernoulliDatum(regularity_index)
    wavenumber_band = symmetric_wavenumber_band(100_000)
    truncated_norm = TWO_PI * float(
        np.sum(np.abs(datum.fourier_coefficients(wavenumber_band)) ** 2)
    )
    relative_deviation = abs(truncated_norm - datum.squared_l2_norm) / datum.squared_l2_norm
    assert relative_deviation < 1e-4, relative_deviation


def test_square_wave_parseval_consistency():
    datum = SquareWaveDatum()
    wavenumber_band = symmetric_wavenumber_band(100_000)
    truncated_norm = TWO_PI * float(
        np.sum(np.abs(datum.fourier_coefficients(wavenumber_band)) ** 2)
    )
    relative_deviation = abs(truncated_norm - datum.squared_l2_norm) / datum.squared_l2_norm
    assert relative_deviation < 1e-4, relative_deviation


# ---------------------------------------------------------------------------
# FFT synthesis (plotting path only)
# ---------------------------------------------------------------------------


def test_synthesise_datum_on_grid_matches_pointwise_values():
    datum = PeriodisedBernoulliDatum(1)  # continuous datum: no Gibbs layer
    grid_points, synthesised_values = synthesise_datum_on_grid(datum, 4096)
    closed_form_values = datum.pointwise_values(grid_points)
    assert np.max(np.abs(synthesised_values - closed_form_values)) < 1e-4


# ---------------------------------------------------------------------------
# Generator symbols, splits, semigroup multipliers
# ---------------------------------------------------------------------------


def test_named_generator_symbols_and_constants():
    generator_g1 = advection_diffusion_reaction()
    generator_g2 = black_scholes_log_price()
    generator_g3 = biharmonic_advection_reaction()

    assert generator_g1.half_order == 1
    assert generator_g1.principal_constant == pytest.approx(0.7)
    assert generator_g1.symbol(np.array([2]))[0] == pytest.approx(-3.2 + 2.6j)

    assert generator_g2.half_order == 1
    assert generator_g2.principal_constant == pytest.approx(0.125)
    assert generator_g2.symbol(np.array([1]))[0] == pytest.approx(-0.155 - 0.095j)

    # (ik)^4 = k^4, so the order-4 coefficient -0.05 yields -0.05 k^4.
    assert generator_g3.half_order == 2
    assert generator_g3.principal_constant == pytest.approx(0.05)
    assert generator_g3.symbol(np.array([1]))[0] == pytest.approx(-0.45 + 1.3j)
    assert generator_g3.symbol(np.array([10]))[0].real == pytest.approx(-500.4)


def test_generator_rejects_odd_max_order():
    with pytest.raises(ValueError, match="even"):
        ConstantCoefficientGenerator({1: 1.0}, name="pure_advection")


def test_split_symbols_and_defect_order():
    generator = advection_diffusion_reaction()
    generator_split = generator.split([2])
    wavenumbers = np.array([-3, 1, 4], dtype=np.int64)
    subset_values = generator_split.subset_symbol(wavenumbers)
    defect_values = generator_split.defect_symbol(wavenumbers)
    np.testing.assert_allclose(subset_values, -0.7 * wavenumbers.astype(float) ** 2)
    np.testing.assert_allclose(
        defect_values, 1.3j * wavenumbers.astype(float) - 0.4
    )
    assert generator_split.defect_order == 1.0
    # The split reconstitutes the full symbol.
    np.testing.assert_allclose(
        subset_values + defect_values, generator.symbol(wavenumbers), rtol=1e-15
    )


def test_split_with_all_orders_has_empty_defect():
    generator = advection_diffusion_reaction()
    generator_split = generator.split([2, 1, 0])
    assert generator_split.defect_order == float("-inf")
    wavenumbers = np.array([1, 5], dtype=np.int64)
    assert np.all(generator_split.defect_symbol(wavenumbers) == 0.0)


def test_split_rejects_unknown_orders():
    generator = advection_diffusion_reaction()
    with pytest.raises(ValueError, match=r"\[3\]"):
        generator.split([2, 3])


def test_semigroup_multiplier_value():
    generator = advection_diffusion_reaction()
    wavenumbers = np.array([2], dtype=np.int64)
    elapsed_time = 0.3
    multiplier = generator.semigroup_multiplier(elapsed_time, wavenumbers, [2])[0]
    assert multiplier == pytest.approx(math.exp(-0.7 * 4.0 * elapsed_time))


def test_semigroup_multiplier_rejects_negative_elapsed_time():
    generator = advection_diffusion_reaction()
    with pytest.raises(ValueError, match="non-negative elapsed time"):
        generator.semigroup_multiplier(-0.1, np.array([1]), [2])


def test_dissipativity_validation_raises_on_antidissipative_subset():
    # (g) An order-2 coefficient of the wrong sign gives the subset symbol
    # a_A(k) = +0.5 k^2 with strictly positive real part, so the semigroup is
    # refused.  (The purely advective subset [1] has Re a_A = 0 and passes;
    # see the companion test below.)
    antidissipative_generator = ConstantCoefficientGenerator(
        {2: -0.5, 1: 1.0}, name="antidissipative_diffusion"
    )
    wavenumbers = symmetric_wavenumber_band(8)
    with pytest.raises(ValueError, match="dissipativity violated"):
        antidissipative_generator.semigroup_multiplier(0.5, wavenumbers, [2])
    with pytest.raises(ValueError, match="dissipativity violated"):
        antidissipative_generator.validate_dissipativity(wavenumbers)


def test_purely_advective_subset_is_marginally_dissipative():
    # The subset symbol of the advection order alone is purely imaginary
    # (Re a_A = 0 <= 1e-12), so the semigroup multiplier has unit modulus and
    # the validation passes.
    generator = advection_diffusion_reaction()
    wavenumbers = symmetric_wavenumber_band(8)
    multiplier = generator.semigroup_multiplier(0.5, wavenumbers, [1])
    np.testing.assert_allclose(np.abs(multiplier), 1.0, rtol=1e-14)


# ---------------------------------------------------------------------------
# Operator-channel floor and its predicted power law
# ---------------------------------------------------------------------------


def test_floor_matches_predicted_constant_for_g1_rho1():
    # (h) G1 with rho = 1 has exponent e = 4p - 2 rho - 1 = 1, so
    # floor(K) = C_pred * K * (1 + o(1)); at K = 2^16 the ratio must be
    # within 5 percent of 1.
    generator = advection_diffusion_reaction()
    datum = PeriodisedBernoulliDatum(1)
    assert predicted_floor_exponent(generator, datum) == 1
    maximum_wavenumber = 2**16
    measured_floor = operator_channel_floor(generator, datum, maximum_wavenumber)
    predicted_constant = predicted_operator_channel_floor_constant(generator, datum)
    ratio = measured_floor / (predicted_constant * maximum_wavenumber)
    assert abs(ratio - 1.0) < 0.05, ratio


def test_floor_saturates_for_negative_exponent():
    # G1 with rho = 2 has e = -1 < 0: the floor tends to a finite limit, and
    # the predicted power-law constant is refused.
    generator = advection_diffusion_reaction()
    datum = PeriodisedBernoulliDatum(2)
    assert predicted_floor_exponent(generator, datum) == -1
    floor_small_band = operator_channel_floor(generator, datum, 2**10)
    floor_large_band = operator_channel_floor(generator, datum, 2**14)
    assert (floor_large_band - floor_small_band) / floor_large_band < 1e-2
    with pytest.raises(ValueError, match="positive growth exponent"):
        predicted_operator_channel_floor_constant(generator, datum)


def test_floor_constant_refused_for_multi_break_datum():
    # The single-break-point floor prediction does not apply verbatim to
    # the square wave.
    generator = advection_diffusion_reaction()
    with pytest.raises(ValueError, match="single break point"):
        predicted_operator_channel_floor_constant(generator, SquareWaveDatum())
