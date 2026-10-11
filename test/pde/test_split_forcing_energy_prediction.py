"""Tests for the power-law datum and the predicted growth of the split forcing energy.

Covered here: the coefficients of :class:`PowerLawCosineDatum` (modulus equal
to that of the Bernoulli datum of regularity one for the exponent two, zero
mean, rejection of non-continuous exponents), the predicted exponent
:math:`e = 2m - 2s_A - 2\\alpha + 1` on a sixth-order generator, the predicted
constant against the closed-form band-truncated energy at a large band edge,
the boundedness of the energy for a negative exponent, the agreement of the
empty-subset prediction with the operator-channel floor constant, and the
rejected subsets.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from learning_option_pricing.pde import (
    ConstantCoefficientGenerator,
    PeriodisedBernoulliDatum,
    PowerLawCosineDatum,
    SplitSemigroupExtension,
    biharmonic_advection_reaction,
    predicted_operator_channel_floor_constant,
    predicted_split_forcing_energy_constant,
    predicted_split_forcing_energy_exponent,
    symmetric_wavenumber_band,
    total_strip_forcing_squared,
)


TERMINAL_TIME = 1.0


def _complete_sixth_order_generator() -> ConstantCoefficientGenerator:
    """Sixth-order generator with a coefficient of modulus one at every order 0..6.

    Every even-order term is dissipative ((ik)^6 = -k^6, (ik)^4 = k^4,
    (ik)^2 = -k^2), so every subset containing order 6 is dissipative.
    """
    return ConstantCoefficientGenerator(
        {6: 1.0, 5: 1.0, 4: -1.0, 3: 1.0, 2: 1.0, 1: 1.0, 0: -1.0},
        name="complete_sixth_order_test_generator",
    )


def test_power_law_datum_of_exponent_two_has_the_bernoulli_modulus():
    wavenumbers = symmetric_wavenumber_band(256)
    power_law_coefficients = PowerLawCosineDatum(2.0).fourier_coefficients(wavenumbers)
    bernoulli_coefficients = PeriodisedBernoulliDatum(1).fourier_coefficients(
        wavenumbers
    )
    np.testing.assert_allclose(
        np.abs(power_law_coefficients), np.abs(bernoulli_coefficients), rtol=1e-14
    )
    assert PowerLawCosineDatum(2.0).fourier_coefficients(np.array([0]))[0] == 0.0


def test_power_law_datum_is_real_even_with_the_stated_critical_index():
    datum = PowerLawCosineDatum(1.25)
    positive_wavenumbers = np.arange(1, 65)
    np.testing.assert_array_equal(
        datum.fourier_coefficients(positive_wavenumbers),
        datum.fourier_coefficients(-positive_wavenumbers),
    )
    assert np.all(np.imag(datum.fourier_coefficients(positive_wavenumbers)) == 0.0)
    assert datum.sobolev_critical_index == pytest.approx(0.75)
    assert datum.coefficient_amplitude == pytest.approx(1.0 / (2.0 * math.pi**2))


@pytest.mark.parametrize("decay_exponent", [1.0, 0.5, -2.0])
def test_power_law_datum_rejects_non_continuous_exponents(decay_exponent):
    with pytest.raises(ValueError, match="exceed 1"):
        PowerLawCosineDatum(decay_exponent)


@pytest.mark.parametrize(
    "subset_orders, expected_exponent",
    [
        # m = 4 > p = 3, s_B = 1 < 3/2: bounded.
        ((5, 6), -1.0),
        # m = 5, s_B = 2 > 3/2: linear growth.
        ((6,), 1.0),
        # Empty subset (constant in time), m = 6, no smoothing: 4p - 2*alpha + 1.
        ((), 9.0),
        # m = 0: bounded.
        ((1, 2, 3, 4, 5, 6), -9.0),
    ],
)
def test_predicted_exponent_on_the_sixth_order_generator(
    subset_orders, expected_exponent
):
    exponent = predicted_split_forcing_energy_exponent(
        _complete_sixth_order_generator(), subset_orders, PowerLawCosineDatum(2.0)
    )
    assert exponent == pytest.approx(expected_exponent)


@pytest.mark.parametrize(
    "subset_orders, decay_exponent, band_edge",
    [
        ((6,), 2.0, 2**14),
        ((6,), 1.25, 2**14),
        ((), 2.0, 2**14),
        ((), 1.25, 2**14),
        # e = 1/2: the constant term of E(K) - C K^(1/2) is of order one, so
        # the relative deviation decays like K^(-1/2) and a larger edge is used.
        ((5, 6), 1.25, 2**18),
    ],
)
def test_predicted_constant_matches_the_closed_form_energy(
    subset_orders, decay_exponent, band_edge
):
    generator = _complete_sixth_order_generator()
    datum = PowerLawCosineDatum(decay_exponent)
    exponent = predicted_split_forcing_energy_exponent(generator, subset_orders, datum)
    assert exponent > 0.0
    constant = predicted_split_forcing_energy_constant(
        generator, subset_orders, datum, TERMINAL_TIME
    )
    extension = SplitSemigroupExtension(datum, generator, subset_orders, TERMINAL_TIME)
    energy = total_strip_forcing_squared(extension, symmetric_wavenumber_band(band_edge))
    assert energy / (constant * band_edge**exponent) == pytest.approx(1.0, abs=1e-2)


def test_energy_is_bounded_for_a_negative_exponent():
    generator = _complete_sixth_order_generator()
    datum = PowerLawCosineDatum(2.0)
    subset_orders = (5, 6)
    assert predicted_split_forcing_energy_exponent(generator, subset_orders, datum) < 0
    extension = SplitSemigroupExtension(datum, generator, subset_orders, TERMINAL_TIME)
    energy_at_half_edge = total_strip_forcing_squared(
        extension, symmetric_wavenumber_band(2**11)
    )
    energy_at_edge = total_strip_forcing_squared(
        extension, symmetric_wavenumber_band(2**12)
    )
    assert (energy_at_edge - energy_at_half_edge) / energy_at_edge < 1e-3


def test_empty_subset_constant_equals_the_operator_channel_floor_constant():
    # Constant in time: E(K) = T * floor(K), and the exponent-two datum has
    # the coefficient modulus of the Bernoulli datum of regularity one.
    generator = biharmonic_advection_reaction()
    split_constant = predicted_split_forcing_energy_constant(
        generator, (), PowerLawCosineDatum(2.0), TERMINAL_TIME
    )
    floor_constant = predicted_operator_channel_floor_constant(
        generator, PeriodisedBernoulliDatum(1)
    )
    assert split_constant == pytest.approx(TERMINAL_TIME * floor_constant, rel=1e-12)


def test_prediction_rejects_uncovered_subsets_and_data():
    generator = _complete_sixth_order_generator()
    datum = PowerLawCosineDatum(2.0)
    with pytest.raises(ValueError, match="maximal order"):
        predicted_split_forcing_energy_exponent(generator, (4,), datum)
    with pytest.raises(ValueError, match="exhausts every order"):
        predicted_split_forcing_energy_exponent(generator, (0, 1, 2, 3, 4, 5, 6), datum)
    with pytest.raises(ValueError, match="positive growth exponent"):
        predicted_split_forcing_energy_constant(generator, (5, 6), datum)
    with pytest.raises(ValueError, match="decay_exponent"):
        predicted_split_forcing_energy_exponent(
            generator, (6,), PeriodisedBernoulliDatum(1)
        )
