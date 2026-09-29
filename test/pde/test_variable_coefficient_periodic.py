r"""Tests of the variable-coefficient periodic module (pre-registration 2026-09-29).

Every closed form is checked against an independent route: the Fourier action
against a pointwise evaluation followed by an FFT, the Galerkin reference
against the exact constant-coefficient semigroup at zero amplitude, the
forcing coefficients against the real-space forcing of the extension field,
the closed-form time integrals against quadrature, and the energies at zero
amplitude against the constant-coefficient library closed forms.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from learning_option_pricing.pde import (
    ConstantCoefficientGenerator,
    ConstantInTimeExtension,
    ConvexRawExtension,
    PeriodisedBernoulliDatum,
    SplitSemigroupExtension,
    bandlimited_bernoulli_cosine_coefficients,
    exact_solution_field,
    symmetric_wavenumber_band,
    total_strip_forcing_squared,
)
from learning_option_pricing.pde.operators import constant_coefficient_operator_parts
from learning_option_pricing.pde.variable_coefficient_periodic import (
    BandLimitedDatum,
    GalerkinReferenceSolution,
    TrigonometricCoefficient,
    VariableCoefficientGenerator,
    build_variable_coefficient_extension,
    full_wavenumber_band,
    galerkin_convergence_deviation,
    galerkin_reference_deviations,
    relative_deviations_from_fields,
    strip_forcing_energy,
    synthesise_dense,
)

TWO_PI = 2.0 * math.pi
PHASE = math.pi / 4.0
TERMINAL_TIME = 1.0


def _lv2_generator(ratio):
    return VariableCoefficientGenerator(
        {
            2: TrigonometricCoefficient(0.125, 0.125 * ratio, PHASE),
            1: TrigonometricCoefficient(-0.095, -0.125 * ratio, PHASE),
            0: -0.03,
        },
        "lv2",
    )


def _lv4_generator(ratio):
    return VariableCoefficientGenerator(
        {4: TrigonometricCoefficient(-0.05, -0.05 * ratio, PHASE), 1: 1.3, 0: -0.4}, "lv4"
    )


def _grid_derivatives(coefficients_dense, x, max_order):
    """Pointwise derivatives of a dense trigonometric polynomial."""
    band = (len(coefficients_dense) - 1) // 2
    k = np.arange(-band, band + 1)
    return {
        order: synthesise_dense(coefficients_dense * (1j * k) ** order, x)
        for order in range(max_order + 1)
    }


def _fft_coefficients(values, band):
    """Dense coefficients k = -band..band of samples on a uniform grid."""
    n = len(values)
    spectrum = np.fft.fft(values) / n
    k = np.arange(-band, band + 1)
    return spectrum[k % n]


@pytest.mark.parametrize("builder", [_lv2_generator, _lv4_generator])
def test_fourier_action_matches_pointwise_evaluation(builder):
    generator = builder(0.6)
    rng = np.random.default_rng(0)
    band = 8
    half = rng.normal(size=band) + 1j * rng.normal(size=band)
    f = np.concatenate([np.conj(half[::-1]), [0.3], half])  # real function
    x = np.linspace(0.0, TWO_PI, 64, endpoint=False)
    derivatives = _grid_derivatives(f, x, generator.principal_order)
    pointwise = sum(c(x) * derivatives[order] for order, c in generator.coefficients.items())
    expected = _fft_coefficients(pointwise, band + 1)
    np.testing.assert_allclose(generator.apply_fourier(f), expected, rtol=1e-11, atol=1e-11)
    # The Galerkin matrix reproduces the action inside the band.
    matrix = generator.galerkin_matrix(band)
    np.testing.assert_allclose(matrix @ f, generator.apply_fourier(f)[1:-1], rtol=1e-12, atol=1e-12)


def test_generator_validation():
    with pytest.raises(ValueError):  # coefficient changes sign
        VariableCoefficientGenerator({2: TrigonometricCoefficient(0.1, 0.2)}, "bad")
    with pytest.raises(ValueError):  # anti-dissipative fourth order
        VariableCoefficientGenerator({4: TrigonometricCoefficient(0.05, 0.01)}, "bad")
    with pytest.raises(ValueError):  # odd highest order
        VariableCoefficientGenerator({2: 0.1, 3: 0.2}, "bad")
    generator = _lv2_generator(0.5)
    assert generator.frozen_principal_coefficients("mean") == {2: 0.125}
    assert generator.frozen_principal_coefficients(0.0)[2] == pytest.approx(
        0.125 * (1 + 0.5 * math.cos(PHASE))
    )


@pytest.mark.parametrize("builder,orders", [(_lv2_generator, {2: 0.125, 1: -0.095, 0: -0.03}),
                                            (_lv4_generator, {4: -0.05, 1: 1.3, 0: -0.4})])
def test_reference_solution_at_zero_amplitude_is_the_exact_semigroup(builder, orders):
    band_edge = 16
    datum = BandLimitedDatum(PeriodisedBernoulliDatum(1), band_edge)
    reference = GalerkinReferenceSolution(builder(0.0), datum, 64, TERMINAL_TIME)
    exact = exact_solution_field(
        orders, bandlimited_bernoulli_cosine_coefficients(band_edge), terminal_time=TERMINAL_TIME
    )
    x = np.linspace(0.0, TWO_PI, 97)
    for time in (0.0, 0.4, 0.95, 1.0):
        t = np.full_like(x, time)
        np.testing.assert_allclose(reference.field(x, t), exact.field(x, t), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reference.terminal_datum_values(x), exact.terminal_datum_values(x),
                               rtol=1e-12, atol=1e-14)


def test_reference_solution_converges_in_the_galerkin_band():
    datum = BandLimitedDatum(PeriodisedBernoulliDatum(1), 16)
    generator = _lv2_generator(0.75)
    x = np.linspace(0.0, TWO_PI, 128, endpoint=False)
    coarse = GalerkinReferenceSolution(generator, datum, 48, TERMINAL_TIME)
    fine = GalerkinReferenceSolution(generator, datum, 96, TERMINAL_TIME)
    for time in (0.0, 0.5, 0.9):
        t = np.full_like(x, time)
        deviation = np.linalg.norm(coarse.field(x, t) - fine.field(x, t)) / np.linalg.norm(fine.field(x, t))
        assert deviation <= 1e-10, (time, deviation)


def test_relative_deviations_from_fields_on_a_known_example():
    times = [0.0, 0.5]
    second = np.array([[3.0, 4.0], [1.0, 0.0]])
    first = second + np.array([[0.0, 0.5], [0.0, 1.0]])
    deviations = relative_deviations_from_fields(first, second, times)
    np.testing.assert_allclose(deviations["per_time"], [0.1, 1.0])
    assert deviations["maximum_over_times"] == pytest.approx(1.0)
    assert deviations["time_of_maximum"] == 0.5
    assert deviations["space_time"] == pytest.approx(math.sqrt(1.25 / 26.0))
    assert deviations["space_time"] <= deviations["maximum_over_times"]


def test_relative_deviations_reject_a_vanishing_normalising_field():
    with pytest.raises(ValueError):
        relative_deviations_from_fields(np.ones((2, 3)), np.zeros((2, 3)), [0.0, 1.0])
    with pytest.raises(ValueError):
        relative_deviations_from_fields(np.ones((2, 3)), np.ones((3, 3)), [0.0, 1.0])


def test_galerkin_reference_deviations_extend_the_convergence_deviation():
    datum = BandLimitedDatum(PeriodisedBernoulliDatum(1), 8)
    generator = _lv4_generator(0.75)
    x = np.linspace(0.0, TWO_PI, 64, endpoint=False)
    times = np.linspace(0.0, TERMINAL_TIME, 5)
    deviations = galerkin_reference_deviations(generator, datum, 16, 24, TERMINAL_TIME, times, x)
    maximum = galerkin_convergence_deviation(generator, datum, 16, 24, TERMINAL_TIME, times, x)
    assert deviations["maximum_over_times"] == pytest.approx(maximum, rel=1e-12, abs=0.0)
    assert len(deviations["per_time"]) == len(times)
    # Both references return the datum at t = T (zero-padded to different bands).
    assert deviations["per_time"][-1] <= 1e-14
    assert deviations["space_time"] <= deviations["maximum_over_times"]


@pytest.mark.parametrize("builder", [_lv2_generator, _lv4_generator])
@pytest.mark.parametrize("variant", ["split_frozen_singular", "split_frozen_mean",
                                     "constant_in_time", "convex_raw"])
def test_forcing_coefficients_match_real_space_forcing(builder, variant):
    """The spectral forcing equals the FFT of d_t h + sum_j c_j(x) d_x^j h, with h
    synthesised independently from its own closed form."""
    band_edge = 12
    generator = builder(0.75)
    datum = BandLimitedDatum(PeriodisedBernoulliDatum(1), band_edge)
    extension = build_variable_coefficient_extension(variant, generator, datum, 0.0, TERMINAL_TIME)
    cosine = bandlimited_bernoulli_cosine_coefficients(band_edge)
    x = np.linspace(0.0, TWO_PI, 128, endpoint=False)
    band = full_wavenumber_band(band_edge + 1)
    for time in (0.0, 0.35, 0.9, 1.0):
        t = np.full_like(x, time)
        if variant.startswith("split"):
            frozen_at = 0.0 if variant == "split_frozen_singular" else "mean"
            field = exact_solution_field(
                generator.frozen_principal_coefficients(frozen_at), cosine, terminal_time=TERMINAL_TIME
            )
            values = {0: field.field(x, t), 1: field.space_derivative(x, t),
                      2: field.second_space_derivative(x, t)}
            if generator.principal_order == 4:
                values[4] = field.fourth_space_derivative(x, t)
            time_derivative = field.time_derivative(x, t)
        else:
            dense = datum.dense_coefficients(band_edge)
            derivatives = _grid_derivatives(dense, x, generator.principal_order)
            factor = 1.0 if variant == "constant_in_time" else time / TERMINAL_TIME
            values = {order: factor * d for order, d in derivatives.items()}
            time_derivative = (
                np.zeros_like(x) if variant == "constant_in_time" else derivatives[0] / TERMINAL_TIME
            )
        forcing = time_derivative + sum(
            c(x) * values[order] for order, c in generator.coefficients.items()
        )
        expected = _fft_coefficients(forcing, band_edge + 1)
        np.testing.assert_allclose(extension.forcing_coefficient(band, time), expected,
                                   rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("variant", ["split_frozen_singular", "split_frozen_mean",
                                     "constant_in_time", "convex_raw"])
def test_closed_form_time_integral_matches_quadrature(variant):
    band_edge = 6
    generator = _lv2_generator(0.5)
    datum = BandLimitedDatum(PeriodisedBernoulliDatum(1), band_edge)
    extension = build_variable_coefficient_extension(variant, generator, datum, 0.0, TERMINAL_TIME)
    band = full_wavenumber_band(band_edge + 1)
    nodes, weights = np.polynomial.legendre.leggauss(400)
    times = 0.5 * TERMINAL_TIME * (nodes + 1.0)
    quadrature = np.zeros(len(band))
    for time, weight in zip(times, weights):
        quadrature += 0.5 * TERMINAL_TIME * weight * np.abs(extension.forcing_coefficient(band, time)) ** 2
    np.testing.assert_allclose(extension.squared_forcing_time_integral(band), quadrature,
                               rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("builder,orders", [(_lv2_generator, {2: 0.125, 1: -0.095, 0: -0.03}),
                                            (_lv4_generator, {4: -0.05, 1: 1.3, 0: -0.4})])
def test_zero_amplitude_energies_equal_constant_coefficient_closed_forms(builder, orders):
    band_edge = 32
    generator = builder(0.0)
    datum = BandLimitedDatum(PeriodisedBernoulliDatum(1), band_edge)
    constant_generator = ConstantCoefficientGenerator(coefficients=orders, name="constant")
    library_datum = PeriodisedBernoulliDatum(1)
    top = max(o for o in orders if o % 2 == 0)
    references = {
        "constant_in_time": ConstantInTimeExtension(library_datum, constant_generator, TERMINAL_TIME),
        "convex_raw": ConvexRawExtension(library_datum, constant_generator, TERMINAL_TIME),
        "split_frozen_singular": SplitSemigroupExtension(library_datum, constant_generator, (top,), TERMINAL_TIME),
        "split_frozen_mean": SplitSemigroupExtension(library_datum, constant_generator, (top,), TERMINAL_TIME),
    }
    for variant, reference in references.items():
        extension = build_variable_coefficient_extension(variant, generator, datum, 0.0, TERMINAL_TIME)
        expected = total_strip_forcing_squared(reference, symmetric_wavenumber_band(band_edge))
        assert strip_forcing_energy(extension) == pytest.approx(expected, rel=1e-11), variant


def test_operator_with_callable_coefficients_matches_closed_form():
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(64, generator=generator, dtype=torch.float64).mul(TWO_PI).requires_grad_(True)
    t = torch.rand(64, generator=generator, dtype=torch.float64).requires_grad_(True)
    u = torch.exp(-t) * torch.sin(3.0 * x)
    diffusion = TrigonometricCoefficient(0.1, 0.05, PHASE)
    fourth = TrigonometricCoefficient(-0.05, -0.02, PHASE)
    parts = constant_coefficient_operator_parts(u, x, t, {4: fourth, 2: diffusion, 1: 1.3, 0: -0.4})
    torch.testing.assert_close(
        parts["diffusion"].detach(),
        (diffusion(x) * (-9.0) * torch.exp(-t) * torch.sin(3.0 * x)).detach(),
        rtol=1e-10, atol=1e-12,
    )
    torch.testing.assert_close(
        parts["higher_order"].detach(),
        (fourth(x) * 81.0 * torch.exp(-t) * torch.sin(3.0 * x)).detach(),
        rtol=1e-10, atol=1e-12,
    )


def test_bypass_matches_autograd_with_variable_coefficients():
    from learning_option_pricing.models.resnet import ResNet
    from learning_option_pricing.models.terminal_ansatz import (
        TerminalAnsatz,
        cross_check_extension_forcing_analytic_versus_autograd,
        make_interpolation_coefficient,
    )

    generator = _lv4_generator(0.75)
    field = exact_solution_field(
        generator.frozen_principal_coefficients(0.0),
        bandlimited_bernoulli_cosine_coefficients(8),
        terminal_time=TERMINAL_TIME,
    )
    torch.manual_seed(0)
    ansatz = TerminalAnsatz(
        ResNet(d_in=2, d_out=1, n=16, M=2, L=2).double(),
        None,
        make_interpolation_coefficient("linear", T=TERMINAL_TIME),
        form="hard_constant",
        extension_fn=field.field,
        extension_derivative_fns=field.derivative_callables(),
    )
    rng = torch.Generator().manual_seed(1)
    x = torch.rand(128, generator=rng, dtype=torch.float64) * TWO_PI
    t = torch.rand(128, generator=rng, dtype=torch.float64) * TERMINAL_TIME
    deviation = cross_check_extension_forcing_analytic_versus_autograd(
        ansatz, x, t, generator_coefficients=generator.runtime_coefficients()
    )
    assert deviation <= 1e-10
