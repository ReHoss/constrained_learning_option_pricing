"""Square-root smoothings with the width law eps(s) = eps_0 sqrt(s/T) (added 2026-10-21).

The width law is that of the exact-terminal-condition field of Zhang, Guo and Lu (2026, page 9); the
transported kind is that field's construction (the smoothing of the datum transported by the orders 1
and 0). Each test pins one property the runs rely on:

* the analytic time derivative, which uses eps eps' = eps_0^2 / (2T), coincides with finite differences;
* the transported kind tends to the transported datum as eps_0 -> 0;
* the closed-form spectral counterpart coincides with the field and with a brute-force time integral;
* the exact-trace band grows like s^{-1/2} and the reported bound uses the multiplier itself;
* at the datum slice the time derivative has the almost-everywhere limit eps_0^2 / (4T |y - a_k|).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from scipy import integrate

from learning_option_pricing.pde import (
    EXTENSION_FIELD_REGISTRY,
    ConstantCoefficientGenerator,
    ExactTraceAdaptiveBandField,
    PeriodisedBernoulliDatum,
    SquareRootTimeSmoothingExtension,
    advection_diffusion_reaction,
    bandlimited_bernoulli_cosine_coefficients,
    chen_mangasarian_multiplier,
    periodised_bernoulli_values,
)
from learning_option_pricing.pde.real_line_butterfly_fields import ButterflyDatum, RealLineButterflyField

TERMINAL_TIME = 1.0
ORDER_TWO = {2: 0.125, 1: -0.095, 0: -0.03}      # G_1 of the paper
ORDER_FOUR = {4: -0.05, 1: 1.3, 0: -0.4}          # G_2 of the paper
SMOOTHING_SCALE = 0.5
KINDS = ("square_root_smoothing_sqrt_time", "square_root_smoothing_sqrt_time_transported")
BAND = 64


def circle_field(kind, coefficients, band=BAND, scale=SMOOTHING_SCALE):
    return EXTENSION_FIELD_REGISTRY[kind](coefficients, bandlimited_bernoulli_cosine_coefficients(band),
                                          terminal_time=TERMINAL_TIME, initial_smoothing_scale=scale)


@pytest.mark.parametrize("coefficients", [ORDER_TWO, ORDER_FOUR])
@pytest.mark.parametrize("kind", KINDS)
def test_circle_time_derivative_coincides_with_central_differences(kind, coefficients):
    field = circle_field(kind, coefficients)
    x = np.linspace(0.05, 2.0 * math.pi - 0.05, 37)
    for time_since_datum in (0.3, 0.02):
        t = np.full_like(x, TERMINAL_TIME - time_since_datum)
        step = 1e-6 * time_since_datum
        finite_difference = (field.field(x, t + step) - field.field(x, t - step)) / (2.0 * step)
        np.testing.assert_allclose(field.time_derivative(x, t), finite_difference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("coefficients", [ORDER_TWO, ORDER_FOUR])
def test_transported_kind_tends_to_the_transported_datum_as_the_scale_vanishes(coefficients):
    smoothed = circle_field("square_root_smoothing_sqrt_time_transported", coefficients, scale=1e-12)
    transported = EXTENSION_FIELD_REGISTRY["transported_datum"](
        coefficients, bandlimited_bernoulli_cosine_coefficients(BAND), terminal_time=TERMINAL_TIME)
    x = np.linspace(0.0, 2.0 * math.pi, 41)
    t = np.full_like(x, TERMINAL_TIME - 0.4)
    np.testing.assert_allclose(smoothed.field(x, t), transported.field(x, t), rtol=0, atol=1e-10)
    np.testing.assert_allclose(smoothed.space_derivative(x, t), transported.space_derivative(x, t), rtol=0, atol=1e-8)


def test_without_transport_the_two_kinds_coincide_for_a_generator_without_orders_one_and_zero():
    coefficients = {2: 0.125}
    plain, transported = (circle_field(kind, coefficients) for kind in KINDS)
    x = np.linspace(0.0, 2.0 * math.pi, 29)
    t = np.full_like(x, TERMINAL_TIME - 0.25)
    np.testing.assert_array_equal(plain.field(x, t), transported.field(x, t))


@pytest.mark.parametrize("coefficients", [ORDER_TWO, ORDER_FOUR])
@pytest.mark.parametrize("kind", KINDS)
def test_torch_and_numpy_paths_coincide(kind, coefficients):
    field = circle_field(kind, coefficients)
    random_generator = np.random.default_rng(3)
    x = random_generator.uniform(0.0, 2.0 * math.pi, 50)
    t = random_generator.uniform(0.0, 0.99 * TERMINAL_TIME, 50)
    for method in ("field", "time_derivative", "second_space_derivative", "forcing_values"):
        np.testing.assert_allclose(getattr(field, method)(torch.tensor(x), torch.tensor(t)).numpy(),
                                   getattr(field, method)(x, t), rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("coefficients", [ORDER_TWO, ORDER_FOUR])
@pytest.mark.parametrize("kind", KINDS)
def test_closed_form_counterpart_coincides_with_the_field(kind, coefficients):
    """The spectral class gives the field's Fourier coefficients and the forcing's time integral."""
    field = circle_field(kind, coefficients, band=8)
    generator = ConstantCoefficientGenerator(coefficients=coefficients, name="test")
    counterpart = SquareRootTimeSmoothingExtension(
        PeriodisedBernoulliDatum(1), generator, SMOOTHING_SCALE, TERMINAL_TIME,
        transported_orders=(0, 1) if kind.endswith("_transported") else ())
    x = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    wavenumbers = np.concatenate([-np.arange(8, 0, -1), np.arange(1, 9)])
    for time_value in (0.1, 0.7, 0.995):
        t = np.full_like(x, time_value)
        synthesis = np.real(np.sum(counterpart.extension_coefficient(wavenumbers, time_value)[:, None]
                                   * np.exp(1j * wavenumbers[:, None] * x[None, :]), axis=0))
        np.testing.assert_allclose(field.field(x, t), synthesis, rtol=0, atol=1e-13)
        forcing_synthesis = np.real(np.sum(counterpart.forcing_coefficient(wavenumbers, time_value)[:, None]
                                           * np.exp(1j * wavenumbers[:, None] * x[None, :]), axis=0))
        np.testing.assert_allclose(field.forcing_values(x, t), forcing_synthesis, rtol=1e-10, atol=1e-11)
    # Adaptive quadrature of one component in tau, s = tau^2 (the integrand is K_0^2-singular in s).
    for wavenumber in (1, 5):
        def integrand(tau, wavenumber=wavenumber):
            coefficient = counterpart.forcing_coefficient(np.array([wavenumber]), TERMINAL_TIME - tau**2)[0]
            return 2.0 * tau * abs(coefficient) ** 2
        brute_force, _ = integrate.quad(integrand, 0.0, math.sqrt(TERMINAL_TIME), limit=400, epsabs=0.0, epsrel=1e-11)
        closed_form = float(counterpart.squared_forcing_time_integral(np.array([wavenumber]))[0])
        assert abs(closed_form - brute_force) <= 1e-6 * brute_force


def test_time_derivative_is_not_finite_at_the_datum_slice_on_the_circle():
    field = circle_field("square_root_smoothing_sqrt_time", ORDER_TWO)
    x = np.array([0.3, 1.0])
    assert not np.any(np.isfinite(field.time_derivative(x, np.full_like(x, TERMINAL_TIME))))


def test_exact_trace_band_grows_like_one_over_square_root_of_s():
    field = ExactTraceAdaptiveBandField(
        EXTENSION_FIELD_REGISTRY["square_root_smoothing_sqrt_time"], ORDER_TWO,
        datum_values=periodised_bernoulli_values,
        cosine_coefficients_for_band=bandlimited_bernoulli_cosine_coefficients,
        datum_amplitude_constant=1.0 / math.pi**2, datum_amplitude_exponent=2.0, terminal_time=TERMINAL_TIME,
        base_band=4096, maximum_band=2**21, builder_keyword_arguments={"initial_smoothing_scale": SMOOTHING_SCALE})
    points = np.array([0.0, 0.7, math.pi])
    np.testing.assert_array_equal(field.field(points, np.full_like(points, TERMINAL_TIME)),
                                  periodised_bernoulli_values(points))
    # K(s) = 40 / (0.5 sqrt(s)): 8000 at s = 1e-4 -> 8192; 8e5 at s = 1e-8 -> 2^20; 4e6 at s = 4e-10 -> capped.
    bands, capped = field.required_band(np.array([1e-4, 1e-8, 4e-10]))
    np.testing.assert_array_equal(bands, [8192, 2**20, 2**21])
    np.testing.assert_array_equal(capped, [False, False, True])
    # The reported bound uses the multiplier m(z) = z K_1(z) at z = c s^(1/2) K.
    time_since_datum, band = 4e-10, 2**21
    expected = (1.0 / math.pi**2) * float(chen_mangasarian_multiplier(SMOOTHING_SCALE * math.sqrt(time_since_datum) * band)) / band
    assert field.omitted_part_bound(time_since_datum, band) == pytest.approx(expected, rel=1e-12)


def test_graded_kind_bound_uses_the_multiplier_and_exceeds_the_exponential_factor():
    field = ExactTraceAdaptiveBandField(
        EXTENSION_FIELD_REGISTRY["graded_chen_mangasarian"], ORDER_TWO,
        datum_values=periodised_bernoulli_values,
        cosine_coefficients_for_band=bandlimited_bernoulli_cosine_coefficients,
        datum_amplitude_constant=1.0 / math.pi**2, datum_amplitude_exponent=2.0, terminal_time=TERMINAL_TIME,
        base_band=4096, maximum_band=2**21, builder_keyword_arguments={"initial_smoothing_scale": SMOOTHING_SCALE})
    time_since_datum, band = 7.510e-6, 2**21
    scaled = SMOOTHING_SCALE * time_since_datum * band
    bound = field.omitted_part_bound(time_since_datum, band)
    assert bound == pytest.approx((1.0 / math.pi**2) * float(chen_mangasarian_multiplier(scaled)) / band, rel=1e-12)
    assert bound > (1.0 / math.pi**2) * math.exp(-scaled) / band


# -- real line --------------------------------------------------------------------------------------

DIFFUSIVITY, DRIFT, DISCOUNT_RATE = 0.125, 0.6, 0.3
REAL_LINE_COEFFICIENTS = {2: DIFFUSIVITY, 1: DRIFT, 0: -DISCOUNT_RATE}
DATUM = ButterflyDatum(half_width=1.0, singular_point=0.0)


def real_line_field(kind, scale=math.sqrt(2.0 * DIFFUSIVITY * TERMINAL_TIME)):
    return RealLineButterflyField(REAL_LINE_COEFFICIENTS, DATUM, extension_kind=kind, terminal_time=TERMINAL_TIME,
                                  initial_smoothing_scale=scale)


@pytest.mark.parametrize("kind", KINDS)
def test_real_line_time_derivative_coincides_with_central_differences(kind):
    field = real_line_field(kind)
    x = np.linspace(-3.0, 3.0, 61) + 0.013
    for time_since_datum in (0.4, 0.01):
        t = np.full_like(x, TERMINAL_TIME - time_since_datum)
        step = 1e-6 * time_since_datum
        finite_difference = (field.field(x, t + step) - field.field(x, t - step)) / (2.0 * step)
        np.testing.assert_allclose(field.time_derivative(x, t), finite_difference, rtol=1e-6, atol=1e-7)


def test_real_line_transported_kind_tends_to_the_transported_datum():
    smoothed = real_line_field("square_root_smoothing_sqrt_time_transported", scale=1e-12)
    transported = RealLineButterflyField(REAL_LINE_COEFFICIENTS, DATUM, extension_kind="transported_datum",
                                         terminal_time=TERMINAL_TIME)
    x = np.linspace(-3.0, 3.0, 61) + 0.013
    t = np.full_like(x, TERMINAL_TIME - 0.5)
    np.testing.assert_allclose(smoothed.field(x, t), transported.field(x, t), rtol=0, atol=1e-11)


@pytest.mark.parametrize("kind", KINDS)
def test_real_line_time_derivative_at_the_datum_is_the_almost_everywhere_limit(kind):
    scale = math.sqrt(2.0 * DIFFUSIVITY * TERMINAL_TIME)
    field = real_line_field(kind, scale)
    x = np.array([-2.5, -0.5, 0.25, 0.75, 1.5])
    rate = -DISCOUNT_RATE if kind.endswith("_transported") else 0.0
    velocity = DRIFT if kind.endswith("_transported") else 0.0
    singular = (scale**2 / (4.0 * TERMINAL_TIME)) * sum(
        jump / np.abs(x - kink) for kink, jump in zip(DATUM.kink_points, DATUM.first_derivative_jumps))
    expected = -(rate * DATUM.values(x) + velocity * DATUM.first_derivative_values(x) + singular)
    np.testing.assert_allclose(field.time_derivative(x, np.full_like(x, TERMINAL_TIME)), expected, rtol=1e-12)
    # Just after the datum the derivative approaches this limit.
    near = field.time_derivative(x, np.full_like(x, TERMINAL_TIME - 1e-12))
    np.testing.assert_allclose(near, expected, rtol=1e-4)
