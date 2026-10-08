"""Closed-form fields of the butterfly datum on the real line (boundary paper, Example 3.1).

Each test pins one property the training runner relies on:

* the terminal identity ``field(x, T) == g(x)`` holds exactly in floating point;
* the analytic derivatives coincide with automatic differentiation;
* each kind has the forcing its retained operator predicts, and the exact solution
  coincides with an independent implementation of the paper's formula;
* the algebraic smoothing is the convolution with the Chen--Mangasarian kernel;
* the line-source correction coincides with adaptive quadrature and solves its
  equation away from the kinks;
* the window mean square coincides with a brute-force tensor-grid value.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import integrate
from scipy.stats import norm

from learning_option_pricing.pde.real_line_butterfly_fields import (
    REAL_LINE_BUTTERFLY_FIELD_KINDS,
    ButterflyDatum,
    RealLineButterflyField,
    datum_line_source_correction_values,
    window_mean_square_of_pointwise_values,
)

# Boundary paper, Figure 2: nu = 0.125, mu = 0.6, rho = 0.3, ell = 1, x* = 0, T = 1.
DIFFUSIVITY, DRIFT, DISCOUNT_RATE = 0.125, 0.6, 0.3
GENERATOR_COEFFICIENTS = {2: DIFFUSIVITY, 1: DRIFT, 0: -DISCOUNT_RATE}
TERMINAL_TIME = 1.0
DATUM = ButterflyDatum(half_width=1.0, singular_point=0.0)
SPATIAL_WINDOW = (-4.8, 4.2)


def build_field(kind: str) -> RealLineButterflyField:
    extra_arguments = {}
    if kind == "graded_gaussian":
        extra_arguments["comparison_diffusivity"] = 0.5 * DIFFUSIVITY
    if kind == "graded_chen_mangasarian":
        extra_arguments["initial_smoothing_scale"] = math.sqrt(2.0 * DIFFUSIVITY * TERMINAL_TIME)
    return RealLineButterflyField(
        GENERATOR_COEFFICIENTS, DATUM, extension_kind=kind, terminal_time=TERMINAL_TIME, **extra_arguments
    )


def interior_points(count: int = 400, seed: int = 0):
    random_generator = np.random.default_rng(seed)
    x = random_generator.uniform(*SPATIAL_WINDOW, count)
    t = random_generator.uniform(0.0, 0.98 * TERMINAL_TIME, count)
    return x, t


def test_datum_is_the_sum_of_three_plus_functions():
    x = np.linspace(-3.0, 3.0, 1201)
    plus_sum = sum(w * np.maximum(x - a, 0.0) for a, w in zip(DATUM.kink_points, DATUM.first_derivative_jumps))
    np.testing.assert_allclose(DATUM.values(x), plus_sum, atol=1e-15)
    assert DATUM.first_derivative_values(np.array([0.0]))[0] == 0.0  # midpoint of +1 and -1
    np.testing.assert_allclose(DATUM.first_derivative_values(np.array([-0.5, 0.5, 2.0])), [1.0, -1.0, 0.0])


@pytest.mark.parametrize("kind", REAL_LINE_BUTTERFLY_FIELD_KINDS)
def test_terminal_identity_is_exact_in_floating_point(kind):
    field = build_field(kind)
    x32 = torch.linspace(SPATIAL_WINDOW[0], SPATIAL_WINDOW[1], 1024, dtype=torch.float32)
    assert torch.equal(field.field(x32, torch.full_like(x32, TERMINAL_TIME)), DATUM.values(x32))
    x64 = np.linspace(SPATIAL_WINDOW[0], SPATIAL_WINDOW[1], 1024)
    assert np.array_equal(field.terminal_datum_values(x64), DATUM.values(x64))
    column = x32[:, None]
    assert torch.equal(field.field(column, torch.full_like(column, TERMINAL_TIME)), DATUM.values(column))


@pytest.mark.parametrize("kind", REAL_LINE_BUTTERFLY_FIELD_KINDS)
def test_analytic_derivatives_coincide_with_autograd(kind):
    field = build_field(kind)
    x_values, t_values = interior_points()
    x = torch.tensor(x_values, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(t_values, dtype=torch.float64, requires_grad=True)
    values = field.field(x, t)
    first_space, first_time = torch.autograd.grad(values.sum(), (x, t), create_graph=True)
    second_space = torch.autograd.grad(first_space.sum(), x)[0]
    with torch.no_grad():
        np.testing.assert_allclose(field.space_derivative(x, t).numpy(), first_space.detach().numpy(),
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(field.second_space_derivative(x, t).numpy(), second_space.numpy(),
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(field.time_derivative(x, t).numpy(), first_time.detach().numpy(),
                                   rtol=1e-10, atol=1e-12)


def test_numpy_and_torch_paths_coincide():
    field = build_field("exact_solution")
    x_values, t_values = interior_points()
    numpy_values = field.forcing_values(x_values, t_values)
    torch_values = field.forcing_values(torch.tensor(x_values), torch.tensor(t_values)).numpy()
    np.testing.assert_array_equal(numpy_values, torch_values)


def test_exact_solution_coincides_with_the_paper_formula():
    """u*(x, t) = e^{-rho s} V(x + mu s, s), V a combination of three call evolutions."""
    x_values, t_values = interior_points()
    time_to_terminal = TERMINAL_TIME - t_values
    standard_deviation = np.sqrt(2.0 * DIFFUSIVITY * time_to_terminal)
    shifted = x_values + DRIFT * time_to_terminal
    heat_evolution = sum(
        w * ((shifted - a) * norm.cdf((shifted - a) / standard_deviation)
             + standard_deviation * norm.pdf((shifted - a) / standard_deviation))
        for a, w in zip(DATUM.kink_points, DATUM.first_derivative_jumps)
    )
    expected = np.exp(-DISCOUNT_RATE * time_to_terminal) * heat_evolution
    np.testing.assert_allclose(build_field("exact_solution").field(x_values, t_values), expected,
                               rtol=1e-12, atol=1e-14)


def test_exact_solution_satisfies_the_equation_by_finite_differences():
    field = build_field("exact_solution")
    x_values, t_values = interior_points(200, seed=1)
    step = 1e-4
    time_derivative = (field.field(x_values, t_values + step) - field.field(x_values, t_values - step)) / (2 * step)
    space_derivative = (field.field(x_values + step, t_values) - field.field(x_values - step, t_values)) / (2 * step)
    second_space_derivative = (
        field.field(x_values + step, t_values) - 2 * field.field(x_values, t_values)
        + field.field(x_values - step, t_values)
    ) / step**2
    residual = (time_derivative + DIFFUSIVITY * second_space_derivative + DRIFT * space_derivative
                - DISCOUNT_RATE * field.field(x_values, t_values))
    assert np.max(np.abs(residual)) < 1e-5


@pytest.mark.parametrize(
    "kind, expected_forcing",
    [
        ("exact_solution", lambda field, x, t: np.zeros_like(x)),
        ("split_diffusion_advection", lambda field, x, t: -DISCOUNT_RATE * field.field(x, t)),
        ("split_diffusion",
         lambda field, x, t: DRIFT * field.space_derivative(x, t) - DISCOUNT_RATE * field.field(x, t)),
        ("transported_datum", lambda field, x, t: np.zeros_like(x)),
        ("graded_gaussian",
         lambda field, x, t: 0.5 * DIFFUSIVITY * field.second_space_derivative(x, t)
         + DRIFT * field.space_derivative(x, t) - DISCOUNT_RATE * field.field(x, t)),
    ],
)
def test_forcing_is_the_remainder_applied_to_the_field(kind, expected_forcing):
    field = build_field(kind)
    x_values, t_values = interior_points()
    np.testing.assert_allclose(field.forcing_values(x_values, t_values), expected_forcing(field, x_values, t_values),
                               atol=1e-13)


def test_chen_mangasarian_profile_is_the_kernel_convolution_of_the_plus_function():
    scale = 0.37

    def kernel(y):
        return 0.5 / scale * (1.0 + (y / scale) ** 2) ** -1.5

    assert integrate.quad(kernel, -np.inf, np.inf)[0] == pytest.approx(1.0, rel=1e-10)
    field = build_field("graded_chen_mangasarian")
    time_value = TERMINAL_TIME - scale * TERMINAL_TIME / field.initial_smoothing_scale
    for point in (-2.3, -1.1, -0.2, 0.0, 0.4, 1.7):
        convolution = sum(
            w * integrate.quad(lambda z, a=a: max(point - a - z, 0.0) * kernel(z), -np.inf, point - a,
                               epsabs=1e-12, epsrel=1e-12)[0]
            for a, w in zip(DATUM.kink_points, DATUM.first_derivative_jumps)
        )
        assert field.field(np.array([point]), np.array([time_value]))[0] == pytest.approx(convolution, abs=1e-9)


@pytest.mark.parametrize("temporal_factor_name", ["constant", "linear"])
def test_line_source_correction_coincides_with_adaptive_quadrature(temporal_factor_name):
    temporal_factor = (np.ones_like if temporal_factor_name == "constant"
                       else (lambda times: times / TERMINAL_TIME))
    points = [(-1.9, 0.0), (-0.7, 0.3), (0.05, 0.6), (1.0, 0.2), (2.4, 0.9), (-1.0, 0.5)]
    computed = datum_line_source_correction_values(
        np.array([p[0] for p in points]), np.array([p[1] for p in points]),
        generator_coefficients=GENERATOR_COEFFICIENTS, datum=DATUM, terminal_time=TERMINAL_TIME,
        temporal_factor=temporal_factor,
    )
    for (x_value, t_value), value in zip(points, computed):
        def integrand(sigma):
            variance = 2.0 * DIFFUSIVITY * sigma
            return sum(
                w * float(temporal_factor(np.array(t_value + sigma))) * math.exp(-DISCOUNT_RATE * sigma)
                * math.exp(-(x_value + DRIFT * sigma - a) ** 2 / (2 * variance)) / math.sqrt(2 * math.pi * variance)
                for a, w in zip(DATUM.kink_points, DATUM.first_derivative_jumps)
            )

        reference = -DIFFUSIVITY * integrate.quad(integrand, 0.0, TERMINAL_TIME - t_value, limit=500,
                                                  epsabs=1e-13, epsrel=1e-12)[0]
        assert value == pytest.approx(reference, abs=1e-10)


def test_line_source_correction_solves_its_equation_away_from_the_kinks():
    x_values = np.array([-2.6, -1.4, -0.45, 0.6, 1.5, 2.8])
    t_values = np.full_like(x_values, 0.4)
    step = 1e-3

    def correction(x, t):
        return datum_line_source_correction_values(
            x, t, generator_coefficients=GENERATOR_COEFFICIENTS, datum=DATUM, terminal_time=TERMINAL_TIME,
            temporal_factor=np.ones_like,
        )

    residual = (
        (correction(x_values, t_values + step) - correction(x_values, t_values - step)) / (2 * step)
        + DIFFUSIVITY * (correction(x_values + step, t_values) - 2 * correction(x_values, t_values)
                         + correction(x_values - step, t_values)) / step**2
        + DRIFT * (correction(x_values + step, t_values) - correction(x_values - step, t_values)) / (2 * step)
        - DISCOUNT_RATE * correction(x_values, t_values)
    )
    assert np.max(np.abs(residual)) < 1e-5
    terminal_values = correction(x_values, np.full_like(x_values, TERMINAL_TIME))
    np.testing.assert_array_equal(terminal_values, 0.0)


def test_window_mean_square_coincides_with_a_brute_force_tensor_grid():
    field = build_field("split_diffusion")
    computed = window_mean_square_of_pointwise_values(
        field.forcing_values, spatial_window=SPATIAL_WINDOW, terminal_time=TERMINAL_TIME,
        refinement_centres=lambda s: np.asarray(DATUM.kink_points),
    )
    tau = np.linspace(0.0, 1.0, 1001)
    x_grid = np.linspace(*SPATIAL_WINDOW, 40001)
    spatial_integrals = np.array([
        np.trapezoid(field.forcing_values(x_grid, np.full_like(x_grid, TERMINAL_TIME - value**2)) ** 2, x_grid)
        for value in tau
    ])
    brute_force = np.trapezoid(2.0 * tau * spatial_integrals, tau) / (
        (SPATIAL_WINDOW[1] - SPATIAL_WINDOW[0]) * TERMINAL_TIME)
    assert computed == pytest.approx(brute_force, rel=2e-5)
    # Times the window measure, it is the forcing energy E_V = 0.3821 of the heat evolution
    # quoted in Figure 1 of the boundary paper (the window contains the support up to 1e-10).
    assert computed * (SPATIAL_WINDOW[1] - SPATIAL_WINDOW[0]) * TERMINAL_TIME == pytest.approx(0.3821, abs=5e-5)


def test_window_mean_square_of_the_mismatched_gaussian_is_resolution_independent():
    field = build_field("graded_gaussian")

    def mean_square(refinement_point_count, time_node_count):
        return window_mean_square_of_pointwise_values(
            field.forcing_values, spatial_window=SPATIAL_WINDOW, terminal_time=TERMINAL_TIME,
            refinement_centres=lambda s: np.asarray(DATUM.kink_points),
            refinement_point_count=refinement_point_count, time_node_count=time_node_count,
        )

    assert mean_square(400, 64) == pytest.approx(mean_square(800, 128), rel=1e-6)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        RealLineButterflyField(GENERATOR_COEFFICIENTS, DATUM, extension_kind="graded_gaussian",
                               terminal_time=1.0)
    with pytest.raises(ValueError):
        RealLineButterflyField(GENERATOR_COEFFICIENTS, DATUM, extension_kind="split_diffusion",
                               terminal_time=1.0, comparison_diffusivity=0.1)
    with pytest.raises(ValueError):
        RealLineButterflyField({2: 0.1, 4: -0.05}, DATUM, extension_kind="exact_solution", terminal_time=1.0)
    with pytest.raises(ValueError):
        ButterflyDatum(half_width=0.0, singular_point=0.0)
