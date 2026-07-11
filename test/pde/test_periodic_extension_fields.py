r"""Tests T1–T5 for the real-space closed-form periodic extension fields.

All tests run in ``float64`` (specification Section 6.2):

* **T1** terminal exactness — ``field(x, T)`` equals the datum sum exactly in
  floating point (fixed summation order; the decay factor at ``t = T`` is
  ``e^0 = 1`` exactly);
* **T2** analytic derivatives against ``torch.autograd`` on random points,
  relative deviation at most ``1e-10``;
* **T3** real-space split identity — for the split extensions,
  ``P h - B h`` relative deviation at most ``1e-12``;
* **T4** exact-solution forcing at most ``1e-12`` of the field scale;
* **T5** quadrature consistency — the trapezoidal strip integral of
  ``(P h)^2`` on a fine tensor grid against ``2 pi sum_k I_k`` from
  :mod:`learning_option_pricing.pde.terminal_data_extensions`, relative
  deviation at most ``1e-6``.

Constructor validation (raising, never silently clamping) is covered at the
end.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from learning_option_pricing.pde import (
    EXTENSION_FIELD_KINDS,
    EXTENSION_FIELD_REGISTRY,
    GradedGaussianExtension,
    PeriodicExtensionField,
    PeriodisedBernoulliDatum,
    SplitSemigroupExtension,
    advection_diffusion_reaction,
    bandlimited_bernoulli_cosine_coefficients,
    black_scholes_log_price,
    build_graded_gaussian_extension_field,
    build_split_diffusion_advection_extension_field,
    build_split_diffusion_extension_field,
    exact_solution_field,
    make_single_component_sine_cell,
    sine_cell_matched_exponential_rate,
    symmetric_wavenumber_band,
    total_strip_forcing_squared,
)

TWO_PI = 2.0 * math.pi
TERMINAL_TIME = 1.0

# Stage-2 generator coefficient mappings (specification Section 1.1).
GENERATOR_G1 = {2: 0.7, 1: 1.3, 0: -0.4}
GENERATOR_G2 = {2: 0.125, 1: -0.095, 0: -0.03}

# numpy >= 2 renames ``np.trapz`` to ``np.trapezoid``; resolve whichever is
# available at import time so numpy 1.x environments also pass.
trapezoid_integrate = getattr(np, "trapezoid", None) or np.trapz


def _build_field(extension_kind, generator_coefficients, truncation_wavenumber):
    """Build a field of the given kind on the band-limited Bernoulli datum."""
    cosine_coefficients = bandlimited_bernoulli_cosine_coefficients(
        truncation_wavenumber
    )
    comparison_diffusivity = (
        0.5 * generator_coefficients[2]
        if extension_kind == "graded_gaussian"
        else None
    )
    return PeriodicExtensionField(
        generator_coefficients,
        cosine_coefficients,
        extension_kind=extension_kind,
        comparison_diffusivity=comparison_diffusivity,
        terminal_time=TERMINAL_TIME,
    )


# ---------------------------------------------------------------------------
# T1 — terminal exactness (bitwise, fixed summation order)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extension_kind", EXTENSION_FIELD_KINDS)
def test_t1_terminal_identity_exact_in_floating_point_torch(extension_kind):
    truncation_wavenumber = 16
    field = _build_field(extension_kind, GENERATOR_G1, truncation_wavenumber)
    x = torch.linspace(0.0, TWO_PI, 97, dtype=torch.float64)
    t_terminal = torch.full_like(x, TERMINAL_TIME)

    # Independent datum sum with the same term values and the same fixed
    # reduction (sum over the trailing component axis).
    wavenumbers = torch.arange(
        1, truncation_wavenumber + 1, dtype=torch.float64
    )
    cosine_amplitudes = torch.as_tensor(
        bandlimited_bernoulli_cosine_coefficients(truncation_wavenumber)
    )
    datum_terms = cosine_amplitudes * torch.cos(x[..., None] * wavenumbers)
    expected_datum_values = datum_terms.sum(-1)

    assert torch.equal(field.field(x, t_terminal), expected_datum_values)
    assert torch.equal(
        field.terminal_datum_values(x), expected_datum_values
    )


@pytest.mark.parametrize("extension_kind", EXTENSION_FIELD_KINDS)
def test_t1_terminal_identity_exact_in_floating_point_numpy(extension_kind):
    truncation_wavenumber = 16
    field = _build_field(extension_kind, GENERATOR_G2, truncation_wavenumber)
    x = np.linspace(0.0, TWO_PI, 97)
    t_terminal = np.full_like(x, TERMINAL_TIME)

    wavenumbers = np.arange(1, truncation_wavenumber + 1, dtype=np.float64)
    cosine_amplitudes = bandlimited_bernoulli_cosine_coefficients(
        truncation_wavenumber
    )
    datum_terms = cosine_amplitudes * np.cos(x[..., None] * wavenumbers)
    expected_datum_values = datum_terms.sum(-1)

    assert np.array_equal(field.field(x, t_terminal), expected_datum_values)


def test_t1_sine_cell_terminal_identity_is_the_sine_datum():
    cell = make_single_component_sine_cell()
    x = torch.linspace(0.0, TWO_PI, 129, dtype=torch.float64)
    t_terminal = torch.full_like(x, cell.terminal_time)
    # Amplitude 1 at wavenumber 1: the component term is bitwise sin(x).
    assert torch.equal(cell.exact_solution.field(x, t_terminal), torch.sin(x))
    assert torch.equal(cell.terminal_datum_values(x), torch.sin(x))


# ---------------------------------------------------------------------------
# T2 — analytic derivatives against torch autograd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extension_kind", EXTENSION_FIELD_KINDS)
def test_t2_analytic_derivatives_match_autograd(extension_kind):
    field = _build_field(extension_kind, GENERATOR_G1, 12)
    generator = torch.Generator().manual_seed(0)
    x = (
        TWO_PI * torch.rand(256, generator=generator, dtype=torch.float64)
    ).requires_grad_(True)
    t = (
        TERMINAL_TIME * torch.rand(256, generator=generator, dtype=torch.float64)
    ).requires_grad_(True)

    field_values = field.field(x, t)
    (autograd_dt,) = torch.autograd.grad(
        field_values.sum(), (t,), create_graph=True
    )
    (autograd_dx,) = torch.autograd.grad(
        field_values.sum(), (x,), create_graph=True
    )
    (autograd_dxx,) = torch.autograd.grad(autograd_dx.sum(), (x,))

    for analytic_values, autograd_values in (
        (field.time_derivative(x, t), autograd_dt),
        (field.space_derivative(x, t), autograd_dx),
        (field.second_space_derivative(x, t), autograd_dxx),
    ):
        relative_deviation = (
            torch.linalg.vector_norm(analytic_values.detach() - autograd_values.detach())
            / torch.linalg.vector_norm(autograd_values.detach())
        ).item()
        assert relative_deviation <= 1.0e-10, (extension_kind, relative_deviation)


# ---------------------------------------------------------------------------
# T3 — real-space split identity P h = B h
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("generator_coefficients", [GENERATOR_G1, GENERATOR_G2])
def test_t3_split_diffusion_identity(generator_coefficients):
    # V3: subset {d_xx}, defect B = mu d_x + r0.
    field = _build_field("split_diffusion", generator_coefficients, 32)
    x = np.linspace(0.0, TWO_PI, 257)[None, :]
    t = np.linspace(0.0, TERMINAL_TIME, 17)[:, None]
    forcing_values = field.forcing_values(x, t)
    defect_values = (
        generator_coefficients[1] * field.space_derivative(x, t)
        + generator_coefficients[0] * field.field(x, t)
    )
    relative_deviation = np.linalg.norm(
        forcing_values - defect_values
    ) / np.linalg.norm(defect_values)
    assert relative_deviation <= 1.0e-12


@pytest.mark.parametrize("generator_coefficients", [GENERATOR_G1, GENERATOR_G2])
def test_t3_split_diffusion_advection_identity(generator_coefficients):
    # V4: subset {d_xx, d_x}, defect B = r0.
    field = _build_field(
        "split_diffusion_advection", generator_coefficients, 32
    )
    x = np.linspace(0.0, TWO_PI, 257)[None, :]
    t = np.linspace(0.0, TERMINAL_TIME, 17)[:, None]
    forcing_values = field.forcing_values(x, t)
    defect_values = generator_coefficients[0] * field.field(x, t)
    relative_deviation = np.linalg.norm(
        forcing_values - defect_values
    ) / np.linalg.norm(defect_values)
    assert relative_deviation <= 1.0e-12


# ---------------------------------------------------------------------------
# T4 — exact-solution forcing vanishes to round-off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("generator_coefficients", [GENERATOR_G1, GENERATOR_G2])
def test_t4_exact_solution_forcing_vanishes(generator_coefficients):
    field = _build_field("exact_solution", generator_coefficients, 32)
    x = np.linspace(0.0, TWO_PI, 257)[None, :]
    t = np.linspace(0.0, TERMINAL_TIME, 17)[:, None]
    forcing_norm = np.linalg.norm(field.forcing_values(x, t))
    field_norm = np.linalg.norm(field.field(x, t))
    assert forcing_norm <= 1.0e-12 * field_norm


def test_t4_sine_cell_exact_solution_closed_form_and_zero_forcing():
    cell = make_single_component_sine_cell()
    x = np.linspace(0.0, TWO_PI, 257)[None, :]
    t = np.linspace(0.0, cell.terminal_time, 17)[:, None]
    closed_form_values = np.exp(
        -cell.diffusivity * (cell.terminal_time - t)
    ) * np.sin(x)
    np.testing.assert_allclose(
        cell.exact_solution.field(x, t), closed_form_values, rtol=1.0e-14
    )
    forcing_norm = np.linalg.norm(cell.exact_solution.forcing_values(x, t))
    field_norm = np.linalg.norm(cell.exact_solution.field(x, t))
    assert forcing_norm <= 1.0e-12 * field_norm


# ---------------------------------------------------------------------------
# T5 — quadrature consistency with the closed-form strip forcing
# ---------------------------------------------------------------------------


def _spectral_reference_extension(extension_kind, generator):
    """The stage-1 spectral-side extension matching the real-space field."""
    datum = PeriodisedBernoulliDatum(1)
    if extension_kind == "split_diffusion":
        return SplitSemigroupExtension(datum, generator, [2], TERMINAL_TIME)
    if extension_kind == "split_diffusion_advection":
        return SplitSemigroupExtension(datum, generator, [2, 1], TERMINAL_TIME)
    # graded_gaussian, mismatched comparison diffusivity nu_c = nu / 2.
    return GradedGaussianExtension(
        datum, generator, 0.5 * generator.coefficients[2], TERMINAL_TIME
    )


@pytest.mark.parametrize(
    "extension_kind",
    ["split_diffusion", "split_diffusion_advection", "graded_gaussian"],
)
def test_t5_trapezoidal_strip_integral_matches_closed_form(extension_kind):
    truncation_wavenumber = 8
    field = _build_field(extension_kind, GENERATOR_G1, truncation_wavenumber)
    spectral_extension = _spectral_reference_extension(
        extension_kind, advection_diffusion_reaction()
    )
    wavenumber_band = symmetric_wavenumber_band(truncation_wavenumber)
    closed_form_value = total_strip_forcing_squared(
        spectral_extension, wavenumber_band
    )

    # Fine tensor grid: the x-integrand is a trigonometric polynomial of
    # degree at most 2 K, for which the periodic trapezoidal rule (endpoint
    # included) is exact; the time direction needs the fine step.
    x_grid = np.linspace(0.0, TWO_PI, 129)
    t_grid = np.linspace(0.0, TERMINAL_TIME, 4097)
    squared_forcing_values = (
        field.forcing_values(x_grid[None, :], t_grid[:, None]) ** 2
    )
    spatial_integrals = trapezoid_integrate(squared_forcing_values, x_grid, axis=1)
    strip_integral = trapezoid_integrate(spatial_integrals, t_grid)

    relative_deviation = abs(strip_integral - closed_form_value) / closed_form_value
    assert relative_deviation <= 1.0e-6, (extension_kind, relative_deviation)


# ---------------------------------------------------------------------------
# Registry, matched rate, and constructor validation
# ---------------------------------------------------------------------------


def test_registry_keys_and_builders():
    assert set(EXTENSION_FIELD_REGISTRY) == set(EXTENSION_FIELD_KINDS)
    cosine_coefficients = bandlimited_bernoulli_cosine_coefficients(4)
    for schema_key, builder in EXTENSION_FIELD_REGISTRY.items():
        built = builder(
            GENERATOR_G2,
            cosine_coefficients,
            comparison_diffusivity=(
                0.0625 if schema_key == "graded_gaussian" else None
            ),
            terminal_time=TERMINAL_TIME,
        )
        assert built.extension_kind == schema_key


def test_matched_exponential_rate_is_explicit_and_correct():
    # Specification D10: gamma = nu k0^2 = 0.125 for the control cell; the
    # unit-interval library default sigma^2 pi^2 / 2 would mismatch by pi^2.
    assert sine_cell_matched_exponential_rate(0.125, 1) == 0.125
    assert sine_cell_matched_exponential_rate(0.125, 2) == 0.5
    cell = make_single_component_sine_cell()
    assert cell.matched_exponential_rate == 0.125
    with pytest.raises(ValueError):
        sine_cell_matched_exponential_rate(0.0, 1)
    with pytest.raises(ValueError):
        sine_cell_matched_exponential_rate(0.125, 0)


def test_bandlimited_bernoulli_coefficients_match_stage1_datum():
    # a_k = 2 c_k on the retained band (real cosine amplitude versus the
    # complex coefficient of the stage-1 datum).
    truncation_wavenumber = 8
    cosine_amplitudes = bandlimited_bernoulli_cosine_coefficients(
        truncation_wavenumber
    )
    positive_wavenumbers = np.arange(1, truncation_wavenumber + 1)
    complex_coefficients = PeriodisedBernoulliDatum(1).fourier_coefficients(
        positive_wavenumbers
    )
    np.testing.assert_allclose(
        cosine_amplitudes, 2.0 * np.real(complex_coefficients), rtol=1.0e-15
    )
    assert np.max(np.abs(np.imag(complex_coefficients))) == 0.0


def test_constructor_validation_raises():
    cosine_coefficients = bandlimited_bernoulli_cosine_coefficients(4)
    with pytest.raises(ValueError):
        PeriodicExtensionField(
            GENERATOR_G1, cosine_coefficients, extension_kind="unknown_kind"
        )
    with pytest.raises(ValueError):
        PeriodicExtensionField(
            GENERATOR_G1,
            cosine_coefficients,
            extension_kind="graded_gaussian",  # missing comparison_diffusivity
        )
    with pytest.raises(ValueError):
        PeriodicExtensionField(
            GENERATOR_G1,
            cosine_coefficients,
            extension_kind="split_diffusion",
            comparison_diffusivity=0.35,  # forbidden outside graded_gaussian
        )
    with pytest.raises(ValueError):
        PeriodicExtensionField(
            {1: 1.3, 0: -0.4},  # missing the mandatory order 2
            cosine_coefficients,
            extension_kind="split_diffusion",
        )
    with pytest.raises(ValueError):
        PeriodicExtensionField(
            {2: -0.7},  # non-positive diffusivity
            cosine_coefficients,
            extension_kind="split_diffusion",
        )
    with pytest.raises(ValueError):
        PeriodicExtensionField(
            GENERATOR_G1,
            cosine_coefficients,
            extension_kind="split_diffusion",
            terminal_time=0.0,
        )
    with pytest.raises(ValueError):
        # Antidissipative exact solution: nu k^2 - r0 < 0 at k = 1.
        PeriodicExtensionField(
            {2: 0.1, 0: 0.5},
            bandlimited_bernoulli_cosine_coefficients(1),
            extension_kind="exact_solution",
        )
    with pytest.raises(ValueError):
        build_split_diffusion_extension_field(
            GENERATOR_G1, cosine_coefficients, comparison_diffusivity=0.35
        )
    with pytest.raises(ValueError):
        build_split_diffusion_advection_extension_field(
            GENERATOR_G1, cosine_coefficients, comparison_diffusivity=0.35
        )
    with pytest.raises(ValueError):
        exact_solution_field(
            GENERATOR_G1, cosine_coefficients, comparison_diffusivity=0.35
        )
    with pytest.raises(ValueError):
        build_graded_gaussian_extension_field(
            GENERATOR_G1, cosine_coefficients, comparison_diffusivity=-0.1
        )
