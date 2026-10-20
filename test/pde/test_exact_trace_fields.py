"""Extension fields with the datum itself as trace and a Fourier band adapted to the time since the datum."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from learning_option_pricing.pde import (
    EXTENSION_FIELD_REGISTRY,
    ExactTraceAdaptiveBandField,
    bandlimited_bernoulli_cosine_coefficients,
    periodised_bernoulli_values,
)

TERMINAL_TIME = 1.0
ORDER_TWO = {2: 0.125, 1: -0.095, 0: -0.03}      # G_1 of the paper
ORDER_FOUR = {4: -0.05, 1: 1.3, 0: -0.4}          # G_2 of the paper
POINTS = np.array([0.0, 1e-3, 0.7, math.pi, 2.0 * math.pi - 1e-3, 5.5])


def adaptive_field(kind, coefficients, *, base_band=4096, maximum_band=2**21):
    return ExactTraceAdaptiveBandField(
        EXTENSION_FIELD_REGISTRY[kind], coefficients,
        datum_values=periodised_bernoulli_values,
        cosine_coefficients_for_band=bandlimited_bernoulli_cosine_coefficients,
        datum_amplitude_constant=1.0 / math.pi**2, datum_amplitude_exponent=2.0,
        terminal_time=TERMINAL_TIME, base_band=base_band, maximum_band=maximum_band)


def band_field(kind, coefficients, band):
    return EXTENSION_FIELD_REGISTRY[kind](coefficients, bandlimited_bernoulli_cosine_coefficients(band),
                                          terminal_time=TERMINAL_TIME)


def test_closed_form_datum_has_the_cosine_series_of_the_bernoulli_datum():
    grid = np.linspace(0.05, 2.0 * math.pi - 0.05, 41)
    series = band_field("exact_solution", ORDER_TWO, 2**16).terminal_datum_values(grid)
    np.testing.assert_allclose(periodised_bernoulli_values(grid), series, atol=1e-5)


@pytest.mark.parametrize("kind,coefficients", [("split_diffusion", ORDER_TWO), ("exact_solution", ORDER_TWO),
                                               ("split_principal", ORDER_FOUR), ("exact_solution", ORDER_FOUR)])
def test_the_trace_is_the_datum_itself(kind, coefficients):
    field = adaptive_field(kind, coefficients)
    np.testing.assert_array_equal(field.field(POINTS, np.full_like(POINTS, TERMINAL_TIME)), periodised_bernoulli_values(POINTS))
    np.testing.assert_array_equal(field.terminal_datum_values(POINTS), periodised_bernoulli_values(POINTS))
    points = torch.as_tensor(POINTS, dtype=torch.float32)
    trace = field.field(points, torch.full_like(points, TERMINAL_TIME))
    assert trace.dtype == torch.float32
    assert torch.equal(trace, periodised_bernoulli_values(points.double()).float())


def test_away_from_the_datum_the_base_band_is_used_unchanged():
    field, base = adaptive_field("split_diffusion", ORDER_TWO), band_field("split_diffusion", ORDER_TWO, 4096)
    times = np.full_like(POINTS, TERMINAL_TIME - 0.5)
    for method in ("field", "space_derivative", "time_derivative", "forcing_values"):
        np.testing.assert_array_equal(getattr(field, method)(POINTS, times), getattr(base, method)(POINTS, times))


def test_required_band_follows_the_decay_criterion():
    order_two, order_four = adaptive_field("split_diffusion", ORDER_TWO), adaptive_field("split_principal", ORDER_FOUR)
    bands, capped = order_two.required_band(np.array([0.5, 1e-6, 1e-9]))
    # K(s) = sqrt(40 / (0.125 s)): 25, 17889, 565686 -> 4096, 32768, 1048576.
    np.testing.assert_array_equal(bands, [4096, 32768, 2**20])
    assert not capped.any()
    bands, _ = order_four.required_band(np.array([1e-12]))
    # K(s) = (40 / (0.05 s))^(1/4) = 5318 -> 8192.
    np.testing.assert_array_equal(bands, [8192])


@pytest.mark.parametrize("method", ["field", "space_derivative", "forcing_values"])
def test_near_the_datum_the_adaptive_band_matches_a_much_larger_band(method):
    field = adaptive_field("split_diffusion", ORDER_TWO)
    reference = band_field("split_diffusion", ORDER_TWO, 2**20)
    base = band_field("split_diffusion", ORDER_TWO, 4096)
    times = np.full_like(POINTS, TERMINAL_TIME - 1e-6)
    adaptive_values = getattr(field, method)(POINTS, times)
    reference_values = getattr(reference, method)(POINTS, times)
    scale = np.max(np.abs(reference_values))
    assert np.max(np.abs(adaptive_values - reference_values)) <= 1e-12 * max(scale, 1.0)
    # The base band alone misses the components beyond 4096 at this time.
    assert np.max(np.abs(getattr(base, method)(POINTS, times) - reference_values)) > 1e-9


def test_points_beyond_the_maximum_band_are_reported_once_then_at_debug(caplog):
    field = adaptive_field("split_diffusion", ORDER_TWO, maximum_band=8192)
    times = np.full(3, TERMINAL_TIME - 1e-9)
    with caplog.at_level(logging.DEBUG, logger="learning_option_pricing.pde.exact_trace_fields"):
        values = field.field(POINTS[:3], times)
        field.field(POINTS[:3], times)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    debugs = [record for record in caplog.records if record.levelno == logging.DEBUG]
    assert len(warnings) == 1 and "maximum 8192" in warnings[0].getMessage()
    assert len(debugs) == 1
    np.testing.assert_array_equal(values, band_field("split_diffusion", ORDER_TWO, 8192).field(POINTS[:3], times))
    assert field.omitted_part_bound(1e-9, 8192) > 1e-6


def test_derivatives_at_the_datum_slice_are_reported_and_band_limited(caplog):
    field = adaptive_field("split_diffusion", ORDER_TWO)
    times = np.full_like(POINTS, TERMINAL_TIME)
    with caplog.at_level(logging.WARNING, logger="learning_option_pricing.pde.exact_trace_fields"):
        values = field.space_derivative(POINTS, times)
    assert any("datum slice" in record.getMessage() for record in caplog.records)
    np.testing.assert_array_equal(values, band_field("split_diffusion", ORDER_TWO, 4096).space_derivative(POINTS, times))


def test_mixed_batches_are_evaluated_point_by_point_with_torch():
    field = adaptive_field("exact_solution", ORDER_TWO)
    x = torch.as_tensor(np.tile(POINTS[:3], 3))
    t = torch.as_tensor(np.repeat([TERMINAL_TIME, TERMINAL_TIME - 1e-6, 0.0], 3))
    values = field.field(x, t).numpy()
    np.testing.assert_array_equal(values[:3], periodised_bernoulli_values(POINTS[:3]))
    np.testing.assert_allclose(values[3:6], band_field("exact_solution", ORDER_TWO, 2**20).field(POINTS[:3], t[3:6].numpy()),
                               atol=1e-13)
    np.testing.assert_array_equal(values[6:], band_field("exact_solution", ORDER_TWO, 4096).field(POINTS[:3], t[6:].numpy()))


def test_derivative_callables_have_the_keys_of_the_base_field():
    for kind, coefficients in (("split_diffusion", ORDER_TWO), ("split_principal", ORDER_FOUR)):
        field = adaptive_field(kind, coefficients)
        assert set(field.derivative_callables()) == set(band_field(kind, coefficients, 4096).derivative_callables())


def test_kinds_without_exponential_decay_are_refused():
    with pytest.raises(ValueError, match="no exponential decay"):
        ExactTraceAdaptiveBandField(
            EXTENSION_FIELD_REGISTRY["graded_chen_mangasarian"], ORDER_TWO,
            datum_values=periodised_bernoulli_values,
            cosine_coefficients_for_band=bandlimited_bernoulli_cosine_coefficients,
            datum_amplitude_constant=1.0 / math.pi**2, datum_amplitude_exponent=2.0, terminal_time=TERMINAL_TIME,
            base_band=4096, maximum_band=2**21, builder_keyword_arguments={"initial_smoothing_scale": 0.5})


def test_times_beyond_the_datum_slice_are_refused():
    with pytest.raises(ValueError, match="beyond the datum slice"):
        adaptive_field("split_diffusion", ORDER_TWO).field(POINTS, np.full_like(POINTS, TERMINAL_TIME + 0.1))
