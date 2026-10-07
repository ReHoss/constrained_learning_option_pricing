"""Full-datum cells of the stage-2 runner (pre-registration 2026-10-08, sampling of singular forcing)."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "python_scripts" / "exp_split_extension_trained"
)
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

runner = importlib.import_module("ablation_split_extension_trained")
catalogue = importlib.import_module("_split_extension_catalogue")

POINTS_AWAY_FROM_THE_BREAK = np.linspace(0.3, 2.0 * math.pi - 0.3, 25)


def test_full_datum_equals_its_fourier_series_away_from_the_break_point():
    wavenumbers = np.arange(1, 200001, dtype=np.float64)
    series = np.cos(np.outer(POINTS_AWAY_FROM_THE_BREAK, wavenumbers)) @ (
        1.0 / (np.pi**2 * wavenumbers**2)
    )
    values = runner.full_bernoulli_datum_values(POINTS_AWAY_FROM_THE_BREAK)
    np.testing.assert_allclose(values, series, atol=1e-6)


def test_full_datum_is_periodic():
    values = runner.full_bernoulli_datum_values(POINTS_AWAY_FROM_THE_BREAK)
    shifted = runner.full_bernoulli_datum_values(POINTS_AWAY_FROM_THE_BREAK + 2.0 * math.pi)
    np.testing.assert_allclose(values, shifted, atol=1e-12)


def test_autograd_derivatives_of_the_full_datum_are_the_classical_ones():
    torch = pytest.importorskip("torch")
    x = torch.tensor(POINTS_AWAY_FROM_THE_BREAK, dtype=torch.float64, requires_grad=True)
    values = runner.full_bernoulli_datum_values(x)
    first = torch.autograd.grad(values.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    unit_position = POINTS_AWAY_FROM_THE_BREAK / (2.0 * math.pi)
    np.testing.assert_allclose(first.detach().numpy(), (2.0 * unit_position - 1.0) / (2.0 * math.pi),
                               rtol=1e-12)
    np.testing.assert_allclose(second.numpy(), np.full_like(unit_position, 1.0 / (2.0 * math.pi**2)),
                               rtol=1e-12)


@pytest.mark.parametrize(
    "cell_name, split_variant",
    [("g2_bernoulli_full", "split_diffusion"), ("g3_bernoulli_full", "split_principal")],
)
def test_full_datum_cells_give_the_full_datum_to_the_raw_variant_only(cell_name, split_variant):
    problem = runner.build_problem(cell_name)
    assert problem["band_edge"] == catalogue.FULL_DATUM_REFERENCE_BAND_EDGE
    assert catalogue.variant_names(cell_name) == ["constant_in_time", split_variant]
    raw_problem = runner.problem_for_variant(
        problem, catalogue.variant_by_name(cell_name, "constant_in_time")
    )
    split_problem = runner.problem_for_variant(
        problem, catalogue.variant_by_name(cell_name, split_variant)
    )
    assert raw_problem["terminal_datum"] is runner.full_bernoulli_datum_values
    assert split_problem["terminal_datum"] is problem["terminal_datum"]
    assert problem["terminal_datum"] is not runner.full_bernoulli_datum_values


def test_band_limited_cells_are_unchanged_by_the_variant_problem():
    problem = runner.build_problem("g2_bernoulli_bandlimited")
    variant = catalogue.variant_by_name("g2_bernoulli_bandlimited", "constant_in_time")
    assert runner.problem_for_variant(problem, variant) is problem


@pytest.mark.parametrize(
    "generator_coefficients",
    [{2: 0.125, 1: -0.095, 0: -0.03}, {4: -0.05, 1: 1.3, 0: -0.4}],
)
def test_line_source_correction_solves_its_equation_mode_by_mode(generator_coefficients):
    band_edge = 16
    grid_size = 64
    problem = {"generator_coefficients": generator_coefficients, "terminal_time": 1.0,
               "band_edge": band_edge}
    x = np.linspace(0.0, 2.0 * math.pi, grid_size, endpoint=False)
    time_value, time_step = 0.4, 1e-5

    def modes(t):
        values = runner.line_source_correction_values(problem, x, np.full_like(x, t))
        return np.fft.fft(values) / grid_size

    wavenumbers = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    in_band = np.abs(wavenumbers) <= band_edge
    time_derivative = (modes(time_value + time_step) - modes(time_value - time_step)) / (2 * time_step)
    symbol = sum(c * (1j * wavenumbers) ** j for j, c in generator_coefficients.items())
    jump = -1.0 / math.pi
    singular_coefficients = (jump / (2.0 * math.pi)) * sum(
        c * (1j * wavenumbers) ** (j - 2) for j, c in generator_coefficients.items() if j >= 2
    )
    residual = time_derivative + symbol * modes(time_value) - singular_coefficients
    assert np.max(np.abs(residual[in_band])) < 1e-6 * max(1.0, np.max(np.abs(singular_coefficients[in_band])))
    terminal_values = runner.line_source_correction_values(problem, x, np.full_like(x, 1.0))
    np.testing.assert_allclose(terminal_values, 0.0, atol=1e-15)
