"""Variable-coefficient cells on the full datum (stage-2 runner, pre-registration 2026-10-13)."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "python_scripts" / "exp_split_extension_trained"
)
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

runner = importlib.import_module("ablation_split_extension_trained")
catalogue = importlib.import_module("_split_extension_catalogue")

FULL_DATUM_CELLS = ("lv2_bernoulli_full_eps0p25", "lv2_bernoulli_full_eps0p75",
                    "lv4_bernoulli_full_eps0p25", "lv4_bernoulli_full_eps0p75")
SMALL_NETWORK = dict(catalogue.DEFAULT_HPARAMS, net_width=8, net_blocks=1, net_layers_per_block=1)


@pytest.fixture(scope="module", params=("lv2_bernoulli_full_eps0p25", "lv4_bernoulli_full_eps0p75"))
def full_datum_problem(request):
    return request.param, runner.build_problem(request.param)


def test_full_datum_cells_differ_from_the_band_limited_ones_only_by_the_datum():
    for cell_name in FULL_DATUM_CELLS:
        full_cell = catalogue.cell_by_name(cell_name)
        band_limited_cell = catalogue.cell_by_name(cell_name.replace("_full_", "_bandlimited_"))
        assert full_cell["generator_coefficients"] == band_limited_cell["generator_coefficients"]
        assert full_cell["datum"] == "bernoulli_full"
        assert full_cell["truncation_wavenumber"] == catalogue.FULL_DATUM_REFERENCE_BAND_EDGE
        assert catalogue.variant_names(cell_name) == [
            "constant_in_time", "split_frozen_singular", "split_frozen_mean"]


def test_reference_equals_the_full_datum_at_the_terminal_time(full_datum_problem):
    _, problem = full_datum_problem
    x = np.linspace(0.0, 2.0 * math.pi, 257, endpoint=False)
    terminal_time = problem["terminal_time"]
    reference = problem["exact_field"]
    np.testing.assert_array_equal(reference.field(x, np.full_like(x, terminal_time)),
                                  runner.full_bernoulli_datum_values(x))
    np.testing.assert_array_equal(reference.terminal_datum_values(x), runner.full_bernoulli_datum_values(x))
    # Before T the wrapper returns the Galerkin solution built on g_{N-1}.
    halfway = np.full_like(x, 0.5 * terminal_time)
    np.testing.assert_array_equal(reference.field(x, halfway), reference.galerkin_reference.field(x, halfway))
    assert reference.galerkin_band == runner.VARIABLE_COEFFICIENT_GALERKIN_BAND_BY_ORDER[
        problem["variable_generator"].principal_order]


def test_reference_just_before_the_terminal_time_is_close_to_the_datum(full_datum_problem):
    _, problem = full_datum_problem
    x = np.linspace(0.0, 2.0 * math.pi, 1024, endpoint=False)
    near_terminal = np.full_like(x, problem["terminal_time"] - 1e-6)
    values = problem["exact_field"].field(x, near_terminal)
    datum = runner.full_bernoulli_datum_values(x)
    assert np.linalg.norm(values - datum) / np.linalg.norm(datum) < 1e-2


def test_only_the_datum_path_sees_the_unprojected_datum(full_datum_problem):
    cell_name, problem = full_datum_problem
    x = torch.linspace(0.1, 6.0, 64, dtype=torch.float64)
    constant_problem = runner.problem_for_variant(problem, catalogue.variant_by_name(cell_name, "constant_in_time"))
    assert torch.equal(constant_problem["terminal_datum"](x), runner.full_bernoulli_datum_values(x))
    for split_name in ("split_frozen_singular", "split_frozen_mean"):
        split_problem = runner.problem_for_variant(problem, catalogue.variant_by_name(cell_name, split_name))
        assert split_problem["terminal_datum"] is problem["terminal_datum"]


@pytest.mark.parametrize("variant_name", ["constant_in_time", "split_frozen_singular", "split_frozen_mean"])
def test_every_variant_meets_its_datum_exactly_and_has_finite_errors(full_datum_problem, variant_name):
    cell_name, problem = full_datum_problem
    variant = catalogue.variant_by_name(cell_name, variant_name)
    variant_problem = runner.problem_for_variant(problem, variant)
    model, _ = runner.build_ansatz(variant, variant_problem, SMALL_NETWORK, model_seed=0)
    error_metrics = runner.compute_error_metrics(model, variant_problem)  # asserts the terminal identity
    assert error_metrics["tc_l2"] == 0.0
    assert all(math.isfinite(value) for value in error_metrics.values())


@pytest.mark.parametrize("variant_name", ["split_frozen_singular", "split_frozen_mean"])
def test_frozen_splits_have_a_finite_closed_form_forcing_floor(full_datum_problem, variant_name):
    cell_name, problem = full_datum_problem
    floor = runner.closed_form_forcing_floor(catalogue.variant_by_name(cell_name, variant_name), problem)
    assert math.isfinite(floor) and floor > 0.0
