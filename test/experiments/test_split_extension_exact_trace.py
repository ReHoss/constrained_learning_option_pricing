"""Exact-trace circle cells of the stage-2 runner (author request 2026-10-20: comparisons on g, not g_K)."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRECTORY = REPOSITORY_ROOT / "experiments" / "python_scripts" / "exp_split_extension_trained"
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

runner = importlib.import_module("ablation_split_extension_trained")
catalogue = importlib.import_module("_split_extension_catalogue")

from learning_option_pricing.pde import ExactTraceAdaptiveBandField, periodised_bernoulli_values

EXACT_TRACE_CELLS = ("g2_bernoulli_exact_trace", "g3_bernoulli_exact_trace")
SPECTRAL_VARIANTS = {"g2_bernoulli_exact_trace": ("split_diffusion", "split_diffusion_advection", "exact_solution"),
                     "g3_bernoulli_exact_trace": ("split_principal", "split_principal_advection", "exact_solution")}
SMALL_TRAINING = dict(
    catalogue.DEFAULT_HPARAMS, net_width=8, net_blocks=1, net_layers_per_block=1,
    n_interior=64, n_terminal=32, n_validation=128, validation_every=2,
)
POINTS = np.array([0.0, 1e-3, 0.7, math.pi, 2.0 * math.pi - 1e-3, 5.5])


def test_the_runner_and_library_closed_forms_of_the_datum_coincide():
    grid = np.linspace(-1.0, 7.0, 101)
    np.testing.assert_array_equal(runner.full_bernoulli_datum_values(grid), periodised_bernoulli_values(grid))


@pytest.mark.parametrize("cell_name", EXACT_TRACE_CELLS)
def test_exact_trace_cells_run_every_choice_but_the_square_root_smoothing(cell_name):
    names = catalogue.variant_names(cell_name)
    assert "graded_chen_mangasarian" not in names
    assert set(SPECTRAL_VARIANTS[cell_name]) | {"constant_in_time", "transported_datum"} <= set(names)
    assert names[-3:] == [variant["name"] for variant in catalogue.SOFT_CONSTRAINT_VARIANTS]


@pytest.mark.parametrize("cell_name", EXACT_TRACE_CELLS)
def test_the_reference_and_every_variant_have_the_datum_itself_as_trace(cell_name):
    problem = runner.build_problem(cell_name)
    terminal = np.full_like(POINTS, problem["terminal_time"])
    assert isinstance(problem["exact_field"], ExactTraceAdaptiveBandField)
    np.testing.assert_array_equal(problem["exact_field"].field(POINTS, terminal), periodised_bernoulli_values(POINTS))
    for variant in catalogue.variants_for_cell(cell_name):
        variant_problem = runner.problem_for_variant(problem, variant)
        np.testing.assert_array_equal(variant_problem["terminal_datum"](POINTS), periodised_bernoulli_values(POINTS))


@pytest.mark.parametrize("cell_name", EXACT_TRACE_CELLS)
def test_spectral_fields_are_exact_trace_and_the_model_imposes_g(cell_name):
    problem = runner.build_problem(cell_name)
    for variant_name in SPECTRAL_VARIANTS[cell_name]:
        variant = catalogue.variant_by_name(cell_name, variant_name)
        variant_problem = runner.problem_for_variant(problem, variant)
        model, extension_field = runner.build_ansatz(variant, variant_problem, SMALL_TRAINING, model_seed=0)
        assert isinstance(extension_field, ExactTraceAdaptiveBandField)
        x = torch.as_tensor(POINTS, dtype=torch.float32)
        with torch.no_grad():
            trace = model(torch.stack([x, torch.full_like(x, problem["terminal_time"])], dim=1)).squeeze(-1)
        assert torch.equal(trace, periodised_bernoulli_values(x.double()).float())


@pytest.mark.parametrize("cell_name", EXACT_TRACE_CELLS)
def test_away_from_the_datum_the_reference_equals_the_full_datum_cell_reference(cell_name):
    exact_trace_reference = runner.build_problem(cell_name)["exact_field"]
    band_reference = runner.build_problem(cell_name.replace("exact_trace", "full"))["exact_field"]
    times = np.full_like(POINTS, 0.5)
    np.testing.assert_array_equal(exact_trace_reference.field(POINTS, times), band_reference.field(POINTS, times))


@pytest.mark.parametrize("cell_name,variant_name", [("g2_bernoulli_exact_trace", "split_diffusion"),
                                                    ("g3_bernoulli_exact_trace", "split_principal")])
def test_short_training_runs_with_the_analytic_forcing_cross_check(cell_name, variant_name):
    variant = catalogue.variant_by_name(cell_name, variant_name)
    problem = runner.problem_for_variant(runner.build_problem(cell_name), variant)
    _, history, cross_check_deviation = runner.train_variant(
        variant, problem, SMALL_TRAINING, num_iterations=4, seed=0, device=torch.device("cpu"), log_every=1)
    assert cross_check_deviation is not None and cross_check_deviation < 1e-3
    assert all(math.isfinite(value) for value in history["loss"])


def test_reevaluation_recomputes_the_errors_of_a_saved_data_path_model_on_the_exact_trace_cell(tmp_path):
    import yaml

    full_cell, exact_cell = "g2_bernoulli_full", "g2_bernoulli_exact_trace"
    variant = catalogue.variant_by_name(full_cell, "constant_in_time")
    full_problem = runner.problem_for_variant(runner.build_problem(full_cell), variant)
    model, _ = runner.build_ansatz(variant, full_problem, SMALL_TRAINING, model_seed=0)
    run_directory = tmp_path / "run"
    (run_directory / "variant_constant_in_time" / "models").mkdir(parents=True)
    torch.save(model.state_dict(), run_directory / "variant_constant_in_time" / "models" / "model.pt")
    recorded = runner.compute_error_metrics(model, full_problem)
    runner.write_summary(run_directory / "metadata.yaml", {"cell": full_cell, "seed": 0, "hparams": SMALL_TRAINING})
    runner.write_summary(run_directory / "summary_constant_in_time.yaml", {"constant_in_time": recorded})
    assert runner.main(["--reevaluate-run", str(run_directory), "--reevaluation-cell", exact_cell, "--device", "cpu"]) == 0
    with open(run_directory / f"reevaluation_{exact_cell}.yaml") as reevaluation_file:
        result = yaml.safe_load(reevaluation_file)["constant_in_time"]
    expected = runner.compute_error_metrics(
        model, runner.problem_for_variant(runner.build_problem(exact_cell),
                                          catalogue.variant_by_name(exact_cell, "constant_in_time")))
    assert result["reevaluated"]["rel_l2"] == pytest.approx(expected["rel_l2"], rel=1e-12)
    assert result["recorded"]["rel_l2"] == pytest.approx(recorded["rel_l2"], rel=1e-12)
    # The two references differ by the truncation of the datum, about 2e-6 of its norm.
    assert abs(result["reevaluated"]["rel_l2"] - recorded["rel_l2"]) <= 1e-4 * recorded["rel_l2"] + 1e-5
