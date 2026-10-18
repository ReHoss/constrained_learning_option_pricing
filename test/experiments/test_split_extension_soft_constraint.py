"""Soft-constraint variants of the stage-2 runner (pre-registration 2026-10-18).

The trial field of a soft-constraint variant is the bare network; the terminal
datum is penalised in the objective with the variant's weight.  These tests use a
small network and a few iterations: they check the plumbing, not the accuracy.
"""

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

SOFT_NAMES = [variant["name"] for variant in catalogue.SOFT_CONSTRAINT_VARIANTS]
CELLS_WITH_SOFT_VARIANTS = ("g2_bernoulli_full", "g3_bernoulli_full", "butterfly_real_line")
SMALL_TRAINING = dict(
    catalogue.DEFAULT_HPARAMS, net_width=8, net_blocks=1, net_layers_per_block=1,
    n_interior=64, n_terminal=32, n_validation=128, validation_every=2,
)


def test_the_runner_restates_the_soft_forms_of_the_library():
    from learning_option_pricing.models.terminal_ansatz import SOFT_FORMS

    assert tuple(SOFT_FORMS) == runner.SOFT_CONSTRAINT_FORMS


def test_three_weights_named_after_their_value():
    assert catalogue.SOFT_CONSTRAINT_TERMINAL_PENALTY_WEIGHTS == (1.0, 10.0, 100.0)
    assert SOFT_NAMES == [
        "soft_constraint_weight_1", "soft_constraint_weight_10", "soft_constraint_weight_100",
    ]
    for variant in catalogue.SOFT_CONSTRAINT_VARIANTS:
        assert variant["form"] == "soft_pinn"
        assert variant["extension"] is None
        assert variant["name"] in catalogue.all_variant_names()


@pytest.mark.parametrize("cell_name", CELLS_WITH_SOFT_VARIANTS)
def test_soft_variants_are_appended_last(cell_name):
    assert catalogue.variant_names(cell_name)[-len(SOFT_NAMES):] == SOFT_NAMES


@pytest.mark.parametrize("cell_name", CELLS_WITH_SOFT_VARIANTS)
def test_soft_variant_is_the_bare_network_and_uses_the_full_datum(cell_name):
    variant = catalogue.variant_by_name(cell_name, "soft_constraint_weight_10")
    problem = runner.problem_for_variant(runner.build_problem(cell_name), variant)
    if not runner.is_real_line_problem(problem):
        assert problem["terminal_datum"] is runner.full_bernoulli_datum_values
    model, extension_field = runner.build_ansatz(variant, problem, SMALL_TRAINING, model_seed=0)
    assert extension_field is None
    assert runner.build_closed_form_extension(variant, problem) is None
    points = torch.rand(16, 2, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(model(points), model.free_network(points))


def test_terminal_target_of_a_soft_variant_is_the_datum():
    cell_name = "g2_bernoulli_full"
    variant = catalogue.variant_by_name(cell_name, "soft_constraint_weight_1")
    problem = runner.problem_for_variant(runner.build_problem(cell_name), variant)
    model, _ = runner.build_ansatz(variant, problem, SMALL_TRAINING, model_seed=0)
    target = runner.compute_terminal_target(model, problem, variant, None)
    grid = runner._evaluation_grid(problem)
    np.testing.assert_allclose(
        target["phi_star_terminal"], runner.full_bernoulli_datum_values(grid), atol=1e-6
    )
    expected_distance = np.linalg.norm(
        target["phi_theta_terminal"] - target["phi_star_terminal"]
    ) / np.linalg.norm(target["phi_star_terminal"])
    assert target["terminal_target_distance"] == pytest.approx(expected_distance)
    assert not target["terminal_target_is_zero_target"]


@pytest.mark.parametrize("cell_name", CELLS_WITH_SOFT_VARIANTS)
def test_error_metrics_measure_the_terminal_mismatch_of_a_soft_variant(cell_name):
    variant = catalogue.variant_by_name(cell_name, "soft_constraint_weight_1")
    problem = runner.problem_for_variant(runner.build_problem(cell_name), variant)
    model, _ = runner.build_ansatz(variant, problem, SMALL_TRAINING, model_seed=0)
    metrics = runner.compute_error_metrics(model, problem)
    assert math.isfinite(metrics["tc_l2"]) and metrics["tc_l2"] > 0.0


def test_spectra_of_a_soft_variant_have_no_forcing_side():
    cell_name = "g2_bernoulli_full"
    variant = catalogue.variant_by_name(cell_name, "soft_constraint_weight_1")
    problem = runner.problem_for_variant(runner.build_problem(cell_name), variant)
    model, _ = runner.build_ansatz(variant, problem, SMALL_TRAINING, model_seed=0)
    spectra = runner.compute_spectra(model, problem, variant, None)
    assert not bool(spectra["forcing_defined"][0])
    assert float(spectra["forcing_power"].max()) == 0.0


@pytest.mark.parametrize("cell_name", ("g2_bernoulli_full", "butterfly_real_line"))
def test_soft_training_records_the_penalised_objective(cell_name):
    variant = catalogue.variant_by_name(cell_name, "soft_constraint_weight_10")
    problem = runner.problem_for_variant(runner.build_problem(cell_name), variant)
    model, history, cross_check_deviation = runner.train_variant(
        variant, problem, SMALL_TRAINING, num_iterations=4, seed=0,
        device=torch.device("cpu"), log_every=1,
    )
    assert cross_check_deviation is None
    assert len(history["validation_terminal_mismatch"]) == len(history["validation_iter"]) == 2
    retained_position = history["validation_iter"].index(history["retained_iter"])
    retained_mismatch = history["validation_terminal_mismatch"][retained_position]
    assert history["validation_terminal_mismatch_at_retained_state"] == retained_mismatch
    assert history["validation_objective_at_retained_state"] == pytest.approx(
        history["validation_residual"][retained_position] + 10.0 * retained_mismatch
    )
    # The objective adds the weighted penalty to the residual, so it exceeds L_pde.
    assert all(
        total > pde for total, pde in zip(history["loss"], history["loss_pde"])
    )
    assert all(math.isfinite(value) for value in history["loss"])


def test_hard_training_is_unchanged_by_the_soft_path():
    cell_name = "g2_bernoulli_full"
    variant = catalogue.variant_by_name(cell_name, "split_diffusion")
    problem = runner.problem_for_variant(runner.build_problem(cell_name), variant)
    _, history, _ = runner.train_variant(
        variant, problem, SMALL_TRAINING, num_iterations=4, seed=0,
        device=torch.device("cpu"), log_every=1,
    )
    assert history["validation_terminal_mismatch"] == []
    assert "validation_objective_at_retained_state" not in history
    assert history["loss"] == history["loss_pde"]
