"""Delta and Gamma of the trained field at the initial date (stage-2 runner, CMAP talk 2026-10-14)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRECTORY = (
    REPOSITORY_ROOT / "experiments" / "python_scripts" / "exp_split_extension_trained"
)
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

runner = importlib.import_module("ablation_split_extension_trained")
catalogue = importlib.import_module("_split_extension_catalogue")

CELL = "butterfly_real_line"
SMALL_NETWORK = dict(catalogue.DEFAULT_HPARAMS, net_width=8, net_blocks=1, net_layers_per_block=1)


@pytest.fixture(scope="module")
def problem():
    return runner.build_problem(CELL)


def build_model(problem, variant_name, *, zero_network_output: bool):
    variant = catalogue.variant_by_name(CELL, variant_name)
    variant_problem = runner.problem_for_variant(problem, variant)
    model, _ = runner.build_ansatz(variant, variant_problem, SMALL_NETWORK, model_seed=0)
    if zero_network_output:
        last_linear_layer = [
            module for module in model.network.modules() if isinstance(module, torch.nn.Linear)
        ][-1]
        with torch.no_grad():
            last_linear_layer.weight.zero_()
            last_linear_layer.bias.zero_()
    return model, variant_problem


def test_control_with_zero_network_reproduces_the_closed_form_greeks(problem):
    model, variant_problem = build_model(problem, "exact_solution", zero_network_output=True)
    greeks = runner.compute_greeks(model, variant_problem)
    for greek in ("price", "delta", "gamma"):
        reference = greeks[f"{greek}_reference"]
        assert np.max(np.abs(greeks[f"{greek}_predicted"] - reference)) <= 1e-5 * np.max(np.abs(reference))
    assert greeks["delta_rel_l2_interior"] < 1e-5 and greeks["gamma_rel_l2_interior"] < 1e-5


def test_constant_extension_with_zero_network_gives_the_classical_derivatives_of_the_datum(problem):
    model, variant_problem = build_model(problem, "constant_in_time", zero_network_output=True)
    greeks = runner.compute_greeks(model, variant_problem)
    datum = variant_problem["butterfly_datum"]
    np.testing.assert_allclose(greeks["price_predicted"], datum.values(greeks["x"]), atol=1e-6)
    np.testing.assert_array_equal(greeks["delta_predicted"], datum.first_derivative_values(greeks["x"]))
    np.testing.assert_array_equal(greeks["gamma_predicted"], 0.0)


@pytest.mark.parametrize("variant_name", ["split_diffusion", "graded_chen_mangasarian"])
def test_greeks_of_a_trained_network_match_central_differences_in_double_precision(problem, variant_name):
    model, variant_problem = build_model(problem, variant_name, zero_network_output=False)
    model = model.double()
    greeks = runner.compute_greeks(model, variant_problem)
    step = 1e-3
    x = torch.as_tensor(greeks["x"], dtype=torch.float64)
    with torch.no_grad():
        def price(shift):
            return model(torch.stack([x + shift, torch.zeros_like(x)], dim=1)).squeeze(-1).numpy()
        centre, right, left = price(0.0), price(step), price(-step)
    np.testing.assert_allclose(greeks["price_predicted"], centre, rtol=0, atol=1e-12)
    central_delta = (right - left) / (2 * step)
    central_gamma = (right - 2 * centre + left) / step**2
    assert np.max(np.abs(greeks["delta_predicted"] - central_delta)) <= 1e-5 * np.max(np.abs(central_delta))
    assert np.max(np.abs(greeks["gamma_predicted"] - central_gamma)) <= 1e-4 * np.max(np.abs(central_gamma))


def write_saved_run(tmp_path, problem, variant_name, recorded_shift=0.0):
    model, variant_problem = build_model(problem, variant_name, zero_network_output=False)
    run_directory = tmp_path / "run"
    variant_directory = run_directory / f"variant_{variant_name}"
    (variant_directory / "models").mkdir(parents=True)
    torch.save(model.state_dict(), variant_directory / "models" / "model.pt")
    recorded_error = runner.compute_error_metrics(model, variant_problem)["rel_l2_t0"] * (1.0 + recorded_shift)
    runner.write_summary(run_directory / "metadata.yaml", {"cell": CELL, "seed": 0, "hparams": SMALL_NETWORK})
    runner.write_summary(run_directory / f"summary_{variant_name}.yaml", {variant_name: {"rel_l2_t0": recorded_error}})
    return run_directory, variant_directory, model, variant_problem


def test_replay_mode_writes_the_greeks_of_the_saved_model(tmp_path, problem):
    run_directory, variant_directory, model, variant_problem = write_saved_run(tmp_path, problem, "split_diffusion")
    assert runner.main(["--compute-greeks", str(run_directory), "--device", "cpu"]) == 0
    saved = np.load(variant_directory / "greeks.npz")
    expected = runner.compute_greeks(model, variant_problem)
    np.testing.assert_array_equal(saved["delta_predicted"], expected["delta_predicted"])
    np.testing.assert_array_equal(saved["gamma_predicted"], expected["gamma_predicted"])
    with open(run_directory / "greeks_summary.yaml") as summary_file:
        summary = yaml.safe_load(summary_file)
    assert summary["split_diffusion"]["gamma_rel_l2_interior"] == pytest.approx(float(saved["gamma_rel_l2_interior"]))
    assert summary["split_diffusion"]["reload_rel_l2_t0_deviation"] < 1e-6


def test_replay_mode_refuses_a_model_that_does_not_reproduce_the_recorded_error(tmp_path, problem):
    run_directory, _, _, _ = write_saved_run(tmp_path, problem, "split_diffusion", recorded_shift=1e-2)
    with pytest.raises(RuntimeError, match="reloaded model"):
        runner.main(["--compute-greeks", str(run_directory), "--device", "cpu"])
