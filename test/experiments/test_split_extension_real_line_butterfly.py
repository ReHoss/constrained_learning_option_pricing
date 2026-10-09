"""Butterfly cell on the real line of the stage-2 runner (pre-registration 2026-10-09)."""

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

CELL = "butterfly_real_line"
SMALL_NETWORK = dict(catalogue.DEFAULT_HPARAMS, net_width=8, net_blocks=1, net_layers_per_block=1)


@pytest.fixture(scope="module")
def problem():
    return runner.build_problem(CELL)


def test_problem_is_posed_on_the_window_with_the_butterfly_datum(problem):
    assert runner.is_real_line_problem(problem)
    assert problem["spatial_window"] == catalogue.BUTTERFLY_SPATIAL_WINDOW
    assert problem["interior_window"] == pytest.approx((-3.6, 3.0))
    assert problem["generator_coefficients"] == {2: 0.125, 1: 0.6, 0: -0.3}
    assert problem["butterfly_datum"].kink_points == (-1.0, 0.0, 1.0)
    assert problem["terminal_datum"] == problem["butterfly_datum"].values
    assert problem["exact_field"].extension_kind == "exact_solution"
    assert runner.spatial_domain_measure(problem) == pytest.approx(9.0)


def test_circle_samples_are_unchanged_and_line_samples_fill_the_window(problem):
    unit_samples = torch.rand(10000, generator=torch.Generator().manual_seed(3))
    circle_problem = runner.build_problem("g2_bernoulli_bandlimited")
    assert torch.equal(runner.scale_unit_samples_to_space(unit_samples, circle_problem),
                       runner.TWO_PI * unit_samples)
    line_samples = runner.scale_unit_samples_to_space(unit_samples, problem)
    assert float(line_samples.min()) >= -4.8 and float(line_samples.max()) < 4.2
    assert float(line_samples.min()) < -4.79 and float(line_samples.max()) > 4.19


@pytest.mark.parametrize("variant_name", catalogue.variant_names(CELL))
def test_every_variant_meets_the_datum_exactly_at_the_terminal_slice(problem, variant_name):
    variant = catalogue.variant_by_name(CELL, variant_name)
    model, extension_field = runner.build_ansatz(variant, problem, SMALL_NETWORK, model_seed=0)
    first_linear_layer = next(
        module for module in model.network.modules() if isinstance(module, torch.nn.Linear)
    )
    assert first_linear_layer.in_features == 2
    x = torch.linspace(-4.8, 4.2, 1024)
    terminal_slice = torch.full_like(x, problem["terminal_time"])
    with torch.no_grad():
        values = model(torch.stack([x, terminal_slice], dim=1)).squeeze(-1)
    assert torch.equal(values, problem["terminal_datum"](x))
    assert (extension_field is None) == (variant["extension"] is None)


@pytest.mark.parametrize(
    "variant_name",
    [name for name in catalogue.variant_names(CELL)
     if catalogue.variant_by_name(CELL, name)["extension"] is not None],
)
def test_analytic_forcing_passes_the_startup_cross_check(problem, variant_name):
    from learning_option_pricing.models.terminal_ansatz import (
        cross_check_extension_forcing_analytic_versus_autograd,
    )

    variant = catalogue.variant_by_name(CELL, variant_name)
    model, _ = runner.build_ansatz(variant, problem, SMALL_NETWORK, model_seed=0)
    generator = torch.Generator().manual_seed(1)
    x = runner.scale_unit_samples_to_space(torch.rand(4096, generator=generator), problem)
    t = problem["terminal_time"] * torch.rand(4096, generator=generator)
    deviation = cross_check_extension_forcing_analytic_versus_autograd(
        model, x, t, generator_coefficients=problem["generator_coefficients"]
    )
    assert deviation < 1e-3


def test_closed_form_floors(problem):
    def floor(name):
        return runner.closed_form_forcing_floor(catalogue.variant_by_name(CELL, name), problem)

    # Times the window measure, the floor of the heat evolution is its forcing energy
    # E_V = 0.3821 of Figure 1 of the boundary paper.
    assert floor("split_diffusion") * 9.0 == pytest.approx(0.3821, abs=5e-5)
    assert floor("exact_solution") < 1e-25
    assert floor("transported_datum") < 1e-25
    assert math.isnan(floor("graded_chen_mangasarian"))
    assert floor("graded_gaussian_mismatched") > floor("split_diffusion")
    # constant_in_time: the integral over x of (mu g' - rho g)^2 is
    # int_0^1 [(0.6 - 0.3 u)^2 + (0.6 + 0.3 u)^2] du = 0.72 + 0.06, divided by |W| = 9.
    assert floor("constant_in_time") == pytest.approx(0.78 / 9.0, rel=1e-8)


def test_limits_of_the_variants_with_singular_forcing(problem):
    x = runner._evaluation_grid(problem)
    t = np.full_like(x, 0.37)
    transported = catalogue.variant_by_name(CELL, "transported_datum")
    transported_field = runner.build_real_line_extension_field(transported, problem)
    np.testing.assert_array_equal(
        runner.real_line_limit_field_values(transported, problem, transported_field, x, t),
        transported_field.field(x, t),
    )
    for name in ("constant_in_time", "convex_raw"):
        variant = catalogue.variant_by_name(CELL, name)
        terminal = np.full_like(x, problem["terminal_time"])
        np.testing.assert_allclose(
            runner.real_line_limit_field_values(variant, problem, None, x, terminal),
            problem["terminal_datum"](x), atol=1e-15,
        )
        limit = runner.real_line_limit_field_values(variant, problem, None, x, t)
        assert np.linalg.norm(limit - problem["exact_field"].field(x, t)) > 0.0
    split = catalogue.variant_by_name(CELL, "split_diffusion")
    assert runner.real_line_limit_field_values(split, problem, None, x, t) is None


def test_spectra_are_recorded_as_absent(problem):
    variant = catalogue.variant_by_name(CELL, "split_diffusion")
    model, _ = runner.build_ansatz(variant, problem, SMALL_NETWORK, model_seed=0)
    spectra = runner.compute_spectra(model, problem, variant, None)
    assert not bool(spectra["forcing_defined"][0])
    assert int(spectra["k_star"][0]) == -1
    assert runner.build_closed_form_extension(variant, problem) is None


def test_summary_holds_plain_values_that_the_safe_loader_reads(problem, tmp_path):
    import yaml

    floor = runner.closed_form_forcing_floor(catalogue.variant_by_name(CELL, "split_diffusion"), problem)
    assert type(floor) is float
    payload = {"split_diffusion": {"forcing_floor_closed_form": np.float64(0.5),
                                   "n_parameters": np.int64(3), "flag": np.bool_(True),
                                   "values": np.array([1.0, 2.0])}}
    summary_path = tmp_path / "summary_split_diffusion.yaml"
    runner.write_summary(summary_path, payload)
    loaded = yaml.safe_load(summary_path.read_text())
    assert loaded == {"split_diffusion": {"forcing_floor_closed_form": 0.5, "n_parameters": 3,
                                          "flag": True, "values": [1.0, 2.0]}}
