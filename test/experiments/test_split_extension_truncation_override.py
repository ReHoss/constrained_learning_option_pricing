"""Band-edge override of the stage-2 runner (truncation study, pre-registration 2026-10-08)."""

from __future__ import annotations

import importlib
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


def test_default_band_edge_is_the_catalogue_value():
    problem = runner.build_problem("g2_bernoulli_bandlimited")
    assert problem["band_edge"] == 128
    assert problem["cosine_coefficients"].shape == (128,)


@pytest.mark.parametrize("band_edge", [32, 512])
def test_override_changes_the_datum_band_and_the_forcing_band(band_edge):
    problem = runner.build_problem("g3_bernoulli_bandlimited", band_edge)
    assert problem["band_edge"] == band_edge
    assert problem["forcing_band_edge"] == band_edge
    assert problem["cosine_coefficients"].shape == (band_edge,)
    wavenumbers = np.arange(1, band_edge + 1, dtype=np.float64)
    np.testing.assert_allclose(
        problem["cosine_coefficients"], 1.0 / (np.pi**2 * wavenumbers**2), rtol=1e-14
    )


def test_override_is_rejected_for_the_single_component_control_cell():
    with pytest.raises(ValueError, match="band-limited data only"):
        runner.build_problem("heat_sine_single_component", 32)


def test_command_line_override_reaches_the_hyperparameters():
    arguments = runner.build_parser().parse_args(
        ["--cell", "g2_bernoulli_bandlimited", "--truncation-wavenumber", "512"]
    )
    assert runner.resolve_hparams(arguments)["truncation_wavenumber"] == 512
    default_arguments = runner.build_parser().parse_args(["--cell", "g2_bernoulli_bandlimited"])
    assert "truncation_wavenumber" not in runner.resolve_hparams(default_arguments)
