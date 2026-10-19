"""Tests of the two diagnostics of the Chen--Mangasarian study (pre-registration of 2026-10-19).

Covered here: the band sum and the verdicts S1-S3 of ``early_time_residual_spectra``
on synthetic records, and the verdicts F1-F4 of
``measure_chen_mangasarian_forcing_energy_growth`` on synthetic energy curves
that follow, or violate, the pre-registered laws.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINED_DIRECTORY = REPOSITORY_ROOT / "experiments" / "python_scripts" / "exp_split_extension_trained"
GENERATOR_DIRECTORY = REPOSITORY_ROOT / "experiments" / "python_scripts" / "exp_extension_split_generator"
sys.path.insert(0, str(TRAINED_DIRECTORY))
sys.path.insert(0, str(GENERATOR_DIRECTORY))

early = importlib.import_module("early_time_residual_spectra")
growth = importlib.import_module("measure_chen_mangasarian_forcing_energy_growth")
runner = importlib.import_module("ablation_split_extension_trained")


def test_band_power_sums_the_closed_band():
    bins = np.arange(0, 11, dtype=float)
    power = np.ones_like(bins)
    assert early.band_power(bins, power, (2, 5)) == 4.0


def _record(cell, role, tau, residual, forcing):
    return {"cell": cell, "role": role, "paper_time_fraction": tau,
            "high_band_residual_power": residual, "high_band_forcing_power": forcing}


def test_early_time_verdicts_on_synthetic_records():
    records = []
    for cell in early.CELLS:
        for seed_offset in (0.9, 1.0, 1.1):
            records += [
                _record(cell, "chen_mangasarian", 0.001, 100.0 * seed_offset, 50.0),
                _record(cell, "highest_order_split", 0.001, 1.0 * seed_offset, 0.1),
                _record(cell, "constant_in_time", 0.001, 500.0 * seed_offset, 52.0),
                _record(cell, "chen_mangasarian", 0.1, 2.0 * seed_offset, 1.0),
                _record(cell, "highest_order_split", 0.1, 1.0 * seed_offset, 0.1),
            ]
    verdicts = early.verdicts(records)
    for cell in early.CELLS:
        assert verdicts[cell]["residual_ratio_by_paper_time"]["0.001"] == pytest.approx(100.0)
        assert verdicts[cell]["s1_relative_deviation_of_forcing"] == pytest.approx(2.0 / 52.0)
        assert verdicts[cell]["s1_holds"] and verdicts[cell]["s2_holds"] and verdicts[cell]["s3_holds"]


def _synthetic_curves(problem, band_edges, follow_law):
    prediction = growth.predicted_chen_mangasarian_constant(problem)
    edges = band_edges.astype(float)
    if prediction["half_order"] == 1:
        chen = 1.0 + prediction["sum_coefficient"] * np.log(edges) * (1.0 if follow_law else 1.5)
    else:
        chen = prediction["sum_coefficient"] / 4.0 * edges**4 * (1.0 if follow_law else 1.5)
    split = 1.0 - 1.0 / edges**3
    constant = edges ** (4 * prediction["half_order"] - 3)
    return {growth.CHEN_MANGASARIAN: chen, "split": split, growth.CONSTANT_IN_TIME: constant}


@pytest.mark.parametrize("follow_law", [True, False])
def test_growth_verdicts_on_synthetic_curves(follow_law):
    band_edges = 2 ** np.arange(4, 21, dtype=np.int64)
    for cell_name in growth.CELLS:
        problem = runner.build_problem(cell_name)
        summary = growth.analyse_cell(cell_name, problem, band_edges,
                                      _synthetic_curves(problem, band_edges, follow_law))
        chen_key = "f1" if summary["prediction"]["half_order"] == 1 else "f2"
        assert summary[chen_key]["holds"] is follow_law
        assert summary["f3"]["holds"] and summary["f4"]["holds"]


def test_predicted_constants_match_the_pre_registration():
    order_two = growth.predicted_chen_mangasarian_constant(runner.build_problem("g2_bernoulli_full"))
    order_four = growth.predicted_chen_mangasarian_constant(runner.build_problem("g3_bernoulli_full"))
    assert math.log(2.0) * order_two["sum_coefficient"] == pytest.approx(6.464e-4, rel=1e-3)
    assert order_four["sum_coefficient"] / 4.0 == pytest.approx(2.789e-5, rel=1e-3)
