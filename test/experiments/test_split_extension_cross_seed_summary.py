r"""Unit tests for the pure helpers of ``split_extension_cross_seed_summary``.

Covered helpers (against a tiny synthetic artefact tree in a temporary
directory, never against real training output):

* the run-directory discovery regex (``_seed<S>`` suffix optional,
  ``_debug_`` excluded, cell / seed capture);
* run-record loading (per-task summary merge, metadata seed cross-check
  raising on contradiction);
* cross-seed aggregation (medians / interquartile ranges, ``k_star``
  sentinel handling, ``hist.npz`` forcing-floor fallback);
* the closed-form band-edge recomputation through the library extensions
  (exact-solution forcing is zero; the convex floor exceeds the split
  floor; the matched graded extension coincides with the split
  :math:`\{\partial_{xx}\}` extension; strip norm / Monte-Carlo floor
  normalisation);
* table assembly (explicitly empty ``not measured`` slots; Markdown
  rendering with capitalised headers).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIRECTORY = (
    REPOSITORY_ROOT
    / "experiments"
    / "python_scripts"
    / "exp_split_extension_trained"
)
sys.path.insert(0, str(EXPERIMENT_DIRECTORY))

aggregator = importlib.import_module("split_extension_cross_seed_summary")


# ---------------------------------------------------------------------------
# Discovery regex
# ---------------------------------------------------------------------------


def test_directory_pattern_accepts_seed_suffix():
    parsed = aggregator.parse_run_directory_name(
        "2026-07-11-00-00-00-000000Z_g2_bernoulli_bandlimited_iters20000_seed2"
    )
    assert parsed == {"cell": "g2_bernoulli_bandlimited", "seed": 2}


def test_directory_pattern_accepts_missing_seed_suffix():
    parsed = aggregator.parse_run_directory_name(
        "2026-07-11-00-00-00-000000Z_heat_sine_single_component_iters20000"
    )
    assert parsed == {"cell": "heat_sine_single_component", "seed": None}


def test_directory_pattern_rejects_debug_directories():
    assert (
        aggregator.parse_run_directory_name(
            "_debug_2026-07-11-00-00-00-000000Z_g1_bernoulli_bandlimited"
            "_iters300_seed0"
        )
        is None
    )


def test_directory_pattern_rejects_names_without_iteration_count():
    assert (
        aggregator.parse_run_directory_name(
            "2026-07-11-00-00-00-000000Z_g1_bernoulli_bandlimited_seed0"
        )
        is None
    )


def test_directory_pattern_rejects_uppercase_cell_names():
    assert (
        aggregator.parse_run_directory_name(
            "2026-07-11-00-00-00-000000Z_G1_Bernoulli_iters20000_seed0"
        )
        is None
    )


# ---------------------------------------------------------------------------
# Synthetic artefact tree
# ---------------------------------------------------------------------------


def _write_run_directory(
    data_root: Path,
    *,
    cell: str,
    seed: int,
    timestamp: str,
    variant_summaries: dict[str, dict],
    metadata_seed: int | None = None,
    forcing_floor_history: dict[str, np.ndarray] | None = None,
    spectra: dict[str, dict[str, np.ndarray]] | None = None,
) -> Path:
    """Create one synthetic run directory following the runner layout."""
    run_directory = data_root / f"{timestamp}Z_{cell}_iters20000_seed{seed}"
    run_directory.mkdir(parents=True)
    with open(run_directory / "metadata.yaml", "w") as handle:
        yaml.dump(
            {
                "cell": cell,
                "seed": seed if metadata_seed is None else metadata_seed,
            },
            handle,
        )
    for variant_name, metrics in variant_summaries.items():
        with open(run_directory / f"summary_{variant_name}.yaml", "w") as handle:
            yaml.dump({variant_name: metrics}, handle)
    for variant_name, history in (forcing_floor_history or {}).items():
        variant_directory = run_directory / f"variant_{variant_name}"
        variant_directory.mkdir(exist_ok=True)
        np.savez_compressed(variant_directory / "hist.npz", **history)
    for variant_name, arrays in (spectra or {}).items():
        variant_directory = run_directory / f"variant_{variant_name}"
        variant_directory.mkdir(exist_ok=True)
        np.savez_compressed(variant_directory / "spectra.npz", **arrays)
    return run_directory


@pytest.fixture
def synthetic_tree(tmp_path):
    """Two seeds of the G2 cell plus one debug directory and one stray file."""
    data_root = tmp_path / "ablation_split_extension_trained"
    _write_run_directory(
        data_root,
        cell="g2_bernoulli_bandlimited",
        seed=0,
        timestamp="2026-07-11-00-00-00-000000",
        variant_summaries={
            "convex_raw": {
                "best_loss": 4.0,
                "rel_l2": 0.20,
                "tc_l2": 0.0,
                "k_star": 16.0,
            },
            "split_diffusion": {
                "best_loss": 1.0e-4,
                "rel_l2": 0.002,
                "tc_l2": 0.0,
                # Sentinel: a non-positive stored k_star encodes "absent"
                # and must never enter the aggregated sample.
                "k_star": -1.0,
            },
        },
        forcing_floor_history={
            "convex_raw": {"forcing_floor": np.asarray([3.0, 5.0, 4.0])}
        },
        spectra={
            "split_diffusion": {
                "wavenumbers": np.arange(0.0, 33.0),
                "running_mean": np.linspace(0.0, 1.0, 33),
                "in_band_mask": np.ones(33, dtype=bool),
                "k_star": np.asarray([8.0]),
            }
        },
    )
    _write_run_directory(
        data_root,
        cell="g2_bernoulli_bandlimited",
        seed=1,
        timestamp="2026-07-11-01-00-00-000000",
        variant_summaries={
            "convex_raw": {
                "best_loss": 6.0,
                "rel_l2": 0.30,
                "tc_l2": 0.0,
                "k_star": 32.0,
            },
            "split_diffusion": {
                "best_loss": 3.0e-4,
                "rel_l2": 0.004,
                "tc_l2": 0.0,
            },
        },
    )
    # A debug run must be excluded from discovery.
    debug_directory = (
        data_root
        / "_debug_2026-07-11-02-00-00-000000Z_g2_bernoulli_bandlimited"
        "_iters300_seed0"
    )
    debug_directory.mkdir(parents=True)
    with open(debug_directory / "summary_convex_raw.yaml", "w") as handle:
        yaml.dump({"convex_raw": {"best_loss": 999.0}}, handle)
    # A stray file whose name matches the glob must be ignored (not a dir).
    (data_root / "2026-07-11-03-00-00-000000Z_g2_bernoulli_bandlimited"
     "_iters20000_seed9").write_text("")
    return data_root


def test_discovery_excludes_debug_and_non_directories(synthetic_tree):
    discovered = aggregator.discover_run_directories(synthetic_tree)
    assert len(discovered) == 2
    assert all("_debug_" not in d.name for d in discovered)


def test_load_run_record_merges_per_task_summaries(synthetic_tree):
    discovered = aggregator.discover_run_directories(synthetic_tree)
    record = aggregator.load_run_record(discovered[0])
    assert record.cell == "g2_bernoulli_bandlimited"
    assert record.seed == 0
    assert set(record.variant_summaries) == {"convex_raw", "split_diffusion"}
    assert record.variant_summaries["convex_raw"]["best_loss"] == 4.0


def test_load_run_record_raises_on_seed_contradiction(tmp_path):
    data_root = tmp_path / "runs"
    run_directory = _write_run_directory(
        data_root,
        cell="g2_bernoulli_bandlimited",
        seed=0,
        timestamp="2026-07-11-00-00-00-000000",
        variant_summaries={"convex_raw": {"best_loss": 1.0}},
        metadata_seed=3,  # contradicts the _seed0 folder suffix
    )
    with pytest.raises(ValueError, match="seed"):
        aggregator.load_run_record(run_directory)


def test_load_run_record_raises_on_cell_contradiction(tmp_path):
    data_root = tmp_path / "runs"
    run_directory = data_root / (
        "2026-07-11-00-00-00-000000Z_g1_bernoulli_bandlimited_iters20000_seed0"
    )
    run_directory.mkdir(parents=True)
    with open(run_directory / "metadata.yaml", "w") as handle:
        yaml.dump({"cell": "g2_bernoulli_bandlimited", "seed": 0}, handle)
    with pytest.raises(ValueError, match="cell"):
        aggregator.load_run_record(run_directory)


# ---------------------------------------------------------------------------
# Aggregation statistics
# ---------------------------------------------------------------------------


def test_median_and_interquartile_range_empty_sample():
    stats = aggregator.median_and_interquartile_range([])
    assert stats == {"median": None, "q25": None, "q75": None, "n": 0}


def test_median_and_interquartile_range_known_values():
    stats = aggregator.median_and_interquartile_range([1.0, 3.0, 2.0, 4.0])
    assert stats["median"] == pytest.approx(2.5)
    assert stats["q25"] == pytest.approx(1.75)
    assert stats["q75"] == pytest.approx(3.25)
    assert stats["n"] == 4


def test_collect_statistics_across_seeds(synthetic_tree):
    records = [
        aggregator.load_run_record(d)
        for d in aggregator.discover_run_directories(synthetic_tree)
    ]
    statistics = aggregator.collect_statistics(records)
    cell_stats = statistics["g2_bernoulli_bandlimited"]
    assert cell_stats["convex_raw"]["metrics"]["best_loss"] == [4.0, 6.0]
    assert cell_stats["convex_raw"]["metrics"]["rel_l2"] == [0.20, 0.30]
    # hist.npz fallback: the seed-0 run has no forcing_floor_median in its
    # summary, so the training median 4.0 is read from hist.npz.
    assert cell_stats["convex_raw"]["metrics"]["forcing_floor_median"] == [4.0]
    # k_star: the convex entries are positive (16, 32); the split seed-0
    # sentinel -1 is replaced by the spectra.npz value 8; the split seed-1
    # run stores neither.
    assert cell_stats["convex_raw"]["metrics"]["k_star"] == [16.0, 32.0]
    assert cell_stats["split_diffusion"]["metrics"]["k_star"] == [8.0]

    summarised = aggregator.summarise_statistics(statistics)
    convex = summarised["g2_bernoulli_bandlimited"]["convex_raw"]
    assert convex["seeds"] == [0, 1]
    assert convex["metrics"]["best_loss"]["median"] == pytest.approx(5.0)
    assert convex["metrics"]["best_loss"]["n"] == 2


def test_collect_statistics_runner_native_schema(tmp_path):
    """The runner's actual key spellings are canonicalised at aggregation.

    Covers: ``forcing_floor_median_train`` (alias of
    ``forcing_floor_median``), the ``terminal_target_distance`` /
    ``terminal_target_is_zero_target`` split into the relative and absolute
    observables, ``k_star: null`` encoding an absent cutoff, and the
    runner's ``spectra.npz`` key names (``wavenumber_bins``,
    ``cancellation_ratio_running_mean``).
    """
    data_root = tmp_path / "runs"
    run_directory = _write_run_directory(
        data_root,
        cell="g2_bernoulli_bandlimited",
        seed=0,
        timestamp="2026-07-11-05-00-00-000000",
        variant_summaries={
            "split_diffusion": {
                "best_loss": 2.0e-4,
                "forcing_floor_median_train": 5.5e-5,
                "terminal_target_distance": 0.04,
                "terminal_target_is_zero_target": False,
                "k_star": 24,
                "k_star_defined": True,
            },
            "exact_solution": {
                "best_loss": 3.0e-9,
                "forcing_floor_median_train": 0.0,
                "terminal_target_distance": 1.5e-4,
                "terminal_target_is_zero_target": True,
                "k_star": None,
                "k_star_defined": False,
            },
        },
        spectra={
            "split_diffusion": {
                "wavenumber_bins": np.arange(0.0, 65.0),
                "cancellation_ratio_running_mean": np.linspace(0.0, 1.0, 65),
                "in_band_mask": np.ones(65, dtype=bool),
                "k_star": np.asarray([24]),
                "k_star_defined": np.asarray([True]),
            },
            "exact_solution": {
                "wavenumber_bins": np.arange(0.0, 65.0),
                "residual_power": np.ones(65),
                "k_star": np.asarray([-1]),
                "k_star_defined": np.asarray([False]),
            },
        },
    )
    record = aggregator.load_run_record(run_directory)
    statistics = aggregator.collect_statistics([record])
    split_metrics = statistics["g2_bernoulli_bandlimited"]["split_diffusion"][
        "metrics"
    ]
    exact_metrics = statistics["g2_bernoulli_bandlimited"]["exact_solution"][
        "metrics"
    ]
    # Alias canonicalised, no duplicate under the runner spelling.
    assert split_metrics["forcing_floor_median"] == [5.5e-5]
    assert "forcing_floor_median_train" not in split_metrics
    # Terminal-target split into distinct observables.
    assert split_metrics["terminal_target_rel_l2"] == [0.04]
    assert "terminal_target_abs_l2" not in split_metrics
    assert exact_metrics["terminal_target_abs_l2"] == [1.5e-4]
    assert "terminal_target_rel_l2" not in exact_metrics
    # k_star: measured for the split variant, absent for the zero-forcing one.
    assert split_metrics["k_star"] == [24.0]
    assert "k_star" not in exact_metrics
    # The flags never enter the metric samples.
    assert "k_star_defined" not in split_metrics
    assert "terminal_target_is_zero_target" not in split_metrics
    # Runner spectra key names resolve through the alias map.
    spectra = aggregator.load_spectra_arrays(run_directory, "split_diffusion")
    assert "wavenumbers" in spectra and "running_mean" in spectra


# ---------------------------------------------------------------------------
# Closed-form band-edge recomputation (library calls)
# ---------------------------------------------------------------------------


def test_closed_form_exact_solution_forcing_is_zero():
    quantities = aggregator.closed_form_band_edge_quantities(
        "g2_bernoulli_bandlimited", "exact_solution", 128
    )
    assert quantities["squared_strip_forcing"] == 0.0
    assert quantities["monte_carlo_floor_expectation"] == 0.0


def test_closed_form_control_cell_matched_factor_is_zero():
    quantities = aggregator.closed_form_band_edge_quantities(
        "heat_sine_single_component", "matched_exponential_factor", 128
    )
    assert quantities["band_edge"] == 1
    assert quantities["squared_strip_forcing"] == 0.0


def test_closed_form_convex_floor_exceeds_split_floor_on_g2():
    convex = aggregator.closed_form_band_edge_quantities(
        "g2_bernoulli_bandlimited", "convex_raw", 128
    )
    split = aggregator.closed_form_band_edge_quantities(
        "g2_bernoulli_bandlimited", "split_diffusion", 128
    )
    assert convex["squared_strip_forcing"] > split["squared_strip_forcing"]
    # Stage-1 finding at the full band: the ratio is of the order 5e5; at
    # the truncated band edge the flat form is smaller by construction, so
    # only a conservative ordering margin is asserted here (the exact
    # band-edge values are what the table reports).
    assert (
        convex["squared_strip_forcing"] / split["squared_strip_forcing"] > 1.0e2
    )


def test_closed_form_matched_graded_coincides_with_split_diffusion():
    graded = aggregator.closed_form_band_edge_quantities(
        "g1_bernoulli_bandlimited", "graded_gaussian_matched", 128
    )
    split = aggregator.closed_form_band_edge_quantities(
        "g1_bernoulli_bandlimited", "split_diffusion", 128
    )
    assert graded["squared_strip_forcing"] == pytest.approx(
        split["squared_strip_forcing"], rel=1.0e-12
    )


def test_closed_form_monte_carlo_normalisation():
    quantities = aggregator.closed_form_band_edge_quantities(
        "g2_bernoulli_bandlimited", "constant_in_time", 128
    )
    assert quantities["monte_carlo_floor_expectation"] == pytest.approx(
        quantities["squared_strip_forcing"]
        / (2.0 * np.pi * aggregator.TERMINAL_TIME),
        rel=1.0e-14,
    )


def test_closed_form_unknown_variant_yields_none():
    assert (
        aggregator.closed_form_band_edge_quantities(
            "g2_bernoulli_bandlimited", "an_unknown_variant", 128
        )
        is None
    )


def test_forcing_mass_above_cutoff_decreases_and_vanishes_at_band_edge():
    extension = aggregator.build_terminal_data_extension(
        "g2_bernoulli_bandlimited", "split_diffusion"
    )
    full_mass = aggregator.forcing_mass_above_cutoff(extension, 128, 0.0)
    partial_mass = aggregator.forcing_mass_above_cutoff(extension, 128, 16.0)
    assert 0.0 < partial_mass < full_mass
    assert aggregator.forcing_mass_above_cutoff(extension, 128, 128.0) == 0.0
    # Cutoff 0 reproduces the Monte-Carlo floor expectation.
    quantities = aggregator.closed_form_band_edge_quantities(
        "g2_bernoulli_bandlimited", "split_diffusion", 128
    )
    assert full_mass == pytest.approx(
        quantities["monte_carlo_floor_expectation"], rel=1.0e-12
    )


def test_single_component_sine_datum_coefficients():
    datum = aggregator.SingleComponentSineDatum(wavenumber=1)
    coefficients = datum.fourier_coefficients(np.asarray([-2, -1, 0, 1, 2]))
    assert coefficients[2] == 0.0
    assert coefficients[3] == pytest.approx(-0.5j)
    assert coefficients[1] == pytest.approx(0.5j)
    # Synthesis check: sum c_k e^{ikx} equals sin(x).
    x = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    band = np.asarray([-2, -1, 1, 2])
    values = np.real(
        (datum.fourier_coefficients(band)[None, :]
         * np.exp(1j * band[None, :] * x[:, None])).sum(axis=1)
    )
    np.testing.assert_allclose(values, np.sin(x), atol=1.0e-14)


# ---------------------------------------------------------------------------
# Table assembly
# ---------------------------------------------------------------------------


def test_render_markdown_table_rejects_lowercase_headers():
    with pytest.raises(ValueError, match="capital"):
        aggregator.render_markdown_table(["variant"], [["a"]])


def test_render_markdown_table_rejects_ragged_rows():
    with pytest.raises(ValueError, match="row length"):
        aggregator.render_markdown_table(["A", "B"], [["only one"]])


def test_stage1_rows_have_explicit_empty_slots_without_runs():
    closed_forms = aggregator.compute_closed_forms({}, 128)
    rows = aggregator.assemble_stage1_comparison_rows(
        "g2_bernoulli_bandlimited", {}, closed_forms
    )
    assert len(rows) == len(aggregator.GENERATOR_CELL_VARIANT_NAMES)
    for row in rows:
        # Closed-form columns are filled; every measured column is the
        # explicit marker.
        assert row[1] != aggregator.NOT_MEASURED
        assert row[3] == aggregator.NOT_MEASURED
        assert row[4] == aggregator.NOT_MEASURED
        assert row[5] == aggregator.NOT_MEASURED


def test_additive_versus_convex_conclusion_empty_without_measurements():
    closed_forms = aggregator.compute_closed_forms({}, 128)
    rows, conclusion = aggregator.assemble_additive_versus_convex_rows(
        "g2_bernoulli_bandlimited", {}, closed_forms, {}
    )
    assert conclusion == aggregator.NOT_MEASURED
    # The closed-form floor row is filled for all four variants ...
    assert all(cell != aggregator.NOT_MEASURED for cell in rows[0][1:])
    # ... every measured row is explicitly empty ...
    for measured_row in rows[1:5]:
        assert all(
            cell == aggregator.NOT_MEASURED for cell in measured_row[1:]
        )
    # ... the closed-form ratio line is computed, the measured one is not.
    assert rows[5][1] != aggregator.NOT_MEASURED
    assert rows[6][1] == aggregator.NOT_MEASURED


def test_additive_versus_convex_conclusion_with_measurements(synthetic_tree):
    records = [
        aggregator.load_run_record(d)
        for d in aggregator.discover_run_directories(synthetic_tree)
    ]
    statistics = aggregator.collect_statistics(records)
    summarised = aggregator.summarise_statistics(statistics)
    closed_forms = aggregator.compute_closed_forms(statistics, 128)
    unreachable = aggregator.compute_unreachable_masses(statistics, 128)
    rows, conclusion = aggregator.assemble_additive_versus_convex_rows(
        "g2_bernoulli_bandlimited", summarised, closed_forms, unreachable
    )
    # Measured best-loss ratio = median(4, 6) / median(1e-4, 3e-4) = 25000.
    assert rows[6][1] == "{:.6e}".format(5.0 / 2.0e-4)
    assert conclusion != aggregator.NOT_MEASURED
    # The unreachable-mass slot of split_diffusion is filled (spectra k_star
    # = 8 was measured for seed 0), the constant_in_time one stays empty.
    header_index = aggregator.ADDITIVE_VERSUS_CONVEX_VARIANTS.index(
        "split_diffusion"
    )
    assert rows[4][1 + header_index] != aggregator.NOT_MEASURED
    constant_index = aggregator.ADDITIVE_VERSUS_CONVEX_VARIANTS.index(
        "constant_in_time"
    )
    assert rows[4][1 + constant_index] == aggregator.NOT_MEASURED


# ---------------------------------------------------------------------------
# End-to-end smoke test on the synthetic tree (no training recomputation)
# ---------------------------------------------------------------------------


def test_main_writes_tables_yaml_and_figures(synthetic_tree, tmp_path):
    out_dir = tmp_path / "aggregate"
    exit_code = aggregator.main(
        [
            "--data-root",
            str(synthetic_tree),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "summary_across_seeds.yaml").exists()
    assert (out_dir / "stage1_comparison_table.md").exists()
    assert (out_dir / "additive_versus_convex_conclusion.md").exists()
    assert (out_dir / "rel_l2_by_cell.png").exists()
    assert (out_dir / "floor_vs_accuracy.png").exists()
    assert (out_dir / "cutoff_by_variant.png").exists()
    assert (out_dir / "residual_spectra_by_cell.png").exists()
    # No terminal-target measurement exists in the synthetic tree: the H3
    # figure must be absent (an explicitly empty slot, not a filler figure).
    assert not (out_dir / "terminal_target.png").exists()

    with open(out_dir / "summary_across_seeds.yaml") as handle:
        payload = yaml.safe_load(handle)
    variants = payload["cells"]["g2_bernoulli_bandlimited"]["variants"]
    assert variants["convex_raw"]["measured"]["best_loss"]["median"] == 5.0
    assert variants["convex_raw"]["seeds"] == [0, 1]
    # Unmeasured variants of the specification keep explicitly empty slots.
    assert variants["exact_solution"]["measured"] is None
    assert variants["exact_solution"]["closed_form_at_band_edge"][
        "squared_strip_forcing"
    ] == 0.0
    conclusion_text = (out_dir / "additive_versus_convex_conclusion.md").read_text()
    assert aggregator.NOT_MEASURED in conclusion_text


def test_main_without_any_run_still_writes_tables(tmp_path):
    empty_root = tmp_path / "no_runs"
    empty_root.mkdir()
    out_dir = tmp_path / "aggregate_empty"
    exit_code = aggregator.main(
        ["--data-root", str(empty_root), "--out-dir", str(out_dir)]
    )
    assert exit_code == 0
    assert (out_dir / "summary_across_seeds.yaml").exists()
    assert (out_dir / "stage1_comparison_table.md").exists()
    assert (out_dir / "additive_versus_convex_conclusion.md").exists()
    # Every figure needs at least one measured run: none may exist here.
    assert not list(out_dir.glob("*.png"))
    table_text = (out_dir / "stage1_comparison_table.md").read_text()
    assert aggregator.NOT_MEASURED in table_text
