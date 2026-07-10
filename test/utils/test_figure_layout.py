"""Tests for the shared figure-layout helpers.

Covered here: ``finalize_figure`` writes a PNG file (Agg backend) for a dummy
figure with an external legend and a formula box; ``formula_box`` returns the
text artist for a non-empty formula and ``None`` for an empty one; the two
experiment-side ``_figure_layout`` compatibility shims re-export the library
implementation (single source of truth).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from learning_option_pricing.utils.figure_layout import (  # noqa: E402
    finalize_figure,
    formula_box,
)


def _build_dummy_figure():
    """A small figure with one solid curve and an external bottom legend."""
    figure, axes = plt.subplots(figsize=(4.5, 3.5))
    axes.plot([0.0, 0.5, 1.0], [0.0, 0.25, 1.0], "-", label="Measured curve")
    axes.set_xlabel("Abscissa $x$")
    axes.set_ylabel("Ordinate $y$")
    legend = axes.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=True
    )
    figure.tight_layout(rect=[0.0, 0.28, 1.0, 1.0])
    return figure, axes, legend


def test_finalize_figure_writes_png_with_formula_box(tmp_path):
    figure, axes, legend = _build_dummy_figure()
    figure_path = tmp_path / "figure_layout_smoke.png"
    finalize_figure(
        figure,
        figure_path,
        legends=[legend],
        axes=[axes],
        formula=r"$y = x^{2}$ (quadratic reference)",
    )
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0


def test_formula_box_returns_artist_or_none():
    figure = plt.figure()
    try:
        assert formula_box(figure, "") is None
        artist = formula_box(figure, r"$y = x$")
        assert artist is not None
        assert artist.get_text() == r"$y = x$"
    finally:
        plt.close(figure)


def test_experiment_shims_reexport_the_library_implementation():
    repository_root = Path(__file__).resolve().parents[2]
    shim_paths = [
        repository_root
        / "experiments/python_scripts/exp_extension_split_generator/_figure_layout.py",
        repository_root
        / "experiments/python_scripts/exp_ansatz_forms_heat/_figure_layout.py",
    ]
    for shim_index, shim_path in enumerate(shim_paths):
        module_name = f"_figure_layout_shim_under_test_{shim_index}"
        specification = importlib.util.spec_from_file_location(
            module_name, shim_path
        )
        shim_module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = shim_module
        specification.loader.exec_module(shim_module)
        try:
            assert shim_module.finalize_figure is finalize_figure
            assert shim_module.formula_box is formula_box
        finally:
            del sys.modules[module_name]
