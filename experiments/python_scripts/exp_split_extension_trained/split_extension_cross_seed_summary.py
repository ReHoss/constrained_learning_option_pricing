r"""Cross-seed aggregation for the trained split-extension ablation (stage 2).

Scans the per-(cell, seed) run directories produced by
``ablation_split_extension_trained.py`` (specification:
``documents/methodology/stage2_trained_ablation_specification.md``, Sections 4
and 6.1), groups the saved run summaries by cell and method variant, and
produces the cross-seed comparison artefacts.  Everything is computed from
saved run artefacts and library closed forms — **no training computation is
ever repeated here**.

Outputs (written to a timestamped folder under
``data/split_extension_cross_seed_summary/``):

* ``summary_across_seeds.yaml`` — per-(cell, variant) medians and
  interquartile ranges over seeds for every recorded scalar metric, beside
  the closed-form band-edge quantities; unmeasured slots are ``null``,
  never a plausible-looking filler;
* ``stage1_comparison_table.md`` — the comparison table against the stage-1
  anchor quantities **recomputed at the datum's band edge**
  :math:`K_g = 128` through the library calls
  (``squared_forcing_time_integral`` / ``total_strip_forcing_squared``); the
  stage-1 :math:`2^{20}` anchor values are never quoted (specification,
  "Band-edge caution");
* ``additive_versus_convex_conclusion.md`` — the additive-versus-convex
  conclusion table, with explicitly empty slots (marker "not measured") for
  every quantity that no saved run supplies yet (decision D9: no placeholder
  value is written anywhere);
* ``rel_l2_by_cell.png`` — relative :math:`L^2` error per cell and variant,
  median with interquartile range over seeds;
* ``floor_vs_accuracy.png`` — hypothesis H1: best training loss against the
  closed-form floor :math:`\mathbb{E}[(P\Psi)^2] = \frac{1}{T}
  \sum_{0<|k|\le K_g} I_k` and against the unreachable forcing mass
  :math:`\mathcal{F}(k_\star) = \frac{1}{T} \sum_{|k| > k_\star,\,|k| \le
  K_g} I_k` evaluated at the *measured* cutoff :math:`k_\star`;
* ``cutoff_by_variant.png`` — hypothesis H2: the measured reachable cutoff
  :math:`k_\star` per variant and cell (invariance across variants);
* ``terminal_target.png`` — hypothesis H3: distance of the trained bare
  network's terminal profile :math:`\Phi_\theta(\cdot, T)` to the exact
  minimiser's profile :math:`\Phi^\star(\cdot, T) = T\,(P\Psi)(\cdot, T)`;
* ``residual_spectra_by_cell.png`` — the residual frequency decomposition of
  specification Section 3.4, aggregated across seeds: per-spectral-component
  cancellation ratio :math:`|\hat r_k|^2 / |\widehat{Lh}(k)|^2` (7-point
  running mean) with the measured cutoff :math:`k_\star` marked.

Expected run-summary schema (the contract shared with the runner; the runner
is written against the same specification).  Each run directory
``data/ablation_split_extension_trained/<timestamp>Z_<cell>_iters<N>_seed<S>/``
holds:

* ``metadata.yaml`` with at least the keys ``cell`` and ``seed``;
* per-task ``summary_<variant>.yaml`` (and/or a combined ``summary.yaml``)
  mapping the variant name to a dictionary of scalar metrics; the metric
  keys consumed here are ``best_loss``, ``best_iter``, ``rel_l2``,
  ``rel_l2_t0``, ``rel_l2_corner_t0``, ``rel_l2_corner_max``, ``tc_l2``,
  ``forcing_floor_median`` (alias ``forcing_floor_median_train``),
  ``forcing_floor_closed_form``, ``wall_time_s``, ``k_star`` (a ``null`` or
  non-positive stored value encodes "cutoff absent"), and the
  terminal-target pair ``terminal_target_distance`` /
  ``terminal_target_is_zero_target``, which is split at aggregation time
  into ``terminal_target_rel_l2`` (non-zero-target variants) and
  ``terminal_target_abs_l2`` (zero-target variants) — the direct spellings
  ``terminal_target_rel_l2`` / ``terminal_target_abs_l2`` are accepted as
  well; every additional scalar key is aggregated too and appears in the
  YAML;
* per-variant ``variant_<name>/hist.npz`` (history channel
  ``forcing_floor``, used as fallback for the training-median floor) and
  ``variant_<name>/spectra.npz`` (arrays of Section 3.4; key aliases are
  accepted, see ``SPECTRA_KEY_ALIASES`` — in particular the runner's
  ``wavenumber_bins`` and ``cancellation_ratio_running_mean``).

Discovery follows specification Section 4 exactly: directory names matching
``Z_(?P<cell>[a-z0-9_]+)_iters\d+(?:_seed\d+)?$`` (the ``_seed<S>`` suffix is
optional so pre-convention runs still match), ``_debug_`` directories
excluded.  A seed suffix that contradicts the ``seed`` field of
``metadata.yaml`` raises ``ValueError`` — validations raise, they never pass
silently.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Make the sibling catalogue importable whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_option_pricing.pde.periodic_spectral_toolbox import (  # noqa: E402
    ConstantCoefficientGenerator,
    PeriodisedBernoulliDatum,
    advection_diffusion_reaction,
    black_scholes_log_price,
    symmetric_wavenumber_band,
)
from learning_option_pricing.pde.terminal_data_extensions import (  # noqa: E402
    ConstantInTimeExtension,
    ConvexRawExtension,
    ExactSolutionExtension,
    GradedGaussianExtension,
    SplitSemigroupExtension,
    TerminalDataExtension,
    total_strip_forcing_squared,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

# The torch-free variant catalogue is authored in parallel with the runner
# (specification Section 1.4 item 1).  When present it supplies the display
# colour and label of each variant so the cross-seed figures match the
# per-run figures; when absent, the specification-pinned fallback table
# below is used and a notice is printed (never a silent divergence).
try:
    import _split_extension_catalogue as _catalogue  # noqa: E402
except ImportError:  # the catalogue is being written in parallel
    _catalogue = None


TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Specification-pinned constants (stage-2 specification, Sections 0, 1, 4)
# ---------------------------------------------------------------------------

RUNNER_SCRIPT_STEM = "ablation_split_extension_trained"

TERMINAL_TIME = 1.0

# Band edge of the truncated Bernoulli datum of the generator cells
# (specification Section 0: K_g = 128).
GENERATOR_CELL_BAND_EDGE = 128

# The control cell's datum sin(x) has the single retained wavenumber k_0 = 1,
# so its band edge for every closed-form sum is 1.
CONTROL_CELL_BAND_EDGE = 1

CONTROL_CELL_NAME = "heat_sine_single_component"

GENERATOR_CELL_NAMES = ("g1_bernoulli_bandlimited", "g2_bernoulli_bandlimited")

CELL_NAMES = (*GENERATOR_CELL_NAMES, CONTROL_CELL_NAME)

# Variant sets per cell (specification Sections 1.2 and 1.3), in display order.
GENERATOR_CELL_VARIANT_NAMES = (
    "convex_raw",
    "constant_in_time",
    "split_diffusion",
    "split_diffusion_advection",
    "graded_gaussian_matched",
    "graded_gaussian_mismatched",
    "exact_solution",
)
CONTROL_CELL_VARIANT_NAMES = ("matched_exponential_factor", "convex_raw")

# Variants whose extension forcing vanishes identically (specification V7 and
# C1): the cancellation ratio of Section 3.4 is undefined for them and the
# cutoff k_star is recorded as absent.
ZERO_FORCING_VARIANTS = {
    ("g1_bernoulli_bandlimited", "exact_solution"),
    ("g2_bernoulli_bandlimited", "exact_solution"),
    (CONTROL_CELL_NAME, "matched_exponential_factor"),
}

# Fallback display properties (used only when the parallel catalogue is not
# importable); colour choices mirror the matplotlib tab10 palette.
FALLBACK_VARIANT_DISPLAY = {
    "convex_raw": {"color": "#7f7f7f", "label": "Convex raw"},
    "constant_in_time": {"color": "#8c564b", "label": "Constant-in-time"},
    "split_diffusion": {
        "color": "#1f77b4",
        "label": r"Split $\{\partial_{xx}\}$",
    },
    "split_diffusion_advection": {
        "color": "#2ca02c",
        "label": r"Split $\{\partial_{xx},\partial_x\}$",
    },
    "graded_gaussian_matched": {
        "color": "#17becf",
        "label": "Graded Gaussian (matched)",
    },
    "graded_gaussian_mismatched": {
        "color": "#ff7f0e",
        "label": "Graded Gaussian (mismatched)",
    },
    "exact_solution": {"color": "#9467bd", "label": "Exact solution"},
    "matched_exponential_factor": {
        "color": "#e377c2",
        "label": "Matched exponential factor",
    },
}

CELL_MARKERS = {
    "g1_bernoulli_bandlimited": "o",
    "g2_bernoulli_bandlimited": "s",
    CONTROL_CELL_NAME: "^",
}

# Metric keys consumed by the figures below (every additional scalar key
# found in a summary is aggregated into the YAML as well).
SUMMARY_METRIC_KEYS = (
    "best_loss",
    "best_iter",
    "network_energy_best_state",
    "forcing_floor_best_state",
    "rel_l2",
    "rel_l2_t0",
    "rel_l2_corner_t0",
    "rel_l2_corner_max",
    "tc_l2",
    "forcing_floor_median",
    "forcing_floor_closed_form",
    "terminal_target_rel_l2",
    "terminal_target_abs_l2",
    "k_star",
    "wall_time_s",
)

# Accepted spellings of summary keys (canonical name -> accepted names, in
# preference order).  The runner records the training-median floor as
# ``forcing_floor_median_train``; both spellings are canonicalised to
# ``forcing_floor_median`` at aggregation time.
SUMMARY_KEY_ALIASES = {
    "forcing_floor_median": ("forcing_floor_median", "forcing_floor_median_train"),
}

# Summary keys with dedicated handling in :func:`collect_statistics` — they
# must not enter the generic scalar-metric loop:
# * ``k_star`` — a stored ``None`` / non-positive sentinel encodes "absent";
# * ``k_star_defined`` — a flag, not a metric;
# * ``terminal_target_distance`` — split into ``terminal_target_rel_l2`` or
#   ``terminal_target_abs_l2`` according to ``terminal_target_is_zero_target``
#   (the two quantities are distinct observables and must not be pooled);
# * ``terminal_target_is_zero_target`` — the flag steering that split.
SPECIALLY_HANDLED_SUMMARY_KEYS = (
    "k_star",
    "k_star_defined",
    "terminal_target_distance",
    "terminal_target_is_zero_target",
)

# Accepted key aliases inside spectra.npz (the canonical names first; the
# long names are those written by the runner's ``compute_spectra``; the
# short names are those of the reused recipe in spectral_bias_periodic_nn.py).
SPECTRA_KEY_ALIASES = {
    "wavenumbers": ("wavenumbers", "wavenumber_bins", "k"),
    "forcing_power": ("forcing_power", "forcing_power_exact", "fpow"),
    "residual_power": ("residual_power", "residual_power_mean", "rpow"),
    "cancellation_ratio": ("cancellation_ratio", "ratio"),
    "running_mean": (
        "running_mean",
        "cancellation_ratio_running_mean",
        "ratio_running_mean",
        "rsm",
    ),
    "in_band_mask": ("in_band_mask", "band", "mask"),
    "k_star": ("k_star", "kstar"),
}

# Explicit marker for a quantity no saved run supplies yet (decision D9:
# an empty slot is honest; a plausible filler is not).
NOT_MEASURED = "not measured"

# Specification Section 4: the run-directory discovery pattern, with the
# optional ``_seed<S>`` suffix (named group added for seed extraction; the
# accepted language is identical to the specification's pattern).
RUN_DIRECTORY_PATTERN = re.compile(
    r"Z_(?P<cell>[a-z0-9_]+)_iters\d+(?:_seed(?P<seed>\d+))?$"
)


# ---------------------------------------------------------------------------
# Discovery (pure helpers, unit-tested)
# ---------------------------------------------------------------------------


def parse_run_directory_name(directory_basename: str) -> dict | None:
    """Parse a run-directory basename into its cell and (optional) seed.

    Args:
        directory_basename: The basename of a candidate run directory, e.g.
            ``2026-07-11-00-00-00-000000Z_g2_bernoulli_bandlimited_iters20000_seed0``.

    Returns:
        ``{"cell": str, "seed": int | None}`` when the name matches the
        specification pattern, ``None`` otherwise.  ``_debug_`` directories
        are rejected here as well (they are exploratory by convention).
    """
    if "_debug_" in directory_basename or directory_basename.startswith("_debug_"):
        return None
    match = RUN_DIRECTORY_PATTERN.search(directory_basename)
    if match is None:
        return None
    seed_text = match.group("seed")
    return {
        "cell": match.group("cell"),
        "seed": int(seed_text) if seed_text is not None else None,
    }


def discover_run_directories(data_root: Path) -> list[Path]:
    """Return the sorted non-debug run directories under ``data_root``."""
    candidates = sorted(
        Path(d)
        for d in glob.glob(str(Path(data_root) / "*_iters*"))
        if os.path.isdir(d)
    )
    return [d for d in candidates if parse_run_directory_name(d.name) is not None]


@dataclass
class RunRecord:
    """One discovered run directory with its parsed identity and summaries."""

    path: Path
    cell: str
    seed: int | None
    variant_summaries: dict[str, dict] = field(default_factory=dict)


def load_run_record(run_directory: Path) -> RunRecord:
    """Load one run directory into a :class:`RunRecord`.

    Merges the combined ``summary.yaml`` (local all-variant mode) with every
    per-task ``summary_<variant>.yaml`` (array mode; per-task entries win).
    Cross-checks the folder-name seed suffix and cell against
    ``metadata.yaml`` when present.

    Raises:
        ValueError: If the folder-name seed suffix contradicts the ``seed``
            field of ``metadata.yaml``, or the folder-name cell contradicts
            its ``cell`` field.  The seed suffix and the logged seed must be
            the same integer (multi-seed folder-name convention).
    """
    run_directory = Path(run_directory)
    parsed = parse_run_directory_name(run_directory.name)
    if parsed is None:
        raise ValueError(
            f"{run_directory} does not match the run-directory pattern "
            f"{RUN_DIRECTORY_PATTERN.pattern!r}"
        )
    cell_name = parsed["cell"]
    seed_value = parsed["seed"]

    metadata_path = run_directory / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as handle:
            metadata = yaml.safe_load(handle) or {}
        metadata_seed = metadata.get("seed")
        if metadata_seed is not None:
            metadata_seed = int(metadata_seed)
            if seed_value is not None and metadata_seed != seed_value:
                raise ValueError(
                    f"{run_directory.name}: folder-name seed suffix "
                    f"{seed_value} contradicts metadata.yaml seed "
                    f"{metadata_seed}; the two must be the same integer"
                )
            seed_value = metadata_seed
        metadata_cell = metadata.get("cell")
        if metadata_cell is not None and str(metadata_cell) != cell_name:
            raise ValueError(
                f"{run_directory.name}: folder-name cell {cell_name!r} "
                f"contradicts metadata.yaml cell {metadata_cell!r}"
            )

    variant_summaries: dict[str, dict] = {}
    combined_path = run_directory / "summary.yaml"
    summary_paths = sorted(run_directory.glob("summary_*.yaml"))
    if combined_path.exists():
        summary_paths = [combined_path, *summary_paths]
    for summary_path in summary_paths:
        with open(summary_path) as handle:
            payload = yaml.safe_load(handle) or {}
        for variant_name, metrics in payload.items():
            if isinstance(metrics, dict):
                variant_summaries[str(variant_name)] = dict(metrics)

    return RunRecord(
        path=run_directory,
        cell=cell_name,
        seed=seed_value,
        variant_summaries=variant_summaries,
    )


# ---------------------------------------------------------------------------
# Saved-artefact readers (hist.npz fallback, spectra.npz)
# ---------------------------------------------------------------------------


def read_forcing_floor_training_median(
    run_directory: Path, variant_name: str
) -> float | None:
    """Median of the ``forcing_floor`` history channel from ``hist.npz``.

    Fallback used only when the run summary does not already record
    ``forcing_floor_median``.  Returns ``None`` when the artefact or the
    channel is absent (an explicitly empty slot, never a filler).
    """
    history_path = Path(run_directory) / f"variant_{variant_name}" / "hist.npz"
    if not history_path.exists():
        return None
    with np.load(history_path) as history:
        if "forcing_floor" not in history:
            return None
        return float(np.median(history["forcing_floor"]))


def load_spectra_arrays(run_directory: Path, variant_name: str) -> dict | None:
    """Load the Section-3.4 spectra arrays of one variant, resolving aliases.

    Returns a dictionary keyed by the canonical names of
    ``SPECTRA_KEY_ALIASES`` (missing entries are simply absent), or ``None``
    when the ``spectra.npz`` artefact does not exist.
    """
    spectra_path = Path(run_directory) / f"variant_{variant_name}" / "spectra.npz"
    if not spectra_path.exists():
        return None
    resolved: dict[str, np.ndarray] = {}
    with np.load(spectra_path) as spectra:
        available_keys = set(spectra.files)
        for canonical_name, aliases in SPECTRA_KEY_ALIASES.items():
            for alias in aliases:
                if alias in available_keys:
                    resolved[canonical_name] = np.asarray(spectra[alias])
                    break
    return resolved


def read_measured_cutoff(
    run_directory: Path, variant_name: str, summary_entry: dict
) -> float | None:
    """The measured reachable cutoff ``k_star`` of one trained run.

    Preference order: the run-summary key ``k_star`` when it holds a valid
    cutoff (finite and strictly positive); else the ``k_star`` entry of
    ``spectra.npz`` (the canonical location per specification Section 3.4).
    A non-finite or non-positive stored value encodes "absent" (the
    zero-forcing variants record no cutoff), so ``None`` is returned when
    neither location holds a valid value.
    """

    def _valid_cutoff(raw) -> float | None:
        if raw is None:
            return None
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            return None
        return value

    summary_cutoff = _valid_cutoff(summary_entry.get("k_star"))
    if summary_cutoff is not None:
        return summary_cutoff
    spectra = load_spectra_arrays(run_directory, variant_name)
    if spectra is not None and "k_star" in spectra:
        k_star_array = np.asarray(spectra["k_star"], dtype=float).reshape(-1)
        if k_star_array.size:
            return _valid_cutoff(k_star_array[0])
    return None


# ---------------------------------------------------------------------------
# Cross-seed statistics (pure helpers, unit-tested)
# ---------------------------------------------------------------------------


def median_and_interquartile_range(values) -> dict:
    """Median and interquartile range (25th / 75th percentiles) of a sample.

    Returns ``{"median", "q25", "q75", "n"}``; an empty sample yields
    ``None`` entries with ``n = 0`` (an explicitly empty slot).
    """
    finite_values = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and np.isfinite(float(v))
    ]
    if not finite_values:
        return {"median": None, "q25": None, "q75": None, "n": 0}
    sample = np.asarray(finite_values, dtype=float)
    return {
        "median": float(np.median(sample)),
        "q25": float(np.percentile(sample, 25.0)),
        "q75": float(np.percentile(sample, 75.0)),
        "n": int(sample.size),
    }


def collect_statistics(records: list[RunRecord]) -> dict:
    """Aggregate the run records into per-(cell, variant) metric samples.

    Returns ``{cell: {variant: {"seeds": [...], "metrics": {name: [values]}}}}``.
    Every scalar numeric key of every summary entry is aggregated; the
    ``forcing_floor_median`` fallback is read from ``hist.npz`` when the
    summary lacks it, and the measured cutoff ``k_star`` is resolved through
    :func:`read_measured_cutoff`.
    """
    aggregated: dict = defaultdict(
        lambda: defaultdict(lambda: {"seeds": [], "metrics": defaultdict(list)})
    )
    alias_to_canonical = {
        accepted: canonical
        for canonical, accepted_names in SUMMARY_KEY_ALIASES.items()
        for accepted in accepted_names
    }
    for record in records:
        for variant_name, summary_entry in record.variant_summaries.items():
            slot = aggregated[record.cell][variant_name]
            slot["seeds"].append(record.seed)
            recorded_canonical: set[str] = set()
            for key, value in summary_entry.items():
                # Keys with dedicated semantics (sentinels, flags, the
                # terminal-target split) never enter the generic loop.
                if key in SPECIALLY_HANDLED_SUMMARY_KEYS:
                    continue
                canonical_key = alias_to_canonical.get(key, key)
                if canonical_key in recorded_canonical:
                    continue
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    slot["metrics"][canonical_key].append(float(value))
                    recorded_canonical.add(canonical_key)
            # Terminal-target split: the runner records one distance plus a
            # zero-target flag; a relative distance and an absolute norm are
            # distinct observables and are aggregated under distinct names.
            target_distance = summary_entry.get("terminal_target_distance")
            if isinstance(target_distance, (int, float)) and np.isfinite(
                float(target_distance)
            ):
                is_zero_target = bool(
                    summary_entry.get(
                        "terminal_target_is_zero_target",
                        (record.cell, variant_name) in ZERO_FORCING_VARIANTS,
                    )
                )
                target_key = (
                    "terminal_target_abs_l2"
                    if is_zero_target
                    else "terminal_target_rel_l2"
                )
                slot["metrics"][target_key].append(float(target_distance))
            if "forcing_floor_median" not in recorded_canonical:
                fallback = read_forcing_floor_training_median(
                    record.path, variant_name
                )
                if fallback is not None:
                    slot["metrics"]["forcing_floor_median"].append(fallback)
            measured_cutoff = read_measured_cutoff(
                record.path, variant_name, summary_entry
            )
            if measured_cutoff is not None:
                slot["metrics"]["k_star"].append(measured_cutoff)
    # Convert the nested defaultdicts into plain dicts for stable YAML output.
    return {
        cell: {
            variant: {
                "seeds": slot["seeds"],
                "metrics": {k: list(v) for k, v in slot["metrics"].items()},
            }
            for variant, slot in variants.items()
        }
        for cell, variants in aggregated.items()
    }


def summarise_statistics(statistics: dict) -> dict:
    """Medians / interquartile ranges of every aggregated metric sample."""
    summarised: dict = {}
    for cell, variants in statistics.items():
        summarised[cell] = {}
        for variant, slot in variants.items():
            distinct_seeds = sorted(
                {s for s in slot["seeds"] if s is not None}
            )
            summarised[cell][variant] = {
                "n_runs": len(slot["seeds"]),
                "seeds": distinct_seeds,
                "metrics": {
                    name: median_and_interquartile_range(values)
                    for name, values in sorted(slot["metrics"].items())
                },
            }
    return summarised


def metric_median(summarised: dict, cell: str, variant: str, metric: str):
    """Convenience accessor: the median of a metric, or ``None`` if absent."""
    try:
        return summarised[cell][variant]["metrics"][metric]["median"]
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Closed-form band-edge recomputation (library calls only; never the 2^20
# anchor values)
# ---------------------------------------------------------------------------


class SingleComponentSineDatum:
    r"""Terminal datum :math:`g(x) = \sin(k_0 x)` with exact Fourier coefficients.

    :math:`\sin(k_0 x) = (e^{i k_0 x} - e^{-i k_0 x}) / (2i)`, so
    :math:`c_{k_0} = -i/2`, :math:`c_{-k_0} = i/2`, and every other
    coefficient vanishes.
    """

    def __init__(self, wavenumber: int = 1) -> None:
        if int(wavenumber) != wavenumber or wavenumber < 1:
            raise ValueError(
                f"wavenumber must be a positive integer, received {wavenumber!r}"
            )
        self.wavenumber = int(wavenumber)

    def fourier_coefficients(self, wavenumbers) -> np.ndarray:
        wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
        coefficient_values = np.zeros(wavenumber_array.shape, dtype=np.complex128)
        coefficient_values[wavenumber_array == self.wavenumber] = -0.5j
        coefficient_values[wavenumber_array == -self.wavenumber] = 0.5j
        return coefficient_values


def build_cell_generator(cell_name: str) -> ConstantCoefficientGenerator:
    """The constant-coefficient generator of a cell (specification Section 1.1)."""
    if cell_name == "g1_bernoulli_bandlimited":
        return advection_diffusion_reaction()
    if cell_name == "g2_bernoulli_bandlimited":
        return black_scholes_log_price()
    if cell_name == CONTROL_CELL_NAME:
        # Pure heat at the G2 diffusivity (specification Section 1.1).
        return ConstantCoefficientGenerator(
            coefficients={2: 0.125}, name="pure_heat_g2_diffusivity"
        )
    raise ValueError(f"Unknown cell {cell_name!r}; expected one of {CELL_NAMES}")


def build_cell_datum(cell_name: str):
    """The terminal datum of a cell, restricted to its retained band."""
    if cell_name in GENERATOR_CELL_NAMES:
        # Restricted to |k| <= K_g, the coefficients of the regularity-index-1
        # periodised Bernoulli datum coincide with those of its band-limited
        # truncation, so the untruncated datum object evaluated on the band
        # is exactly the truncated datum's closed form.
        return PeriodisedBernoulliDatum(regularity_index=1)
    if cell_name == CONTROL_CELL_NAME:
        return SingleComponentSineDatum(wavenumber=1)
    raise ValueError(f"Unknown cell {cell_name!r}; expected one of {CELL_NAMES}")


def cell_band_edge(cell_name: str, generator_band_edge: int) -> int:
    """The band edge of every closed-form sum for a cell."""
    if cell_name in GENERATOR_CELL_NAMES:
        return int(generator_band_edge)
    return CONTROL_CELL_BAND_EDGE


def build_terminal_data_extension(
    cell_name: str, variant_name: str
) -> TerminalDataExtension | None:
    """The library extension object of a (cell, variant) pair, or ``None``.

    Returns ``None`` for a variant name outside the specification's variant
    set (the caller records an explicitly empty closed-form slot and prints
    a notice).
    """
    generator = build_cell_generator(cell_name)
    datum = build_cell_datum(cell_name)
    diffusivity = generator.coefficients[2]
    if variant_name == "convex_raw":
        return ConvexRawExtension(datum, generator, TERMINAL_TIME)
    if variant_name == "constant_in_time":
        return ConstantInTimeExtension(datum, generator, TERMINAL_TIME)
    if variant_name == "split_diffusion":
        return SplitSemigroupExtension(datum, generator, (2,), TERMINAL_TIME)
    if variant_name == "split_diffusion_advection":
        return SplitSemigroupExtension(datum, generator, (2, 1), TERMINAL_TIME)
    if variant_name == "graded_gaussian_matched":
        return GradedGaussianExtension(datum, generator, diffusivity, TERMINAL_TIME)
    if variant_name == "graded_gaussian_mismatched":
        return GradedGaussianExtension(
            datum, generator, 0.5 * diffusivity, TERMINAL_TIME
        )
    if variant_name == "exact_solution":
        return ExactSolutionExtension(datum, generator, TERMINAL_TIME)
    if variant_name == "matched_exponential_factor":
        # The control-cell extension Psi(x, t) = e^{-nu (T - t)} sin x equals
        # the exact solution of the pure-heat cell (specification Section
        # 1.3), so its forcing closed form is that of the exact-solution
        # extension: identically zero.
        return ExactSolutionExtension(datum, generator, TERMINAL_TIME)
    return None


def forcing_mass_above_cutoff(
    extension: TerminalDataExtension,
    band_edge: int,
    cutoff: float,
) -> float:
    r"""Unreachable forcing mass :math:`\mathcal{F}(k_\star)` at a cutoff.

    Evaluates, through the closed-form per-wavenumber time integrals
    :math:`I_k` of the library (never a quadrature),

    .. math::

        \mathcal{F}(k_\star)
        = \frac{1}{T} \sum_{|k| > k_\star,\ 0 < |k| \le K_g} I_k .

    With ``cutoff = 0`` this is the full Monte-Carlo floor expectation
    :math:`\mathbb{E}[(P\Psi)^2]` over the uniform sampling measure
    (specification Section 3.3 item 5:
    :math:`\mathbb{E}[(P\Psi)^2] = \|Lh\|^2_{\mathrm{strip}} / (2\pi T)`
    and the strip norm is :math:`2\pi \sum_k I_k`).
    """
    band = symmetric_wavenumber_band(int(band_edge))
    per_wavenumber_integrals = extension.squared_forcing_time_integral(band)
    above_cutoff_mask = np.abs(band) > float(cutoff)
    return float(
        np.sum(per_wavenumber_integrals[above_cutoff_mask])
        / extension.terminal_time
    )


def closed_form_band_edge_quantities(
    cell_name: str, variant_name: str, generator_band_edge: int
) -> dict | None:
    """Closed-form comparison quantities of a variant at the datum band edge.

    Returns ``{"band_edge", "squared_strip_forcing",
    "monte_carlo_floor_expectation"}`` computed through the library calls
    (``squared_forcing_time_integral`` summed over the symmetric band), or
    ``None`` for an unknown variant name.
    """
    extension = build_terminal_data_extension(cell_name, variant_name)
    if extension is None:
        return None
    band_edge = cell_band_edge(cell_name, generator_band_edge)
    band = symmetric_wavenumber_band(band_edge)
    squared_strip_forcing = total_strip_forcing_squared(extension, band)
    return {
        "band_edge": band_edge,
        "squared_strip_forcing": squared_strip_forcing,
        # Monte-Carlo expectation over the uniform sampling measure:
        # E[(P Psi)^2] = ||Lh||^2_strip / (2 pi T) (specification 3.3 item 5).
        "monte_carlo_floor_expectation": squared_strip_forcing
        / (TWO_PI * extension.terminal_time),
    }


def compute_closed_forms(
    statistics: dict, generator_band_edge: int
) -> dict:
    """Closed-form quantities for every (cell, variant) in the display tables.

    Covers the union of the specification's variant set and every variant
    found on disk (an unknown variant receives an explicitly empty slot and
    a printed notice).
    """
    closed_forms: dict = {}
    for cell in CELL_NAMES:
        specified_variants = (
            CONTROL_CELL_VARIANT_NAMES
            if cell == CONTROL_CELL_NAME
            else GENERATOR_CELL_VARIANT_NAMES
        )
        discovered_variants = tuple(statistics.get(cell, {}).keys())
        ordered = list(specified_variants) + [
            v for v in discovered_variants if v not in specified_variants
        ]
        closed_forms[cell] = {}
        for variant in ordered:
            quantities = closed_form_band_edge_quantities(
                cell, variant, generator_band_edge
            )
            if quantities is None:
                print(
                    f"NOTICE: variant {variant!r} of cell {cell!r} is outside "
                    "the specification's variant set; its closed-form slots "
                    "stay empty."
                )
            closed_forms[cell][variant] = quantities
    return closed_forms


def compute_unreachable_masses(
    statistics: dict, generator_band_edge: int
) -> dict:
    r"""Per-(cell, variant) unreachable masses at the *measured* cutoffs.

    For every run whose ``k_star`` was measured, the closed form
    :math:`\mathcal{F}(k_\star)` is evaluated at that run's cutoff; the
    per-run values are then summarised by median / interquartile range.
    Variants without any measured cutoff receive an explicitly empty slot
    (decision D9).
    """
    unreachable: dict = {}
    for cell, variants in statistics.items():
        unreachable[cell] = {}
        for variant, slot in variants.items():
            cutoffs = slot["metrics"].get("k_star", [])
            extension = (
                build_terminal_data_extension(cell, variant)
                if cell in CELL_NAMES
                else None
            )
            if extension is None or not cutoffs:
                unreachable[cell][variant] = median_and_interquartile_range([])
                continue
            band_edge = cell_band_edge(cell, generator_band_edge)
            masses = [
                forcing_mass_above_cutoff(extension, band_edge, cutoff)
                for cutoff in cutoffs
            ]
            unreachable[cell][variant] = median_and_interquartile_range(masses)
    return unreachable


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def variant_display_properties(variant_name: str) -> dict:
    """Colour and label of a variant (catalogue first, fallback second)."""
    if _catalogue is not None:
        try:
            for entry in _catalogue.METHOD_VARIANTS:
                if entry.get("name") == variant_name:
                    return {
                        "color": entry.get(
                            "color",
                            FALLBACK_VARIANT_DISPLAY.get(variant_name, {}).get(
                                "color", "#444444"
                            ),
                        ),
                        "label": entry.get("label", variant_name),
                    }
        except (AttributeError, TypeError):
            pass
    return FALLBACK_VARIANT_DISPLAY.get(
        variant_name, {"color": "#444444", "label": variant_name}
    )


def format_quantity(value, value_format: str = "{:.6e}") -> str:
    """Format a scalar for a table cell; ``None``/NaN yields ``NOT_MEASURED``."""
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float) and not np.isfinite(value):
        return NOT_MEASURED
    return value_format.format(value)


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table (headers begin with a capital letter)."""
    for header in headers:
        if header and not header[0].isupper() and header[0].isalpha():
            raise ValueError(
                f"table column headers must begin with a capital letter, "
                f"received {header!r}"
            )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(
                f"row length {len(row)} does not match header length "
                f"{len(headers)}: {row!r}"
            )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tables (pure assembly, unit-tested)
# ---------------------------------------------------------------------------

STAGE1_TABLE_HEADERS = [
    "Variant",
    "Squared strip forcing at the band edge (closed form)",
    "Floor expectation $\\mathbb{E}[(P\\Psi)^2]$ (closed form)",
    "Forcing-floor channel, training median (measured)",
    "Best training loss (measured, median over seeds)",
    "Relative $L^2$ error (measured, median over seeds)",
]


def assemble_stage1_comparison_rows(
    cell: str,
    summarised: dict,
    closed_forms: dict,
) -> list[list[str]]:
    """Rows of the stage-1 comparison table for one cell.

    Closed-form columns are the band-edge recomputations; measured columns
    are read from the aggregated run summaries and stay explicitly empty
    (``NOT_MEASURED``) when no saved run supplies them.
    """
    rows: list[list[str]] = []
    for variant, quantities in closed_forms.get(cell, {}).items():
        display = variant_display_properties(variant)
        rows.append(
            [
                display["label"],
                format_quantity(
                    None
                    if quantities is None
                    else quantities["squared_strip_forcing"]
                ),
                format_quantity(
                    None
                    if quantities is None
                    else quantities["monte_carlo_floor_expectation"]
                ),
                format_quantity(
                    metric_median(summarised, cell, variant, "forcing_floor_median")
                ),
                format_quantity(
                    metric_median(summarised, cell, variant, "best_loss")
                ),
                format_quantity(metric_median(summarised, cell, variant, "rel_l2")),
            ]
        )
    return rows


ADDITIVE_VERSUS_CONVEX_VARIANTS = (
    "convex_raw",
    "constant_in_time",
    "split_diffusion",
    "split_diffusion_advection",
)

ADDITIVE_VERSUS_CONVEX_HEADERS = [
    "Quantity",
    "Convex raw (hard convex)",
    "Constant-in-time (additive)",
    "Split $\\{\\partial_{xx}\\}$ (additive)",
    "Split $\\{\\partial_{xx},\\partial_x\\}$ (additive)",
]


def assemble_additive_versus_convex_rows(
    cell: str,
    summarised: dict,
    closed_forms: dict,
    unreachable_masses: dict,
) -> tuple[list[list[str]], str]:
    """Rows and conclusion line of the additive-versus-convex table (one cell).

    The conclusion line is computed only when every measured input exists;
    otherwise it is the explicit ``NOT_MEASURED`` marker (decision D9 — no
    placeholder value is written anywhere).
    """

    def closed_floor(variant):
        quantities = closed_forms.get(cell, {}).get(variant)
        return None if quantities is None else quantities[
            "monte_carlo_floor_expectation"
        ]

    def measured(variant, metric):
        return metric_median(summarised, cell, variant, metric)

    def unreachable(variant):
        entry = unreachable_masses.get(cell, {}).get(variant)
        return None if entry is None else entry["median"]

    quantity_rows = [
        (
            "Closed-form floor $\\mathbb{E}[(P\\Psi)^2]$ at the band edge",
            closed_floor,
        ),
        (
            "Best training loss (median over seeds)",
            lambda v: measured(v, "best_loss"),
        ),
        (
            "Relative $L^2$ error (median over seeds)",
            lambda v: measured(v, "rel_l2"),
        ),
        (
            "Measured cutoff $k_\\star$ (median over seeds)",
            lambda v: measured(v, "k_star"),
        ),
        (
            "Unreachable mass $\\mathcal{F}(k_\\star)$ at the measured cutoff",
            unreachable,
        ),
    ]
    rows = [
        [quantity_name]
        + [format_quantity(getter(v)) for v in ADDITIVE_VERSUS_CONVEX_VARIANTS]
        for quantity_name, getter in quantity_rows
    ]

    # Ratio lines: closed-form ratio is always computable; the measured
    # best-loss ratio only once both losses are measured.
    convex_floor = closed_floor("convex_raw")
    split_floor = closed_floor("split_diffusion")
    if (
        convex_floor is not None
        and split_floor is not None
        and split_floor > 0.0
    ):
        closed_ratio_text = format_quantity(convex_floor / split_floor)
    else:
        closed_ratio_text = NOT_MEASURED
    convex_loss = measured("convex_raw", "best_loss")
    split_loss = measured("split_diffusion", "best_loss")
    if convex_loss is not None and split_loss is not None and split_loss > 0.0:
        measured_ratio = convex_loss / split_loss
        measured_ratio_text = format_quantity(measured_ratio)
        conclusion = (
            f"Measured best-loss ratio convex-raw / split-diffusion = "
            f"{measured_ratio:.3e}; closed-form floor ratio at the band edge "
            f"= {closed_ratio_text}. The hypothesis H1 comparison at the "
            f"measured cutoff is given in the unreachable-mass row."
        )
    else:
        measured_ratio_text = NOT_MEASURED
        conclusion = NOT_MEASURED
    rows.append(
        [
            "Floor ratio convex raw / split $\\{\\partial_{xx}\\}$ (closed form)",
            closed_ratio_text,
            "",
            "",
            "",
        ]
    )
    rows.append(
        [
            "Best-loss ratio convex raw / split $\\{\\partial_{xx}\\}$ (measured)",
            measured_ratio_text,
            "",
            "",
            "",
        ]
    )
    return rows, conclusion


def write_stage1_comparison_table(
    path: Path, summarised: dict, closed_forms: dict, generator_band_edge: int
) -> None:
    sections = [
        "# Stage-1 comparison quantities recomputed at the datum band edge\n",
        (
            f"Closed forms evaluated at the band edge $K_g = "
            f"{generator_band_edge}$ for the generator cells and at the "
            f"datum band edge $k_0 = {CONTROL_CELL_BAND_EDGE}$ for the "
            "control cell, through the library closed-form time integrals "
            "(`squared_forcing_time_integral`, summed over the symmetric "
            "band; Parseval normalisation "
            "$\\mathbb{E}[(P\\Psi)^2] = \\|Lh\\|^2_{\\mathrm{strip}} / "
            "(2\\pi T)$).  The stage-1 anchors at band edge $2^{20}$ are "
            "intentionally not quoted: the trained datum is truncated, so "
            "every comparison quantity is recomputed at the truncation "
            "band edge (specification, band-edge caution).  Measured "
            f'columns marked "{NOT_MEASURED}" await the corresponding '
            "saved runs.\n"
        ),
    ]
    for cell in CELL_NAMES:
        if cell not in closed_forms:
            continue
        sections.append(f"## Cell `{cell}`\n")
        rows = assemble_stage1_comparison_rows(cell, summarised, closed_forms)
        sections.append(render_markdown_table(STAGE1_TABLE_HEADERS, rows))
        sections.append("")
    Path(path).write_text("\n".join(sections))


def write_additive_versus_convex_table(
    path: Path,
    summarised: dict,
    closed_forms: dict,
    unreachable_masses: dict,
) -> None:
    sections = [
        "# Additive-versus-convex conclusion table\n",
        (
            "Comparison of the convex baseline (`hard_convex`, extension "
            "$\\Psi = \\lambda(t) g$) against the additive `hard_constant` "
            "family (constant-in-time and split extensions).  Every slot "
            "that no saved run supplies yet is explicitly marked "
            f'"{NOT_MEASURED}" — an inferred value is never reported as a '
            "measured one.\n"
        ),
    ]
    for cell in GENERATOR_CELL_NAMES:
        sections.append(f"## Cell `{cell}`\n")
        rows, conclusion = assemble_additive_versus_convex_rows(
            cell, summarised, closed_forms, unreachable_masses
        )
        sections.append(
            render_markdown_table(ADDITIVE_VERSUS_CONVEX_HEADERS, rows)
        )
        sections.append(f"Conclusion: {conclusion}\n")
    Path(path).write_text("\n".join(sections))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _cells_present(summarised: dict) -> list[str]:
    return [c for c in CELL_NAMES if c in summarised] + [
        c for c in summarised if c not in CELL_NAMES
    ]


def _variant_order(cell: str, summarised: dict) -> list[str]:
    specified = (
        CONTROL_CELL_VARIANT_NAMES
        if cell == CONTROL_CELL_NAME
        else GENERATOR_CELL_VARIANT_NAMES
    )
    present = summarised.get(cell, {})
    return [v for v in specified if v in present] + [
        v for v in present if v not in specified
    ]


def plot_rel_l2_by_cell(summarised: dict, out_dir: Path) -> bool:
    """Grouped bar chart of the relative L2 error (median, IQR over seeds)."""
    cells = _cells_present(summarised)
    all_variants: list[str] = []
    for cell in cells:
        for v in _variant_order(cell, summarised):
            if v not in all_variants:
                all_variants.append(v)
    has_data = any(
        metric_median(summarised, cell, v, "rel_l2") is not None
        for cell in cells
        for v in all_variants
    )
    if not has_data:
        print(
            "NOTICE: rel_l2_by_cell.png not generated — no saved run records "
            "a rel_l2 value yet (explicitly empty)."
        )
        return False
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_variants = max(len(all_variants), 1)
    bar_width = 0.8 / n_variants
    x_positions = np.arange(len(cells))
    for j, variant in enumerate(all_variants):
        display = variant_display_properties(variant)
        medians, lower_quartiles, upper_quartiles = [], [], []
        for cell in cells:
            stats = (
                summarised.get(cell, {})
                .get(variant, {})
                .get("metrics", {})
                .get("rel_l2", {"median": None})
            )
            median = stats.get("median")
            if median is None:
                medians.append(np.nan)
                lower_quartiles.append(np.nan)
                upper_quartiles.append(np.nan)
            else:
                medians.append(median)
                lower_quartiles.append(stats["q25"])
                upper_quartiles.append(stats["q75"])
        bar_centres = x_positions + j * bar_width
        ax.bar(
            bar_centres,
            np.clip(np.asarray(medians, dtype=float), 1e-30, None),
            bar_width,
            color=display["color"],
            label=display["label"],
        )
        # Interquartile interval drawn directly (per-seed spread over seeds).
        ax.vlines(
            bar_centres,
            np.clip(np.asarray(lower_quartiles, dtype=float), 1e-30, None),
            np.clip(np.asarray(upper_quartiles, dtype=float), 1e-30, None),
            color="black",
            lw=1.0,
        )
    ax.set_yscale("log")
    ax.set_xticks(x_positions + 0.4 - bar_width / 2)
    ax.set_xticklabels(cells, fontsize=8)
    # The quantity is defined in full in the formula box below the figure.
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.set_title(
        "Accuracy per extension variant and cell (median, interquartile "
        "range over seeds)"
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)
    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8,
        frameon=True,
    )
    fig.tight_layout(rect=[0.04, 0.30, 1, 1])
    finalize_figure(
        fig,
        out_dir / "rel_l2_by_cell.png",
        legends=[legend],
        axes=[ax],
        formula=(
            r"$\mathrm{rel}\,L^2 = \|\hat u - u^\star\|_{L^2(G_{\mathrm{eval}})}"
            r"\,/\,\|u^\star\|_{L^2(G_{\mathrm{eval}})}$; "
            r"$u^\star$ the exact spectral-component sum; median with "
            r"interquartile range over seeds"
        ),
    )
    return True


def plot_floor_vs_accuracy(
    summarised: dict,
    closed_forms: dict,
    unreachable_masses: dict,
    out_dir: Path,
) -> bool:
    """H1 figure: best loss against the closed-form floor and against
    the unreachable mass at the measured cutoff."""
    points_floor = []  # (floor, best_loss, cell, variant)
    points_unreachable = []
    for cell in _cells_present(summarised):
        for variant in _variant_order(cell, summarised):
            best_loss = metric_median(summarised, cell, variant, "best_loss")
            if best_loss is None:
                continue
            quantities = closed_forms.get(cell, {}).get(variant)
            if quantities is not None and quantities[
                "monte_carlo_floor_expectation"
            ] > 0.0:
                points_floor.append(
                    (
                        quantities["monte_carlo_floor_expectation"],
                        best_loss,
                        cell,
                        variant,
                    )
                )
            mass_entry = unreachable_masses.get(cell, {}).get(variant)
            if (
                mass_entry is not None
                and mass_entry["median"] is not None
                and mass_entry["median"] > 0.0
            ):
                points_unreachable.append(
                    (mass_entry["median"], best_loss, cell, variant)
                )
    if not points_floor and not points_unreachable:
        print(
            "NOTICE: floor_vs_accuracy.png not generated — no saved run "
            "records a best_loss value yet (explicitly empty)."
        )
        return False

    fig, (ax_floor, ax_mass) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, points, x_label, title, empty_text in (
        (
            ax_floor,
            points_floor,
            r"Closed-form floor $\mathbb{E}[(P\Psi)^2]$ at the band edge",
            "Best loss against the closed-form floor",
            "No variant with a strictly positive closed-form\nfloor has a "
            "saved best loss yet (slot explicitly empty)",
        ),
        (
            ax_mass,
            points_unreachable,
            r"Unreachable mass $\mathcal{F}(k_\star)$ at the measured cutoff",
            "Best loss against the unreachable mass (H1)",
            "No measured cutoff $k_\\star$ saved yet\n(slot explicitly "
            "empty, decision D9)",
        ),
    ):
        if points:
            values = np.asarray([[p[0], p[1]] for p in points], dtype=float)
            span = np.asarray(
                [values.min() * 0.3, values.max() * 3.0], dtype=float
            )
            ax.plot(
                span, span, "--", color="black", lw=1.0,
                label=r"Identity $y = x$",
            )
            for x_value, y_value, cell, variant in points:
                display = variant_display_properties(variant)
                ax.scatter(
                    x_value,
                    y_value,
                    c=display["color"],
                    marker=CELL_MARKERS.get(cell, "o"),
                    s=90,
                    edgecolors="black",
                    linewidths=0.5,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            ax.text(
                0.5,
                0.5,
                empty_text,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"Best training loss $\mathbb{E}[(P\hat u)^2]$ (median)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    from matplotlib.lines import Line2D

    seen_variants: list[str] = []
    for _, _, _, variant in points_floor + points_unreachable:
        if variant not in seen_variants:
            seen_variants.append(variant)
    variant_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=variant_display_properties(v)["color"],
            markeredgecolor="black", markersize=9,
            label=variant_display_properties(v)["label"],
        )
        for v in seen_variants
    ]
    cell_handles = [
        Line2D(
            [0], [0], marker=CELL_MARKERS[c], color="w",
            markerfacecolor="grey", markeredgecolor="black", markersize=9,
            label=c,
        )
        for c in CELL_MARKERS
        if c in summarised
    ]
    legend = fig.legend(
        handles=variant_handles + cell_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.24),
        ncol=3,
        fontsize=7,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.34, 1, 1])
    finalize_figure(
        fig,
        out_dir / "floor_vs_accuracy.png",
        legends=[legend],
        axes=[ax_floor, ax_mass],
        formula=(
            r"$\mathbb{E}[(P\Psi)^2]=\frac{1}{T}\sum_{0<|k|\leq K_g}I_k$;  "
            r"$\mathcal{F}(k_\star)=\frac{1}{T}\sum_{k_\star<|k|\leq K_g}I_k$;  "
            r"$I_k=\int_0^T|\widehat{Lh}(k,t)|^2\,dt$ (closed forms)"
        ),
        formula_fontsize=7,
    )
    return True


def plot_cutoff_by_variant(summarised: dict, statistics: dict, out_dir: Path) -> bool:
    """H2 figure: the measured cutoff k_star per variant and cell."""
    has_cutoff = any(
        slot["metrics"].get("k_star")
        for variants in statistics.values()
        for slot in variants.values()
    )
    if not has_cutoff:
        print(
            "NOTICE: cutoff_by_variant.png not generated — no saved run "
            "records a measured cutoff k_star yet (explicitly empty, "
            "decision D9)."
        )
        return False
    cells = _cells_present(summarised)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x_tick_positions, x_tick_labels = [], []
    cell_group_centres: list[tuple[float, str]] = []
    cell_group_edges: list[float] = []
    position = 0.0
    plotted_variants: list[str] = []
    for cell_index, cell in enumerate(cells):
        variants = _variant_order(cell, summarised)
        group_start = position
        if cell_index > 0:
            cell_group_edges.append(position - 0.9)
        for variant in variants:
            values = (
                statistics.get(cell, {})
                .get(variant, {})
                .get("metrics", {})
                .get("k_star", [])
            )
            display = variant_display_properties(variant)
            if values:
                jitter = np.linspace(-0.15, 0.15, len(values))
                ax.scatter(
                    position + jitter,
                    values,
                    c=display["color"],
                    s=45,
                    edgecolors="black",
                    linewidths=0.4,
                )
                ax.scatter(
                    [position],
                    [float(np.median(values))],
                    marker="_",
                    s=500,
                    c=display["color"],
                )
                if variant not in plotted_variants:
                    plotted_variants.append(variant)
            x_tick_positions.append(position)
            x_tick_labels.append(variant_display_properties(variant)["label"])
            position += 1.0
        cell_group_centres.append((0.5 * (group_start + position - 1.0), cell))
        position += 0.8  # gap between cells
    ax.set_yscale("log", base=2)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels, fontsize=7, rotation=30, ha="right")
    # Cell group annotation above the axes; dotted separators between cells
    # (auxiliary annotation lines per the stroke convention).
    import matplotlib.transforms as mtransforms

    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for centre, cell in cell_group_centres:
        ax.text(
            centre, 1.02, cell, transform=blended, ha="center", va="bottom",
            fontsize=8,
        )
    for edge in cell_group_edges:
        ax.axvline(edge, ls=":", color="grey", lw=0.8)
    ax.set_ylabel(r"Measured reachable cutoff $k_\star$")
    ax.set_title(
        "Cutoff invariance across variants (H2): one point per seed, dash = "
        "median",
        pad=28,
    )
    ax.grid(True, axis="y", which="both", alpha=0.3)
    # No separate legend: each colour-coded point group sits directly above
    # its own variant tick label, which already names the variant.
    fig.tight_layout(rect=[0.04, 0.20, 1, 1])
    finalize_figure(
        fig,
        out_dir / "cutoff_by_variant.png",
        legends=[],
        axes=[ax],
        formula=(
            r"$k_\star$ = first in-band wavenumber at which the 7-point "
            r"running mean of $|\hat r_k|^2 / |\widehat{Lh}(k)|^2$ reaches "
            r"$1/2$ (specification Section 3.4); absent for the zero-forcing "
            r"variants"
        ),
    )
    return True


def plot_terminal_target(summarised: dict, out_dir: Path) -> bool:
    """H3 figure: terminal-target distances of the trained bare network."""
    relative_entries = []  # (cell, variant, stats)
    absolute_entries = []
    for cell in _cells_present(summarised):
        for variant in _variant_order(cell, summarised):
            metrics = summarised[cell][variant]["metrics"]
            if metrics.get("terminal_target_rel_l2", {}).get("median") is not None:
                relative_entries.append(
                    (cell, variant, metrics["terminal_target_rel_l2"])
                )
            if metrics.get("terminal_target_abs_l2", {}).get("median") is not None:
                absolute_entries.append(
                    (cell, variant, metrics["terminal_target_abs_l2"])
                )
    if not relative_entries and not absolute_entries:
        print(
            "NOTICE: terminal_target.png not generated — no saved run "
            "records a terminal-target distance yet (explicitly empty)."
        )
        return False
    fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, entries, y_label, title in (
        (
            ax_rel,
            relative_entries,
            r"$\|\Phi_\theta(\cdot,T) - \Phi^\star(\cdot,T)\|_2 \,/\, "
            r"\|\Phi^\star(\cdot,T)\|_2$",
            "Distance to the exact minimiser's terminal profile (H3)",
        ),
        (
            ax_abs,
            absolute_entries,
            r"$\|\Phi_\theta(\cdot,T)\|_2$",
            "Zero-target variants: absolute terminal norm",
        ),
    ):
        if entries:
            positions = np.arange(len(entries))
            medians = [e[2]["median"] for e in entries]
            lower_quartiles = [e[2]["q25"] for e in entries]
            upper_quartiles = [e[2]["q75"] for e in entries]
            colors = [variant_display_properties(e[1])["color"] for e in entries]
            ax.bar(
                positions,
                np.clip(medians, 1e-30, None),
                0.7,
                color=colors,
            )
            # Interquartile interval drawn directly (spread over seeds).
            ax.vlines(
                positions,
                np.clip(np.asarray(lower_quartiles, dtype=float), 1e-30, None),
                np.clip(np.asarray(upper_quartiles, dtype=float), 1e-30, None),
                color="black",
                lw=1.0,
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [
                    f"{variant_display_properties(v)['label']}"
                    f" ({c.split('_')[0]})"
                    for c, v, _ in entries
                ],
                fontsize=6,
                rotation=30,
                ha="right",
            )
            ax.set_yscale("log")
        else:
            ax.text(
                0.5, 0.5, "No saved measurement yet\n(slot explicitly empty)",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
            )
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    finalize_figure(
        fig,
        out_dir / "terminal_target.png",
        legends=[],
        axes=[ax_rel, ax_abs],
        formula=(
            r"$\Phi^\star(\cdot,T) = -\,(P\Psi)(\cdot,T)\,/\,d_T'(T) = "
            r"T\,(P\Psi)(\cdot,T)$ for the linear factor; median with "
            r"interquartile range over seeds"
        ),
    )
    return True


def plot_residual_spectra_by_cell(
    records: list[RunRecord], summarised: dict, out_dir: Path
) -> bool:
    """Residual frequency decomposition (specification Section 3.4),
    aggregated across seeds: per-seed running-mean cancellation ratios."""
    # Gather the per-(cell, variant, seed) running means from spectra.npz.
    curves: dict = defaultdict(list)  # (cell, variant) -> list of (k, rsm, mask)
    cutoffs: dict = defaultdict(list)
    for record in records:
        for variant in record.variant_summaries:
            if (record.cell, variant) in ZERO_FORCING_VARIANTS:
                continue  # cancellation ratio undefined (zero forcing)
            spectra = load_spectra_arrays(record.path, variant)
            if spectra is None:
                continue
            if "wavenumbers" not in spectra or "running_mean" not in spectra:
                continue
            mask = spectra.get("in_band_mask")
            curves[(record.cell, variant)].append(
                (
                    np.asarray(spectra["wavenumbers"], dtype=float),
                    np.asarray(spectra["running_mean"], dtype=float),
                    None if mask is None else np.asarray(mask, dtype=bool),
                )
            )
            cutoff = read_measured_cutoff(
                record.path, variant, record.variant_summaries[variant]
            )
            if cutoff is not None:
                cutoffs[(record.cell, variant)].append(cutoff)
    if not curves:
        print(
            "NOTICE: residual_spectra_by_cell.png not generated — no saved "
            "spectra.npz with a running-mean cancellation ratio yet "
            "(explicitly empty)."
        )
        return False

    cells_with_curves = [
        c for c in _cells_present(summarised) if any(k[0] == c for k in curves)
    ]
    fig, axes = plt.subplots(
        1,
        len(cells_with_curves),
        figsize=(max(9.0, 6.0 * len(cells_with_curves)), 5.2),
        squeeze=False,
    )
    legend_labels_done: set[str] = set()
    all_legends = []
    for ax, cell in zip(axes[0], cells_with_curves):
        for (curve_cell, variant), seed_curves in curves.items():
            if curve_cell != cell:
                continue
            display = variant_display_properties(variant)
            for wavenumbers, running_mean, mask in seed_curves:
                keep = (wavenumbers > 0)
                if mask is not None and mask.shape == wavenumbers.shape:
                    keep &= mask
                label = (
                    display["label"]
                    if variant not in legend_labels_done
                    else None
                )
                ax.semilogx(
                    wavenumbers[keep],
                    np.clip(running_mean[keep], 0.0, 1.3),
                    "-",
                    color=display["color"],
                    lw=1.2,
                    alpha=0.7,
                    label=label,
                )
                if label is not None:
                    legend_labels_done.add(variant)
            variant_cutoffs = cutoffs.get((cell, variant), [])
            if variant_cutoffs:
                ax.axvline(
                    float(np.median(variant_cutoffs)),
                    ls=":",
                    color=display["color"],
                    lw=1.0,
                )
        ax.axhline(0.5, ls=":", color="black", lw=0.8)
        ax.set_xlabel(r"Wavenumber $k$")
        ax.set_ylabel("Cancellation ratio (running mean)")
        ax.set_title(f"Cell {cell}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
    legend = fig.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.22), ncol=3, fontsize=7,
        frameon=True,
    )
    all_legends.append(legend)
    fig.tight_layout(rect=[0.04, 0.32, 1, 1])
    finalize_figure(
        fig,
        out_dir / "residual_spectra_by_cell.png",
        legends=all_legends,
        axes=list(axes[0]),
        formula=(
            r"$|\hat r_k|^2/|\widehat{Lh}(k)|^2$: slice-averaged residual "
            r"FFT power over exact forcing power; 7-point running mean; "
            r"dotted: median $k_\star$; one curve per seed"
        ),
        formula_fontsize=7,
    )
    return True


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def build_yaml_payload(
    summarised: dict,
    closed_forms: dict,
    unreachable_masses: dict,
    data_root: Path,
    generator_band_edge: int,
) -> dict:
    payload: dict = {
        "generated_by": Path(__file__).name,
        "timestamp_utc": utc_timestamp(),
        "data_root": str(data_root),
        "band_edge_note": (
            "Closed-form quantities recomputed at the datum band edge "
            f"(K_g = {generator_band_edge} for the generator cells, "
            f"k_0 = {CONTROL_CELL_BAND_EDGE} for the control cell) through "
            "the library closed-form time integrals; the stage-1 anchors at "
            "band edge 2^20 are not quoted. Null entries are explicitly "
            "empty slots (not measured), never inferred values."
        ),
        "cells": {},
    }
    # Union of the specified cells and any additional cell found on disk
    # (an unexpected cell is reported, never silently dropped).
    all_cells = list(CELL_NAMES) + [c for c in summarised if c not in CELL_NAMES]
    for cell in all_cells:
        if cell not in summarised and cell not in closed_forms:
            continue
        if cell not in CELL_NAMES:
            print(
                f"NOTICE: cell {cell!r} is outside the specification's cell "
                "set; its closed-form slots stay empty."
            )
        cell_payload: dict = {"variants": {}}
        variant_names = list(closed_forms.get(cell, {}).keys())
        for variant in summarised.get(cell, {}):
            if variant not in variant_names:
                variant_names.append(variant)
        for variant in variant_names:
            measured = summarised.get(cell, {}).get(variant)
            quantities = closed_forms.get(cell, {}).get(variant)
            mass_entry = unreachable_masses.get(cell, {}).get(variant)
            cell_payload["variants"][variant] = {
                "n_runs": None if measured is None else measured["n_runs"],
                "seeds": [] if measured is None else measured["seeds"],
                "measured": (
                    None if measured is None else measured["metrics"]
                ),
                "closed_form_at_band_edge": quantities,
                "unreachable_mass_at_measured_cutoff": (
                    None
                    if mass_entry is None or mass_entry["median"] is None
                    else mass_entry
                ),
            }
        payload["cells"][cell] = cell_payload
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=(
            "Folder holding the per-(cell, seed) run directories to "
            f"aggregate (default: data/{RUNNER_SCRIPT_STEM})."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Where to write the aggregate artefacts (default: a timestamped "
            "folder under data/split_extension_cross_seed_summary/)."
        ),
    )
    parser.add_argument(
        "--band-edge",
        type=int,
        default=GENERATOR_CELL_BAND_EDGE,
        help=(
            "Band edge of the closed-form recomputation for the generator "
            f"cells (default: the specification value "
            f"{GENERATOR_CELL_BAND_EDGE}; the control cell always uses its "
            f"datum band edge {CONTROL_CELL_BAND_EDGE})."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Mark this aggregation as exploratory (prefixes the output "
            "folder with _debug_)."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if _catalogue is None:
        print(
            "NOTICE: _split_extension_catalogue is not importable yet "
            "(authored in parallel with the runner); the "
            "specification-pinned fallback colours and labels are used."
        )
    else:
        catalogue_stem = getattr(_catalogue, "RUNNER_SCRIPT_STEM", None)
        if catalogue_stem is not None and catalogue_stem != RUNNER_SCRIPT_STEM:
            raise ValueError(
                f"catalogue RUNNER_SCRIPT_STEM {catalogue_stem!r} differs "
                f"from this aggregator's pinned stem {RUNNER_SCRIPT_STEM!r};"
                " reconcile the two before aggregating."
            )

    data_root = (
        Path(args.data_root)
        if args.data_root
        else script_data_dir(Path(__file__).parent / f"{RUNNER_SCRIPT_STEM}.py")
    )

    run_directories = discover_run_directories(data_root)
    records = [load_run_record(d) for d in run_directories]
    records = [r for r in records if r.variant_summaries]
    if not records:
        print(
            f"NOTICE: no (non-debug) run directory with summaries found "
            f"under {data_root}; the closed-form tables are still written, "
            "with every measured slot explicitly empty."
        )

    statistics = collect_statistics(records)
    summarised = summarise_statistics(statistics)
    closed_forms = compute_closed_forms(statistics, args.band_edge)
    unreachable_masses = compute_unreachable_masses(statistics, args.band_edge)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        debug_prefix = "_debug_" if args.debug else ""
        n_cells = len(summarised)
        n_seeds = max(
            (
                len(summarised[cell][variant]["seeds"])
                for cell in summarised
                for variant in summarised[cell]
            ),
            default=0,
        )
        out_dir = (
            script_data_dir(__file__)
            / f"{debug_prefix}{utc_timestamp()}_{n_cells}cell_{n_seeds}seed"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Aggregated {len(records)} run directories over cells "
        f"{sorted(summarised)} into {out_dir}"
    )

    # Tables (always written; measured slots stay explicitly empty until the
    # corresponding runs exist).
    write_stage1_comparison_table(
        out_dir / "stage1_comparison_table.md",
        summarised,
        closed_forms,
        args.band_edge,
    )
    write_additive_versus_convex_table(
        out_dir / "additive_versus_convex_conclusion.md",
        summarised,
        closed_forms,
        unreachable_masses,
    )
    with open(out_dir / "summary_across_seeds.yaml", "w") as handle:
        yaml.dump(
            build_yaml_payload(
                summarised, closed_forms, unreachable_masses, data_root,
                args.band_edge,
            ),
            handle,
            default_flow_style=False,
            sort_keys=False,
            width=float("inf"),
        )

    # Figures (each is skipped, with an explicit notice, when no saved run
    # supplies its measured inputs — a missing figure is an empty slot, not
    # an error).
    written = ["stage1_comparison_table.md",
               "additive_versus_convex_conclusion.md",
               "summary_across_seeds.yaml"]
    if plot_rel_l2_by_cell(summarised, out_dir):
        written.append("rel_l2_by_cell.png")
    if plot_floor_vs_accuracy(summarised, closed_forms, unreachable_masses, out_dir):
        written.append("floor_vs_accuracy.png")
    if plot_cutoff_by_variant(summarised, statistics, out_dir):
        written.append("cutoff_by_variant.png")
    if plot_terminal_target(summarised, out_dir):
        written.append("terminal_target.png")
    if plot_residual_spectra_by_cell(records, summarised, out_dir):
        written.append("residual_spectra_by_cell.png")
    print("wrote: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
