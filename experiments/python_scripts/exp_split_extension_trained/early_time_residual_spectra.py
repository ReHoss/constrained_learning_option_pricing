#!/usr/bin/env python3
r"""Residual and forcing spectra of saved trained fields at early times, without retraining.

Pre-registration:
``documents/methodology/2026-10-19_preregistration_chen_mangasarian_forcing_and_early_residual.md``
(part 2).

Question.  At the datum slice the Chen--Mangasarian extension has the initial
time derivative zero, so its forcing tends to that of the constant-in-time
extension, whereas the retained evolution has the initial time derivative
:math:`Ag`.  The production spectra of the runner are averaged over the slices
:math:`(0.1, \dots, 0.9)\,T`, which do not reach the times at which this
difference is located.  This diagnostic re-evaluates the saved trained fields
at early paper times :math:`\tau T`, :math:`\tau \in \{0.001, 0.01, 0.03, 0.1,
0.5\}`, one slice at a time, with the production routine
:func:`compute_spectra`.

Time convention.  The runner places the datum at :math:`t = T` (backward
convention); a paper time :math:`\tau T` after the datum is the runner slice
fraction :math:`1 - \tau`.

Measured per run and paper time: the residual power of the trained field and
the closed-form forcing power of its extension, per wavenumber, and summed over
the low band and the high band.  Predictions (pre-registered), with
:math:`R(\tau)` the ratio of the median over seeds of the high-band residual
power of the Chen--Mangasarian fields to that of the highest-order splits:

* S1: at :math:`\tau = 0.001`, the high-band forcing power of the
  Chen--Mangasarian extension lies within ``FORCING_RELATIVE_TOLERANCE`` of that
  of the constant-in-time extension, on both cells;
* S2: :math:`R(0.001) > R(0.1)` on both cells;
* S3: :math:`R(0.001) \ge` ``RESIDUAL_RATIO_THRESHOLD`` on both cells.

No training and no gradient step on the weights occur: only forward passes and
the automatic differentiation of the residual that :func:`compute_spectra`
performs.  The job is a post-processing one.

Artefacts: ``early_time_spectra.npz`` (per-wavenumber powers of every run and
time), ``summary.yaml``, ``run_metadata.json``, ``command.txt``, ``run.log``,
``early_time_high_band_residual.png``; ``--replot RUN_DIR`` rebuilds the figure.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ablation_split_extension_trained import compute_spectra  # noqa: E402
from slice_count_convergence import discover_variant_runs, load_trained_variant  # noqa: E402
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    collect_run_metadata,
    configure_cli_script_logging,
    find_repo_root,
    init_logging,
    log_parsed_args,
    script_data_dir,
    utc_timestamp,
    write_command_txt,
    write_json,
)

LOGGER = logging.getLogger(Path(__file__).stem)

DEFAULT_PAPER_TIME_FRACTIONS = (0.001, 0.01, 0.03, 0.1, 0.5)
LOW_BAND = (1, 8)
HIGH_BAND = (32, 256)
# Pre-registered thresholds (section 3 of the pre-registration).
FORCING_RELATIVE_TOLERANCE = 0.10
RESIDUAL_RATIO_THRESHOLD = 10.0
S1_S3_PAPER_TIME = 0.001
S2_LATER_PAPER_TIME = 0.1

CELLS = ("g2_bernoulli_full", "g3_bernoulli_full")
CHEN_MANGASARIAN = "graded_chen_mangasarian"
CONSTANT_IN_TIME = "constant_in_time"
HIGHEST_ORDER_SPLIT = {"g2_bernoulli_full": "split_diffusion", "g3_bernoulli_full": "split_principal"}
ROLE_OF_VARIANT = {CHEN_MANGASARIAN: "chen_mangasarian", CONSTANT_IN_TIME: "constant_in_time",
                   "split_diffusion": "highest_order_split", "split_principal": "highest_order_split"}
ROLE_LABELS = {"chen_mangasarian": "Chen-Mangasarian (algebraic kernel)",
               "highest_order_split": "Highest-order split", "constant_in_time": "Raw constant extension"}
ROLE_COLOURS = {"chen_mangasarian": "#696969", "highest_order_split": "#185d82", "constant_in_time": "#b2182b"}

SPECTRA_FILENAME = "early_time_spectra.npz"
SUMMARY_FILENAME = "summary.yaml"
FIGURE_FILENAME = "early_time_high_band_residual.png"


def band_power(wavenumber_bins: np.ndarray, power: np.ndarray, band: tuple[int, int]) -> float:
    """Sum of a power spectrum over the wavenumbers of a closed band."""
    selected = (wavenumber_bins >= band[0]) & (wavenumber_bins <= band[1])
    return float(np.sum(power[selected]))


def measure_run(run_directory: Path, variant_name: str, paper_times, device: str) -> dict:
    """Residual and forcing power of one saved trained field at each paper time."""
    loaded = load_trained_variant(run_directory, variant_name, device)
    rows = []
    for paper_time in paper_times:
        spectra = compute_spectra(loaded["model"], loaded["problem"], loaded["variant"],
                                  loaded["closed_form_extension"], slice_fractions=[1.0 - paper_time])
        bins = spectra["wavenumber_bins"]
        rows.append({
            "paper_time_fraction": float(paper_time),
            "wavenumber_bins": bins,
            "residual_power": spectra["residual_power"],
            "forcing_power": spectra["forcing_power"],
        })
    return {"cell": loaded["cell"], "seed": loaded["seed"], "rows": rows}


def median_over_seeds(records: list[dict], cell: str, role: str, paper_time: float, quantity: str) -> float | None:
    """Median over the seeds of a band power of one role on one cell at one paper time."""
    values = [record[quantity] for record in records
              if record["cell"] == cell and record["role"] == role
              and np.isclose(record["paper_time_fraction"], paper_time)]
    return float(np.median(values)) if values else None


def verdicts(records: list[dict]) -> dict:
    """The pre-registered verdicts S1-S3 and the ratio R at every paper time."""
    paper_times = sorted({record["paper_time_fraction"] for record in records})
    result = {}
    for cell in CELLS:
        ratios = {}
        for paper_time in paper_times:
            chen = median_over_seeds(records, cell, "chen_mangasarian", paper_time, "high_band_residual_power")
            split = median_over_seeds(records, cell, "highest_order_split", paper_time, "high_band_residual_power")
            ratios[paper_time] = None if chen is None or split in (None, 0.0) else chen / split
        chen_forcing = median_over_seeds(records, cell, "chen_mangasarian", S1_S3_PAPER_TIME, "high_band_forcing_power")
        constant_forcing = median_over_seeds(records, cell, "constant_in_time", S1_S3_PAPER_TIME,
                                             "high_band_forcing_power")
        s1_deviation = (None if chen_forcing is None or not constant_forcing
                        else abs(chen_forcing - constant_forcing) / constant_forcing)
        early, later = ratios.get(S1_S3_PAPER_TIME), ratios.get(S2_LATER_PAPER_TIME)
        result[cell] = {
            "residual_ratio_by_paper_time": {f"{key:g}": value for key, value in ratios.items()},
            "s1_relative_deviation_of_forcing": s1_deviation,
            "s1_holds": None if s1_deviation is None else s1_deviation <= FORCING_RELATIVE_TOLERANCE,
            "s2_holds": None if early is None or later is None else early > later,
            "s3_holds": None if early is None else early >= RESIDUAL_RATIO_THRESHOLD,
        }
    return result


def render_figure(run_directory: Path) -> Path:
    """Median high-band residual power against the paper time, per role and cell."""
    summary = yaml.safe_load((run_directory / SUMMARY_FILENAME).read_text())
    records = summary["per_run"]
    figure, axes = plt.subplots(1, len(CELLS), figsize=(10.0, 4.0))
    titles = {"g2_bernoulli_full": "(a) Order two, $G_1$", "g3_bernoulli_full": "(b) Order four, $G_2$"}
    for axis, cell in zip(axes, CELLS):
        for role in ("chen_mangasarian", "highest_order_split", "constant_in_time"):
            paper_times = sorted({r["paper_time_fraction"] for r in records if r["cell"] == cell and r["role"] == role})
            if not paper_times:
                continue
            residual = [median_over_seeds(records, cell, role, tau, "high_band_residual_power") for tau in paper_times]
            forcing = [median_over_seeds(records, cell, role, tau, "high_band_forcing_power") for tau in paper_times]
            # Solid: residual of the trained fields (measured); dotted: closed-form forcing, for context.
            axis.loglog(paper_times, residual, "-o", color=ROLE_COLOURS[role], markersize=4,
                        label=f"{ROLE_LABELS[role]}: trained residual")
            if all(value and value > 0 for value in forcing):
                axis.loglog(paper_times, forcing, ":", color=ROLE_COLOURS[role],
                            label=f"{ROLE_LABELS[role]}: forcing of the extension")
        axis.set_title(titles[cell])
        axis.set_xlabel(r"Paper time $t/T$ after the datum")
        axis.set_ylabel(f"Power, ${HIGH_BAND[0]}\\leq k\\leq {HIGH_BAND[1]}$")
        axis.grid(True, which="both", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2,
                           fontsize=8, frameon=True)
    formula = (r"Residual power: $\sum_{32\leq k\leq 256}|\widehat{\mathcal{L}\Phi^h_\theta}(k,t)|^2$ of the "
               r"trained field, median over 3 seeds; forcing power: $\sum_{32\leq k\leq 256}|\widehat{\mathcal{L}h}(k,t)|^2$"
               "\n"
               r"of the extension in closed form; Fourier coefficients of the residual on 1024 equally spaced points.")
    figure.tight_layout(rect=[0, 0.16, 1, 1])
    path = run_directory / FIGURE_FILENAME
    finalize_figure(figure, path, legends=[legend], formula=formula, axes=list(axes))
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, action="append", default=[],
                        help="Series folder holding per-run directories (repeatable).")
    parser.add_argument("--paper-time-fractions", type=str,
                        default=",".join(f"{value:g}" for value in DEFAULT_PAPER_TIME_FRACTIONS))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0,
                        help="master seed, logged only: the diagnostic consumes no random number generator")
    parser.add_argument("--debug", action="store_true",
                        help="prepend '_debug_' to the output folder name and evaluate only seed 0")
    parser.add_argument("--replot", type=Path, default=None, metavar="RUN_DIR")
    arguments = parser.parse_args(argv)

    if arguments.replot is not None:
        configure_cli_script_logging(verbose=False)
        LOGGER.info("Replotted figure from saved artefacts: %s", render_figure(arguments.replot.resolve()))
        return 0
    if not arguments.data_root:
        parser.error("at least one --data-root is required")

    import torch

    start_time = time.perf_counter()
    debug_prefix = "_debug_" if arguments.debug else ""
    run_directory = script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_chen_mangasarian_and_split"
    run_directory.mkdir(parents=True, exist_ok=False)
    init_logging(run_dir=run_directory)
    LOGGER.info("Full command line: %s", " ".join(sys.argv))
    LOGGER.info("Runtime: Python %s | numpy %s | torch %s | device %s", sys.version.split()[0], np.__version__,
                torch.__version__, arguments.device)
    log_parsed_args(LOGGER, arguments)
    LOGGER.info("Master seed %d (no random number generator is consumed; each run rebuilds its ansatz from its "
                "own saved master seed)", arguments.seed)
    run_metadata = collect_run_metadata(
        run_dir=run_directory, repo_root=find_repo_root(Path(__file__)), script_name=Path(__file__).stem,
        command=list(sys.argv), params={key: str(value) for key, value in sorted(vars(arguments).items())},
    )
    write_json(run_directory / "run_metadata.json", run_metadata)
    write_command_txt(run_directory / "command.txt", list(sys.argv))

    paper_times = tuple(float(value) for value in arguments.paper_time_fractions.split(","))
    LOGGER.info("Paper times %s (runner slice fractions %s); low band %s, high band %s", paper_times,
                tuple(1.0 - value for value in paper_times), LOW_BAND, HIGH_BAND)
    wanted = {CHEN_MANGASARIAN, CONSTANT_IN_TIME, *HIGHEST_ORDER_SPLIT.values()}
    records, arrays, excluded = [], {}, []
    for data_root in arguments.data_root:
        for run, variant_name in discover_variant_runs(data_root):
            if variant_name not in wanted:
                continue
            metadata = yaml.safe_load((run / "metadata.yaml").read_text())
            if metadata["cell"] not in CELLS or (arguments.debug and int(metadata["seed"]) != 0):
                continue
            run_start = time.perf_counter()
            try:
                measured = measure_run(run, variant_name, paper_times, arguments.device)
            except Exception as exception:  # noqa: BLE001 - reported, the run is excluded
                LOGGER.error("%s / %s not admissible: %s", run.name, variant_name, exception)
                excluded.append({"run": run.name, "variant": variant_name, "reason": str(exception)})
                continue
            for row in measured["rows"]:
                record = {
                    "cell": measured["cell"], "seed": measured["seed"], "variant": variant_name,
                    "role": ROLE_OF_VARIANT[variant_name], "run": run.name,
                    "paper_time_fraction": row["paper_time_fraction"],
                    "low_band_residual_power": band_power(row["wavenumber_bins"], row["residual_power"], LOW_BAND),
                    "high_band_residual_power": band_power(row["wavenumber_bins"], row["residual_power"], HIGH_BAND),
                    "low_band_forcing_power": band_power(row["wavenumber_bins"], row["forcing_power"], LOW_BAND),
                    "high_band_forcing_power": band_power(row["wavenumber_bins"], row["forcing_power"], HIGH_BAND),
                }
                if not all(np.isfinite(value) for key, value in record.items() if key.endswith("_power")):
                    raise ValueError(f"non-finite power for {run.name} / {variant_name}")
                records.append(record)
                key = f"{measured['cell']}__{variant_name}__seed{measured['seed']}__tau{row['paper_time_fraction']:g}"
                arrays[f"residual_power__{key}"] = row["residual_power"]
                arrays[f"forcing_power__{key}"] = row["forcing_power"]
                arrays["wavenumber_bins"] = row["wavenumber_bins"]
                LOGGER.info("%s seed %d %-24s tau %-6g residual high %.3e low %.3e | forcing high %.3e low %.3e",
                            measured["cell"], measured["seed"], variant_name, row["paper_time_fraction"],
                            record["high_band_residual_power"], record["low_band_residual_power"],
                            record["high_band_forcing_power"], record["low_band_forcing_power"])
            LOGGER.info("%s / %s done in %.1f s", run.name, variant_name, time.perf_counter() - run_start)

    summary = {
        "paper_time_fractions": list(paper_times),
        "low_band": list(LOW_BAND),
        "high_band": list(HIGH_BAND),
        "forcing_relative_tolerance": FORCING_RELATIVE_TOLERANCE,
        "residual_ratio_threshold": RESIDUAL_RATIO_THRESHOLD,
        "n_records": len(records),
        "excluded_runs": excluded,
        "verdicts": verdicts(records),
        "per_run": records,
    }
    np.savez(run_directory / SPECTRA_FILENAME, **arrays)
    (run_directory / SUMMARY_FILENAME).write_text(yaml.safe_dump(summary, sort_keys=False))
    LOGGER.info("Saved %s and %s under %s", SPECTRA_FILENAME, SUMMARY_FILENAME, run_directory)
    LOGGER.info("Saved figure %s", render_figure(run_directory))
    LOGGER.info("Verdicts: %s", summary["verdicts"])
    LOGGER.info("Total wall-clock time %.1f s", time.perf_counter() - start_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
