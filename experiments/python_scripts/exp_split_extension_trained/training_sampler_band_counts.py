#!/usr/bin/env python3
r"""Points of a finished exact-trace run on each band, by replay of its seeded samplers.

On the exact-trace cells, every spectral field is an
:class:`~learning_option_pricing.pde.ExactTraceAdaptiveBandField`: a point at time
:math:`s` since the datum is evaluated on the smallest power-of-two band
:math:`K\ge K_0` with :math:`s\,(cK^{r}-b)\ge Z`, up to a maximum band beyond which a
reported truncation applies. The field logs the number of points above the maximum
band at its first activation only (WARNING), then at DEBUG level, so the total over a
training run is absent from a log written at INFO level.

This diagnostic recovers it without training. The interior sampler and the validation
set are drawn from their own CPU generators, independently of the network, so calling
:func:`make_samplers` and :func:`draw_validation_points` with the run's derived seeds,
and following the training loop's draw rule (:func:`is_logged_iteration`), reproduces
the training times exactly. For each run and each exact-trace field it reports the
number of training and validation points on each band, the number above the maximum
band, the first training iteration with such a point, and the band required at given
probe times.

Post-processing only (no network is built beyond the extension fields): route it to a
non-billed partition (Jean Zay ``prepost``).
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _split_extension_catalogue as catalogue  # noqa: E402
from ablation_split_extension_trained import (  # noqa: E402
    build_ansatz,
    build_problem,
    derive_seed,
    draw_validation_points,
    is_logged_iteration,
    logging_interval,
    make_samplers,
    problem_for_variant,
)

LOGGER = logging.getLogger("training_sampler_band_counts")
DEFAULT_PROBE_TIMES = (3e-12,)


def replay_training_times(problem: dict, hparams: dict, master_seed: int) -> np.ndarray:
    """Interior training times of a run, one row per iteration, as drawn during its training."""
    import torch

    sample_interior, sample_terminal = make_samplers(
        problem, hparams, sampler_seed=derive_seed(master_seed, "sampler"),
        device=torch.device("cpu"),
    )
    num_iterations = int(hparams["num_iterations"])
    log_every = logging_interval(num_iterations)
    training_times = np.empty((num_iterations, int(hparams["n_interior"])), dtype=np.float32)
    for iteration in range(1, num_iterations + 1):
        _, interior_times = sample_interior()
        training_times[iteration - 1] = interior_times.detach().numpy()
        if is_logged_iteration(iteration, log_every, num_iterations):
            sample_terminal()
    return training_times


def band_counts(field, times_by_row: np.ndarray, terminal_time: float) -> dict:
    """Points per band of ``field`` for the times in ``times_by_row`` (one row per iteration).

    The time since the datum is formed as the field forms it: the float32 time cast to
    float64 and subtracted from the terminal time.
    """
    points_per_band: collections.Counter = collections.Counter()
    capped_points, first_capped, datum_slice_points = 0, None, 0
    smallest_positive = float("inf")
    for row_index, row in enumerate(times_by_row):
        time_since_datum = terminal_time - row.astype(np.float64)
        datum_slice_points += int(np.count_nonzero(time_since_datum <= 0.0))
        positive = time_since_datum[time_since_datum > 0.0]
        if positive.size == 0:
            continue
        smallest_positive = min(smallest_positive, float(np.min(positive)))
        bands, capped = field.required_band(positive)
        points_per_band.update(bands.tolist())
        capped_points += int(np.count_nonzero(capped))
        if first_capped is None and np.any(capped):
            first_capped = {"row": row_index + 1, "smallest_time_since_datum": float(np.min(positive[capped]))}
    return {
        "points": int(times_by_row.size),
        "points_on_datum_slice": datum_slice_points,
        "smallest_positive_time_since_datum": smallest_positive,
        "points_above_maximum_band": capped_points,
        "first_row_above_maximum_band": first_capped,
        "points_per_band": {int(band): int(count) for band, count in sorted(points_per_band.items())},
    }


def run_band_counts(run_directory: Path, probe_times: tuple[float, ...]) -> dict | None:
    """Band counts of every exact-trace field of one run directory, or ``None`` off the exact-trace cells."""
    from learning_option_pricing.pde import ExactTraceAdaptiveBandField

    metadata = yaml.safe_load((run_directory / "metadata.yaml").read_text())
    cell_name, master_seed, hparams = metadata["cell"], int(metadata["seed"]), metadata["hparams"]
    problem = build_problem(cell_name)
    if not isinstance(problem["exact_field"], ExactTraceAdaptiveBandField):
        LOGGER.info("%s: cell %s has no exact-trace field; skipped", run_directory.name, cell_name)
        return None
    terminal_time = float(problem["terminal_time"])
    training_times = replay_training_times(problem, hparams, master_seed)
    _, validation_times = draw_validation_points(
        problem, hparams, validation_seed=derive_seed(master_seed, "validation"))
    validation_times = validation_times.numpy()[None, :]
    LOGGER.info("%s: replayed %d training iterations of %d points (seed %d)",
                run_directory.name, *training_times.shape, master_seed)
    fields = {}
    for variant in catalogue.variants_for_cell(cell_name):
        variant_problem = problem_for_variant(problem, variant)
        _, extension_field = build_ansatz(
            variant, variant_problem, hparams, model_seed=derive_seed(master_seed, "model_init"))
        if not isinstance(extension_field, ExactTraceAdaptiveBandField):
            continue
        fields[variant["name"]] = {
            "decay_order": int(extension_field.decay_order),
            "decay_coefficient": float(extension_field.decay_coefficient),
            "decay_rate_offset": float(extension_field.decay_rate_offset),
            "maximum_band": int(extension_field.maximum_band),
            "band_at_probe_times": {
                float(time): int(extension_field.required_band(np.array([time]))[0][0]) for time in probe_times},
            "training": band_counts(extension_field, training_times, terminal_time),
            "validation": band_counts(extension_field, validation_times, terminal_time),
        }
    return {"run": run_directory.name, "cell": cell_name, "seed": master_seed,
            "run_code_revision": metadata.get("git", {}).get("commit"), "fields": fields}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--series-directory", type=Path, required=True,
                        help="Series folder holding the per-run directories (metadata.yaml each).")
    parser.add_argument("--probe-times", type=str, default=",".join(map(str, DEFAULT_PROBE_TIMES)),
                        help="Comma-separated times since the datum at which the required band is reported.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output YAML (default <series>/diagnostics/training_sampler_band_counts.yaml).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s")

    import torch
    LOGGER.info("command: %s", " ".join([Path(__file__).name, *(argv if argv is not None else sys.argv[1:])]))
    LOGGER.info("python %s, numpy %s, torch %s, device cpu (no training)",
                sys.version.split()[0], np.__version__, torch.__version__)
    probe_times = tuple(float(value) for value in args.probe_times.split(","))

    run_directories = sorted(path for path in args.series_directory.glob("*_iters*_seed*")
                             if path.is_dir() and not path.name.startswith("_debug_"))
    results = [result for result in (run_band_counts(path, probe_times) for path in run_directories)
               if result is not None]
    out_path = args.out or args.series_directory / "diagnostics" / "training_sampler_band_counts.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump({"series": str(args.series_directory), "probe_times": list(probe_times),
                                        "runs": results}, sort_keys=False))
    LOGGER.info("wrote %s", out_path)

    for result in results:
        for name, record in result["fields"].items():
            training = record["training"]
            LOGGER.info("%s seed %d %-28s training: %d points, %d above the maximum band, first %s, "
                        "largest band used %d; validation: %d above; band at probe times %s",
                        result["cell"], result["seed"], name, training["points"],
                        training["points_above_maximum_band"], training["first_row_above_maximum_band"],
                        max(training["points_per_band"]), record["validation"]["points_above_maximum_band"],
                        record["band_at_probe_times"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
