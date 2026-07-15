#!/usr/bin/env python3
r"""Convergence of the measured cancellation cutoff in the number of time slices.

The stage-two cancellation ratio :math:`\rho_k` of the methodology report is a
mean over five interior time slices,
:math:`(0.1, 0.3, 0.5, 0.7, 0.9)\,T`, and the cutoff :math:`k^\star` is the
wavenumber at which the seven-point running mean of :math:`\rho_k` first reaches
:math:`\tfrac12`. The report justifies the *lower* bound on the slice count --
fewer slices lose the crossing -- but does not establish that five is
*converged*: that a denser set leaves :math:`k^\star` unchanged. This diagnostic
settles that, without retraining.

For every saved ``model.pt`` of a stage-two run it rebuilds the ansatz
deterministically (:func:`build_ansatz`), loads the trained weights, and calls
the production spectra routine :func:`compute_spectra` at several slice counts
:math:`N`, all spanning the same interior range :math:`[0.1, 0.9]\,T` so that
:math:`N=5` reproduces the production set exactly and larger :math:`N` refine it.
The per-run cutoffs are written to a YAML table together with a per-(cell,
variant) convergence verdict: :math:`k^\star` is deemed converged when it is
constant across all :math:`N\ge5`.

No training and no gradient step on the weights occur -- only forward passes and
the automatic-differentiation of the PDE residual that :func:`compute_spectra`
already performs. The job is therefore a post-processing one and is routed to a
non-billed partition by the launcher.
"""

from __future__ import annotations

import argparse
import glob
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
    build_closed_form_extension,
    build_problem,
    compute_spectra,
    derive_seed,
)

LOGGER = logging.getLogger("slice_count_convergence")

# The interior range is fixed; only the number of slices in it varies. At N = 5
# this is exactly the production set (0.1, 0.3, 0.5, 0.7, 0.9), so the first
# column reproduces the reported cutoff and the rest refine it.
SLICE_RANGE = (0.1, 0.9)
DEFAULT_SLICE_COUNTS = (5, 11, 21, 41)


def slice_fractions(count: int) -> np.ndarray:
    """``count`` interior slice fractions evenly spaced over :data:`SLICE_RANGE`."""
    return np.linspace(SLICE_RANGE[0], SLICE_RANGE[1], count)


def cutoff_at_slice_counts(
    run_directory: Path, variant_name: str, slice_counts, device: str
) -> dict:
    """The cutoff of one trained variant at each requested slice count."""
    import torch

    metadata = yaml.safe_load((run_directory / "metadata.yaml").read_text())
    cell_name = metadata["cell"]
    master_seed = int(metadata["seed"])
    hparams = metadata["hparams"]

    problem = build_problem(cell_name)
    variant = catalogue.variant_by_name(cell_name, variant_name)
    model_seed = derive_seed(master_seed, "model_init")

    model, _ = build_ansatz(variant, problem, hparams, model_seed=model_seed)
    state = torch.load(
        run_directory / f"variant_{variant_name}" / "models" / "model.pt",
        map_location=device,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    closed_form_extension = build_closed_form_extension(variant, problem)

    cutoffs: dict[int, int | None] = {}
    for count in slice_counts:
        spectra = compute_spectra(
            model, problem, variant, closed_form_extension,
            slice_fractions=slice_fractions(count),
        )
        defined = bool(spectra["k_star_defined"][0])
        cutoffs[count] = int(spectra["k_star"][0]) if defined else None
        LOGGER.info(
            "%s / %s: N=%d -> k_star=%s",
            run_directory.name, variant_name, count,
            cutoffs[count] if cutoffs[count] is not None else "absent",
        )
    return {
        "cell": cell_name,
        "seed": master_seed,
        "cutoff_by_slice_count": cutoffs,
    }


def converged(cutoffs: dict[int, int | None], baseline: int = 5) -> bool:
    """Whether the cutoff is constant across all slice counts at or above ``baseline``."""
    at_or_above = [
        value for count, value in cutoffs.items() if count >= baseline
    ]
    return len(set(map(str, at_or_above))) == 1


def discover_variant_runs(data_root: Path):
    """Yield ``(run_directory, variant_name)`` for every saved ``model.pt``."""
    for model_path in sorted(data_root.glob("*/variant_*/models/model.pt")):
        variant_directory = model_path.parent.parent
        run_directory = variant_directory.parent
        if (run_directory / "metadata.yaml").is_file():
            yield run_directory, variant_directory.name.replace("variant_", "")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--data-root", type=Path, required=True,
        help="Folder holding the stage-two per-run directories (each with "
        "metadata.yaml and variant_*/models/model.pt).",
    )
    parser.add_argument(
        "--slice-counts", type=str, default=",".join(map(str, DEFAULT_SLICE_COUNTS)),
        help="Comma-separated interior slice counts to compare (default "
        f"{','.join(map(str, DEFAULT_SLICE_COUNTS))}; 5 reproduces production).",
    )
    parser.add_argument(
        "--variant", type=str, action="append", default=[],
        help="Restrict to these variant names (repeatable); default: all with "
        "a non-zero forcing, since the zero-forcing variants have no cutoff.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
    )

    slice_counts = tuple(int(s) for s in args.slice_counts.split(","))
    if 5 not in slice_counts:
        LOGGER.warning("slice count 5 (production) is not in the requested set")

    import torch
    LOGGER.info("command: %s", " ".join([Path(__file__).name, *(argv or sys.argv[1:])]))
    LOGGER.info("python %s, torch %s, device %s",
                sys.version.split()[0], torch.__version__, args.device)

    rows = []
    for run_directory, variant_name in discover_variant_runs(args.data_root):
        if args.variant and variant_name not in args.variant:
            continue
        try:
            row = cutoff_at_slice_counts(
                run_directory, variant_name, slice_counts, args.device
            )
        except Exception as exception:  # noqa: BLE001 - report and continue
            LOGGER.error("%s / %s failed: %s", run_directory.name, variant_name, exception)
            continue
        row["variant"] = variant_name
        row["run"] = run_directory.name
        row["converged_from_5"] = converged(row["cutoff_by_slice_count"])
        rows.append(row)

    # Per-(cell, variant) verdict across seeds.
    verdict: dict = {}
    for row in rows:
        key = f"{row['cell']}::{row['variant']}"
        verdict.setdefault(key, []).append(row["converged_from_5"])
    summary = {
        "slice_range": list(SLICE_RANGE),
        "slice_counts": list(slice_counts),
        "n_runs": len(rows),
        "per_run": rows,
        "converged_all_runs": bool(rows) and all(r["converged_from_5"] for r in rows),
        "per_variant_converged": {
            key: (all(values) if values else None) for key, values in verdict.items()
        },
    }

    out_path = args.out or (args.data_root / "slice_count_convergence.yaml")
    out_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    LOGGER.info("wrote %s", out_path)

    print("\n=== cutoff k* by interior slice count ===")
    header = ["cell", "variant", "seed"] + [f"N={c}" for c in slice_counts] + ["converged"]
    print("  ".join(f"{h:>12}" for h in header))
    for row in rows:
        cells = [row["cell"], row["variant"], row["seed"]]
        cells += [
            row["cutoff_by_slice_count"][c]
            if row["cutoff_by_slice_count"][c] is not None else "absent"
            for c in slice_counts
        ]
        cells += ["yes" if row["converged_from_5"] else "NO"]
        print("  ".join(f"{str(c):>12}" for c in cells))
    print(f"\nconverged across all runs: {summary['converged_all_runs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
