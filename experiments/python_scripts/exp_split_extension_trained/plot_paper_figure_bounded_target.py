#!/usr/bin/env python
r"""Paper figure: the split bounds the free-field target, the extended-datum forms do not.

Regenerates, from the *saved* slice artefacts of an
``ablation_split_extension_trained`` run (no retraining), the single-column
figure used in the Bermudan report to validate Proposition 5(iii) (bounded
target) against Proposition 6 (unbounded target), together with its spectral
reading.

The figure is centred on the singular point ``x_star`` of the datum, with the
periodic wrap of the circle handled explicitly, so the first-derivative
discontinuity (and the associated curvature spike of the band-limited datum)
sits at the centre rather than at a plot edge.

Four panels (2x2), read from each variant's ``slices.npz``:

* (A) the datum ``g`` (band-limited tent) and its curvature ``g''``;
* (B) the free-field terminal trace ``|phi(., t_k)|`` on a log axis, exact
  target ``phi^star`` dashed, trained network ``phi_theta`` solid;
* (C) the free-field spectrum ``|hat phi(k)|`` on log-log axes, with reference
  slopes ``|k|`` (first order, split) and ``k^2`` (second order, extended-datum
  forms) and the datum band limit ``k_star`` marked;
* (D) the spatial solution error ``|u_theta - u_ref|`` at ``t_0`` on a log axis.

Plot convention (repository standard): solid = trained network, dashed =
analytical reference/exact target. One extension per colour.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = [
    ("variant_constant_in_time", r"Non-convex  $h_k=v_k$", "C3"),
    ("variant_convex_raw", r"Convex  $h_k=(1-d_k)\,v_k$", "C1"),
    ("variant_split_diffusion", r"Split  $h_k=e^{sA}v_k$", "C0"),
]

DEFAULT_RUN = (
    "data/ablation_split_extension_trained/"
    "2026-07-12-01-17-05-804637Z_g2_bernoulli_bandlimited_iters20000_seed0"
)


def _periodic_centre(x_grid: np.ndarray, corner_point: float):
    xc = np.mod(x_grid - corner_point + np.pi, 2.0 * np.pi) - np.pi
    order = np.argsort(xc)
    return xc[order], order


def _spectrum(field: np.ndarray, x_grid: np.ndarray):
    wavenumbers = np.fft.rfftfreq(len(field), d=(x_grid[1] - x_grid[0])) * 2.0 * np.pi
    amplitude = np.abs(np.fft.rfft(field)) / len(field)
    return wavenumbers, amplitude


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--corner-point", type=float, default=0.0)
    parser.add_argument("--half-window", type=float, default=0.18)
    parser.add_argument("--band-limit", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    reference_slice = np.load(
        os.path.join(args.run_dir, "variant_split_diffusion", "slices.npz"),
        allow_pickle=True,
    )
    x_grid = reference_slice["x"]
    xc, order = _periodic_centre(x_grid, args.corner_point)
    window = np.abs(xc) <= args.half_window

    plt.rcParams.update({"font.size": 8.5})
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    ax_datum, ax_trace = axes[0, 0], axes[0, 1]
    ax_spectrum, ax_error = axes[1, 0], axes[1, 1]

    # Panel A: datum and its curvature.
    datum = reference_slice["g"][order]
    curvature = np.gradient(np.gradient(datum, xc), xc)
    ax_datum.plot(xc[window], datum[window], "k-", lw=1.3)
    twin = ax_datum.twinx()
    twin.plot(xc[window], curvature[window], ":", color="0.5", lw=1.1)
    ax_datum.set_title(r"(A) Datum $g$ (solid), curvature $g''$ (dotted)")
    ax_datum.set_xlabel(r"$x-x^{\ast}$")
    ax_datum.set_ylabel(r"$g$")
    twin.set_ylabel(r"$g''$")

    for variant_dir, label, colour in VARIANTS:
        data = np.load(
            os.path.join(args.run_dir, variant_dir, "slices.npz"),
            allow_pickle=True,
        )
        phi_star = data["phi_star_tT"]
        phi_theta = data["phi_theta_tT"]
        solution_error = (
            np.abs(data["u_pred_t0"][order] - data["u_ref_t0"][order]) + 1e-12
        )
        # Panel B: free-field terminal trace, real space, centred.
        ax_trace.semilogy(
            xc[window], np.abs(phi_star[order][window]) + 1e-12, "--", color=colour, lw=1.4
        )
        ax_trace.semilogy(
            xc[window], np.abs(phi_theta[order][window]) + 1e-12, "-", color=colour, lw=1.4,
            label=label,
        )
        # Panel C: free-field spectrum, log-log.
        wavenumbers, amp_star = _spectrum(phi_star, x_grid)
        _, amp_theta = _spectrum(phi_theta, x_grid)
        positive = wavenumbers > 0
        ax_spectrum.loglog(wavenumbers[positive], amp_star[positive] + 1e-14, "--", color=colour, lw=1.3)
        ax_spectrum.loglog(wavenumbers[positive], amp_theta[positive] + 1e-14, "-", color=colour, lw=1.3)
        # Panel D: solution error at t0.
        ax_error.semilogy(xc[window], solution_error[window], "-", color=colour, lw=1.4, label=label)

    for axis in (ax_datum, ax_trace, ax_error):
        axis.axvline(0.0, color="0.4", ls=":", lw=0.9)

    ax_trace.set_title(r"(B) Free-field trace $|\Psi(\cdot,t_k)|$")
    ax_trace.set_xlabel(r"$x-x^{\ast}$")
    ax_trace.set_ylabel(r"$|\Psi(\cdot,t_k)|$")
    ax_spectrum.set_title(r"(C) Free-field spectrum $|\hat\Psi(k)|$")
    ax_spectrum.set_xlabel(r"$k$")
    ax_spectrum.set_ylabel(r"$|\hat\Psi(k)|$")
    ax_error.set_title(r"(D) Solution error $|u_\theta-u^{\mathrm{ref}}|$ at $t_0$")
    ax_error.set_xlabel(r"$x-x^{\ast}$")
    ax_error.set_ylabel(r"$|u_\theta-u^{\mathrm{ref}}|(\cdot,t_0)$")

    handles, labels = ax_trace.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        r"Solid: trained $\Psi_\theta$.  Dashed: exact target $\Psi^{\ast}$.", fontsize=8.5, y=1.00
    )
    figure.tight_layout(rect=[0, 0.05, 1, 0.98])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print("saved:", args.output)


if __name__ == "__main__":
    main()
