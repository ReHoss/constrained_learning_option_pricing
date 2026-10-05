r"""Learned similarity profile (E_theta', Section 5.3 of the note) against the
analytic enrichment (E, Section 5.2), at equal master seed.

Loads the two saved models (never retrains) and evaluates, on one (s, t) grid:

- the price error of each estimator against the Reiner-Rubinstein closed form,
  Phi_theta - V_DO;
- the direct contribution of the profile error to the price,
  E_theta' - E = chi(s) Delta (Lambda_theta'(xi) - erf(xi)), i.e. the error the
  learned estimator would have if its network Psi_theta and regular part h
  coincided with those of E;
- the profile error |Lambda_theta' - erf| and the similarity-equation defect
  |Lambda_theta'' + 2 xi Lambda_theta'| as functions of xi.

The training trajectories are read from the saved artefacts: the learned run's
``history_eps0.yaml`` and, for the analytic run (trained before the history
file existed), the per-iteration lines of its ``training.log``; both runs log
every 1000 iterations, so the two trajectories have the same sampling.

Every evaluated quantity is saved to ``fields.pt`` with a ``summary.yaml``;
``--replot <out_dir>`` rebuilds the four figures from that file alone:

- ``error_maps.png``: |Phi_theta - V_DO| for E and E_theta', and
  |E_theta' - E|, on one logarithmic colour scale;
- ``error_slices.png``: Phi_theta - V_DO and E_theta' - E along s at fixed t;
- ``training_trajectories.png``: losses of both runs; profile error and defect
  integral I of the learned run; gradient norms;
- ``profile_error_vs_xi.png``: |Lambda_theta' - erf| and the pointwise defect.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/compare_learned_and_analytic_similarity_profile.py \
        --learned-run-dir data/pilot_down_and_out_put/<learned run> \
        --analytic-run-dir data/pilot_down_and_out_put/<enrichment run>

    python3 experiments/python_scripts/exp_barrier_option/compare_learned_and_analytic_similarity_profile.py \
        --replot data/compare_learned_and_analytic_similarity_profile/<out_dir>
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learning_option_pricing.pricing.barrier import reiner_rubinstein_down_and_out_put  # noqa: E402
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import find_repo_root, get_git_metadata, script_data_dir  # noqa: E402
from pilot_down_and_out_put import DEVICE, load_trained_model, read_run_metadata  # noqa: E402

logger = logging.getLogger("compare_learned_and_analytic_similarity_profile")

DEFAULT_LEARNED_RUN_DIR = (
    "data/pilot_down_and_out_put/"
    "20261005_152619_iters50000_eps0_seed0_learnedprofile_blackscholes_d00.1_d10.3_w32_l2"
)
DEFAULT_ANALYTIC_RUN_DIR = (
    "data/pilot_down_and_out_put/20260921_003313_iters50000_eps0_seed0_enrichment_blackscholes_d00.1_d10.3"
)
DEFAULT_TIMES = [0.0, 0.5, 0.9, 0.99]
#: Floor of the logarithmic colour scale of the error maps; values below it are
#: drawn in the lowest colour (a display choice, the saved fields are unclamped).
ERROR_MAP_COLOUR_FLOOR = 1e-6

ANALYTIC_COLOUR = "#0072B2"   # Okabe-Ito blue: E (analytic profile)
LEARNED_COLOUR = "#D55E00"    # Okabe-Ito vermilion: E_theta' (learned profile)
PROFILE_COLOUR = "#009E73"    # Okabe-Ito green: E_theta' - E (profile contribution)
DEFECT_COLOUR = "#CC79A7"     # Okabe-Ito purple: defect integral / pointwise defect

FORMULA_TEXT_PRICE = (
    r"$\Phi_\theta=E+h+d_{\partial_pQ}\Psi_\theta$ with $E=\chi(s)\,\Delta\,\mathrm{erf}(\xi)$ (analytic profile, Section 5.2) "
    r"or $E_{\theta'}=\chi(s)\,\Delta\,\Lambda_{\theta'}(\xi)$ (learned profile, Section 5.3);  "
    r"$\Lambda_{\theta'}=\Lambda_0+\omega\,g_{\theta'}$, $\Lambda_0(\xi)=1-e^{-\xi}$, $\omega(\xi)=e^{-\xi}(1-e^{-\xi})$"
    "\n"
    r"$\xi(s,t)=\frac{\ln(s/B)}{\sigma\sqrt{2(T-t)}}$, $\Delta=K-B$, $h=\pi-\chi\,\pi(B,\cdot)$ with $\pi$ the Black-Scholes put, "
    r"$d_{\partial_pQ}=(T-t)(s-B)$;  $V_{DO}$: Reiner-Rubinstein closed form;  "
    r"$E_{\theta'}-E=\chi\,\Delta\,(\Lambda_{\theta'}-\mathrm{erf})(\xi)$: profile contribution"
    "\n"
    r"Equal master seed (0): identical initialisation of $\Psi_\theta$ and identical collocation sequence; 1 seed, not a seed average"
)
FORMULA_TEXT_TRAJECTORIES = (
    r"Loss: $\frac{1}{n_f}\sum_i(\mathcal{L}^{BS}\Phi_\theta)^2(s_i,t_i)$ on one batch of $n_f=4096$ uniform collocation points (single-batch values, logged every 1000 iterations);  "
    r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\partial_sV-rV$"
    "\n"
    r"$\|\Lambda_{\theta'}-\mathrm{erf}\|_{L^\infty}$ and $I=\int_0^\infty(\Lambda_{\theta'}''+2\xi\Lambda_{\theta'}')^2d\xi$ (equation (32)) on a uniform grid of $[0,20]$;  "
    r"$|\nabla_{\theta'}|$: gradient norm on the parameters of $g_{\theta'}$, $|\nabla|$: on all parameters;  dotted: iteration of the best loss (restored state)"
)
FORMULA_TEXT_PROFILE = (
    r"$\Lambda_{\theta'}=\Lambda_0+\omega\,g_{\theta'}$ (Definition 9), restored best-loss state;  $\Lambda=\mathrm{erf}$: unique bounded solution of "
    r"$\Lambda''+2\xi\Lambda'=0$, $\Lambda(0)=0$, $\Lambda(+\infty)=1$;  $\Lambda_0(\xi)=1-e^{-\xi}$: base profile (initialisation)"
)

_LOG_LINE = re.compile(r"iter\s+(\d+)/\d+\s+loss=([0-9.eE+-]+)\s+\|grad\|=([0-9.eE+-]+)")


def read_loss_trajectory_from_log(log_path: Path) -> dict:
    """(iteration, loss, gradient norm) of every logged iteration of a pilot
    ``training.log`` -- the only saved trajectory of runs trained before the
    pilot wrote ``history_eps<E>.yaml``."""
    history = {"iter": [], "loss": [], "grad_norm": []}
    for line in log_path.read_text().splitlines():
        match = _LOG_LINE.search(line)
        if match:
            history["iter"].append(int(match.group(1)))
            history["loss"].append(float(match.group(2)))
            history["grad_norm"].append(float(match.group(3)))
    if not history["iter"]:
        raise ValueError(f"no logged iteration found in {log_path}")
    return history


def _summary(run_dir: Path) -> dict:
    with open(run_dir / "summary_eps0.yaml") as f:
        return yaml.safe_load(f)


def evaluate_fields(learned_run_dir: Path, analytic_run_dir: Path, n_s: int, n_t: int, s_max: float,
                    times: list[float], n_xi: int) -> dict:
    """Every field the figures need, as CPU float64 tensors and plain lists."""
    learned_meta = read_run_metadata(learned_run_dir)
    analytic_meta = read_run_metadata(analytic_run_dir)
    contract = learned_meta["contract"]
    if contract != analytic_meta["contract"]:
        raise ValueError(f"the two runs price different contracts: {contract} vs {analytic_meta['contract']}")
    K, B, r, sigma, T = (contract[k] for k in ("K", "B", "r", "sigma", "T"))
    learned_model = load_trained_model(learned_run_dir, 0.0, learned_meta)
    analytic_model = load_trained_model(analytic_run_dir, 0.0, analytic_meta)
    if not getattr(learned_model.g2, "similarity_profile_is_learned", False):
        raise ValueError(f"{learned_run_dir} is not a learned-similarity-profile run.")
    if getattr(analytic_model.g2, "similarity_profile_is_learned", False) or not hasattr(analytic_model.g2, "enrichment"):
        raise ValueError(f"{analytic_run_dir} is not an analytic-enrichment run.")
    dtype = torch.get_default_dtype()

    def price_and_profile_part(model, s: torch.Tensor, t: torch.Tensor):
        with torch.no_grad():
            x = torch.stack([s, t], dim=1).to(DEVICE).to(dtype)
            price = model(x).squeeze(-1).double().cpu()
            enrichment = model.g2.enrichment(x[:, 0], x[:, 1]).double().cpu()
        return price, enrichment

    s_grid = torch.linspace(B + 1e-4, s_max, n_s, dtype=torch.float64)
    t_grid = torch.linspace(0.0, T - 1e-4, n_t, dtype=torch.float64)
    ss, tt = torch.meshgrid(s_grid, t_grid, indexing="ij")
    reference = reiner_rubinstein_down_and_out_put(ss, K, B, r, sigma, T - tt)
    fields = {"s_grid": s_grid, "t_grid": t_grid, "reference": reference}
    for name, model in (("learned", learned_model), ("analytic", analytic_model)):
        price, enrichment = price_and_profile_part(model, ss.reshape(-1), tt.reshape(-1))
        fields[f"{name}_error"] = price.reshape(ss.shape) - reference
        fields[f"{name}_enrichment"] = enrichment.reshape(ss.shape)
    fields["profile_contribution"] = fields["learned_enrichment"] - fields["analytic_enrichment"]

    slice_s = torch.linspace(B + 1e-5, s_max, 2000, dtype=torch.float64)
    slices = []
    for t_value in times:
        t_slice = torch.full_like(slice_s, t_value)
        slice_reference = reiner_rubinstein_down_and_out_put(slice_s, K, B, r, sigma, T - t_slice)
        learned_price, learned_enrichment = price_and_profile_part(learned_model, slice_s, t_slice)
        analytic_price, analytic_enrichment = price_and_profile_part(analytic_model, slice_s, t_slice)
        slices.append({
            "t": t_value,
            "learned_error": learned_price - slice_reference,
            "analytic_error": analytic_price - slice_reference,
            "profile_contribution": learned_enrichment - analytic_enrichment,
        })
    fields["slice_s"] = slice_s
    fields["slices"] = slices

    network = learned_model.g2.similarity_profile_network
    parameter = next(network.parameters())
    xi = torch.linspace(0.0, 8.0, n_xi, dtype=torch.float64)
    with torch.no_grad():
        value, first, second = network.profile_value_and_derivatives(xi.to(device=parameter.device, dtype=parameter.dtype))
    value, first, second = value.double().cpu(), first.double().cpu(), second.double().cpu()
    fields["xi"] = xi
    fields["profile_error"] = value - torch.erf(xi)
    fields["profile_defect"] = second + 2.0 * xi * first
    base = 1.0 - torch.exp(-xi)
    fields["base_profile_error"] = base - torch.erf(xi)
    fields["base_profile_defect"] = (2.0 * xi - 1.0) * torch.exp(-xi)

    with open(learned_run_dir / "history_eps0.yaml") as f:
        fields["learned_history"] = yaml.safe_load(f)
    fields["analytic_history"] = read_loss_trajectory_from_log(analytic_run_dir / "training.log")
    fields["learned_summary"] = _summary(learned_run_dir)
    fields["analytic_summary"] = _summary(analytic_run_dir)
    fields["contract"] = contract
    fields["corner_window"] = learned_meta["hyperparameters"]["corner_window"]
    fields["runs"] = {"learned": str(learned_run_dir), "analytic": str(analytic_run_dir)}
    return fields


def plot_error_maps(fields: dict, path: Path, s_plot_max: float) -> None:
    s = fields["s_grid"].numpy()
    t = fields["t_grid"].numpy()
    shown = s <= s_plot_max
    B = fields["contract"]["B"]
    panels = (
        (fields["analytic_error"], r"$|\Phi_\theta-V_{DO}|$, analytic profile $E$"),
        (fields["learned_error"], r"$|\Phi_\theta-V_{DO}|$, learned profile $E_{\theta'}$"),
        (fields["profile_contribution"], r"$|E_{\theta'}-E|=\chi\Delta|\Lambda_{\theta'}-\mathrm{erf}|(\xi)$"),
    )
    upper = max(float(field[shown].abs().max()) for field, _ in panels)
    norm = LogNorm(vmin=ERROR_MAP_COLOUR_FLOOR, vmax=upper)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, (field, title) in zip(axes, panels):
        magnitude = field.abs().numpy()[shown].clip(min=ERROR_MAP_COLOUR_FLOOR)
        mesh = ax.pcolormesh(t, s[shown], magnitude, shading="auto", cmap="magma", norm=norm)
        ax.axhline(B, color="white", linestyle=":", lw=1.0)
        ax.axhline(fields["contract"]["K"], color="white", linestyle=":", lw=1.0)
        ax.set_xlabel("Calendar time $t$")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("Underlying price $s$")
    fig.subplots_adjust(bottom=0.32, right=0.9, wspace=0.12)
    colourbar_axis = fig.add_axes([0.915, 0.32, 0.012, 0.56])
    fig.colorbar(mesh, cax=colourbar_axis, label=f"Absolute value (log scale, floor {ERROR_MAP_COLOUR_FLOOR:g})")
    finalize_figure(fig, path, formula=FORMULA_TEXT_PRICE, axes=list(axes))


def plot_error_slices(fields: dict, path: Path, s_plot_max: float) -> None:
    s = fields["slice_s"].numpy()
    shown = s <= s_plot_max
    B, K = fields["contract"]["B"], fields["contract"]["K"]
    slices = fields["slices"]
    fig, axes = plt.subplots(1, len(slices), figsize=(4.4 * len(slices), 4.4), sharey=True)
    for ax, data in zip(np.atleast_1d(axes), slices):
        ax.plot(s[shown], data["analytic_error"].numpy()[shown], color=ANALYTIC_COLOUR, lw=1.5,
                label=r"$\Phi_\theta-V_{DO}$, analytic profile $E$")
        ax.plot(s[shown], data["learned_error"].numpy()[shown], color=LEARNED_COLOUR, lw=1.5,
                label=r"$\Phi_\theta-V_{DO}$, learned profile $E_{\theta'}$")
        ax.plot(s[shown], data["profile_contribution"].numpy()[shown], color=PROFILE_COLOUR, lw=1.3,
                label=r"$E_{\theta'}-E=\chi\Delta(\Lambda_{\theta'}-\mathrm{erf})(\xi)$")
        ax.axvline(B, color="black", linestyle=":", lw=0.9)
        ax.axvline(K, color="grey", linestyle=":", lw=0.9)
        ax.axhline(0.0, color="black", linestyle=":", lw=0.6)
        ax.set_title(f"$t = {data['t']:g}$")
        ax.set_xlabel("Underlying price $s$")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Signed difference")
    legend = np.atleast_1d(axes)[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(right=0.82, bottom=0.34)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_TEXT_PRICE, axes=list(np.atleast_1d(axes)))


def plot_training_trajectories(fields: dict, path: Path) -> None:
    learned = fields["learned_history"]
    analytic = fields["analytic_history"]
    learned_best = fields["learned_summary"]["best_iter"]
    analytic_best = fields["analytic_summary"]["best_iter"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    ax.semilogy(analytic["iter"], analytic["loss"], color=ANALYTIC_COLOUR, lw=1.5, label=r"Analytic profile $E$")
    ax.semilogy(learned["iter"], learned["loss"], color=LEARNED_COLOUR, lw=1.5, label=r"Learned profile $E_{\theta'}$")
    ax.axvline(analytic_best, color=ANALYTIC_COLOUR, linestyle=":", lw=1.0)
    ax.axvline(learned_best, color=LEARNED_COLOUR, linestyle=":", lw=1.0)
    ax.set_title("Interior loss (single batch)")
    ax = axes[1]
    ax.semilogy(learned["iter"], learned["similarity_profile_sup_distance_to_error_function"], color=PROFILE_COLOUR,
                lw=1.5, label=r"$\|\Lambda_{\theta'}-\mathrm{erf}\|_{L^\infty}$")
    ax.semilogy(learned["iter"], learned["similarity_profile_defect_integral"], color=DEFECT_COLOUR, lw=1.5,
                label=r"$I$ (equation (32))")
    ax.axvline(learned_best, color=LEARNED_COLOUR, linestyle=":", lw=1.0)
    ax.set_title(r"Learned profile $\Lambda_{\theta'}$")
    ax = axes[2]
    ax.semilogy(analytic["iter"], analytic["grad_norm"], color=ANALYTIC_COLOUR, lw=1.5, label=r"$|\nabla|$, $E$")
    ax.semilogy(learned["iter"], learned["grad_norm"], color=LEARNED_COLOUR, lw=1.5, label=r"$|\nabla|$, $E_{\theta'}$")
    ax.semilogy(learned["iter"], learned["similarity_profile_network_grad_norm"], color=PROFILE_COLOUR, lw=1.5,
                label=r"$|\nabla_{\theta'}|$, $E_{\theta'}$")
    ax.set_title("Gradient norms")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.3, which="both")
    legends = [axis.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), fontsize=8) for axis in axes]
    fig.subplots_adjust(bottom=0.42, wspace=0.28)
    finalize_figure(fig, path, legends=legends, formula=FORMULA_TEXT_TRAJECTORIES, axes=list(axes))


def plot_profile_error_vs_xi(fields: dict, path: Path) -> None:
    xi = fields["xi"].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    ax.semilogy(xi, np.abs(fields["profile_error"].numpy()), color=LEARNED_COLOUR, lw=1.6,
                label=r"$|\Lambda_{\theta'}-\mathrm{erf}|$ (trained)")
    ax.semilogy(xi, np.abs(fields["base_profile_error"].numpy()), color="grey", linestyle=":", lw=1.3,
                label=r"$|\Lambda_0-\mathrm{erf}|$ (initialisation)")
    ax.set_title("Distance to the analytic profile")
    ax = axes[1]
    ax.semilogy(xi, np.abs(fields["profile_defect"].numpy()), color=LEARNED_COLOUR, lw=1.6,
                label=r"$|\Lambda_{\theta'}''+2\xi\Lambda_{\theta'}'|$ (trained)")
    ax.semilogy(xi, np.abs(fields["base_profile_defect"].numpy()), color="grey", linestyle=":", lw=1.3,
                label=r"$|\Lambda_0''+2\xi\Lambda_0'|=|2\xi-1|e^{-\xi}$ (initialisation)")
    ax.set_title("Defect in the similarity equation (zero for erf)")
    for axis in axes:
        axis.set_xlabel(r"Similarity variable $\xi$")
        axis.grid(alpha=0.3, which="both")
    legends = [axis.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=8) for axis in axes]
    fig.subplots_adjust(bottom=0.4, wspace=0.25)
    finalize_figure(fig, path, legends=legends, formula=FORMULA_TEXT_PROFILE, axes=list(axes))


def build_summary(fields: dict) -> dict:
    window = fields["corner_window"]
    B, T = fields["contract"]["B"], fields["contract"]["T"]
    ss, tt = torch.meshgrid(fields["s_grid"], fields["t_grid"], indexing="ij")
    outside = (ss - B).abs() + (T - tt) > window

    def norm(field: torch.Tensor, mask: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(field[mask]))

    reference_norm = norm(fields["reference"], outside)
    return {
        "runs": fields["runs"],
        "grid": {"n_s": len(fields["s_grid"]), "n_t": len(fields["t_grid"]),
                 "s_max": float(fields["s_grid"][-1]), "corner_window": window},
        "relative_l2_outside_corner": {
            "analytic_error": norm(fields["analytic_error"], outside) / reference_norm,
            "learned_error": norm(fields["learned_error"], outside) / reference_norm,
            "profile_contribution": norm(fields["profile_contribution"], outside) / reference_norm,
        },
        "max_abs_whole_grid": {
            name: {"value": float(fields[name].abs().max()),
                   "s": float(ss.reshape(-1)[fields[name].abs().argmax()]),
                   "t": float(tt.reshape(-1)[fields[name].abs().argmax()])}
            for name in ("analytic_error", "learned_error", "profile_contribution")
        },
        "learned_best_iter": fields["learned_summary"]["best_iter"],
        "analytic_best_iter": fields["analytic_summary"]["best_iter"],
    }


def make_figures(fields: dict, out_dir: Path, s_plot_max: float) -> None:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_error_maps(fields, figures / "error_maps.png", s_plot_max)
    plot_error_slices(fields, figures / "error_slices.png", s_plot_max)
    plot_training_trajectories(fields, figures / "training_trajectories.png")
    plot_profile_error_vs_xi(fields, figures / "profile_error_vs_xi.png")
    logger.info(f"Figures written to {figures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learned-run-dir", type=str, default=DEFAULT_LEARNED_RUN_DIR,
                        help="Pilot run directory of --corner-treatment learned_similarity_profile.")
    parser.add_argument("--analytic-run-dir", type=str, default=DEFAULT_ANALYTIC_RUN_DIR,
                        help="Pilot run directory of --corner-treatment enrichment (same seed and configuration).")
    parser.add_argument("--n-s", type=int, default=600, help="Number of s points of the error maps.")
    parser.add_argument("--n-t", type=int, default=400, help="Number of t points of the error maps.")
    parser.add_argument("--s-max", type=float, default=3.0, help="Upper end of the evaluated s range (training domain).")
    parser.add_argument("--s-plot-max", type=float, default=3.0, help="Upper end of the PLOTTED s range.")
    parser.add_argument("--times", nargs="+", type=float, default=DEFAULT_TIMES, help="Calendar times of the slices.")
    parser.add_argument("--n-xi", type=int, default=1601, help="Number of xi points on [0, 8].")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory override.")
    parser.add_argument("--replot", type=str, default=None, metavar="OUT_DIR",
                        help="Rebuild the figures from OUT_DIR/fields.pt without evaluating any model.")
    args = parser.parse_args()

    if args.replot:
        out_dir = Path(args.replot)
    else:
        repo_root = find_repo_root(Path(__file__).resolve())
        learned_run_dir = (repo_root / args.learned_run_dir).resolve()
        analytic_run_dir = (repo_root / args.analytic_run_dir).resolve()
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        seed = read_run_metadata(learned_run_dir)["hyperparameters"]["seed"]
        out_dir = Path(args.out_dir) if args.out_dir else script_data_dir(__file__) / f"{timestamp}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "compare.log")])
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, device {DEVICE}")
    logger.info(f"Output directory: {out_dir}")

    if args.replot:
        fields = torch.load(out_dir / "fields.pt", weights_only=False)
        logger.info(f"--replot: fields read from {out_dir / 'fields.pt'} (no model evaluated)")
    else:
        if read_run_metadata(learned_run_dir)["hyperparameters"]["seed"] != read_run_metadata(analytic_run_dir)["hyperparameters"]["seed"]:
            logger.warning("The two runs have DIFFERENT master seeds: the comparison is not at equal initialisation.")
        logger.info(f"Learned-profile run: {learned_run_dir}")
        logger.info(f"Analytic-enrichment run: {analytic_run_dir}")
        logger.info(f"Grid: n_s={args.n_s}, n_t={args.n_t}, s in (B, {args.s_max}); slices at t = {args.times}")
        fields = evaluate_fields(learned_run_dir, analytic_run_dir, args.n_s, args.n_t, args.s_max, args.times, args.n_xi)
        torch.save(fields, out_dir / "fields.pt")
        logger.info(f"Fields saved -> {out_dir / 'fields.pt'}")
        with open(out_dir / "metadata.yaml", "w") as f:
            yaml.dump({"command": " ".join(sys.argv), "timestamp": datetime.now().astimezone().isoformat(),
                       "git": get_git_metadata(find_repo_root(Path(__file__).resolve())),
                       "arguments": vars(args)}, f, sort_keys=False)

    summary = build_summary(fields)
    with open(out_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f, sort_keys=False)
    for key, value in summary["relative_l2_outside_corner"].items():
        logger.info(f"  relative L2 outside the corner window, {key}: {value:.4e}")
    for key, value in summary["max_abs_whole_grid"].items():
        logger.info(f"  max absolute value on the grid, {key}: {value['value']:.4e} at (s, t) = "
                    f"({value['s']:.4f}, {value['t']:.4f})")
    make_figures(fields, out_dir, args.s_plot_max)


if __name__ == "__main__":
    main()
