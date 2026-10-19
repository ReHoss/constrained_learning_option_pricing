r"""Compare the corner treatments E, E_theta', P, H2 and H2- (and the samplers of their collocation
points) on ONE fixed validation set shared by every run, after choosing again the restored state of
each run among its model snapshots.

Extension, to every corner treatment of the controlled comparison, of the measurements of
``reselect_restored_state_on_fixed_validation_set.py`` (written for E and E_theta' only): same fixed
validation set, same geometric time-to-maturity grid, same price, Delta and Gamma metrics, so that the
numbers of E and E_theta' are reproduced exactly (checked bit for bit at equal thread count). The
present script is self-contained (it imports only the pilot and the library).

Runs and groups. A group is a corner treatment and a collocation sampler, read from ``metadata.yaml``:

- corner treatments: ``enrichment`` (E), ``learned_similarity_profile`` (E_theta'), ``penalty`` (P),
  ``smoothing`` with :math:`\varepsilon = 0.1` (H2) and :math:`\varepsilon = 0.02` (H2-);
- samplers: ``uniform`` (no ``--corner-sampling``), ``corner`` (``--corner-sampling``, law in the
  similarity variables :math:`(\xi, \ln\tau)`), ``triangle`` (``--corner-sampling-law l1_triangle``).

Measurements, with no training, for every run (``models/snapshots_eps<EPSILON>/model_iter<IT>.pt``,
written by the pilot's ``--save-model-snapshot-every``):

1. the selection criterion of every snapshot on fixed sets drawn once for every run:

   - interior term: the weighted interior loss on the corner mixture
     :class:`~learning_option_pricing.pricing.corner_similarity_sampling.CornerSimilaritySampling`
     (fraction 0.5, :math:`\tau` log-uniform on :math:`[10^{-6}, T]`, :math:`\xi` uniform on :math:`[0, 5]`),
     :math:`n = 65\,536`, weights :math:`u/m`, seed ``derive_seed(--validation-master-seed, "validation_set")``;
   - for P only, the full objective of its training: the interior term plus
     :math:`\lambda_T` and :math:`\lambda_B` times the weighted terminal and barrier penalties on a fixed
     boundary set of :math:`65\,536` points per face drawn from the boundary sampler of the run
     (:class:`~learning_option_pricing.pricing.boundary_penalty.BarrierBoundaryPenaltySampling`), seed
     ``derive_seed(--validation-master-seed, "validation_boundary_set")``;

   the selected state is the snapshot of minimum criterion (interior term for the hard modes, full
   objective for P, as in training);
2. for the final state (last snapshot) and the selected state:

   - the interior validation loss, and for P the two penalties and the full objective;
   - the constraint violations :math:`\max_s|\Phi_\theta(s,T) - g(s)|` and :math:`\max_t|\Phi_\theta(B,t)|`
     (:func:`~learning_option_pricing.pricing.boundary_penalty.boundary_constraint_violations`: geometric
     and uniform grids of the distance to the corner on each face, from :math:`10^{-6}`, corner point
     excluded), for every treatment -- by construction zero for E and E_theta' up to float32 rounding,
     and different from zero in the corner layer for H2 and H2-;
   - on each of the 13 slices :math:`\tau = 10^{-6+k/2}` of the geometric time-to-maturity grid (200
     points uniform in :math:`\xi\in[0,5]`, 100 points uniform on the rest of :math:`(B, s_\infty]`,
     prices rounded to float32): :math:`\max_s|\Phi_\theta - V_{DO}|`, the relative Delta error
     :math:`\max_s|\partial_s\Phi_\theta - \partial_s V_{DO}| / \max_s|\partial_s V_{DO}|`, the relative
     Gamma error defined in the same way with :math:`\partial_{ss}`, and the relative Gamma error on the
     window :math:`B < s \le B + 0.1` (``--near-barrier-window``); the point :math:`s = B` is excluded from
     the Delta and Gamma errors; maxima over the 13 slices;
   - for E_theta': :math:`\|\Lambda_{\theta'} - \operatorname{erf}\|_{L^\infty}` on the uniform grid of
     :math:`[0, 20]` (4001 points) of the pilot, and the values of :math:`\Lambda_{\theta'}` on the uniform
     grid of :math:`[0, 5]` (1001 points) for the profile figures.

Outputs (``data/compare_corner_treatments_on_fixed_validation_set/<timestamp>_runs<N>_validationseed<S>/``):
``results.pt`` (every number above, per run and per snapshot), ``summary.yaml``, ``table.md`` ([min, max]
over the seeds of each group, selected and final states), ``evaluation.log`` and the figures under
``figures/``. ``--replot <out_dir>`` rebuilds the table and the figures from ``results.pt`` alone.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/compare_corner_treatments_on_fixed_validation_set.py \
        --run-dirs data/pilot_down_and_out_put/20261013_001259_* data/pilot_down_and_out_put/20261020_014500_*

    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/compare_corner_treatments_on_fixed_validation_set.py \
        --replot data/compare_corner_treatments_on_fixed_validation_set/<out_dir>
"""
from __future__ import annotations

import argparse
import logging
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_option_pricing.pricing.barrier import (  # noqa: E402
    reiner_rubinstein_down_and_out_put, reiner_rubinstein_down_and_out_put_gamma,
)
from learning_option_pricing.pricing.boundary_penalty import (  # noqa: E402
    BarrierBoundaryPenaltySampling, boundary_constraint_violations,
)
from learning_option_pricing.pricing.corner_similarity_sampling import CornerSimilaritySampling  # noqa: E402
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import find_repo_root, get_git_metadata, script_data_dir  # noqa: E402
from pilot_down_and_out_put import (  # noqa: E402
    DEFAULT_VALIDATION_SET_SIZE, DEVICE, boundary_validation_penalties, derive_seed, draw_boundary_validation_set,
    draw_validation_set, load_trained_model, read_run_metadata, validation_loss,
)

logger = logging.getLogger("compare_corner_treatments_on_fixed_validation_set")

#: Geometric time-to-maturity grid (identical to price_error_on_geometric_time_to_maturity_grid.py).
TIME_TO_MATURITY_LEVELS = np.geomspace(1e-6, 1.0, 13)
SLICE_SIMILARITY_VARIABLE_MAX = 5.0
SLICE_NUMBER_OF_SIMILARITY_POINTS = 200
SLICE_NUMBER_OF_REMAINING_POINTS = 100
#: Uniform grid of xi for the profile figures.
PROFILE_FIGURE_SIMILARITY_VARIABLE_MAX = 5.0
PROFILE_FIGURE_NUMBER_OF_POINTS = 1001

# (corner treatment, sampler) -> display. Okabe-Ito: blue family for E, orange family for E_theta'
# (reddish purple for its triangle control), black for P, green family for H2 / H2-.
GROUPS = [
    (("enrichment", "uniform"), {"label": r"$E$, uniform sampler", "table": "E, uniform", "colour": "#0072B2"}),
    (("enrichment", "corner"), {"label": r"$E$, corner sampler", "table": "E, corner", "colour": "#56B4E9"}),
    (("learned_similarity_profile", "uniform"),
     {"label": r"$E_{\theta'}$, uniform sampler", "table": "E_θ′, uniform", "colour": "#D55E00"}),
    (("learned_similarity_profile", "corner"),
     {"label": r"$E_{\theta'}$, corner sampler", "table": "E_θ′, corner", "colour": "#E69F00"}),
    (("learned_similarity_profile", "triangle"),
     {"label": r"$E_{\theta'}$, $\ell^1$-triangle sampler (control)", "table": "E_θ′, ℓ¹ triangle", "colour": "#CC79A7"}),
    (("penalty", "corner"), {"label": r"$P$, corner sampler", "table": "P, corner", "colour": "#000000"}),
    (("smoothing_eps0.1", "corner"),
     {"label": r"$H_2$ ($\varepsilon=0.1$), corner sampler", "table": "H2 (ε = 0.1), corner", "colour": "#009E73"}),
    (("smoothing_eps0.02", "corner"),
     {"label": r"$H_2^-$ ($\varepsilon=0.02$), corner sampler", "table": "H2− (ε = 0.02), corner", "colour": "#7FCDBB"}),
]
GROUP_DISPLAY = dict(GROUPS)
SEED_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]

FORMULA_TEXT = (
    r"$E$, $E_{\theta'}$: $\Phi_\theta=g_1\Psi_\theta+g_2$ with $g_2=\chi(s)\Delta\Lambda(\xi)+\pi-\chi\pi(B,\cdot)$, "
    r"$\Lambda=\mathrm{erf}$ ($E$) or $\Lambda_{\theta'}=\Lambda_0+\omega g_{\theta'}$ ($E_{\theta'}$), $\xi=\frac{\ln(s/B)}{\sigma\sqrt{2\tau}}$, $\tau=T-t$;  "
    r"$H_2$: $g_2=h_\varepsilon=\zeta((s-B)/\varepsilon)V^e$;  "
    r"$P$: $\Phi_\theta=\frac{s_\infty-s}{s_\infty-B}\Psi_\theta+\pi(s_\infty,t)$, terminal and barrier conditions penalised"
    "\n"
    r"Interior validation loss: $\frac{1}{n}\sum_i w_i(\mathcal{L}^{BS}\Phi_\theta)^2(x_i)$, $n=65\,536$ points of the corner mixture "
    r"(fraction 0.5, $\tau$ log-uniform on $[10^{-6},1]$, $\xi$ uniform on $[0,5]$), $w=u/m$, ONE fixed draw for every run"
    "\n"
    r"$P$: full objective = interior term $+\lambda_T\frac{1}{n}\sum_j w_j(\Phi_\theta(s_j,T)-g(s_j))^2+\lambda_B\frac{1}{n}\sum_j w_j\Phi_\theta(B,t_j)^2$, "
    r"$\lambda_T=\lambda_B=1$, fixed boundary set of $65\,536$ points per face; selection on it (interior term alone for the hard modes)"
    "\n"
    r"Samplers: uniform on $(B,s_\infty)\times(0,T)$; corner: the validation mixture; $\ell^1$ triangle: corner half uniform on "
    r"$\{(s-B)+(T-t)\leq0.1\}$, $m=\frac{1}{2}u+\frac{1}{2}\frac{2}{\delta^2}\mathbf{1}_{\mathcal{T}_\delta}$;  "
    r"$\partial_s$, $\partial_{ss}$: automatic derivatives (network float32, $V_{DO}$ float64), $s=B$ excluded"
    "\n"
    r"Comparison caveat: the validation set is the training law of the corner sampler (not of the uniform and $\ell^1$-triangle "
    r"samplers), and $P$ is selected on its full objective; the price, Delta and Gamma errors are measured identically for every run"
)


# ---------------------------------------------------------------------------
# Run classification
# ---------------------------------------------------------------------------

def run_group(meta: dict) -> tuple[str, str]:
    """(corner treatment label, sampler) of a run, from its metadata."""
    hyper = meta["hyperparameters"]
    treatment = hyper.get("corner_treatment", "smoothing")
    if treatment == "smoothing":
        (epsilon,) = hyper["epsilons"]
        treatment = f"smoothing_eps{epsilon:g}"
    corner_sampling = hyper.get("corner_sampling")
    if not corner_sampling:
        sampler = "uniform"
    else:
        sampler = {"similarity_variables": "corner", "l1_triangle": "triangle"}[
            corner_sampling.get("law", "similarity_variables")]
    if (treatment, sampler) not in GROUP_DISPLAY:
        raise ValueError(f"no display group for corner treatment {treatment!r} with sampler {sampler!r}")
    return treatment, sampler


def run_epsilon(meta: dict) -> float:
    """The epsilon of the run's file names (0 placeholder for the treatments without corner layer)."""
    (epsilon,) = meta["hyperparameters"]["epsilons"]
    return float(epsilon)


def snapshot_paths(run_directory: Path, epsilon: float) -> list[tuple[int, Path]]:
    paths = sorted((run_directory / "models" / f"snapshots_eps{epsilon:g}").glob("model_iter*.pt"))
    return [(int(re.search(r"model_iter(\d+)\.pt", p.name).group(1)), p) for p in paths]


# ---------------------------------------------------------------------------
# Geometric time-to-maturity grid (same points as price_error_on_geometric_time_to_maturity_grid.py)
# ---------------------------------------------------------------------------

def build_slice_points(tau_nominal: float, B: float, sigma: float, s_inf: float, T: float,
                       xi_max: float, n_xi: int, n_rest: int) -> dict:
    """Price points of one time level, rounded to float32 as the network sees them.

    float32(B) = 0.60000002 > B for B = 0.6: the network computes s - B in float32 and sees the xi = 0
    point ON the barrier (output exactly 0), whereas the float64 closed form, floored at s >= B(1 + 1e-6),
    would return the price at B(1 + 1e-6). Such points are therefore mapped to s = B. The closed form is
    evaluated at the effective time to maturity T - float32(T - tau).
    """
    xi_nominal = torch.linspace(0.0, xi_max, n_xi, dtype=torch.float64)
    width = sigma * np.sqrt(2.0 * tau_nominal)
    s_similarity = B * torch.exp(xi_nominal * width)
    inside = s_similarity <= s_inf
    s_points = [s_similarity[inside]]
    s_similarity_end = B * np.exp(xi_max * width)
    if s_similarity_end < s_inf:
        s_points.append(torch.linspace(s_similarity_end, s_inf, n_rest + 1, dtype=torch.float64)[1:])
    s = torch.unique(torch.cat(s_points).to(torch.float32)).to(torch.float64)  # sorted, float32-exact
    on_barrier_for_network = s.to(torch.float32) == torch.tensor(B, dtype=torch.float32)
    s = torch.where(on_barrier_for_network, torch.full_like(s, B), s)
    below_closed_form_floor = (s > B) & (s < B * (1.0 + 1e-6))
    if below_closed_form_floor.any():
        raise ValueError(f"{int(below_closed_form_floor.sum())} points lie in (B, B(1+1e-6)), where the closed form "
                         f"is evaluated at its floor; refine the grid choice (tau = {tau_nominal:g}).")
    t_float32 = torch.tensor(T - tau_nominal, dtype=torch.float32)
    tau_effective = T - float(t_float32.double())
    return {"s": s, "t": float(t_float32), "tau_nominal": tau_nominal, "tau_effective": tau_effective,
            "xi": torch.log(s / B) / (sigma * np.sqrt(2.0 * tau_effective))}


def reference_delta(points: dict, K: float, B: float, r: float, sigma: float) -> torch.Tensor:
    """d V_DO / d s on the slice, by automatic differentiation of the closed form in float64."""
    s = points["s"].clone().requires_grad_(True)
    tau = torch.full_like(s, points["tau_effective"])
    (delta,) = torch.autograd.grad(reiner_rubinstein_down_and_out_put(s, K, B, r, sigma, tau).sum(), s)
    return delta.detach()


def build_slices(K: float, B: float, r: float, sigma: float, T: float, s_inf: float, near_barrier_window: float) -> list[dict]:
    slices = [build_slice_points(float(tau), B, sigma, s_inf, T, SLICE_SIMILARITY_VARIABLE_MAX,
                                 SLICE_NUMBER_OF_SIMILARITY_POINTS, SLICE_NUMBER_OF_REMAINING_POINTS)
              for tau in TIME_TO_MATURITY_LEVELS]
    for points in slices:
        tau = torch.full_like(points["s"], points["tau_effective"])
        points["reference"] = reiner_rubinstein_down_and_out_put(points["s"], K, B, r, sigma, tau)
        points["reference_delta"] = reference_delta(points, K, B, r, sigma)
        points["reference_gamma"] = reiner_rubinstein_down_and_out_put_gamma(points["s"], K, B, r, sigma, tau)
        points["above_barrier"] = points["s"] > B
        points["near_barrier"] = points["above_barrier"] & (points["s"] <= B + near_barrier_window)
    return slices


def price_error_per_slice(model, slices: list[dict]) -> list[dict]:
    dtype = next(model.parameters()).dtype
    per_slice = []
    for points in slices:
        s = points["s"].to(DEVICE).to(dtype).requires_grad_(True)
        x = torch.stack([s, torch.full_like(s, points["t"])], dim=1)
        price = model(x).squeeze(-1)
        (delta,) = torch.autograd.grad(price.sum(), s, create_graph=True)
        (gamma,) = torch.autograd.grad(delta.sum(), s)
        error = price.detach().double().cpu() - points["reference"]
        above = points["above_barrier"]
        delta_error = (delta.detach().double().cpu() - points["reference_delta"])[above]
        reference_delta_sup = float(points["reference_delta"][above].abs().max())
        gamma_error_all = gamma.detach().double().cpu() - points["reference_gamma"]
        gamma_error = gamma_error_all[above]
        reference_gamma_sup = float(points["reference_gamma"][above].abs().max())
        near = points["near_barrier"]
        near_barrier_gamma_error = float(gamma_error_all[near].abs().max())
        near_barrier_reference_gamma_sup = float(points["reference_gamma"][near].abs().max())
        index = int(error.abs().argmax())
        per_slice.append({"tau_effective": points["tau_effective"], "max_abs_error": float(error.abs().max()),
                          "argmax_s": float(points["s"][index]), "argmax_xi": float(points["xi"][index]),
                          "max_abs_delta_error": float(delta_error.abs().max()),
                          "reference_delta_sup": reference_delta_sup,
                          "relative_delta_error": float(delta_error.abs().max()) / reference_delta_sup,
                          "max_abs_gamma_error": float(gamma_error.abs().max()),
                          "reference_gamma_sup": reference_gamma_sup,
                          "relative_gamma_error": float(gamma_error.abs().max()) / reference_gamma_sup,
                          "number_of_points_near_barrier": int(near.sum()),
                          "relative_gamma_error_near_barrier": near_barrier_gamma_error / near_barrier_reference_gamma_sup})
    return per_slice


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def selection_terms(model, validation_set: dict, boundary_validation_set: dict | None, hyper: dict,
                    r: float, sigma: float, K: float) -> dict:
    """Interior validation loss, and for P the two penalties and the full objective; ``criterion`` is
    the quantity minimised by the selection (the full objective for P, the interior term otherwise)."""
    interior = validation_loss(model, validation_set, r, sigma)
    if boundary_validation_set is None:
        return {"interior_validation_loss": interior, "criterion": interior}
    terminal, barrier = boundary_validation_penalties(model, boundary_validation_set, K)
    full = interior + hyper["penalty_terminal_weight"] * terminal + hyper["penalty_barrier_weight"] * barrier
    return {"interior_validation_loss": interior, "terminal_validation_penalty": terminal,
            "barrier_validation_penalty": barrier, "full_validation_objective": full, "criterion": full}


def similarity_profile_on_grid(model) -> dict:
    xi = torch.linspace(0.0, PROFILE_FIGURE_SIMILARITY_VARIABLE_MAX, PROFILE_FIGURE_NUMBER_OF_POINTS, dtype=torch.float64)
    parameter = next(model.g2.similarity_profile_network.parameters())
    with torch.no_grad():
        value, _, _ = model.g2.similarity_profile_value_and_derivatives(xi.to(device=parameter.device, dtype=parameter.dtype))
    return {"xi": xi, "profile": value.detach().double().cpu()}


def measure_state(model, terms: dict, slices: list[dict], K: float, B: float, T: float, s_inf: float,
                  learned: bool) -> dict:
    per_slice = price_error_per_slice(model, slices)
    measured = {key: value for key, value in terms.items() if key != "criterion"}
    measured.update({
        "selection_criterion": terms["criterion"],
        "max_price_error_per_slice": per_slice,
        "max_price_error_over_the_grid": max(entry["max_abs_error"] for entry in per_slice),
        "max_delta_error_over_the_grid": max(entry["max_abs_delta_error"] for entry in per_slice),
        "max_relative_delta_error_over_the_grid": max(entry["relative_delta_error"] for entry in per_slice),
        "max_gamma_error_over_the_grid": max(entry["max_abs_gamma_error"] for entry in per_slice),
        "max_relative_gamma_error_over_the_grid": max(entry["relative_gamma_error"] for entry in per_slice),
        "max_relative_gamma_error_near_barrier_over_the_grid": max(
            entry["relative_gamma_error_near_barrier"] for entry in per_slice),
        "constraint_violations": boundary_constraint_violations(model, K, B, T, s_inf),
        "profile_sup_distance_to_error_function": None,
        "similarity_profile_on_grid": None,
    })
    if learned:
        measured["profile_sup_distance_to_error_function"] = model.g2.similarity_profile_diagnostics()[
            "sup_distance_to_error_function"]
        measured["similarity_profile_on_grid"] = similarity_profile_on_grid(model)
    return measured


def evaluate_runs(run_directories: list[Path], validation_master_seed: int, near_barrier_window: float) -> dict:
    metadata = {path: read_run_metadata(path) for path in run_directories}
    contracts = {(tuple(sorted(m["contract"].items())), m["domain"]["s_inf"]) for m in metadata.values()}
    if len(contracts) != 1:
        raise ValueError(f"the runs price different contracts or domains: {contracts}")
    first = next(iter(metadata.values()))
    K, B, r, sigma, T = (first["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
    s_inf = first["domain"]["s_inf"]

    sampling = CornerSimilaritySampling(barrier=B, truncation_price=s_inf, maturity=T, volatility=sigma)
    validation_seed = derive_seed(validation_master_seed, "validation_set")
    validation_set = draw_validation_set(sampling, DEFAULT_VALIDATION_SET_SIZE, validation_seed)
    logger.info(f"Fixed validation set: {DEFAULT_VALIDATION_SET_SIZE} points, seed {validation_seed} "
                f"(derive_seed({validation_master_seed}, 'validation_set')), statistics {validation_set['statistics']}")
    boundary_validation_seed = derive_seed(validation_master_seed, "validation_boundary_set")
    boundary_validation_sets: dict[tuple, dict] = {}  # one per boundary-sampler configuration (one in practice)

    slices = build_slices(K, B, r, sigma, T, s_inf, near_barrier_window)
    logger.info(f"Geometric time-to-maturity grid: {len(slices)} slices, tau effective "
                + str(['%.3e' % points['tau_effective'] for points in slices]))
    logger.info(f"Window near the barrier for the Gamma error: B < s <= B + {near_barrier_window:g}; points per slice: "
                f"{[int(p['near_barrier'].sum()) for p in slices]}")

    runs = []
    for run_directory in run_directories:
        meta = metadata[run_directory]
        hyper = meta["hyperparameters"]
        group = run_group(meta)
        epsilon = run_epsilon(meta)
        learned = hyper.get("corner_treatment") == "learned_similarity_profile"
        boundary_validation_set = None
        if hyper.get("corner_treatment") == "penalty":
            configuration = tuple(hyper[key] for key in ("penalty_smallest_distance_to_barrier",
                                                         "penalty_smallest_time_to_maturity", "penalty_log_uniform_fraction"))
            if configuration not in boundary_validation_sets:
                boundary_sampling = BarrierBoundaryPenaltySampling(
                    barrier=B, truncation_price=s_inf, maturity=T, smallest_distance_to_barrier=configuration[0],
                    smallest_time_to_maturity=configuration[1], log_uniform_fraction=configuration[2])
                boundary_validation_sets[configuration] = draw_boundary_validation_set(
                    boundary_sampling, DEFAULT_VALIDATION_SET_SIZE, boundary_validation_seed)
                logger.info(f"Fixed boundary validation set of P: {DEFAULT_VALIDATION_SET_SIZE} points per face, seed "
                            f"{boundary_validation_seed} (derive_seed({validation_master_seed}, 'validation_boundary_set')), "
                            f"boundary sampler (smallest distance, smallest tau, log-uniform fraction) = {configuration}")
            boundary_validation_set = boundary_validation_sets[configuration]
        model = load_trained_model(run_directory, epsilon, meta)
        snapshots = snapshot_paths(run_directory, epsilon)
        if not snapshots:
            raise FileNotFoundError(f"no model snapshot in {run_directory} (run without --save-model-snapshot-every?)")
        start = time.time()
        iterations, terms_per_snapshot = [], []
        for iteration, path in snapshots:
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            iterations.append(iteration)
            terms_per_snapshot.append(selection_terms(model, validation_set, boundary_validation_set, hyper, r, sigma, K))
        criteria = [terms["criterion"] for terms in terms_per_snapshot]
        selected_index = int(np.argmin(criteria))
        states = {}
        for name, index in (("final", len(iterations) - 1), ("selected", selected_index)):
            model.load_state_dict(torch.load(snapshots[index][1], map_location=DEVICE, weights_only=True))
            states[name] = {"iteration": iterations[index],
                            **measure_state(model, terms_per_snapshot[index], slices, K, B, T, s_inf, learned)}
        with open(run_directory / f"summary_eps{epsilon:g}.yaml") as f:
            run_summary = yaml.safe_load(f)
        runs.append({"run_directory": str(run_directory), "corner_treatment": group[0], "sampler": group[1],
                     "seed": hyper["seed"], "epsilon": epsilon, "snapshot_iterations": iterations,
                     "selection_criteria": criteria,
                     "interior_validation_losses": [terms["interior_validation_loss"] for terms in terms_per_snapshot],
                     "in_training_selection": {"best_iter": run_summary["best_iter"],
                                               "validation_set_seed": (hyper.get("validation_set") or {}).get("seed")},
                     "final": states["final"], "selected": states["selected"]})
        selected = states["selected"]
        violations = selected["constraint_violations"]
        logger.info(
            f"[{group[0]}, {group[1]}, seed {hyper['seed']}] {len(snapshots)} snapshots in {time.time() - start:.0f}s; "
            f"selected iteration {selected['iteration']} (in-training selection {run_summary['best_iter']}); "
            f"selected state: criterion {selected['selection_criterion']:.4e}, interior validation loss "
            f"{selected['interior_validation_loss']:.4e}, max price error {selected['max_price_error_over_the_grid']:.4e}, "
            f"relative Delta error {selected['max_relative_delta_error_over_the_grid']:.4e}, relative Gamma error "
            f"{selected['max_relative_gamma_error_over_the_grid']:.4e}, max|Phi(s,T)-g| "
            f"{violations['terminal_condition_max_abs_violation']:.3e}, max|Phi(B,t)| "
            f"{violations['barrier_condition_max_abs_violation']:.3e}"
            + ("" if not learned else f", ||Lambda-erf||_inf {selected['profile_sup_distance_to_error_function']:.3e}"))
        del model
    return {"runs": runs, "validation_set_seed": validation_seed, "boundary_validation_set_seed": boundary_validation_seed,
            "validation_master_seed": validation_master_seed, "near_barrier_window": near_barrier_window,
            "contract": first["contract"], "s_inf": s_inf}


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def group_runs(results: dict) -> dict:
    grouped = {key: [] for key, _ in GROUPS}
    for run in results["runs"]:
        grouped[(run["corner_treatment"], run["sampler"])].append(run)
    for runs in grouped.values():
        runs.sort(key=lambda run: run["seed"])
    return grouped


def interval(values: list, fmt: str = "{:.3g}") -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    return "[" + fmt.format(min(values)) + ", " + fmt.format(max(values)) + "]"


#: (column header, function of a measured state -> value or None).
TABLE_COLUMNS = [
    ("Interior validation loss", lambda state: state["interior_validation_loss"]),
    ("Full objective (P)", lambda state: state.get("full_validation_objective")),
    ("Max price error", lambda state: state["max_price_error_over_the_grid"]),
    ("Relative Delta error", lambda state: state["max_relative_delta_error_over_the_grid"]),
    ("Relative Gamma error", lambda state: state["max_relative_gamma_error_over_the_grid"]),
    ("Relative Gamma error, B < s ≤ B + 0.1", lambda state: state["max_relative_gamma_error_near_barrier_over_the_grid"]),
    ("max|Φ(s,T) − g(s)|", lambda state: state["constraint_violations"]["terminal_condition_max_abs_violation"]),
    ("max|Φ(B,t)|", lambda state: state["constraint_violations"]["barrier_condition_max_abs_violation"]),
    ("‖Λ_θ′ − erf‖_∞", lambda state: state["profile_sup_distance_to_error_function"]),
]


def build_table(results: dict) -> tuple[str, dict]:
    grouped = group_runs(results)
    sections, summary = [], {}
    for state, title in (("selected", "Selected state (minimum of the selection criterion on the common fixed sets)"),
                         ("final", "Final state (last snapshot)")):
        header = "| Group (seeds) | Iteration | " + " | ".join(name for name, _ in TABLE_COLUMNS) + " |"
        lines = [f"### {title}", "", header, "|" + "---|" * (len(TABLE_COLUMNS) + 2)]
        for key, display in GROUPS:
            runs = grouped[key]
            if not runs:
                continue
            row = {"seeds": [run["seed"] for run in runs], "iteration": [run[state]["iteration"] for run in runs]}
            for name, extract in TABLE_COLUMNS:
                row[name] = [extract(run[state]) for run in runs]
            summary.setdefault(display["table"], {})[state] = row
            lines.append(f"| {display['table']} ({','.join(map(str, row['seeds']))}) | {interval(row['iteration'], '{:d}')} | "
                         + " | ".join(interval(row[name]) for name, _ in TABLE_COLUMNS) + " |")
        sections.append("\n".join(lines))
    preamble = (f"[min, max] over the seeds of each group. Common validation set: seed {results['validation_set_seed']} "
                f"(derive_seed({results['validation_master_seed']}, 'validation_set')); boundary set of P: seed "
                f"{results['boundary_validation_set_seed']}. Price, Delta and Gamma errors: maxima over the 13 slices "
                f"tau = 1e-6 ... 1 of the geometric grid. Constraint violations: corner point excluded, distances to the "
                f"corner from 1e-6.\n")
    return preamble + "\n" + "\n\n".join(sections) + "\n", summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_metric_vs_time_to_maturity(results: dict, path: Path, metric: str, ylabel: str) -> None:
    grouped = group_runs(results)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    for ax, state, title in ((axes[0], "final", "Final state (last snapshot)"),
                             (axes[1], "selected", "State selected on the common fixed sets")):
        for key, display in GROUPS:
            runs = grouped[key]
            if not runs:
                continue
            tau = [entry["tau_effective"] for entry in runs[0][state]["max_price_error_per_slice"]]
            values = np.array([[entry[metric] for entry in run[state]["max_price_error_per_slice"]] for run in runs])
            ax.fill_between(tau, values.min(axis=0), values.max(axis=0), color=display["colour"], alpha=0.15, lw=0)
            ax.loglog(tau, np.median(values, axis=0), color=display["colour"], lw=1.8, marker="o", ms=3,
                      label=display["label"] + f" (median of {len(runs)} seeds)")
        ax.set_title(title)
        ax.set_xlabel(r"Time to maturity $\tau=T-t$")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(ylabel)
    legend = axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
                            title="Shaded band: min-max over the seeds", title_fontsize=8)
    fig.subplots_adjust(right=0.68, bottom=0.40, wspace=0.08)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_TEXT, axes=list(axes))


def plot_selection_criterion_vs_iteration(results: dict, path: Path) -> None:
    grouped = group_runs(results)
    panels = [("Analytic profile $E$", ["enrichment"]), (r"Learned profile $E_{\theta'}$", ["learned_similarity_profile"]),
              ("Penalty baseline $P$", ["penalty"]), ("Smoothing $H_2$, $H_2^-$", ["smoothing_eps0.1", "smoothing_eps0.02"])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), sharey=True)
    handles, labels = [], []
    for ax, (title, treatments) in zip(axes, panels):
        for key, display in GROUPS:
            if key[0] not in treatments:
                continue
            for index, run in enumerate(grouped[key]):
                (line,) = ax.semilogy(run["snapshot_iterations"], run["selection_criteria"], color=display["colour"], lw=1.0,
                                      alpha=0.85)
                (marker,) = ax.semilogy([run["selected"]["iteration"]], [run["selected"]["selection_criterion"]],
                                        linestyle="none", marker="o", ms=5, color=display["colour"])
                if index == 0:
                    handles += [line, marker]
                    labels += [display["label"] + " (one line per seed)", display["label"] + ": selected state"]
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Selection criterion on the common fixed sets\n(interior loss; full objective for $P$)")
    legend = axes[-1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(right=0.78, bottom=0.40, wspace=0.08)
    finalize_figure(fig, path, legends=[legend], formula=FORMULA_TEXT, axes=list(axes))


def plot_learned_profiles_against_error_function(results: dict, key: tuple[str, str], path: Path) -> None:
    """Lambda_theta'(xi) of every seed (selected state) over erf on [0, 5], and Lambda_theta' - erf."""
    runs = group_runs(results)[key]
    display = GROUP_DISPLAY[key]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    xi = runs[0]["selected"]["similarity_profile_on_grid"]["xi"].numpy()
    error_function = torch.erf(torch.from_numpy(xi)).numpy()
    for index, run in enumerate(runs):
        profile = run["selected"]["similarity_profile_on_grid"]["profile"].numpy()
        label = (f"Seed {run['seed']} (iteration {run['selected']['iteration']}, "
                 f"$\\|\\Lambda_{{\\theta'}}-\\mathrm{{erf}}\\|_{{L^\\infty[0,20]}}$ = "
                 f"{run['selected']['profile_sup_distance_to_error_function']:.2e})")
        colour = SEED_COLOURS[index % len(SEED_COLOURS)]
        axes[0].plot(xi, profile, color=colour, lw=1.4, label=label)
        axes[1].plot(xi, profile - error_function, color=colour, lw=1.4)
    axes[0].plot(xi, error_function, color="black", lw=1.2, linestyle="--", label=r"Analytic profile $\mathrm{erf}(\xi)$")
    axes[1].axhline(0.0, color="black", lw=0.8, linestyle=":")
    axes[0].set_ylabel(r"$\Lambda_{\theta'}(\xi)$")
    axes[1].set_ylabel(r"$\Lambda_{\theta'}(\xi)-\mathrm{erf}(\xi)$")
    axes[0].set_title("Learned similarity profile and erf")
    axes[1].set_title(r"Difference $\Lambda_{\theta'}-\mathrm{erf}$")
    for ax in axes:
        ax.set_xlabel(r"Similarity variable $\xi=\ln(s/B)/(\sigma\sqrt{2\tau})$")
        ax.set_xlim(0.0, PROFILE_FIGURE_SIMILARITY_VARIABLE_MAX)
        ax.grid(alpha=0.3)
    fig.suptitle(display["label"] + f": {len(runs)} seeds, state selected on the common validation set")
    legend = axes[1].legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(right=0.68, bottom=0.36, wspace=0.25)
    formula = (r"$\Lambda_{\theta'}=\Lambda_0+\omega\,g_{\theta'}$, $\Lambda_0=1-e^{-\xi}$, $\omega=e^{-\xi}(1-e^{-\xi})$, "
               r"$g_{\theta'}$ a one-variable network (width 32, depth 2), trained jointly with $\Psi_\theta$;  "
               r"$\xi=\ln(s/B)/(\sigma\sqrt{2\tau})$, $\tau=T-t$"
               "\n"
               r"Selected state: snapshot (every 1000 iterations) of minimum interior loss on ONE fixed validation set "
               r"($n=65\,536$ points of the corner mixture, $w=u/m$) common to every run;  "
               r"$\|\cdot\|_{L^\infty[0,20]}$: uniform grid of 4001 points"
               "\n" + {"corner": r"Collocation: corner mixture (fraction 0.5, $\tau$ log-uniform on $[10^{-6},1]$, $\xi$ uniform on $[0,5]$)",
                       "uniform": r"Collocation: uniform on $(B,s_\infty)\times(0,T)$",
                       "triangle": r"Collocation: corner half uniform on the triangle $\{(s-B)+(T-t)\leq0.1\}$"}[key[1]])
    finalize_figure(fig, path, legends=[legend], formula=formula, axes=list(axes))


def write_outputs(results: dict, out_dir: Path) -> None:
    table, table_summary = build_table(results)
    (out_dir / "table.md").write_text(table)

    def scalar_state(state: dict) -> dict:
        return {k: v for k, v in state.items() if k not in ("max_price_error_per_slice", "similarity_profile_on_grid")}

    with open(out_dir / "summary.yaml", "w") as f:
        yaml.dump({"validation_set_seed": results["validation_set_seed"],
                   "boundary_validation_set_seed": results["boundary_validation_set_seed"],
                   "validation_master_seed": results["validation_master_seed"],
                   "near_barrier_window": results["near_barrier_window"],
                   "groups_min_max_over_seeds": table_summary,
                   "runs": [{key: run[key] for key in ("run_directory", "corner_treatment", "sampler", "seed", "epsilon",
                                                       "in_training_selection")}
                            | {state: scalar_state(run[state]) for state in ("final", "selected")}
                            for run in results["runs"]]},
                  f, sort_keys=False, default_flow_style=None, allow_unicode=True)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_metric_vs_time_to_maturity(results, figures / "max_price_error_vs_time_to_maturity_by_group.png", "max_abs_error",
                                    r"$\max_s|\Phi_\theta-V_{DO}|$ on the slice")
    plot_metric_vs_time_to_maturity(
        results, figures / "relative_delta_error_vs_time_to_maturity_by_group.png", "relative_delta_error",
        r"$\max_s|\partial_s\Phi_\theta-\partial_s V_{DO}|\,/\,\max_s|\partial_s V_{DO}|$ on the slice")
    plot_metric_vs_time_to_maturity(
        results, figures / "relative_gamma_error_vs_time_to_maturity_by_group.png", "relative_gamma_error",
        r"$\max_s|\partial_{ss}\Phi_\theta-\partial_{ss}V_{DO}|\,/\,\max_s|\partial_{ss}V_{DO}|$ on the slice")
    plot_metric_vs_time_to_maturity(
        results, figures / "relative_gamma_error_near_barrier_vs_time_to_maturity_by_group.png",
        "relative_gamma_error_near_barrier",
        r"$\max_{B<s\leq B+0.1}|\partial_{ss}\Phi_\theta-\partial_{ss}V_{DO}|\,/\,\max_{B<s\leq B+0.1}|\partial_{ss}V_{DO}|$")
    plot_selection_criterion_vs_iteration(results, figures / "selection_criterion_vs_iteration.png")
    grouped = group_runs(results)
    profile_figures = figures / "learned_similarity_profile"
    for key in (("learned_similarity_profile", "corner"), ("learned_similarity_profile", "uniform"),
                ("learned_similarity_profile", "triangle")):
        if grouped[key]:
            profile_figures.mkdir(exist_ok=True)
            plot_learned_profiles_against_error_function(
                results, key, profile_figures / f"learned_profile_against_error_function_{key[1]}_sampler.png")
    logger.info(f"Table, summary and figures written to {out_dir}\n{table}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", default=None, help="Finished pilot runs with model snapshots.")
    parser.add_argument("--validation-master-seed", type=int, default=0,
                        help="Master seed of the common fixed sets (derive_seed(<this>, 'validation_set') and "
                             "derive_seed(<this>, 'validation_boundary_set')).")
    parser.add_argument("--near-barrier-window", type=float, default=0.1,
                        help="Width of the window B < s <= B + w of the near-barrier Gamma error (0.1: the inner radius "
                             "delta_0 of the cutoff of E, used for every treatment).")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Torch intra-op threads; float32 evaluations are bit-comparable only at equal thread count.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory override.")
    parser.add_argument("--replot", type=str, default=None, metavar="OUT_DIR",
                        help="Rebuild the table and the figures from OUT_DIR/results.pt without evaluating any model.")
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)

    repo_root = find_repo_root(Path(__file__).resolve())
    if args.replot:
        out_dir = Path(args.replot)
    else:
        if not args.run_dirs:
            parser.error("--run-dirs is required unless --replot is given")
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir) if args.out_dir else (
            script_data_dir(__file__) / f"{timestamp}_runs{len(args.run_dirs)}_validationseed{args.validation_master_seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "evaluation.log")])
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
                f"CUDA {torch.version.cuda if torch.cuda.is_available() else 'not available'}, device {DEVICE}, "
                f"host {platform.node()}, threads {torch.get_num_threads()}; no random number generator other than "
                f"the generators of the two fixed validation sets is used")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Output directory: {out_dir}")

    if args.replot:
        results = torch.load(out_dir / "results.pt", weights_only=False)
        logger.info(f"--replot: results read from {out_dir / 'results.pt'} (no model evaluated)")
    else:
        run_directories = sorted((repo_root / path).resolve() for path in args.run_dirs)
        logger.info(f"{len(run_directories)} runs:\n" + "\n".join(f"  {path}" for path in run_directories))
        start = time.time()
        results = evaluate_runs(run_directories, args.validation_master_seed, args.near_barrier_window)
        torch.save(results, out_dir / "results.pt")
        logger.info(f"Results saved -> {out_dir / 'results.pt'} (evaluation wall-clock time {time.time() - start:.0f}s)")
        with open(out_dir / "metadata.yaml", "w") as f:
            yaml.dump({"command": " ".join(sys.argv), "timestamp": datetime.now().astimezone().isoformat(),
                       "host": platform.node(), "git": get_git_metadata(repo_root), "arguments": vars(args)},
                      f, sort_keys=False)
    write_outputs(results, out_dir)


if __name__ == "__main__":
    main()
