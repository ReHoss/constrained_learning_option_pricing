r"""Check of the importance weights of ``--corner-sampling`` on fixed trained states, before any run.

For each restored state (``models/model_eps0.pt`` of a finished pilot run; no training), the
interior loss of ``pilot_down_and_out_put.compute_loss`` is evaluated on independent batches of
:math:`n_f` points drawn by

- the plain sampler: uniform on :math:`\Omega = (B, s_\infty)\times(0,T)`, unweighted mean of the
  squared residual (``sample_collocation``, the sampler of every existing run);
- the mixture sampler of ``--corner-sampling``
  (:class:`~learning_option_pricing.pricing.corner_similarity_sampling.CornerSimilaritySampling`),
  mean of the squared residual weighted by :math:`w = u/m`.

Both are unbiased estimators of the same quantity
:math:`\mathcal{J}(\theta) = \int_\Omega u\,(\mathcal{L}^{BS}\Phi_\theta)^2` (proof in the module
docstring of the sampler), so for a fixed :math:`\theta` the two batch means must agree within
their standard errors. Reported per state:

- the mean, standard deviation and standard error of the per-batch loss for each sampler;
- the difference of the two means, its combined standard error and the ratio of the two
  (the z-score);
- the variance ratio (plain / mixture) of the per-batch loss at equal :math:`n_f`, i.e. the
  factor by which the mixture reduces the variance of the loss estimator;
- the mean importance weight over all mixture points (expected value 1, since
  :math:`\mathbb{E}_m[u/m] = \int_\Omega u = 1`);
- the per-batch gradient :math:`g = \nabla\widehat{\mathcal{J}}` on each parameter group --
  :math:`\theta'` (the similarity profile network :math:`g_{\theta'}`, learned-profile states only)
  and :math:`\theta` (the network :math:`\Psi_\theta`) -- for each sampler: the mean and standard
  deviation of :math:`\|g\|`, the norm of the mean gradient :math:`\|\bar g\|`, the trace of the
  covariance :math:`\operatorname{tr}\operatorname{Cov}(g) = \mathbb{E}\|g - \bar g\|^2` (unbiased,
  divisor :math:`N-1`), the ratio of the two traces (plain / mixture), and the distance
  :math:`\|\bar g_{\mathrm{uniform}} - \bar g_{\mathrm{mixture}}\|` with its standard error
  :math:`\big((\operatorname{tr}\operatorname{Cov}_{\mathrm{u}} + \operatorname{tr}\operatorname{Cov}_{\mathrm{m}})/N\big)^{1/2}`
  (both gradient estimators are unbiased for :math:`\nabla\mathcal{J}`, the weights not depending
  on the parameters);
- the weighted loss on the fixed validation set of the pilot's ``--best-state-selection validation``
  (``draw_validation_set``, same seed derivation as in training), as a check of that code path.

The two samplers use independent generators derived from the master ``--seed`` with the role
tags ``corner_sampling_verification_uniform`` and ``corner_sampling_verification_mixture``.
Every per-batch scalar is written to ``summary.yaml``; the per-batch gradient vectors on
:math:`\theta'` (1153 parameters) to ``profile_network_gradients.pt``.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/verify_corner_sampling_importance_weights.py
"""
from __future__ import annotations

import argparse
import logging
import math
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learning_option_pricing.pricing.corner_similarity_sampling import CornerSimilaritySampling  # noqa: E402
from learning_option_pricing.utils.run_context import find_repo_root, get_git_metadata, script_data_dir  # noqa: E402
from pilot_down_and_out_put import (  # noqa: E402
    DEFAULT_VALIDATION_SET_SIZE, DEVICE, compute_loss, derive_seed, draw_validation_set, load_trained_model,
    read_run_metadata, sample_collocation, validation_loss,
)

logger = logging.getLogger("verify_corner_sampling_importance_weights")

#: Restored states of the report "profil appris" (seed 0, with the Dirichlet condition at s_inf):
#: learned profile (restored iteration 2058) and analytic enrichment (restored iteration 25471).
DEFAULT_RUN_DIRECTORIES = [
    "data/pilot_down_and_out_put/"
    "20261005_171313_iters50000_eps0_seed0_learnedprofile_blackscholes_d00.1_d10.3_w32_l2_farfield",
    "data/pilot_down_and_out_put/20261002_005610_iters50000_eps0_seed0_enrichment_blackscholes_d00.1_d10.3_farfield",
]


def batch_statistics(values: list[float]) -> dict:
    tensor = torch.tensor(values, dtype=torch.float64)
    standard_deviation = float(tensor.std(unbiased=True))
    return {"mean": float(tensor.mean()), "standard_deviation": standard_deviation,
            "standard_error": standard_deviation / math.sqrt(len(values)),
            "minimum": float(tensor.min()), "maximum": float(tensor.max())}


def parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """theta' = parameters of g2 (the similarity profile network; empty for a fixed g2),
    theta = every other parameter (the network Psi_theta)."""
    profile_parameters = list(model.g2.parameters()) if isinstance(model.g2, torch.nn.Module) else []
    profile_parameter_ids = {id(parameter) for parameter in profile_parameters}
    groups = {"theta_prime_similarity_profile_network": profile_parameters,
              "theta_network_psi": [p for p in model.parameters() if id(p) not in profile_parameter_ids]}
    return {name: parameters for name, parameters in groups.items() if parameters}


def loss_and_gradients(model: torch.nn.Module, groups: dict, s: torch.Tensor, t: torch.Tensor, r: float,
                       sigma: float, weights: torch.Tensor | None) -> tuple[float, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    loss = compute_loss(model, s, t, r, sigma, weights)
    loss.backward()
    gradients = {name: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                  for p in parameters]).double().cpu()
                 for name, parameters in groups.items()}
    return float(loss.detach()), gradients


def gradient_statistics(vectors: list[torch.Tensor]) -> dict:
    stacked = torch.stack(vectors)
    norms = stacked.norm(dim=1)
    mean_vector = stacked.mean(dim=0)
    trace_covariance = float(((stacked - mean_vector) ** 2).sum() / (len(vectors) - 1))
    return {"mean_of_norm": float(norms.mean()), "standard_deviation_of_norm": float(norms.std(unbiased=True)),
            "variance_of_norm": float(norms.var(unbiased=True)),
            "norm_of_mean_gradient": float(mean_vector.norm()), "trace_of_covariance": trace_covariance,
            "relative_noise_sqrt_trace_over_norm_of_mean": math.sqrt(trace_covariance) / float(mean_vector.norm()),
            "per_batch_norms": [float(n) for n in norms]}, mean_vector


def verify_one_state(run_directory: Path, number_of_batches: int, number_of_points: int, master_seed: int,
                     sampling_arguments: dict) -> dict:
    meta = read_run_metadata(run_directory)
    K, B, r, sigma, T = (meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
    s_inf = meta["domain"]["s_inf"]
    if meta["hyperparameters"].get("dtype") == "float64":
        raise ValueError("this check is written for the float32 runs of the report")
    model = load_trained_model(run_directory, 0.0, meta)
    with open(run_directory / "summary_eps0.yaml") as f:
        best_iter = yaml.safe_load(f)["best_iter"]
    sampler = CornerSimilaritySampling(barrier=B, truncation_price=s_inf, maturity=T, volatility=sigma, **sampling_arguments)
    effective_fraction = sampler.effective_corner_fraction(number_of_points)
    logger.info(f"State: {run_directory}")
    logger.info(f"  corner treatment {meta['hyperparameters'].get('corner_treatment')}, far-field Dirichlet "
                f"{meta['hyperparameters'].get('far_field_dirichlet', False)}, restored iteration {best_iter}, "
                f"{sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"  mixture: {sampler.number_of_corner_points(number_of_points)} corner points of {number_of_points} "
                f"(effective fraction {effective_fraction:g}), tau log-uniform on "
                f"[{sampler.smallest_time_to_maturity:g}, {T:g}], xi uniform on [0, {sampler.similarity_variable_max:g}], "
                f"acceptance probability {sampler.acceptance_probability():.6f}, weight bound "
                f"{1.0 / (1.0 - effective_fraction):g}")

    uniform_seed = derive_seed(master_seed, "corner_sampling_verification_uniform")
    mixture_seed = derive_seed(master_seed, "corner_sampling_verification_mixture")
    logger.info(f"  generator seeds: uniform {uniform_seed}, mixture {mixture_seed} (master seed {master_seed})")
    uniform_generator = torch.Generator(device="cpu")
    uniform_generator.manual_seed(uniform_seed)
    mixture_generator = torch.Generator(device="cpu")
    mixture_generator.manual_seed(mixture_seed)

    groups = parameter_groups(model)
    logger.info("  parameter groups: " + ", ".join(f"{name} ({sum(p.numel() for p in parameters)} parameters)"
                                                   for name, parameters in groups.items()))
    uniform_gradients = {name: [] for name in groups}
    mixture_gradients = {name: [] for name in groups}
    uniform_losses, mixture_losses = [], []
    weight_sum, weight_square_sum, weight_count, minimum_weight, maximum_weight, rejected = 0.0, 0.0, 0, math.inf, 0.0, 0
    start = time.time()
    for batch in range(number_of_batches):
        s_f, t_f = sample_collocation(number_of_points, B, s_inf, T, uniform_generator)
        loss_value, gradients = loss_and_gradients(model, groups, s_f, t_f, r, sigma, None)
        uniform_losses.append(loss_value)
        for name, vector in gradients.items():
            uniform_gradients[name].append(vector)

        s_m, t_m, weights, statistics = sampler.sample_mixture(number_of_points, mixture_generator,
                                                               torch.get_default_dtype())
        s_m = s_m.to(DEVICE).requires_grad_(True)
        t_m = t_m.to(DEVICE).requires_grad_(True)
        loss_value, gradients = loss_and_gradients(model, groups, s_m, t_m, r, sigma, weights.to(DEVICE))
        mixture_losses.append(loss_value)
        for name, vector in gradients.items():
            mixture_gradients[name].append(vector)
        weight_sum += float(weights.double().sum())
        weight_square_sum += float((weights.double() ** 2).sum())
        weight_count += weights.numel()
        minimum_weight = min(minimum_weight, statistics["minimum_weight"])
        maximum_weight = max(maximum_weight, statistics["maximum_weight"])
        rejected += statistics["number_of_rejected_corner_draws"]
        if (batch + 1) % 20 == 0:
            logger.info(f"  batch {batch + 1}/{number_of_batches}: running means uniform "
                        f"{sum(uniform_losses) / len(uniform_losses):.6e}, mixture "
                        f"{sum(mixture_losses) / len(mixture_losses):.6e}  ({time.time() - start:.1f}s)")

    gradient_results = {}
    for name in groups:
        uniform_statistics, uniform_mean = gradient_statistics(uniform_gradients[name])
        mixture_statistics, mixture_mean = gradient_statistics(mixture_gradients[name])
        distance = float((uniform_mean - mixture_mean).norm())
        distance_standard_error = math.sqrt((uniform_statistics["trace_of_covariance"]
                                             + mixture_statistics["trace_of_covariance"]) / number_of_batches)
        gradient_results[name] = {
            "number_of_parameters": int(uniform_mean.numel()),
            "uniform_sampler": uniform_statistics,
            "mixture_sampler": mixture_statistics,
            "trace_of_covariance_ratio_uniform_over_mixture": (uniform_statistics["trace_of_covariance"]
                                                               / mixture_statistics["trace_of_covariance"]),
            "variance_of_norm_ratio_uniform_over_mixture": (uniform_statistics["variance_of_norm"]
                                                            / mixture_statistics["variance_of_norm"]),
            "distance_between_mean_gradients": distance,
            "distance_between_mean_gradients_standard_error": distance_standard_error,
        }
        for sampler_name, statistics in (("uniform", uniform_statistics), ("mixture", mixture_statistics)):
            logger.info(f"  gradient on {name}, {sampler_name} sampler: |g| mean {statistics['mean_of_norm']:.4e}, "
                        f"standard deviation {statistics['standard_deviation_of_norm']:.4e}; |mean g| "
                        f"{statistics['norm_of_mean_gradient']:.4e}; tr Cov(g) {statistics['trace_of_covariance']:.4e}; "
                        f"sqrt(tr Cov)/|mean g| {statistics['relative_noise_sqrt_trace_over_norm_of_mean']:.3f}")
        logger.info(f"  gradient on {name}: tr Cov ratio uniform/mixture "
                    f"{gradient_results[name]['trace_of_covariance_ratio_uniform_over_mixture']:.3f}, variance of |g| "
                    f"ratio {gradient_results[name]['variance_of_norm_ratio_uniform_over_mixture']:.3f}; "
                    f"|mean g_uniform - mean g_mixture| = {distance:.4e} (standard error {distance_standard_error:.4e}, "
                    f"ratio {distance / distance_standard_error:.2f})")

    validation_seed = derive_seed(master_seed, "validation_set")
    validation_set = draw_validation_set(sampler, DEFAULT_VALIDATION_SET_SIZE, validation_seed)
    state_validation_loss = validation_loss(model, validation_set, r, sigma)
    logger.info(f"  loss on the pilot's fixed validation set ({DEFAULT_VALIDATION_SET_SIZE} points, seed "
                f"{validation_seed}): {state_validation_loss:.6e}")

    uniform = batch_statistics(uniform_losses)
    mixture = batch_statistics(mixture_losses)
    difference = mixture["mean"] - uniform["mean"]
    combined_standard_error = math.sqrt(uniform["standard_error"] ** 2 + mixture["standard_error"] ** 2)
    result = {
        "run_directory": str(run_directory),
        "restored_iteration": best_iter,
        "number_of_batches": number_of_batches,
        "number_of_points_per_batch": number_of_points,
        "generator_seeds": {"uniform": uniform_seed, "mixture": mixture_seed},
        "uniform_sampler": uniform,
        "mixture_sampler": mixture,
        "difference_of_means_mixture_minus_uniform": difference,
        "relative_difference_of_means": difference / uniform["mean"],
        "combined_standard_error": combined_standard_error,
        "z_score": difference / combined_standard_error,
        "variance_ratio_uniform_over_mixture": (uniform["standard_deviation"] / mixture["standard_deviation"]) ** 2,
        "mean_importance_weight": weight_sum / weight_count,
        # Points of one batch are not independent (fixed group sizes); the per-point standard error
        # is an approximation of the standard error of the mean weight.
        "mean_importance_weight_standard_error": math.sqrt(
            (weight_square_sum / weight_count - (weight_sum / weight_count) ** 2) / weight_count),
        "minimum_importance_weight": minimum_weight,
        "maximum_importance_weight": maximum_weight,
        "number_of_rejected_corner_draws": rejected,
        "per_batch_losses": {"uniform": uniform_losses, "mixture": mixture_losses},
        "gradients": gradient_results,
        "validation_set_loss": state_validation_loss,
        "validation_set_seed": validation_seed,
    }
    profile_group = "theta_prime_similarity_profile_network"
    if profile_group in groups:
        result["_profile_network_gradient_vectors"] = {"uniform": torch.stack(uniform_gradients[profile_group]),
                                                       "mixture": torch.stack(mixture_gradients[profile_group])}
    logger.info(f"  uniform sampler: mean {uniform['mean']:.6e}, standard deviation {uniform['standard_deviation']:.4e}, "
                f"standard error {uniform['standard_error']:.4e}, range [{uniform['minimum']:.3e}, {uniform['maximum']:.3e}]")
    logger.info(f"  mixture sampler: mean {mixture['mean']:.6e}, standard deviation {mixture['standard_deviation']:.4e}, "
                f"standard error {mixture['standard_error']:.4e}, range [{mixture['minimum']:.3e}, {mixture['maximum']:.3e}]")
    logger.info(f"  difference of means (mixture - uniform) {difference:.4e} (relative {result['relative_difference_of_means']:+.3%}), "
                f"combined standard error {combined_standard_error:.4e}, z-score {result['z_score']:+.2f}")
    logger.info(f"  variance ratio uniform/mixture of the per-batch loss: {result['variance_ratio_uniform_over_mixture']:.3f}")
    logger.info(f"  importance weights: mean {result['mean_importance_weight']:.5f} (expected 1; standard error {result['mean_importance_weight_standard_error']:.5f}), range "
                f"[{minimum_weight:.3e}, {maximum_weight:.4f}]; {rejected} corner redraws in total")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", default=DEFAULT_RUN_DIRECTORIES, help="Finished pilot run directories.")
    parser.add_argument("--number-of-batches", type=int, default=100)
    parser.add_argument("--n-f", type=int, default=4096, help="Points per batch (the training value).")
    parser.add_argument("--corner-sampling-fraction", type=float, default=0.5)
    parser.add_argument("--corner-sampling-smallest-time-to-maturity", type=float, default=1e-6)
    parser.add_argument("--corner-sampling-similarity-variable-max", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0, help="Master seed of the two verification generators.")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Torch intra-op threads; float32 evaluations are bit-comparable only at equal thread count.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory override.")
    args = parser.parse_args()
    torch.set_num_threads(args.num_threads)

    repo_root = find_repo_root(Path(__file__).resolve())
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (
        script_data_dir(__file__) / f"{timestamp}_batches{args.number_of_batches}_nf{args.n_f}"
        f"_fraction{args.corner_sampling_fraction:g}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "verification.log")])
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Python {platform.python_version()}, PyTorch {torch.__version__}, "
                f"CUDA {torch.version.cuda if torch.cuda.is_available() else 'not available'}, device {DEVICE}")
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        logger.info(f"GPU {properties.name}, {properties.total_memory / 2**30:.1f} GiB")
    logger.info(f"Threads: {torch.get_num_threads()}; output directory: {out_dir}")
    logger.info(f"Arguments: {vars(args)}")

    sampling_arguments = {
        "corner_fraction": args.corner_sampling_fraction,
        "smallest_time_to_maturity": args.corner_sampling_smallest_time_to_maturity,
        "similarity_variable_max": args.corner_sampling_similarity_variable_max,
    }
    results = [verify_one_state((repo_root / run_directory).resolve(), args.number_of_batches, args.n_f, args.seed,
                                sampling_arguments) for run_directory in args.run_dirs]
    profile_network_gradients = {result["run_directory"]: result.pop("_profile_network_gradient_vectors")
                                 for result in results if "_profile_network_gradient_vectors" in result}
    torch.save(profile_network_gradients, out_dir / "profile_network_gradients.pt")
    logger.info(f"Per-batch gradient vectors on theta' saved -> {out_dir / 'profile_network_gradients.pt'}")
    with open(out_dir / "summary.yaml", "w") as f:
        yaml.dump({"command": " ".join(sys.argv), "timestamp": datetime.now().astimezone().isoformat(),
                   "git": get_git_metadata(repo_root), "arguments": vars(args), "states": results},
                  f, sort_keys=False, default_flow_style=None)
    logger.info(f"Summary saved -> {out_dir / 'summary.yaml'}")


if __name__ == "__main__":
    main()
