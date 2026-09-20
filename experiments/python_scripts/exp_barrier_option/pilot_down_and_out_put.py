r"""Pilot: down-and-out put trained with the corner-regularised ETCNN ansatz.

Reference: working note "A rigorous statement of exact-constraint learning at
a conflicting constraint corner: the knock-out barrier option" (S. Ouaissi
internship, 2026-06-24). Full mapping in
``docs_travail/barrier_start_map.md`` and ``documents/methodology/barrier_option.md``.

The trial solution is the ordinary ETCNN ansatz

    U_theta(s, t) = g1(s, t) * u_theta(s, t) + g2(s, t)

with:

- ``g1 = barrier_composite_distance(s, t, B, T) = (T-t)(s-B)`` -- the
  composite distance of Definition 4, vanishing exactly on the terminal lid
  Sigma_T and the barrier face Sigma_B, including at the corner
  c = (B, T).
- ``g2 = h_epsilon(s, t)`` -- the corner-regularised extension of
  Definition 5 (:func:`learning_option_pricing.pricing.barrier.\
make_corner_regularised_extension`), matching the terminal payoff exactly
  outside the epsilon-corner-layer and the (identically zero) barrier datum
  exactly everywhere.

Because both hard constraints already hold by construction away from the
corner, training minimises the interior PDE residual alone (no terminal- or
barrier-condition loss term) -- Section 4's stated goal in the note.

``--corner-treatment`` selects how the conflicting corner (B, T) is treated
(Table 1 of the note):

- ``smoothing`` (default, the construction above): the jump K - B is spread
  over the corner layer of bandwidth epsilon by the cutoff zeta; the
  residual of the extension grows like (K-B)/epsilon in that layer.
- ``subtraction`` (Method 1, Section 5.1 of the note, Definition 7): the
  jump is reproduced exactly by the closed-form down-and-out digital price,
  g2 = (K-B) V_DOD + pi - pi(B, .), with pi the terminal profile selected by
  the payoff flags (raw payoff, --black-scholes-payoff or --split-payoff);
  no corner layer, no epsilon (the sweep collapses to a single placeholder
  value eps=0 in file names), and the interior residual of g2 is the bounded
  residual of the regular part alone (Proposition 4), assembled analytically
  (:class:`learning_option_pricing.pricing.barrier.SubtractedDigitalCornerExtension`).

Trained models are compared, for each epsilon, to the exact closed-form
reference (method of images / Reiner-Rubinstein,
:func:`learning_option_pricing.pricing.barrier.reiner_rubinstein_down_and_out_put`),
both globally and restricted to a corner window, to see whether the residual
nuisance introduced by h_epsilon stays localised as the note's construction
predicts (Proposition 2).

Usage:
    python3 experiments/python_scripts/exp_barrier_option/pilot_down_and_out_put.py \
        --epsilons 0.2 0.1 0.05 0.02 0.01 --iters 20000

    Replot from a previous run without retraining:
    python3 experiments/python_scripts/exp_barrier_option/pilot_down_and_out_put.py \
        --replot data/pilot_down_and_out_put/<run_dir>

    Smoke test (fast, for CI / sanity-checking the wiring):
    python3 experiments/python_scripts/exp_barrier_option/pilot_down_and_out_put.py \
        --debug --iters 200 --epsilons 0.1 0.02
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import random
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt  # noqa: E402

from learning_option_pricing.models.etcnn import ETCNN, InputNormalization  # noqa: E402
from learning_option_pricing.models.resnet import ResNet  # noqa: E402
from learning_option_pricing.pricing.barrier import (  # noqa: E402
    SplitSemigroupCornerExtension,
    barrier_composite_distance,
    barrier_composite_distance_with_far_field,
    make_corner_regularised_extension,
    make_corner_regularised_extension_split,
    SPLIT_PROFILE_ROUTES,
    make_corner_regularised_extension_with_black_scholes_payoff,
    BlackScholesCornerExtension,
    make_corner_regularised_extension_with_smoothed_payoff,
    make_subtracted_digital_extension,
    SUBTRACTION_TERMINAL_PROFILES,
    reiner_rubinstein_down_and_out_put,
)
from learning_option_pricing.pricing.terminal import bsm_operator  # noqa: E402
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    find_repo_root,
    get_git_metadata,
    script_data_dir,
)

logger = logging.getLogger("pilot_down_and_out_put")

# Below this many total iterations, a run MUST carry --debug (smoke-test guard).
SMOKE_TEST_ITERS_THRESHOLD = 1000

# Defaults = the note's pilot case (Figure 1 legend): K=1, B=0.6, sigma=0.3, r=0.03, T=1.
DEFAULT_K = 1.0
DEFAULT_B = 0.6
DEFAULT_R = 0.03
DEFAULT_SIGMA = 0.3
DEFAULT_T = 1.0
DEFAULT_S_INF = 3.0  # s_infty >> K (Remark 2's domain truncation)
_CORNER_REJECTION_MAX_PASSES = 50  # guard; one pass suffices at w=0.1 (0.2% of the area)
DEFAULT_EPSILONS = (0.2, 0.1, 0.05, 0.02, 0.01)
DEFAULT_EPS0 = 0.05  # Chen-Mangasarian smoothed-payoff bandwidth (--smoothed-payoff only)
DEFAULT_GRADING = "time_graded"

# Split-semigroup payoff (--split-payoff only; make_corner_regularised_extension_split,
# Proposition 7 / Example 7 of the note). n_quad=8000 matches
# GaussianSemigroupExtensionField's own tested default and is chosen for training
# viability, NOT for the ~1e-6 pointwise accuracy the unit tests target near
# maturity (test/pricing/test_barrier.py needed n_quad up to 1_000_000 for that,
# each single-batch query costing seconds -- see that test class's development
# notes). GaussianSemigroupExtensionField recomputes its quadrature nodes and the
# full batch convolution on every call with no caching, so this cost is paid
# EVERY training iteration; --split-n-quad is deliberately exposed so it can be
# raised if training is unstable, with the understanding that doing so multiplies
# the per-iteration cost roughly linearly.
DEFAULT_SPLIT_N_QUAD = 8000
# Evaluation route of the split-semigroup profile. "closed_form" evaluates the
# explicit Gaussian convolution of the put payoff
# (PutPayoffGaussianSemigroupExtensionField): exact, no quadrature floor, O(n_f)
# per iteration. "quadrature" is the fixed-grid route of
# GaussianSemigroupExtensionField at --split-n-quad nodes, O(n_f x n_quad) per
# iteration (measured: 0.31 s vs 0.002 s per F(g2) call at n_f=4096, float32,
# 4 CPU threads); it is the route of every run made before the closed form
# existed, and load_trained_model falls back to it for metadata that predates
# the "split_profile" key so those runs replot faithfully.
DEFAULT_SPLIT_PROFILE = "closed_form"
# Quadrature domain padding, in units of the diffusion length
# comparison_volatility*sqrt(T), beyond the evaluation window (B, s_infty) in
# log-price -- calibrated in test/pricing/test_barrier.py (>=6 diffusion lengths
# leaves the domain-truncation error many orders of magnitude below the
# resolution error, which is what actually limits accuracy near maturity).
DEFAULT_SPLIT_PADDING_DIFFUSION_LENGTHS = 6.0

#: The two corner treatments of Table 1 of the note retained here (see the
#: module docstring); the leading-order enrichment of Section 5.2 is not
#: implemented yet.
CORNER_TREATMENTS = ("smoothing", "subtraction")
# In subtraction mode there is no corner layer; the epsilon loop, file names
# (summary_eps<E>.yaml, model_eps<E>.pt) and directory tag use this single
# placeholder so the artefact layout stays identical to the smoothing runs.
SUBTRACTION_EPSILON_PLACEHOLDER = 0.0
# Corner window of the evaluation metrics in subtraction mode: epsilon carries
# no meaning there, so the window defaults to the canonical value of the
# smoothing comparison (epsilon = 0.1) to keep rel_l2_outside_corner on the
# same region across treatments.
DEFAULT_SUBTRACTION_CORNER_WINDOW = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_device_arg(device_arg: str) -> None:
    global DEVICE
    if device_arg == "cpu":
        DEVICE = torch.device("cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        DEVICE = torch.device("cuda")
    # "auto": keep the module-level default (CUDA if available, else CPU).


# ---------------------------------------------------------------------------
# Seeding — master seed -> deterministic role-tagged per-role seeds
# ---------------------------------------------------------------------------

def derive_seed(master_seed: int, role: str) -> int:
    """Deterministically derive a per-role seed from the master seed.

    Identical construction to the repo's other ablation scripts (e.g.
    ``exp_split_extension_trained/ablation_split_extension_trained.py::derive_seed``):
    a stable blake2b hash of ``"<master_seed>:<role>"``, independent of
    ``PYTHONHASHSEED``. The role tag is the only decorrelation key: every
    epsilon in the sweep shares the same ``model_init``/``sampler`` seeds
    (shared-seeding policy), so an observed difference between epsilon
    values reflects the regularisation bandwidth, not RNG noise.
    """
    digest = hashlib.blake2b(f"{master_seed}:{role}".encode(), digest_size=8).hexdigest()
    return int(digest, 16) % (2**31 - 1)


def _capture_rng_state() -> dict:
    state: dict = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python_random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu().to(torch.uint8))
    np.random.set_state(state["numpy"])
    random.setstate(state["python_random"])
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [s.cpu().to(torch.uint8) for s in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


# ---------------------------------------------------------------------------
# Collocation
# ---------------------------------------------------------------------------

def sample_collocation(
    n_f: int, B: float, s_inf: float, T: float, generator: torch.Generator,
    corner_exclusion_window: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Interior collocation points, uniform on (B, s_infty) x (0, T).

    With ``corner_exclusion_window = w``, points falling in the ell^1 corner
    window :math:`(s-B)+(T-t)\le w` are rejected and redrawn, so the sampler
    is uniform on the domain **minus** that window -- the same mask the
    evaluation metrics remove.  The interior residual is then never enforced
    at the conflicting corner :math:`(B,T)`, which isolates the treatment of
    the payoff singularity at :math:`s=K` from the treatment of the corner.

    Rejection is by redraw of the rejected points only, so the returned tensors
    always hold exactly ``n_f`` points and the draw stays a deterministic
    function of ``generator``.  At :math:`w=0.1` the window covers
    :math:`w^2/2 = 0.005` of a domain of area :math:`(s_\infty-B)T`, i.e.
    about 0.2 per cent, so the loop converges immediately in practice.

    Args:
        n_f: Number of collocation points to return.
        B: Knock-out barrier (lower end of the price range).
        s_inf: Domain truncation (upper end of the price range).
        T: Maturity.
        generator: RNG handle, propagated explicitly.
        corner_exclusion_window: Half-width ``w`` of the ell^1 corner window to
            exclude, or ``None`` to sample the whole domain (default).

    Returns:
        ``(s_f, t_f)``, each of shape ``(n_f,)``, on ``DEVICE``, requiring grad.
    """
    s_raw = torch.rand(n_f, generator=generator) * (s_inf - B) + B
    t_raw = torch.rand(n_f, generator=generator) * T

    if corner_exclusion_window is not None:
        for _ in range(_CORNER_REJECTION_MAX_PASSES):
            inside_corner = (s_raw - B) + (T - t_raw) <= corner_exclusion_window
            n_rejected = int(inside_corner.sum())
            if n_rejected == 0:
                break
            s_raw[inside_corner] = torch.rand(n_rejected, generator=generator) * (s_inf - B) + B
            t_raw[inside_corner] = torch.rand(n_rejected, generator=generator) * T
        else:
            raise RuntimeError(
                f"corner rejection sampling did not converge in "
                f"{_CORNER_REJECTION_MAX_PASSES} passes with window "
                f"{corner_exclusion_window}; the window is too large for the domain."
            )

    s_f = s_raw.to(DEVICE).requires_grad_(True)
    t_f = t_raw.to(DEVICE).requires_grad_(True)
    return s_f, t_f


# ---------------------------------------------------------------------------
# Loss — interior PDE residual only (both hard constraints already hold by
# construction away from the corner; see the module docstring).
# ---------------------------------------------------------------------------

def compute_loss(model: ETCNN, s_f: torch.Tensor, t_f: torch.Tensor, r: float, sigma: float) -> torch.Tensor:
    r"""Interior PDE residual loss, mean(F(U_theta)^2).

    Two routes, selected by whether ``model.g2`` exposes
    ``black_scholes_residual`` (true only for the split-semigroup mode,
    :func:`~learning_option_pricing.pricing.barrier.make_corner_regularised_extension_split`):

    - Ordinary route (raw/mangasarian/black-scholes-payoff g2, all plain
      closed forms with no quadrature): autograd differentiates the FULL
      trial solution U_theta = g1*u_theta + g2 in one graph, exactly as
      before this function grew a second branch.
    - Split-semigroup route: g2 is a fixed-grid quadratured convolution
      (GaussianSemigroupExtensionField), and autograd through it near the
      terminal slice or the corner would divide that quadrature's own
      discretisation error of the VALUE by h^2 for a second-derivative-scale
      perturbation, amplifying it by orders of magnitude (see
      make_corner_regularised_extension_split's docstring, and
      test/pricing/test_barrier.py's development notes for measurements of
      that error). F(U_theta) is instead assembled from the two terms of
      F(g1*u_theta + g2) = F(g1*u_theta) + F(g2) (F is linear): the first by
      autograd on ETCNN.forward_neural_manifold (g1*u_theta alone, smooth,
      no quadrature, safe for autograd), the second from g2's own analytic
      black_scholes_residual -- never autograd, never a finite difference.
      g2 does not depend on theta, so its contribution is computed under
      torch.no_grad() and enters the loss as a constant additive shift; the
      gradient the optimiser sees is exactly F(g1*u_theta)'s.
    """
    x_f = torch.stack([s_f, t_f], dim=1)
    g2 = model.g2
    # Capability test, not an isinstance check.  The docstring above has always
    # described the route as selected by whether g2 exposes an analytic
    # residual; the code tested `isinstance(g2, SplitSemigroupCornerExtension)`
    # instead, so BlackScholesCornerExtension -- which does expose one -- fell
    # silently into the ordinary route and the --analytic-residual flag was
    # inert (verified: identical loss to 5 significant digits over 300
    # iterations with and without it).
    if hasattr(g2, "black_scholes_residual"):
        neural_manifold = model.forward_neural_manifold(x_f).squeeze()
        F_neural_manifold = bsm_operator(neural_manifold, s_f, t_f, r, 0.0, sigma)
        with torch.no_grad():
            F_g2 = g2.black_scholes_residual(s_f.detach(), t_f.detach(), r, sigma)
        F_u = F_neural_manifold + F_g2
    else:
        u_f = model(x_f).squeeze()
        F_u = bsm_operator(u_f, s_f, t_f, r, 0.0, sigma)
    return torch.mean(F_u**2)


# ---------------------------------------------------------------------------
# Optimiser — Adam + two-stage exponential LR decay (identical schedule to
# experiments/python_scripts/exp1/phase3_training.py::build_lr_lambda).
# ---------------------------------------------------------------------------

def build_lr_lambda(total_iters: int):
    gamma = 0.85

    def lr_lambda(step: int) -> float:
        if step <= 10_000:
            decays = step // 2000
        else:
            decays = 10_000 // 2000
            decays += (step - 10_000) // 5000
        return gamma**decays

    return lr_lambda


def _adaptive_log_every(total_iters: int, n_target: int = 50) -> int:
    raw = max(1, total_iters / n_target)
    mag = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 5, 10):
        candidate = int(factor * mag)
        if candidate >= raw:
            return candidate
    return int(10 * mag)


# ---------------------------------------------------------------------------
# Checkpointing (model + optimizer + scheduler + RNG + history), identical
# contract to phase3_training.py's _save_training_checkpoint/_load_training_checkpoint.
# ---------------------------------------------------------------------------

def _save_checkpoint(
    checkpoint_path: Path,
    iter_done: int,
    total_iters: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    history: dict,
    best_loss: float,
    best_iter: int,
    best_model_state: dict,
    label: str,
    sampler_generator: torch.Generator,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "iter_done": iter_done,
        "total_iters": total_iters,
        "label": label,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "rng_state": _capture_rng_state(),
        # The collocation sampler draws from an EXPLICIT generator, which the
        # global torch RNG state does not cover.  Without it a resumed run
        # re-seeds that generator from scratch and replays, from iteration 1,
        # the very batches it has already trained on -- a systematic
        # repetition, not merely a loss of bit-reproducibility.  Cluster jobs
        # are requeued on time limits, so this path is taken in practice.
        "sampler_generator_state": sampler_generator.get_state(),
        "best_loss": best_loss,
        "best_iter": best_iter,
        "best_model_state": best_model_state,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(checkpoint_path)
    logger.info(f"[{label}] checkpoint saved at iter {iter_done}/{total_iters} -> {checkpoint_path}")


def _load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    sampler_generator: torch.Generator,
) -> tuple[int, int, dict, float, int, dict]:
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    _restore_rng_state(payload["rng_state"])
    sampler_state = payload.get("sampler_generator_state")
    if sampler_state is None:
        logger.warning(
            "checkpoint predates the sampler-generator state being stored; the collocation "
            "sampler will restart its sequence from iteration 1 and replay batches already "
            "trained on. Restart this run from scratch rather than resuming it."
        )
    else:
        sampler_generator.set_state(sampler_state)
    return (
        int(payload["iter_done"]),
        int(payload["total_iters"]),
        payload["history"],
        float(payload["best_loss"]),
        int(payload["best_iter"]),
        payload["best_model_state"],
    )


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def default_split_quadrature_bounds(
    B: float, s_inf: float, T: float, comparison_volatility: float,
    padding_diffusion_lengths: float = DEFAULT_SPLIT_PADDING_DIFFUSION_LENGTHS,
) -> tuple[float, float]:
    """Default (y_lo, y_hi) log-price quadrature support for --split-payoff.

    Pads the evaluation window (B, s_infty), converted to log-price, by
    ``padding_diffusion_lengths`` diffusion lengths
    ``comparison_volatility*sqrt(T)`` on each side -- see
    GaussianSemigroupExtensionField's own docstring for why the datum must be
    supplied this far beyond the query window (never by zero-padding), and
    DEFAULT_SPLIT_PADDING_DIFFUSION_LENGTHS's comment for the calibration
    reference.
    """
    diffusion_length = comparison_volatility * math.sqrt(T)
    padding = padding_diffusion_lengths * diffusion_length
    return math.log(B) - padding, math.log(s_inf) + padding


def build_model(
    K: float, B: float, T: float, epsilon: float, model_seed: int,
    smoothed_payoff: bool = False, eps0: float = DEFAULT_EPS0, grading: str = DEFAULT_GRADING,
    black_scholes_payoff: bool = False, r: float = DEFAULT_R, sigma: float = DEFAULT_SIGMA,
    analytic_residual: bool = False,
    split_payoff: bool = False, s_inf: float = DEFAULT_S_INF,
    comparison_volatility: float | None = None,
    split_y_lo: float | None = None, split_y_hi: float | None = None,
    split_n_quad: int = DEFAULT_SPLIT_N_QUAD,
    split_profile: str = DEFAULT_SPLIT_PROFILE,
    far_field_dirichlet: bool = False,
    corner_treatment: str = "smoothing",
    subtraction_terminal_profile: str = "raw",
) -> ETCNN:
    """Build the ETCNN ansatz U_theta = g1 * u_theta + g2.

    ``corner_treatment="subtraction"`` replaces every corner-regularised
    ``g2`` below by the exact-subtraction extension
    :func:`make_subtracted_digital_extension` for the profile
    ``subtraction_terminal_profile`` (one of ``SUBTRACTION_TERMINAL_PROFILES``;
    the split profile takes ``comparison_volatility``/``split_profile``/
    ``split_y_lo``/``split_y_hi``/``split_n_quad`` exactly as the smoothing
    split mode does). ``epsilon`` and the ``smoothed_payoff``/
    ``black_scholes_payoff``/``split_payoff``/``analytic_residual`` switches
    are then unused: the returned ``g2`` always exposes
    ``black_scholes_residual``, so ``compute_loss`` takes the two-term route.

    ``far_field_dirichlet`` replaces ``g1 = (T-t)(s-B)`` by
    :func:`barrier_composite_distance_with_far_field`, which also vanishes on
    the far segment ``s = s_inf`` of the truncated training domain, so that
    the trial solution equals ``g2(s_inf, t)`` there (Dirichlet condition; see
    that function's docstring for why the truncated problem needs one).

    ``g2`` is one of four mutually exclusive terminal-function modes,
    exactly one of which is active at a time (see the module docstring):

    - raw-payoff (default): :func:`make_corner_regularised_extension`.
    - Chen-Mangasarian smoothed payoff, when ``smoothed_payoff`` is set:
      :func:`make_corner_regularised_extension_with_smoothed_payoff`.
      ``epsilon`` (corner-layer bandwidth) and ``eps0`` (Chen-Mangasarian
      smoothing bandwidth) are independent parameters; ``eps0``/``grading``
      are unused unless ``smoothed_payoff`` is ``True``.
    - exact Black-Scholes European put price, when ``black_scholes_payoff``
      is set: :func:`make_corner_regularised_extension_with_black_scholes_payoff`.
      ``r``/``sigma`` are unused unless ``black_scholes_payoff`` is ``True``.
    - split-semigroup profile (Proposition 7 / Example 7 of the note), when
      ``split_payoff`` is set: :func:`make_corner_regularised_extension_split`.
      ``comparison_volatility`` defaults to the contract's own ``sigma``
      (the matched split, whose remainder forcing is bounded uniformly up to
      the terminal slice); ``split_profile`` selects the evaluation route
      (``"closed_form"``, exact and O(n_f) per call, or ``"quadrature"``,
      the fixed-grid route at ``split_n_quad`` nodes on the support
      ``split_y_lo``/``split_y_hi``, which default to
      :func:`default_split_quadrature_bounds`); all are unused unless
      ``split_payoff`` is ``True``. The returned ``g2`` exposes
      ``black_scholes_residual(s, t, r, sigma)`` -- ``compute_loss`` uses
      this to route the interior PDE residual through analytic derivatives
      instead of autograd for this mode (see that function's docstring for
      why: autograd through g2 here would differentiate the fixed-grid
      quadrature convolution of ``GaussianSemigroupExtensionField``, which
      amplifies its own discretisation error rather than being merely slow).
    """
    if corner_treatment not in CORNER_TREATMENTS:
        raise ValueError(f"corner_treatment must be one of {CORNER_TREATMENTS}; got {corner_treatment!r}.")
    torch.manual_seed(model_seed)
    resnet = ResNet()

    def g1(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if far_field_dirichlet:
            return barrier_composite_distance_with_far_field(s, t, B, T, s_inf)
        return barrier_composite_distance(s, t, B, T)

    if corner_treatment == "subtraction":
        resolved_comparison_volatility = comparison_volatility if comparison_volatility is not None else sigma
        if subtraction_terminal_profile == "split" and split_profile == "quadrature" and (
            split_y_lo is None or split_y_hi is None
        ):
            default_y_lo, default_y_hi = default_split_quadrature_bounds(B, s_inf, T, resolved_comparison_volatility)
            split_y_lo = split_y_lo if split_y_lo is not None else default_y_lo
            split_y_hi = split_y_hi if split_y_hi is not None else default_y_hi
        g2 = make_subtracted_digital_extension(
            K, B, r, sigma, T, subtraction_terminal_profile,
            comparison_volatility=resolved_comparison_volatility,
            y_lo=split_y_lo, y_hi=split_y_hi, n_quad=split_n_quad, split_profile=split_profile,
        )
    elif smoothed_payoff:
        g2 = make_corner_regularised_extension_with_smoothed_payoff(K, B, epsilon, T, eps0, grading=grading)
    elif black_scholes_payoff:
        # Same field either way; the class additionally exposes an analytic
        # black_scholes_residual, whose presence makes compute_loss assemble
        # the interior residual as F(g1*u_theta) + F(g2) instead of
        # differentiating the full trial solution as one autograd graph.
        # Control arm for the terminal-function comparison: the split mode was
        # otherwise the only one trained through that route.
        g2 = (BlackScholesCornerExtension(K, B, epsilon, r, sigma, T) if analytic_residual
              else make_corner_regularised_extension_with_black_scholes_payoff(K, B, epsilon, r, sigma, T))
    elif split_payoff:
        resolved_comparison_volatility = comparison_volatility if comparison_volatility is not None else sigma
        if split_y_lo is None or split_y_hi is None:
            default_y_lo, default_y_hi = default_split_quadrature_bounds(
                B, s_inf, T, resolved_comparison_volatility,
            )
            split_y_lo = split_y_lo if split_y_lo is not None else default_y_lo
            split_y_hi = split_y_hi if split_y_hi is not None else default_y_hi
        g2 = make_corner_regularised_extension_split(
            K, B, epsilon, T, resolved_comparison_volatility, split_y_lo, split_y_hi, n_quad=split_n_quad,
            profile=split_profile,
        )
    else:
        g2 = make_corner_regularised_extension(K, B, epsilon)
    normalizer = InputNormalization(K)
    return ETCNN(resnet=resnet, g1=g1, g2=g2, normalizer=normalizer)


# ---------------------------------------------------------------------------
# Training loop for a single epsilon
# ---------------------------------------------------------------------------

def train_one_epsilon(
    *,
    epsilon: float,
    K: float, B: float, r: float, sigma: float, T: float, s_inf: float,
    total_iters: int, n_f: int, log_every: int,
    seed: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    resume: bool,
    smoothed_payoff: bool = False,
    eps0: float = DEFAULT_EPS0,
    grading: str = DEFAULT_GRADING,
    black_scholes_payoff: bool = False,
    analytic_residual: bool = False,
    corner_exclusion_window: float | None = None,
    split_payoff: bool = False,
    comparison_volatility: float | None = None,
    split_y_lo: float | None = None,
    split_y_hi: float | None = None,
    split_n_quad: int = DEFAULT_SPLIT_N_QUAD,
    split_profile: str = DEFAULT_SPLIT_PROFILE,
    far_field_dirichlet: bool = False,
    corner_treatment: str = "smoothing",
    subtraction_terminal_profile: str = "raw",
) -> tuple[ETCNN, dict, float, int]:
    """Train one ETCNN for one epsilon (or, in subtraction mode, the single
    placeholder epsilon). Returns (best_model, history, best_loss, best_iter)."""
    label = (f"subtraction:{subtraction_terminal_profile}" if corner_treatment == "subtraction"
             else f"eps={epsilon:g}")
    model_seed = derive_seed(seed, "model_init")
    sampler_seed = derive_seed(seed, "sampler")

    model = build_model(
        K, B, T, epsilon, model_seed,
        smoothed_payoff=smoothed_payoff, eps0=eps0, grading=grading,
        black_scholes_payoff=black_scholes_payoff, r=r, sigma=sigma,
        analytic_residual=analytic_residual,
        split_payoff=split_payoff, s_inf=s_inf, comparison_volatility=comparison_volatility,
        split_y_lo=split_y_lo, split_y_hi=split_y_hi, split_n_quad=split_n_quad,
        split_profile=split_profile,
        far_field_dirichlet=far_field_dirichlet,
        corner_treatment=corner_treatment,
        subtraction_terminal_profile=subtraction_terminal_profile,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[{label}] model parameters: {n_params}")
    # Report the interior-residual route from the built g2 object itself (the
    # capability test compute_loss performs), not from the CLI flags: this is
    # the line to read to know which route a run actually trained through.
    residual_route = (
        "two-term analytic: F(g1*u_theta) by autograd + F(g2) from g2.black_scholes_residual"
        if hasattr(model.g2, "black_scholes_residual")
        else "ordinary: autograd through the full trial solution g1*u_theta + g2"
    )
    logger.info(f"[{label}] interior residual route (from g2={type(model.g2).__name__}): {residual_route}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(total_iters))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampler_seed)

    history = {"iter": [], "loss": [], "grad_norm": [], "lr": []}
    best_loss = math.inf
    best_iter = 0
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    start_iter = 1

    if resume and checkpoint_path.exists():
        (iter_done, total_iters_saved, history, best_loss, best_iter, best_model_state) = _load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, generator,
        )
        if total_iters_saved != total_iters:
            logger.warning(
                f"[{label}] resume target total_iters={total_iters} differs from "
                f"checkpointed total_iters={total_iters_saved}; using the new target."
            )
        start_iter = iter_done + 1
        logger.info(f"[{label}] resumed from {checkpoint_path} (iter {iter_done}); continuing to {total_iters}.")

    model.train()
    t0 = time.time()
    for it in range(start_iter, total_iters + 1):
        optimizer.zero_grad()
        s_f, t_f = sample_collocation(n_f, B, s_inf, T, generator, corner_exclusion_window)
        loss = compute_loss(model, s_f, t_f, r, sigma)
        loss.backward()

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_iter = it
            best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if it % log_every == 0 or it == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            history["iter"].append(it)
            history["loss"].append(loss_val)
            history["grad_norm"].append(total_norm)
            history["lr"].append(lr_now)
            elapsed = time.time() - t0
            logger.info(
                f"[{label}] iter {it:>6d}/{total_iters}  loss={loss_val:.6e}  "
                f"|grad|={total_norm:.2e}  lr={lr_now:.6f}  best={best_loss:.6e}@{best_iter}  ({elapsed:.1f}s)"
            )

        if checkpoint_every > 0 and it % checkpoint_every == 0 and it < total_iters:
            _save_checkpoint(
                checkpoint_path, it, total_iters, model, optimizer, scheduler,
                history, best_loss, best_iter, best_model_state, label, generator,
            )

    elapsed_total = time.time() - t0
    sec_per_iter = elapsed_total / max(1, total_iters - start_iter + 1)
    logger.info(
        f"[{label}] training done in {elapsed_total:.1f}s "
        f"({sec_per_iter:.4f}s/iter); best loss {best_loss:.6e} at iter {best_iter}"
    )

    if total_iters > 0:
        _save_checkpoint(
            checkpoint_path, total_iters, total_iters, model, optimizer, scheduler,
            history, best_loss, best_iter, best_model_state, label, generator,
        )

    # Restore the best-loss state before returning (CLAUDE.md: the last iter
    # is not always the best one).
    model.load_state_dict(best_model_state)
    model.eval()
    return model, history, best_loss, best_iter


# ---------------------------------------------------------------------------
# Evaluation against the closed form
# ---------------------------------------------------------------------------

def far_field_truncation_error_bound(model, K: float, B: float, r: float, sigma: float, T: float,
                                     s_inf: float, n_t: int = 1001) -> dict:
    """sup_t |g2(s_inf, t) - V_DO(s_inf, t)| on a fine t grid: by the weak maximum
    principle for the Black-Scholes operator (zeroth-order coefficient -r <= 0),
    the solution of the truncated problem with the Dirichlet datum g2(s_inf, .)
    on the far segment differs from the exact price by at most this number on
    the whole truncated domain. Evaluated in float64 from the closed form."""
    t = torch.linspace(0.0, T, n_t, dtype=torch.float64)
    s = torch.full_like(t, s_inf)
    reference = reiner_rubinstein_down_and_out_put(s, K, B, r, sigma, T - t)
    with torch.no_grad():
        datum = model.g2(s.to(DEVICE).to(torch.get_default_dtype()),
                         t.to(DEVICE).to(torch.get_default_dtype())).double().cpu()
    return {
        "bound": float((datum - reference).abs().max()),
        "sup_reference": float(reference.abs().max()),
        "sup_datum": float(datum.abs().max()),
    }


def evaluate_against_closed_form(
    model: torch.nn.Module,
    K: float, B: float, r: float, sigma: float, T: float, s_inf: float,
    corner_window: float,
    n_s: int = 300, n_t: int = 100,
) -> dict:
    """Relative L2 error of the trained price against the closed form, on a
    dense (s, t) grid, on three regions:

    - ``global``: the whole grid (B, s_inf) x (0, T), corner window INCLUDED.
      Because the reference is of order K-B inside the window while the
      barrier condition forces the trial solution to 0 on s=B, this metric is
      dominated by the corner discontinuity whenever the window is not
      negligible; it is kept for continuity with earlier runs, not as the
      comparison metric.
    - ``corner``: restricted to the window {|s-B| + (T-t) <= corner_window}
      (the note's ell^1 corner-distance, Definition 5's ``N_epsilon`` shape,
      at a fixed window size so epsilon values are compared on the same
      window). A diagnostic of the corner treatment only: with
      --exclude-corner-from-collocation the residual is never enforced there.
    - ``outside_corner``: the complement of the window, i.e. exactly the
      region where the PDE residual is enforced when the corner is excluded
      from collocation. This is the metric on which terminal-function modes
      are compared.
    """
    s_grid = torch.linspace(B + 1e-4, s_inf, n_s, dtype=torch.float64)
    t_grid = torch.linspace(0.0, T - 1e-4, n_t, dtype=torch.float64)
    ss, tt = torch.meshgrid(s_grid, t_grid, indexing="ij")

    with torch.no_grad():
        x = torch.stack([ss.reshape(-1), tt.reshape(-1)], dim=1).to(DEVICE).to(torch.get_default_dtype())
        learned = model(x).squeeze().double().cpu().reshape(ss.shape)

    reference = reiner_rubinstein_down_and_out_put(ss, K, B, r, sigma, T - tt)

    error = learned - reference
    corner_mask = (ss - B).abs() + (T - tt) <= corner_window

    def _rel_l2(err: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> float:
        num = torch.linalg.vector_norm(err[mask])
        den = torch.linalg.vector_norm(ref[mask])
        return float(num / den) if den > 0 else float("nan")

    outside_corner_mask = ~corner_mask
    return {
        "rel_l2_global": _rel_l2(error, reference, torch.ones_like(corner_mask, dtype=torch.bool)),
        "rel_l2_corner": _rel_l2(error, reference, corner_mask),
        "rel_l2_outside_corner": _rel_l2(error, reference, outside_corner_mask),
        "max_abs_error_global": float(error.abs().max()),
        "max_abs_error_corner": float(error[corner_mask].abs().max()) if corner_mask.any() else float("nan"),
        "max_abs_error_outside_corner": (
            float(error[outside_corner_mask].abs().max()) if outside_corner_mask.any() else float("nan")
        ),
        "corner_window_grid_fraction": float(corner_mask.double().mean()),
        "s_grid": s_grid, "t_grid": t_grid, "learned": learned, "reference": reference,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

FORMULA_TEXT_RAW_PAYOFF = (
    r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\partial_sV-rV$"
    "\n"
    r"$g_1(s,t)=(T-t)(s-B)$,  $g_2(s,t)=h_\varepsilon(s,t)=\zeta((s-B)/\varepsilon)\,(K-s)^+$"
    "\n"
    r"reference: $V_{DO}$ = Reiner-Rubinstein closed form (method of images, $\mathcal{L}^{BS}$-exact)"
)


def formula_text_smoothed_payoff(eps0: float, grading: str) -> str:
    """Formula textbox for the Chen-Mangasarian smoothed-payoff variant of g2.

    Labels the figure with the actually-used payoff (raw vs. smoothed) so a
    comparison across runs is not visually mistaken for a like-for-like one
    (CLAUDE.md: "Label figures when you think comparison are unfair.").
    """
    eps_of_t = r"\varepsilon_0(T-t)/T" if grading == "time_graded" else r"\varepsilon_0"
    return (
        r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\partial_sV-rV$"
        "\n"
        r"$g_1(s,t)=(T-t)(s-B)$,  $g_2(s,t)=\zeta((s-B)/\varepsilon)\,g_{\varepsilon_0}(s,t)$"
        "\n"
        r"$g_{\varepsilon_0}(s,t)=\frac{1}{2}\left(K-s+\sqrt{(K-s)^2+\varepsilon(t)^2}\right)$, "
        rf"$\varepsilon(t)={eps_of_t}$, $\varepsilon_0={eps0:g}$ (grading={grading})"
        "\n"
        r"reference: $V_{DO}$ = Reiner-Rubinstein closed form (method of images, $\mathcal{L}^{BS}$-exact)"
    )


FORMULA_TEXT_BLACK_SCHOLES_PAYOFF = (
    r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\partial_sV-rV$"
    "\n"
    r"$g_1(s,t)=(T-t)(s-B)$,  $g_2(s,t)=\zeta((s-B)/\varepsilon)\,V^e(s,t)$"
    "\n"
    r"$V^e(s,t)=K e^{-r(T-t)}N(\tilde d_2)-sN(\tilde d_1)$ (exact Black-Scholes European put)"
    "\n"
    r"reference: $V_{DO}$ = Reiner-Rubinstein closed form (method of images, $\mathcal{L}^{BS}$-exact)"
)


def formula_text_split_payoff(
    comparison_volatility: float, n_quad: int, profile: str = DEFAULT_SPLIT_PROFILE,
) -> str:
    """Formula textbox for the split-semigroup variant of g2 (Proposition 7 /
    Example 7): labels the comparison diffusivity and the evaluation route
    (closed form, or quadrature at the resolution actually used), since both
    change the achieved accuracy (CLAUDE.md: "Label figures when you think
    comparison are unfair.")."""
    route_text = (
        r"$\pi$ evaluated in closed form: $K\,\Phi(c)-s\,e^{\nu_c(T-t)}\Phi(c-m)$, "
        r"$m=\sigma_c\sqrt{T-t}$, $c=\ln(K/s)/m$ (no quadrature)"
        if profile == "closed_form"
        else rf"$n_{{\rm quad}}={n_quad:g}$ (fixed-grid quadrature, no caching across calls)"
    )
    return "\n".join([
        r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\partial_sV-rV$",
        r"$g_1(s,t)=(T-t)(s-B)$,  $g_2(s,t)=\zeta((s-B)/\varepsilon)\,\pi(s,t)$,  "
        r"$\pi(\cdot,t)=e^{(T-t)\nu_c\partial_{xx}}(K-e^{(\cdot)})^+$ at $x=\ln s$",
        rf"$\nu_c=\sigma_c^2/2$, $\sigma_c={comparison_volatility:g}$ (comparison volatility), "
        + route_text,
        r"reference: $V_{DO}$ = Reiner-Rubinstein closed form (method of images, $\mathcal{L}^{BS}$-exact)",
    ])


SUBTRACTION_PROFILE_FORMULAS = {
    "raw": r"$\pi(s,t)=(K-s)^+$ (raw payoff, no strike treatment)",
    "black_scholes": r"$\pi(s,t)=V^e(s,t)=Ke^{-r(T-t)}N(\tilde d_2)-sN(\tilde d_1)$ (exact Black-Scholes European put)",
    "split": r"$\pi(\cdot,t)=e^{(T-t)\nu_c\partial_{xx}}(K-e^{(\cdot)})^+$ at $x=\ln s$ (split-semigroup profile)",
}


def formula_text_subtraction(
    terminal_profile: str, comparison_volatility: float | None = None,
    split_profile: str = DEFAULT_SPLIT_PROFILE, n_quad: int = DEFAULT_SPLIT_N_QUAD,
) -> str:
    """Formula textbox for the exact-subtraction ansatz (Method 1, Definition 7
    of the note), labelled with the terminal profile actually used."""
    profile_line = SUBTRACTION_PROFILE_FORMULAS[terminal_profile]
    if terminal_profile == "split":
        route = ("closed form, no quadrature" if split_profile == "closed_form"
                 else rf"quadrature, $n_{{\rm quad}}={n_quad:g}$")
        profile_line += rf",  $\nu_c=\sigma_c^2/2$, $\sigma_c={comparison_volatility:g}$ ({route})"
    return "\n".join([
        r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\partial_sV-rV$;  "
        r"$\Phi_\theta=\Delta\,V_{DOD}+h+g_1u_\theta$, $\Delta=K-B$, $g_1(s,t)=(T-t)(s-B)$ (exact subtraction, no corner layer)",
        r"$V_{DOD}(s,t)=e^{-r(T-t)}\left[N(d_-(s,t))-(s/B)^{1-2r/\sigma^2}N(d_-(B^2/s,t))\right]$, "
        r"$d_-(s,t)=\frac{\ln(s/B)+(r-\sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$;  "
        r"$h(s,t)=\pi(s,t)-\pi(B,t)$,  " + profile_line,
        r"reference: $V_{DO}$ = Reiner-Rubinstein closed form (method of images, $\mathcal{L}^{BS}$-exact)",
    ])


def plot_subtraction_decomposition(
    model: ETCNN, eval_result: dict, K: float, B: float, T: float, out_path: Path, formula_text: str,
) -> None:
    """Slices at fixed t of the three terms of the subtracted estimator,
    Phi_theta = Delta V_DOD + h + g1 u_theta, against the closed form: shows
    what the closed-form digital reproduces, what the regular extension h
    adds and what is left to the network."""
    s_grid = eval_result["s_grid"]
    t_grid = eval_result["t_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    g2 = model.g2
    for ax, t_target in zip(axes, (0.0, 0.5, 0.9)):
        j = int(np.argmin(np.abs(t_grid.numpy() - t_target)))
        s = s_grid.to(DEVICE).to(torch.get_default_dtype())
        t = torch.full_like(s, float(t_grid[j]))
        with torch.no_grad():
            digital = g2.digital_price(s, t).double().cpu().numpy()
            regular = g2.subtracted_data_extension(s, t).double().cpu().numpy()
            manifold = model.forward_neural_manifold(torch.stack([s, t], dim=1)).squeeze(-1).double().cpu().numpy()
        s_np = s_grid.numpy()
        ax.plot(s_np, eval_result["reference"].numpy()[:, j], linestyle="--", color="black", lw=1.8, label=r"$V_{DO}$ (closed form)")
        ax.plot(s_np, eval_result["learned"].numpy()[:, j], color="tab:blue", lw=1.6, label=r"$\Phi_\theta$ (trained)")
        ax.plot(s_np, digital, color="tab:red", lw=1.4, label=r"$\Delta\,V_{DOD}$ (subtracted singular part)")
        ax.plot(s_np, regular, color="tab:green", lw=1.4, label=r"$h=\pi-\pi(B,\cdot)$ (regular extension)")
        ax.plot(s_np, manifold, color="tab:purple", lw=1.4, label=r"$g_1u_\theta$ (network)")
        ax.axvline(B, color="black", linestyle=":", lw=1.0)
        ax.axvline(K, color="grey", linestyle=":", lw=1.0)
        ax.set_xlabel("Underlying price $s$")
        ax.set_title(f"$t = {float(t_grid[j]):.2f}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Value")
    legend = axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(right=0.78, bottom=0.36)
    finalize_figure(fig, out_path, legends=[legend], formula=formula_text, axes=list(axes))


def plot_error_vs_epsilon(summaries: list[dict], out_path: Path, formula_text: str = FORMULA_TEXT_RAW_PAYOFF) -> None:
    epsilons = [s["epsilon"] for s in summaries]
    rel_l2_global = [s["rel_l2_global"] for s in summaries]
    rel_l2_corner = [s["rel_l2_corner"] for s in summaries]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.loglog(epsilons, rel_l2_global, marker="o", linestyle="-", color="tab:blue", label="Global rel. $L^2$")
    ax.loglog(epsilons, rel_l2_corner, marker="s", linestyle="-", color="tab:red", label="Corner-window rel. $L^2$")
    ax.set_xlabel(r"Corner-regularisation bandwidth $\varepsilon$")
    ax.set_ylabel(r"Relative $L^2$ error vs. closed form")
    ax.set_title("Down-and-out put: error vs. $\\varepsilon$", fontsize=11)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.subplots_adjust(right=0.62, bottom=0.30)
    finalize_figure(fig, out_path, legends=[legend], formula=formula_text, axes=[ax])


def plot_price_surface(
    eval_result: dict, epsilon: float, K: float, B: float, out_path: Path,
    formula_text: str = FORMULA_TEXT_RAW_PAYOFF,
) -> None:
    s_grid = eval_result["s_grid"].numpy()
    t_grid = eval_result["t_grid"].numpy()
    learned = eval_result["learned"].numpy()
    reference = eval_result["reference"].numpy()
    diff = learned - reference

    vmin = min(learned.min(), reference.min())
    vmax = max(learned.max(), reference.max())
    dmax = np.abs(diff).max()

    # epsilon = 0 is never a smoothing bandwidth (rejected by the builders): it
    # is the placeholder of the exact-subtraction runs.
    trained_title = ("Trained (exact subtraction, no corner layer)" if epsilon == SUBTRACTION_EPSILON_PLACEHOLDER
                     else f"Trained ($\\varepsilon={epsilon:g}$)")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    panels = (
        (axes[0], learned,   trained_title,                            "viridis", vmin,  vmax,  "Price $V(s,t)$"),
        (axes[1], reference, "Closed form (Reiner-Rubinstein)",       "viridis", vmin,  vmax,  "Price $V(s,t)$"),
        (axes[2], diff,      "Trained $-$ closed form",               "RdBu_r", -dmax,  dmax,  "Error"),
    )
    for ax, data, title, cmap, lo, hi, label in panels:
        mesh = ax.pcolormesh(t_grid, s_grid, data, shading="auto",
                            cmap=cmap, vmin=lo, vmax=hi)
        ax.axhline(B, color="black", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Calendar time $t$")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, label=label)
    axes[0].set_ylabel("Underlying price $s$")
    fig.subplots_adjust(bottom=0.34)
    finalize_figure(fig, out_path, formula=formula_text, axes=list(axes))

def plot_log_slice(
    eval_result: dict, epsilon: float, B: float, out_path: Path,
    formula_text: str = FORMULA_TEXT_RAW_PAYOFF,
) -> None:
    """Coupe V(s) à t fixé, échelle log : révèle les écarts invisibles en linéaire."""
    s_grid = eval_result["s_grid"].numpy()
    t_grid = eval_result["t_grid"].numpy()
    learned = eval_result["learned"].numpy()
    reference = eval_result["reference"].numpy()

    floor = 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, t_target in zip(axes, (0.0, 0.5, 0.9)):
        j = int(np.argmin(np.abs(t_grid - t_target)))
        ax.semilogy(s_grid, np.maximum(learned[:, j], floor), lw=2, label="trained")
        ax.semilogy(s_grid, np.maximum(reference[:, j], floor), lw=2, ls="--", label="closed form")
        ax.axvline(B, color="black", lw=1.0)
        ax.set_xlabel("Underlying price $s$")
        ax.set_title(f"$t = {t_grid[j]:.2f}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$V(s,t)$  (log scale)")
    axes[0].legend(loc="lower left")
    fig.subplots_adjust(bottom=0.34)
    finalize_figure(fig, out_path, formula=formula_text, axes=list(axes))

# ---------------------------------------------------------------------------
# Summary I/O (per-epsilon, so --replot never needs to retrain)
# ---------------------------------------------------------------------------

def _summary_path(out_dir: Path, epsilon: float) -> Path:
    return out_dir / f"summary_eps{epsilon:g}.yaml"


def _write_summary(out_dir: Path, epsilon: float, payload: dict) -> None:
    path = _summary_path(out_dir, epsilon)
    serialisable = {k: v for k, v in payload.items() if k not in ("s_grid", "t_grid", "learned", "reference")}
    with open(path, "w") as f:
        yaml.dump(serialisable, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  Summary saved -> {path}")


def read_run_metadata(run_dir: Path) -> dict:
    """Read a run's ``metadata.yaml`` (written once per run by ``main``)."""
    with open(run_dir / "metadata.yaml") as f:
        return yaml.safe_load(f)


def formula_text_for_run(meta: dict) -> str:
    """The figure formula box matching the terminal-function mode recorded in
    a run's metadata. ``.get(..., default)`` keeps runs recorded before a mode
    existed readable (absent key -> the raw-payoff default)."""
    hyper = meta["hyperparameters"]
    if hyper.get("corner_treatment", "smoothing") == "subtraction":
        comparison_volatility = hyper.get("comparison_volatility", None)
        return formula_text_subtraction(
            hyper["subtraction_terminal_profile"],
            comparison_volatility if comparison_volatility is not None else meta["contract"]["sigma"],
            hyper.get("split_profile", DEFAULT_SPLIT_PROFILE), hyper.get("split_n_quad", DEFAULT_SPLIT_N_QUAD),
        )
    if hyper.get("smoothed_payoff", False):
        return formula_text_smoothed_payoff(hyper.get("eps0", DEFAULT_EPS0), hyper.get("grading", DEFAULT_GRADING))
    if hyper.get("black_scholes_payoff", False):
        return FORMULA_TEXT_BLACK_SCHOLES_PAYOFF
    if hyper.get("split_payoff", False):
        comparison_volatility = hyper.get("comparison_volatility", None)
        return formula_text_split_payoff(
            comparison_volatility if comparison_volatility is not None else meta["contract"]["sigma"],
            hyper.get("split_n_quad", DEFAULT_SPLIT_N_QUAD),
            # Runs recorded before the closed form existed trained through the
            # quadrature route; their metadata has no "split_profile" key.
            hyper.get("split_profile", "quadrature"),
        )
    return FORMULA_TEXT_RAW_PAYOFF


def load_trained_model(run_dir: Path, epsilon: float, meta: dict | None = None) -> ETCNN:
    """Rebuild the trial solution of a finished run from its metadata and load
    the saved final weights ``models/model_eps<EPSILON>.pt`` (evaluation mode).

    The terminal-function mode, its parameters and the contract are taken from
    ``metadata.yaml`` so that the rebuilt ``g1``/``g2`` are those the run was
    trained with; the model seed is irrelevant because the weights are
    overwritten. Used by ``--replot`` and by the aggregation scripts, which
    must never retrain or re-derive anything from the run's command line.
    """
    meta = meta if meta is not None else read_run_metadata(run_dir)
    hyper = meta["hyperparameters"]
    K, B, r, sigma, T = (meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
    model = build_model(
        K, B, T, epsilon, model_seed=0,
        smoothed_payoff=hyper.get("smoothed_payoff", False),
        eps0=hyper.get("eps0", DEFAULT_EPS0), grading=hyper.get("grading", DEFAULT_GRADING),
        black_scholes_payoff=hyper.get("black_scholes_payoff", False), r=r, sigma=sigma,
        analytic_residual=hyper.get("analytic_residual", False),
        split_payoff=hyper.get("split_payoff", False), s_inf=meta["domain"]["s_inf"],
        comparison_volatility=hyper.get("comparison_volatility", None),
        split_y_lo=hyper.get("split_y_lo", None), split_y_hi=hyper.get("split_y_hi", None),
        split_n_quad=hyper.get("split_n_quad", DEFAULT_SPLIT_N_QUAD),
        # Metadata without the key predates the closed form: those runs
        # trained through the quadrature route and must be rebuilt with it.
        split_profile=hyper.get("split_profile", "quadrature"),
        far_field_dirichlet=hyper.get("far_field_dirichlet", False),
        # Metadata without the key predates the exact-subtraction ansatz.
        corner_treatment=hyper.get("corner_treatment", "smoothing"),
        subtraction_terminal_profile=hyper.get("subtraction_terminal_profile", "raw"),
    )
    model_path = run_dir / "models" / f"model_eps{epsilon:g}.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


EVALUATION_METRIC_KEYS = (
    "rel_l2_global", "rel_l2_corner", "rel_l2_outside_corner",
    "max_abs_error_global", "max_abs_error_corner", "max_abs_error_outside_corner",
    "corner_window_grid_fraction",
)


def _evaluation_metrics(eval_result: dict) -> dict:
    """The scalar evaluation metrics of ``evaluate_against_closed_form``, as
    written to ``summary_eps<EPSILON>.yaml`` (tensors excluded)."""
    return {key: eval_result[key] for key in EVALUATION_METRIC_KEYS}


def _read_summaries(out_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(out_dir.glob("summary_eps*.yaml")):
        with open(path) as f:
            summaries.append(yaml.safe_load(f))
    summaries.sort(key=lambda s: s["epsilon"])
    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot — down-and-out put, corner-regularised ETCNN ansatz")
    parser.add_argument("--epsilons", nargs="+", type=float, default=list(DEFAULT_EPSILONS),
                         help="Corner-regularisation bandwidths to sweep (Definition 5's epsilon).")
    parser.add_argument("--K", type=float, default=DEFAULT_K, help="Strike price.")
    parser.add_argument("--B", type=float, default=DEFAULT_B, help="Knock-out barrier, 0 < B < K.")
    parser.add_argument("--r", type=float, default=DEFAULT_R, help="Risk-free rate.")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA, help="Volatility.")
    parser.add_argument("--T", type=float, default=DEFAULT_T, help="Maturity.")
    parser.add_argument("--s-inf", type=float, default=DEFAULT_S_INF,
                         help="Domain truncation s_infty (Remark 2); the far-field decay "
                              "condition is NOT hard-enforced in this pilot (see methodology doc).")
    parser.add_argument("--corner-window", type=float, default=None,
                         help="ell^1 corner-window half-width used to report the localised "
                              "error metric (default: max of --epsilons, so the window covers "
                              "the coarsest regularisation tested).")
    parser.add_argument("--smoothed-payoff", action="store_true",
                         help="Use the Chen-Mangasarian smoothed put payoff "
                              "(make_corner_regularised_extension_with_smoothed_payoff) instead "
                              "of the raw (K-s)^+ payoff in g2. Default: raw payoff (unchanged).")
    parser.add_argument("--eps0", type=float, default=DEFAULT_EPS0,
                         help="Chen-Mangasarian smoothing bandwidth epsilon_0 (only used with "
                              "--smoothed-payoff; independent of --epsilons, the corner-layer "
                              "bandwidth).")
    parser.add_argument("--grading", type=str, default=DEFAULT_GRADING,
                         choices=["constant", "time_graded"],
                         help="Time-dependence of the smoothing bandwidth epsilon_0(t) (only used "
                              "with --smoothed-payoff): \"constant\" = epsilon_0 everywhere, "
                              "\"time_graded\" = epsilon_0*(T-t)/T (0 exactly at t=T).")
    parser.add_argument("--black-scholes-payoff", action="store_true",
                         help="Use the exact Black-Scholes European put price "
                              "(make_corner_regularised_extension_with_black_scholes_payoff) "
                              "instead of the raw (K-s)^+ payoff in g2. Mutually exclusive with "
                              "--smoothed-payoff. Default: raw payoff (unchanged).")
    parser.add_argument("--far-field-dirichlet", action="store_true",
                         help="Hard-enforce a Dirichlet condition on the far segment s = --s-inf of the "
                              "truncated training domain: g1 = (T-t)(s-B)(s_inf-s)/(s_inf-B) vanishes "
                              "there, so the trial solution equals g2(s_inf, t). Without it the truncated "
                              "problem is not well posed (any solution of the homogeneous PDE vanishing on "
                              "s=B and t=T with arbitrary far trace has zero interior residual). The "
                              "truncation-error bound max_t |g2(s_inf,t) - V_DO(s_inf,t)| (weak maximum "
                              "principle) is logged and recorded. Directory tag _farfield.")
    parser.add_argument("--corner-treatment", type=str, default="smoothing", choices=list(CORNER_TREATMENTS),
                         help="Treatment of the conflicting corner (B, T) (Table 1 of the note). 'smoothing' "
                              "(default): the corner-regularised extension zeta((s-B)/epsilon) * pi, swept over "
                              "--epsilons. 'subtraction' (Method 1, Section 5.1, Definition 7): g2 = (K-B) V_DOD "
                              "+ pi - pi(B, .), with V_DOD the closed-form down-and-out digital price reproducing "
                              "the jump exactly and pi the terminal profile selected by the payoff flags (raw "
                              "payoff by default, --black-scholes-payoff, --split-payoff; --smoothed-payoff is "
                              "refused). No corner layer: --epsilons is ignored (a single placeholder eps=0 names "
                              "the artefacts), --corner-window defaults to "
                              f"{DEFAULT_SUBTRACTION_CORNER_WINDOW:g} (the canonical smoothing window, so the "
                              "outside-corner metric is evaluated on the same region), the interior residual is "
                              "always assembled through the two-term analytic route (the digital's residual is "
                              "exactly zero, Proposition 4; autograd through it would difference unbounded terms "
                              "at the corner). Directory tag _subtraction_<profile>.")
    parser.add_argument("--exclude-corner-from-collocation", action="store_true",
                         help="Reject interior collocation points falling in the ell^1 corner window "
                              "(s-B)+(T-t) <= --corner-window, so the PDE residual is never enforced at the "
                              "conflicting corner (B,T). The evaluation then reports rel_l2_outside_corner on "
                              "exactly the region where the residual is enforced (rel_l2_global still "
                              "includes the window and rel_l2_corner is the window alone), isolating the "
                              "treatment of the payoff singularity at s=K from that of the corner.")
    parser.add_argument("--analytic-residual", action="store_true",
                         help="With --black-scholes-payoff only: build g2 as BlackScholesCornerExtension, "
                              "which exposes an analytic black_scholes_residual. compute_loss then assembles "
                              "the interior residual as F(g1*u_theta) + F(g2) (the two-term route used by "
                              "--split-payoff) instead of differentiating the full trial solution as one "
                              "autograd graph. The FIELD is unchanged (bit-identical); only the loss "
                              "assembly differs. Control arm removing the training-route confound from the "
                              "terminal-function comparison: F(g2) is exactly 0 where zeta is constant, "
                              "where the ordinary route injects autograd noise of order 1e-7 instead.")
    parser.add_argument("--split-payoff", action="store_true",
                         help="Use the split-semigroup terminal profile (Proposition 7 / Example 7 "
                              "of the note; make_corner_regularised_extension_split) instead of the "
                              "raw (K-s)^+ payoff in g2. Mutually exclusive with --smoothed-payoff "
                              "and --black-scholes-payoff. The interior PDE residual for this mode "
                              "is assembled from analytic derivatives (g2.black_scholes_residual), "
                              "not autograd through g2 -- see compute_loss's docstring. The profile "
                              "is evaluated by the route selected with --split-profile (closed form "
                              "by default). Default: raw payoff (unchanged).")
    parser.add_argument("--split-profile", type=str, default=DEFAULT_SPLIT_PROFILE,
                         choices=list(SPLIT_PROFILE_ROUTES),
                         help="Evaluation route of the split-semigroup profile (only used with "
                              "--split-payoff). 'closed_form' (default): the explicit Gaussian "
                              "convolution of the put payoff, K Phi(c) - s exp(nu_c (T-t)) Phi(c-m), "
                              "with closed-form derivatives -- exact, no quadrature floor, O(n_f) per "
                              "iteration. 'quadrature': the fixed-grid route of "
                              "GaussianSemigroupExtensionField at --split-n-quad nodes, O(n_f x "
                              "split_n_quad) per iteration with no caching across calls (about 140x "
                              "slower per F(g2) call at the defaults); the route of every run made "
                              "before the closed form existed, kept for reproducing them and as a "
                              "cross-check.")
    parser.add_argument("--comparison-volatility", type=float, default=None,
                         help="sigma_c of the split-semigroup profile's comparison heat semigroup "
                              "(only used with --split-payoff). Default: the contract's own --sigma "
                              "(the matched split of Example 7, whose remainder forcing is bounded "
                              "uniformly up to the terminal slice; a mismatched value reinstates an "
                              "unbounded second-order channel -- see "
                              "GaussianSemigroupExtensionField's tests).")
    parser.add_argument("--split-y-lo", type=float, default=None,
                         help="Lower end of the split-semigroup profile's log-price quadrature "
                              "support (only used with --split-payoff). Default: "
                              "default_split_quadrature_bounds(B, s_inf, T, comparison_volatility).")
    parser.add_argument("--split-y-hi", type=float, default=None,
                         help="Upper end of the split-semigroup profile's log-price quadrature "
                              "support (only used with --split-payoff). Default: see --split-y-lo.")
    parser.add_argument("--split-n-quad", type=int, default=DEFAULT_SPLIT_N_QUAD,
                         help="Number of quadrature nodes for the split-semigroup profile (only "
                              "used with --split-payoff --split-profile quadrature). Controls an accuracy/cost tradeoff "
                              "measured in test/pricing/test_barrier.py: the default trades "
                              "training-time viability against the ~1e-6 pointwise accuracy that "
                              "unit test targeted near maturity (which needed up to 1_000_000 "
                              "there). Raise it if training with --split-payoff is unstable.")
    parser.add_argument("--iters", type=int, default=20_000, help="Training iterations per epsilon.")
    parser.add_argument("--n-f", type=int, default=4096, help="Interior PDE collocation points per step.")
    parser.add_argument("--log-every", type=int, default=None, help="Log interval (default: adaptive).")
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="Checkpoint period in iterations (0 disables periodic checkpoints).")
    parser.add_argument("--resume", action="store_true", help="Resume each epsilon from its checkpoint if present.")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="Output directory override. Default: a fresh timestamped directory under "
                              "data/<script>/ derived from the configuration. Pass an EXISTING run "
                              "directory together with --resume to continue that run from its checkpoint "
                              "(e.g. after migrating it to another machine); its metadata.yaml is then "
                              "kept and a 'resumes' entry (command, host, timestamp) is appended to it.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed (shared across all epsilons).")
    parser.add_argument("--num-threads", type=int, default=None,
                         help="Intra-op thread count passed to torch.set_num_threads before any tensor "
                              "work. Runs that differ only in this value are NOT comparable in float32 "
                              "(threaded reductions change the last bits; 20000 iterations amplify "
                              "that into a different local minimum -- measured factor 1.98 on the "
                              "Delta's relative L2 error between two otherwise identical runs). Set it "
                              "explicitly and identically across every run of a comparison. Default: "
                              "leave torch's own default (OMP_NUM_THREADS or the core count).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--debug", action="store_true",
                         help="Smoke-test mode: prefixes the output folder with '_debug_' and "
                              "waives the minimum-iteration guard. Required whenever --iters is "
                              f"below {SMOKE_TEST_ITERS_THRESHOLD}.")
    parser.add_argument("--replot", type=str, default=None, metavar="RUN_DIR",
                         help="Regenerate figures from a previous run's saved summaries, without retraining.")
    args = parser.parse_args()

    # Pin the thread count before any tensor is created, so that every threaded
    # reduction of the run uses the same partition (see --num-threads's help).
    if args.num_threads is not None:
        if args.num_threads < 1:
            print(f"ERROR: --num-threads must be >= 1; got {args.num_threads}.", file=sys.stderr)
            sys.exit(2)
        torch.set_num_threads(args.num_threads)

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    _apply_device_arg(args.device)

    # ---- --replot path: no training, no torch RNG. Reads the saved summaries
    # (scalars) for the aggregate error-vs-epsilon plot, and reloads each
    # saved model checkpoint to recompute the price-surface grids (never
    # retrains) so every figure is rebuilt from artefacts on disk.
    if args.replot is not None:
        out_dir = Path(args.replot)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
        logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
        logger.info(f"--replot: reading summaries and metadata from {out_dir}")
        summaries = _read_summaries(out_dir)
        if not summaries:
            logger.error(f"No summary_eps*.yaml found in {out_dir}")
            sys.exit(1)
        meta = read_run_metadata(out_dir)
        K, B, r, sigma, T = (meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
        s_inf = meta["domain"]["s_inf"]
        corner_window = meta["hyperparameters"]["corner_window"]
        formula_text = formula_text_for_run(meta)
        replot_subtraction = meta["hyperparameters"].get("corner_treatment", "smoothing") == "subtraction"
        if meta["hyperparameters"].get("dtype") == "float64":
            torch.set_default_dtype(torch.float64)
        (out_dir / "figures").mkdir(exist_ok=True)

        for summary in summaries:
            epsilon = summary["epsilon"]
            model_path = out_dir / "models" / f"model_eps{epsilon:g}.pt"
            if not model_path.exists():
                logger.warning(f"[eps={epsilon:g}] no saved model at {model_path}, skipping its price-surface plot.")
                continue
            model = load_trained_model(out_dir, epsilon, meta)
            eval_result = evaluate_against_closed_form(model, K, B, r, sigma, T, s_inf, corner_window)
            # Refresh the saved metrics from the saved model: the evaluation
            # grid is deterministic, so this reproduces the training-time
            # values and fills in metrics added after the run was trained
            # (e.g. rel_l2_outside_corner) without retraining.
            refreshed = _evaluation_metrics(eval_result)
            changed = {k: (summary.get(k), v) for k, v in refreshed.items()
                       if summary.get(k) is None or abs(summary[k] - v) > 1e-12 * max(1.0, abs(v))}
            summary.update(refreshed)
            _write_summary(out_dir, epsilon, summary)
            logger.info(
                f"[eps={epsilon:g}] evaluation refreshed from {model_path}: "
                f"rel_l2_outside_corner={eval_result['rel_l2_outside_corner']:.4e}  "
                f"rel_l2_global={eval_result['rel_l2_global']:.4e}  rel_l2_corner={eval_result['rel_l2_corner']:.4e}"
                + (f"  (summary keys added/changed: {sorted(changed)})" if changed else "  (summary unchanged)")
            )
            plot_price_surface(eval_result, epsilon, K, B, out_dir / "figures" / f"price_surface_eps{epsilon:g}.png", formula_text=formula_text)
            plot_log_slice(eval_result, epsilon, B, out_dir / "figures" / f"log_slice_eps{epsilon:g}.png", formula_text=formula_text)
            if replot_subtraction:
                plot_subtraction_decomposition(
                    model, eval_result, K, B, T, out_dir / "figures" / "subtraction_decomposition.png", formula_text,
                )
            logger.info(f"[eps={epsilon:g}] price-surface figure rebuilt from {model_path}")

        if replot_subtraction:
            logger.info("--replot: error_vs_epsilon.png not produced (subtraction mode, no epsilon sweep).")
        else:
            plot_error_vs_epsilon(summaries, out_dir / "figures" / "error_vs_epsilon.png", formula_text=formula_text)
        logger.info(f"--replot: done ({len(summaries)} epsilon values)")
        return

    if not args.debug and args.iters < SMOKE_TEST_ITERS_THRESHOLD:
        print(
            f"ERROR: --iters {args.iters} is below the smoke-test threshold "
            f"({SMOKE_TEST_ITERS_THRESHOLD}). Pass --debug for short/smoke runs, "
            f"or raise --iters for a real run.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not (0.0 < args.B < args.K):
        print(f"ERROR: need 0 < B < K (reverse knock-out regime); got B={args.B}, K={args.K}.", file=sys.stderr)
        sys.exit(2)

    if args.smoothed_payoff and args.eps0 <= 0.0:
        print(f"ERROR: --eps0 must be > 0; got {args.eps0}.", file=sys.stderr)
        sys.exit(2)

    payoff_modes_requested = sum([args.smoothed_payoff, args.black_scholes_payoff, args.split_payoff])
    if payoff_modes_requested > 1:
        print(
            "ERROR: --smoothed-payoff, --black-scholes-payoff and --split-payoff are mutually exclusive.",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- corner treatment: in subtraction mode the terminal profile is read
    # off the payoff flags, the epsilon sweep collapses to one placeholder and
    # the evaluation window defaults to the canonical smoothing value.
    subtraction = args.corner_treatment == "subtraction"
    subtraction_terminal_profile = None
    if subtraction:
        if args.smoothed_payoff:
            print("ERROR: --corner-treatment subtraction does not accept --smoothed-payoff (the Chen-Mangasarian "
                  "family was set aside in section 10 of the methodology document; the terminal profiles of the "
                  f"subtraction ansatz are {SUBTRACTION_TERMINAL_PROFILES}).", file=sys.stderr)
            sys.exit(2)
        subtraction_terminal_profile = ("black_scholes" if args.black_scholes_payoff
                                        else "split" if args.split_payoff else "raw")
        if args.epsilons != list(DEFAULT_EPSILONS) and args.epsilons != [SUBTRACTION_EPSILON_PLACEHOLDER]:
            print(f"WARNING: --corner-treatment subtraction has no corner layer; --epsilons {args.epsilons} is "
                  f"ignored (placeholder eps={SUBTRACTION_EPSILON_PLACEHOLDER:g} names the artefacts).", file=sys.stderr)
        args.epsilons = [SUBTRACTION_EPSILON_PLACEHOLDER]
        if args.corner_window is None:
            args.corner_window = DEFAULT_SUBTRACTION_CORNER_WINDOW

    # ---- split-semigroup mode: resolve defaults now (not inside build_model)
    # so the resolved values are logged and recorded in metadata.yaml exactly
    # once, identically for every epsilon in the sweep.
    comparison_volatility = args.comparison_volatility
    split_y_lo = args.split_y_lo
    split_y_hi = args.split_y_hi
    if args.split_payoff:  # both corner treatments of the split profile
        comparison_volatility = comparison_volatility if comparison_volatility is not None else args.sigma
        if split_y_lo is None or split_y_hi is None:
            default_y_lo, default_y_hi = default_split_quadrature_bounds(
                args.B, args.s_inf, args.T, comparison_volatility,
            )
            split_y_lo = split_y_lo if split_y_lo is not None else default_y_lo
            split_y_hi = split_y_hi if split_y_hi is not None else default_y_hi

    corner_window = args.corner_window if args.corner_window is not None else max(args.epsilons)
    if subtraction:
        formula_text = formula_text_subtraction(
            subtraction_terminal_profile, comparison_volatility, args.split_profile, args.split_n_quad,
        )
    elif args.smoothed_payoff:
        formula_text = formula_text_smoothed_payoff(args.eps0, args.grading)
    elif args.black_scholes_payoff:
        formula_text = FORMULA_TEXT_BLACK_SCHOLES_PAYOFF
    elif args.split_payoff:
        formula_text = formula_text_split_payoff(comparison_volatility, args.split_n_quad, args.split_profile)
    else:
        formula_text = FORMULA_TEXT_RAW_PAYOFF

    # ---- output directory ----
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    debug_prefix = "_debug_" if args.debug else ""
    eps_tag = "_".join(f"{e:g}" for e in sorted(args.epsilons))
    if subtraction:
        # The treatment and its terminal profile name the run: aggregation
        # scripts key on this tag (aggregate_terminal_function_comparison.py).
        payoff_tag = f"_subtraction_{subtraction_terminal_profile.replace('_', '')}"
        if subtraction_terminal_profile == "split":
            payoff_tag += f"_nuc{comparison_volatility:g}" + (
                "_closedform" if args.split_profile == "closed_form" else f"_nquad{args.split_n_quad}"
            )
    elif args.smoothed_payoff:
        payoff_tag = f"_smoothed_eps0{args.eps0:g}_{args.grading}"
    elif args.black_scholes_payoff:
        payoff_tag = "_blackscholes" + ("_analyticres" if args.analytic_residual else "")
    elif args.split_payoff:
        payoff_tag = f"_split_nuc{comparison_volatility:g}" + (
            "_closedform" if args.split_profile == "closed_form" else f"_nquad{args.split_n_quad}"
        )
    else:
        payoff_tag = ""
    # The corner-exclusion flag belongs in the directory name: it changes what
    # the run IS, and metadata.yaml is not what one reads when listing data/.
    corner_tag = "_nocorner" if args.exclude_corner_from_collocation else ""
    far_field_tag = "_farfield" if args.far_field_dirichlet else ""
    out_dir = (Path(args.out_dir) if args.out_dir is not None else script_data_dir(__file__) / (
        f"{debug_prefix}{timestamp}_iters{args.iters}_eps{eps_tag}_seed{args.seed}{payoff_tag}{corner_tag}{far_field_tag}"
    ))
    resuming_existing_run = args.resume and (out_dir / "metadata.yaml").exists()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "training.log")],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    if resuming_existing_run and args.split_payoff:
        # A resumed run keeps the profile route it was started with: the
        # checkpointed weights and loss history belong to that g2.  Metadata
        # without the key predates the closed form and means "quadrature".
        with open(out_dir / "metadata.yaml") as f:
            split_profile_on_disk = yaml.safe_load(f)["hyperparameters"].get("split_profile", "quadrature")
        if split_profile_on_disk != args.split_profile:
            logger.warning(
                f"--resume: the run on disk trained the split-semigroup profile through the "
                f"'{split_profile_on_disk}' route but the command line asks for "
                f"'{args.split_profile}'; the on-disk route is kept so that the continuation "
                f"is faithful (pass --split-profile {split_profile_on_disk} to silence this)."
            )
            args.split_profile = split_profile_on_disk
            formula_text = formula_text_split_payoff(comparison_volatility, args.split_n_quad, args.split_profile)

    logger.info("Pilot — down-and-out put, corner-regularised ETCNN ansatz")
    logger.info(f"  Output directory: {out_dir}")
    logger.info(f"  Log file (follow in real time): {out_dir / 'training.log'}")
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Host: {socket.gethostname()}")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  Torch threads (effective): {torch.get_num_threads()}  "
                f"(--num-threads={args.num_threads}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
    if args.num_threads is None:
        logger.warning(
            "  --num-threads not given: the thread count is inherited from the environment. "
            "Runs of one comparison must share it (float32 reductions are thread-count dependent)."
        )
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)")
    logger.info(f"  Device (requested): {args.device}  (resolved: {DEVICE})")
    logger.info(f"  Contract: K={args.K}, B={args.B}, r={args.r}, sigma={args.sigma}, T={args.T}")
    if args.far_field_dirichlet:
        logger.info(f"  Domain: s in ({args.B}, {args.s_inf})  [hard Dirichlet on s = s_inf via "
                    "g1 = (T-t)(s-B)(s_inf-s)/(s_inf-B): the trial solution equals g2(s_inf, t) there]")
    else:
        logger.info(f"  Domain: s in ({args.B}, {args.s_inf})  [NO far-field condition: the truncated problem "
                    "is not well posed without one, see --far-field-dirichlet; Remark 2 of the methodology doc]")
    logger.info(f"  Corner treatment: {args.corner_treatment}")
    if subtraction:
        logger.info(f"  Epsilons swept: none (exact subtraction has no corner layer; placeholder "
                    f"eps={SUBTRACTION_EPSILON_PLACEHOLDER:g} names the artefacts)")
    else:
        logger.info(f"  Epsilons swept: {sorted(args.epsilons)}")
    logger.info(f"  Corner window (evaluation only): {corner_window:g}")
    logger.info(f"  Iterations per epsilon: {args.iters}, n_f={args.n_f}")
    logger.info(f"  Master seed: {args.seed}")
    logger.info(f"    -> model_init seed: {derive_seed(args.seed, 'model_init')}")
    logger.info(f"    -> sampler seed:    {derive_seed(args.seed, 'sampler')}")
    if subtraction:
        logger.info(
            f"  Ansatz: exact subtraction (Method 1, Section 5.1, Definition 7): "
            f"Phi = (K-B) V_DOD + h + g1 u_theta with h = pi - pi(B, .), Delta = K - B = {args.K - args.B:g}; "
            f"terminal profile pi = {subtraction_terminal_profile} "
            f"(make_subtracted_digital_extension)."
        )
        logger.info(
            "    Interior residual: two-term analytic route, F(g1 u_theta) by autograd + F(h) in closed form; "
            "F((K-B) V_DOD) = 0 exactly (Proposition 4) and is omitted. The corner is "
            + ("EXCLUDED from collocation (--exclude-corner-from-collocation)." if args.exclude_corner_from_collocation
               else "INCLUDED in collocation: the residual is enforced up to the corner, where it stays bounded.")
        )
        if args.analytic_residual:
            logger.info("    --analytic-residual is implied by the subtraction ansatz (no effect).")
        if subtraction_terminal_profile == "split":
            logger.info(f"    comparison_volatility={comparison_volatility:g} "
                        f"({'matched to --sigma' if comparison_volatility == args.sigma else 'MISMATCHED from --sigma'}), "
                        f"profile route: {args.split_profile}.")
    else:
        logger.info(
            "  Note: h_epsilon's transition has scale ~1/epsilon in the first "
            "derivative and ~1/epsilon^2 in the second; small epsilon sharpens "
            "the interior residual near the corner and may need more collocation "
            "density / iterations there to resolve well."
        )
    if subtraction:
        pass  # the terminal profile was logged with the ansatz above
    elif args.smoothed_payoff:
        logger.info(
            f"  Terminal payoff: Chen-Mangasarian smoothed (make_corner_regularised_extension_with_smoothed_payoff), "
            f"eps0={args.eps0:g}, grading={args.grading} "
            f"(independent of the corner-layer bandwidth epsilon above)."
        )
    elif args.black_scholes_payoff:
        logger.info(
            "  Terminal payoff: exact Black-Scholes European put price "
            "(make_corner_regularised_extension_with_black_scholes_payoff)."
        )
    elif args.split_payoff:
        logger.info(
            "  Terminal payoff: split-semigroup profile (Proposition 7 / Example 7, "
            "make_corner_regularised_extension_split)."
        )
        logger.info(
            f"    comparison_volatility={comparison_volatility:g} "
            f"({'matched to --sigma' if comparison_volatility == args.sigma else 'MISMATCHED from --sigma'}), "
            f"profile route: {args.split_profile}."
        )
        if args.split_profile == "closed_form":
            logger.info(
                "    pi(s,t) = K Phi(c) - s exp(nu_c (T-t)) Phi(c-m), m = sigma_c sqrt(T-t), "
                "c = ln(K/s)/m, with closed-form d_x, d_xx and d_t = -nu_c d_xx "
                "(PutPayoffGaussianSemigroupExtensionField): exact, no quadrature support, "
                "no near-maturity floor; --split-y-lo/--split-y-hi/--split-n-quad are ignored."
            )
        else:
            logger.info(
                f"    log-price quadrature support ({split_y_lo:.6g}, {split_y_hi:.6g}), "
                f"n_quad={args.split_n_quad}."
            )
            logger.info(
                "    WARNING: GaussianSemigroupExtensionField recomputes its quadrature nodes and "
                "the full (n_f x n_quad) convolution on every call, with no caching across training "
                "iterations. This dominates the per-iteration wall-clock cost at n_f/n_quad of this "
                "size (about 140x the closed-form route per F(g2) call at the defaults); prefer "
                "--split-profile closed_form unless reproducing an older quadrature run."
            )
        logger.info(
            "    Interior PDE residual for this mode is assembled from g2's analytic "
            "black_scholes_residual, not autograd through g2 (see compute_loss's docstring); "
            "the two-term split costs one extra forward pass of the network manifold per "
            "iteration."
        )
    else:
        logger.info("  Terminal payoff: raw (K-s)^+ (make_corner_regularised_extension).")

    metadata = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contract": {"K": args.K, "B": args.B, "r": args.r, "sigma": args.sigma, "T": args.T},
        # Recorded because two runs that differ only in these are NOT
        # comparable: in float32 the order of the threaded reductions changes
        # the last bits, and 20000 iterations amplify that into a different
        # local minimum.  Two Black-Scholes runs of this pilot, same seed and
        # same configuration but different thread counts, differed by a factor
        # 1.98 on the relative L2 error of the Delta.  torch.get_num_threads()
        # is the effective value, which OMP_NUM_THREADS may leave unset.
        "environment": {
            "host": socket.gethostname(),
            "num_threads_flag": args.num_threads,
            "torch_num_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "git": get_git_metadata(find_repo_root(Path(__file__).resolve())),
        },
        "domain": {"B": args.B, "s_inf": args.s_inf},
        "hyperparameters": {
            "epsilons": sorted(args.epsilons),
            "iters": args.iters,
            "n_f": args.n_f,
            "seed": args.seed,
            "corner_window": corner_window,
            "dtype": args.dtype,
            "smoothed_payoff": args.smoothed_payoff,
            "eps0": args.eps0,
            "grading": args.grading,
            "black_scholes_payoff": args.black_scholes_payoff,
            "analytic_residual": args.analytic_residual,
            "corner_exclusion_window": (corner_window if args.exclude_corner_from_collocation else None),
            "far_field_dirichlet": args.far_field_dirichlet,
            "split_payoff": args.split_payoff,
            "comparison_volatility": comparison_volatility,
            "split_y_lo": split_y_lo,
            "split_y_hi": split_y_hi,
            "split_n_quad": args.split_n_quad,
            "split_profile": args.split_profile,
            "corner_treatment": args.corner_treatment,
            "subtraction_terminal_profile": subtraction_terminal_profile,
        },
    }
    if resuming_existing_run:
        # The run's identity (original command, seeds, git commit, thread
        # count) is the one recorded at its first launch; a resume only adds
        # its own provenance so the continuation is traceable.
        with open(out_dir / "metadata.yaml") as f:
            metadata_on_disk = yaml.safe_load(f)
        metadata_on_disk.setdefault("resumes", []).append({
            "command": metadata["command"],
            "timestamp": metadata["timestamp"],
            "environment": metadata["environment"],
        })
        metadata = metadata_on_disk
        logger.info(f"  --resume into an existing run directory: metadata.yaml kept, resume #{len(metadata['resumes'])} "
                    f"recorded (host {socket.gethostname()}).")
        if metadata["environment"].get("torch_num_threads") != torch.get_num_threads():
            logger.warning(
                f"  Resuming with {torch.get_num_threads()} torch threads while the run started with "
                f"{metadata['environment'].get('torch_num_threads')}: float32 reductions may differ from here on."
            )
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    t_start = time.time()
    summaries: list[dict] = []
    for epsilon in sorted(args.epsilons):
        logger.info("=" * 70)
        logger.info(f"epsilon = {epsilon:g}")
        logger.info("=" * 70)
        checkpoint_path = out_dir / "models" / f"checkpoint_eps{epsilon:g}.pt"
        log_every = args.log_every or _adaptive_log_every(args.iters)

        model, history, best_loss, best_iter = train_one_epsilon(
            epsilon=epsilon, K=args.K, B=args.B, r=args.r, sigma=args.sigma, T=args.T,
            s_inf=args.s_inf, total_iters=args.iters, n_f=args.n_f, log_every=log_every,
            seed=args.seed, checkpoint_path=checkpoint_path,
            checkpoint_every=args.checkpoint_every, resume=args.resume,
            smoothed_payoff=args.smoothed_payoff, eps0=args.eps0, grading=args.grading,
            black_scholes_payoff=args.black_scholes_payoff,
            analytic_residual=args.analytic_residual,
            corner_exclusion_window=(corner_window if args.exclude_corner_from_collocation else None),
            split_payoff=args.split_payoff, comparison_volatility=comparison_volatility,
            split_y_lo=split_y_lo, split_y_hi=split_y_hi, split_n_quad=args.split_n_quad,
            split_profile=args.split_profile,
            far_field_dirichlet=args.far_field_dirichlet,
            corner_treatment=args.corner_treatment,
            subtraction_terminal_profile=subtraction_terminal_profile or "raw",
        )
        truncation_bound = None
        if args.far_field_dirichlet:
            truncation_bound = far_field_truncation_error_bound(model, args.K, args.B, args.r, args.sigma, args.T, args.s_inf)
            logger.info(
                f"[eps={epsilon:g}] far-field truncation-error bound (weak maximum principle): "
                f"sup_t |g2(s_inf,t) - V_DO(s_inf,t)| = {truncation_bound['bound']:.3e}  "
                f"(sup_t |V_DO(s_inf,t)| = {truncation_bound['sup_reference']:.3e}, "
                f"sup_t |g2(s_inf,t)| = {truncation_bound['sup_datum']:.3e}, s_inf={args.s_inf:g})"
            )

        eval_result = evaluate_against_closed_form(
            model, args.K, args.B, args.r, args.sigma, args.T, args.s_inf, corner_window,
        )
        logger.info(
            f"[eps={epsilon:g}] vs closed form: rel_l2_outside_corner={eval_result['rel_l2_outside_corner']:.4e}  "
            f"rel_l2_global={eval_result['rel_l2_global']:.4e}  "
            f"rel_l2_corner={eval_result['rel_l2_corner']:.4e}  "
            f"max_abs_outside_corner={eval_result['max_abs_error_outside_corner']:.4e}  "
            f"max_abs_global={eval_result['max_abs_error_global']:.4e}  "
            f"max_abs_corner={eval_result['max_abs_error_corner']:.4e}  "
            f"(corner window = {100 * eval_result['corner_window_grid_fraction']:.2f}% of the evaluation grid)"
        )

        model_path = out_dir / "models" / f"model_eps{epsilon:g}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"[eps={epsilon:g}] final model saved -> {model_path}")

        summary = {
            "epsilon": epsilon,
            "best_loss": best_loss,
            "best_iter": best_iter,
            **_evaluation_metrics(eval_result),
            "final_history_loss": history["loss"][-1] if history["loss"] else None,
            **({"far_field_truncation_error_bound": truncation_bound["bound"]} if truncation_bound else {}),
        }
        _write_summary(out_dir, epsilon, summary)
        summaries.append({**summary, **{k: eval_result[k] for k in ("s_grid", "t_grid", "learned", "reference")}})

        plot_price_surface(eval_result, epsilon, args.K, args.B, out_dir / "figures" / f"price_surface_eps{epsilon:g}.png", formula_text=formula_text)
        plot_log_slice(eval_result, epsilon, args.B, out_dir / "figures" / f"log_slice_eps{epsilon:g}.png", formula_text=formula_text)
        if subtraction:
            plot_subtraction_decomposition(
                model, eval_result, args.K, args.B, args.T,
                out_dir / "figures" / "subtraction_decomposition.png", formula_text,
            )
    if subtraction:
        logger.info("  error_vs_epsilon.png not produced: no epsilon sweep in subtraction mode.")
    else:
        plot_error_vs_epsilon(summaries, out_dir / "figures" / "error_vs_epsilon.png", formula_text=formula_text)

    elapsed_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info("JOINT SUMMARY")
    logger.info("=" * 70)
    for s in summaries:
        logger.info(
            f"  eps={s['epsilon']:<7g} best_loss={s['best_loss']:.4e}@{s['best_iter']:<6d} "
            f"rel_l2_outside_corner={s['rel_l2_outside_corner']:.4e}  "
            f"rel_l2_global={s['rel_l2_global']:.4e}  rel_l2_corner={s['rel_l2_corner']:.4e}"
        )
    logger.info(f"Total wall-clock time: {elapsed_total:.1f}s ({elapsed_total/len(summaries):.1f}s/epsilon)")
    logger.info(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
