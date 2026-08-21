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
    barrier_composite_distance,
    make_corner_regularised_extension,
    reiner_rubinstein_down_and_out_put,
)
from learning_option_pricing.pricing.terminal import bsm_operator  # noqa: E402
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import script_data_dir  # noqa: E402

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
DEFAULT_EPSILONS = (0.2, 0.1, 0.05, 0.02, 0.01)

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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interior collocation points, uniform on (B, s_infty) x (0, T)."""
    s_f = (
        torch.rand(n_f, generator=generator) * (s_inf - B) + B
    ).to(DEVICE).requires_grad_(True)
    t_f = (
        torch.rand(n_f, generator=generator) * T
    ).to(DEVICE).requires_grad_(True)
    return s_f, t_f


# ---------------------------------------------------------------------------
# Loss — interior PDE residual only (both hard constraints already hold by
# construction away from the corner; see the module docstring).
# ---------------------------------------------------------------------------

def compute_loss(model: torch.nn.Module, s_f: torch.Tensor, t_f: torch.Tensor, r: float, sigma: float) -> torch.Tensor:
    x_f = torch.stack([s_f, t_f], dim=1)
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
) -> tuple[int, int, dict, float, int, dict]:
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    _restore_rng_state(payload["rng_state"])
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

def build_model(K: float, B: float, T: float, epsilon: float, model_seed: int) -> ETCNN:
    torch.manual_seed(model_seed)
    resnet = ResNet()

    def g1(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return barrier_composite_distance(s, t, B, T)

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
) -> tuple[ETCNN, dict, float, int]:
    """Train one ETCNN for one epsilon. Returns (best_model, history, best_loss, best_iter)."""
    label = f"eps={epsilon:g}"
    model_seed = derive_seed(seed, "model_init")
    sampler_seed = derive_seed(seed, "sampler")

    model = build_model(K, B, T, epsilon, model_seed).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[{label}] model parameters: {n_params}")

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
            checkpoint_path, model, optimizer, scheduler,
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
        s_f, t_f = sample_collocation(n_f, B, s_inf, T, generator)
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
                history, best_loss, best_iter, best_model_state, label,
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
            history, best_loss, best_iter, best_model_state, label,
        )

    # Restore the best-loss state before returning (CLAUDE.md: the last iter
    # is not always the best one).
    model.load_state_dict(best_model_state)
    model.eval()
    return model, history, best_loss, best_iter


# ---------------------------------------------------------------------------
# Evaluation against the closed form
# ---------------------------------------------------------------------------

def evaluate_against_closed_form(
    model: torch.nn.Module,
    K: float, B: float, r: float, sigma: float, T: float, s_inf: float,
    corner_window: float,
    n_s: int = 300, n_t: int = 100,
) -> dict:
    """Relative L2 error of the trained price against the closed form, on a
    dense (s, t) grid, both globally and restricted to the corner window
    {|s-B| + (T-t) <= corner_window} (the note's ell^1 corner-distance,
    Definition 5's ``N_epsilon`` shape, evaluated at a fixed window size so
    epsilon values are compared on the same window).
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

    return {
        "rel_l2_global": _rel_l2(error, reference, torch.ones_like(corner_mask, dtype=torch.bool)),
        "rel_l2_corner": _rel_l2(error, reference, corner_mask),
        "max_abs_error_global": float(error.abs().max()),
        "max_abs_error_corner": float(error[corner_mask].abs().max()) if corner_mask.any() else float("nan"),
        "s_grid": s_grid, "t_grid": t_grid, "learned": learned, "reference": reference,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

FORMULA_TEXT = (
    r"$\mathcal{L}^{BS}V=\partial_tV+\frac{1}{2}\sigma^2s^2\partial_{ss}V+rs\partial_sV-rV$"
    "\n"
    r"$g_1(s,t)=(T-t)(s-B)$,  $g_2(s,t)=h_\varepsilon(s,t)=\zeta((s-B)/\varepsilon)\,(K-s)^+$"
    "\n"
    r"reference: $V_{DO}$ = Reiner-Rubinstein closed form (method of images, $\mathcal{L}^{BS}$-exact)"
)


def plot_error_vs_epsilon(summaries: list[dict], out_path: Path) -> None:
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
    finalize_figure(fig, out_path, legends=[legend], formula=FORMULA_TEXT, axes=[ax])


def plot_price_surface(eval_result: dict, epsilon: float, K: float, B: float, out_path: Path) -> None:
    s_grid = eval_result["s_grid"].numpy()
    t_grid = eval_result["t_grid"].numpy()
    learned = eval_result["learned"].numpy()
    reference = eval_result["reference"].numpy()
    diff = learned - reference

    vmin = min(learned.min(), reference.min())
    vmax = max(learned.max(), reference.max())
    dmax = np.abs(diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    panels = (
        (axes[0], learned,   f"Trained ($\\varepsilon={epsilon:g}$)", "viridis", vmin,  vmax,  "Price $V(s,t)$"),
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
    finalize_figure(fig, out_path, formula=FORMULA_TEXT, axes=list(axes))

def plot_log_slice(eval_result: dict, epsilon: float, B: float, out_path: Path) -> None:
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
    finalize_figure(fig, out_path, formula=FORMULA_TEXT, axes=list(axes))

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
    parser.add_argument("--iters", type=int, default=20_000, help="Training iterations per epsilon.")
    parser.add_argument("--n-f", type=int, default=4096, help="Interior PDE collocation points per step.")
    parser.add_argument("--log-every", type=int, default=None, help="Log interval (default: adaptive).")
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="Checkpoint period in iterations (0 disables periodic checkpoints).")
    parser.add_argument("--resume", action="store_true", help="Resume each epsilon from its checkpoint if present.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed (shared across all epsilons).")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--debug", action="store_true",
                         help="Smoke-test mode: prefixes the output folder with '_debug_' and "
                              "waives the minimum-iteration guard. Required whenever --iters is "
                              f"below {SMOKE_TEST_ITERS_THRESHOLD}.")
    parser.add_argument("--replot", type=str, default=None, metavar="RUN_DIR",
                         help="Regenerate figures from a previous run's saved summaries, without retraining.")
    args = parser.parse_args()

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
        with open(out_dir / "metadata.yaml") as f:
            meta = yaml.safe_load(f)
        K, B, r, sigma, T = (meta["contract"][k] for k in ("K", "B", "r", "sigma", "T"))
        s_inf = meta["domain"]["s_inf"]
        corner_window = meta["hyperparameters"]["corner_window"]
        if meta["hyperparameters"].get("dtype") == "float64":
            torch.set_default_dtype(torch.float64)
        (out_dir / "figures").mkdir(exist_ok=True)

        for summary in summaries:
            epsilon = summary["epsilon"]
            model_path = out_dir / "models" / f"model_eps{epsilon:g}.pt"
            if not model_path.exists():
                logger.warning(f"[eps={epsilon:g}] no saved model at {model_path}, skipping its price-surface plot.")
                continue
            model = build_model(K, B, T, epsilon, model_seed=0)  # seed irrelevant: weights are overwritten below
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE).eval()
            eval_result = evaluate_against_closed_form(model, K, B, r, sigma, T, s_inf, corner_window)
            plot_price_surface(eval_result, epsilon, K, B, out_dir / "figures" / f"price_surface_eps{epsilon:g}.png")
            plot_log_slice(eval_result, epsilon, B, out_dir / "figures" / f"log_slice_eps{epsilon:g}.png")
            logger.info(f"[eps={epsilon:g}] price-surface figure rebuilt from {model_path}")

        plot_error_vs_epsilon(summaries, out_dir / "figures" / "error_vs_epsilon.png")
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

    corner_window = args.corner_window if args.corner_window is not None else max(args.epsilons)

    # ---- output directory ----
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    debug_prefix = "_debug_" if args.debug else ""
    eps_tag = "_".join(f"{e:g}" for e in sorted(args.epsilons))
    out_dir = script_data_dir(__file__) / f"{debug_prefix}{timestamp}_iters{args.iters}_eps{eps_tag}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "training.log")],
    )
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    logger.info("Pilot — down-and-out put, corner-regularised ETCNN ansatz")
    logger.info(f"  Output directory: {out_dir}")
    logger.info(f"  Log file (follow in real time): {out_dir / 'training.log'}")
    logger.info(f"  Command: {' '.join(sys.argv)}")
    logger.info(f"  Python: {sys.version.split()[0]}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB)")
    logger.info(f"  Device (requested): {args.device}  (resolved: {DEVICE})")
    logger.info(f"  Contract: K={args.K}, B={args.B}, r={args.r}, sigma={args.sigma}, T={args.T}")
    logger.info(f"  Domain: s in ({args.B}, {args.s_inf})  [no hard far-field condition, Remark 2 -- see methodology doc]")
    logger.info(f"  Epsilons swept: {sorted(args.epsilons)}")
    logger.info(f"  Corner window (evaluation only): {corner_window:g}")
    logger.info(f"  Iterations per epsilon: {args.iters}, n_f={args.n_f}")
    logger.info(f"  Master seed: {args.seed}")
    logger.info(f"    -> model_init seed: {derive_seed(args.seed, 'model_init')}")
    logger.info(f"    -> sampler seed:    {derive_seed(args.seed, 'sampler')}")
    logger.info(
        "  Note: h_epsilon's transition has scale ~1/epsilon in the first "
        "derivative and ~1/epsilon^2 in the second; small epsilon sharpens "
        "the interior residual near the corner and may need more collocation "
        "density / iterations there to resolve well."
    )

    metadata = {
        "command": " ".join(sys.argv),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contract": {"K": args.K, "B": args.B, "r": args.r, "sigma": args.sigma, "T": args.T},
        "domain": {"B": args.B, "s_inf": args.s_inf},
        "hyperparameters": {
            "epsilons": sorted(args.epsilons),
            "iters": args.iters,
            "n_f": args.n_f,
            "seed": args.seed,
            "corner_window": corner_window,
            "dtype": args.dtype,
        },
    }
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
        )

        eval_result = evaluate_against_closed_form(
            model, args.K, args.B, args.r, args.sigma, args.T, args.s_inf, corner_window,
        )
        logger.info(
            f"[eps={epsilon:g}] vs closed form: rel_l2_global={eval_result['rel_l2_global']:.4e}  "
            f"rel_l2_corner={eval_result['rel_l2_corner']:.4e}  "
            f"max_abs_global={eval_result['max_abs_error_global']:.4e}  "
            f"max_abs_corner={eval_result['max_abs_error_corner']:.4e}"
        )

        model_path = out_dir / "models" / f"model_eps{epsilon:g}.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"[eps={epsilon:g}] final model saved -> {model_path}")

        summary = {
            "epsilon": epsilon,
            "best_loss": best_loss,
            "best_iter": best_iter,
            "rel_l2_global": eval_result["rel_l2_global"],
            "rel_l2_corner": eval_result["rel_l2_corner"],
            "max_abs_error_global": eval_result["max_abs_error_global"],
            "max_abs_error_corner": eval_result["max_abs_error_corner"],
            "final_history_loss": history["loss"][-1] if history["loss"] else None,
        }
        _write_summary(out_dir, epsilon, summary)
        summaries.append({**summary, **{k: eval_result[k] for k in ("s_grid", "t_grid", "learned", "reference")}})

        plot_price_surface(eval_result, epsilon, args.K, args.B, out_dir / "figures" / f"price_surface_eps{epsilon:g}.png")
        plot_log_slice(eval_result, epsilon, args.B, out_dir / "figures" / f"log_slice_eps{epsilon:g}.png")
    plot_error_vs_epsilon(summaries, out_dir / "figures" / "error_vs_epsilon.png")

    elapsed_total = time.time() - t_start
    logger.info("=" * 70)
    logger.info("JOINT SUMMARY")
    logger.info("=" * 70)
    for s in summaries:
        logger.info(
            f"  eps={s['epsilon']:<7g} best_loss={s['best_loss']:.4e}@{s['best_iter']:<6d} "
            f"rel_l2_global={s['rel_l2_global']:.4e}  rel_l2_corner={s['rel_l2_corner']:.4e}"
        )
    logger.info(f"Total wall-clock time: {elapsed_total:.1f}s ({elapsed_total/len(summaries):.1f}s/epsilon)")
    logger.info(f"All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
