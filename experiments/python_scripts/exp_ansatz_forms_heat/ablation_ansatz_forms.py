r"""Ablation: terminal-condition enforcement forms for backward-heat PINNs.

Compares four trial-solution *forms* (two hard boundary-constrained ansatze, a
soft-penalty PINN, and a pure unconstrained-network control) on three
initial (terminal) conditions of the backward heat equation
``P u = d_t u + (sigma^2 / 2) d_xx u = 0``.  See
``_ansatz_forms_catalogue.py`` for the form/IC definitions and the scientific
report ``2026_01_29_constrained_learning_pde_lehalle_hosseinkhan`` for the
derivation.

Each component of the training objective is monitored separately.  For the hard
forms the stage residual is split, in the notation of the report's
``rem:residual-decomposition``, into

    P u_hat = R_theta + P Psi,
    R_theta = (1 - lambda) P Phi - lambda' Phi   (network contribution),
    P Psi                                          (extension forcing),

with the loss decomposing as
``L = E[R_theta^2] + 2 E[R_theta P Psi] + E[(P Psi)^2]`` (last = floor).

Run modes
---------
* single variant (array task):
    ``--ic sine --variant hard_constant_linear --ablation-dir DIR``
* all variants for one IC (local convenience):
    ``--ic sine``
* regenerate comparison plots without training:
    ``--replot DIR``
* create directory + metadata + per-variant configs and exit (login node):
    ``--ic sine --init-only --ablation-dir DIR``

Smoke runs MUST pass ``--debug`` (and are prefixed ``_debug_`` on disk); any run
below ``SMOKE_TEST_NUM_ITERATIONS_THRESHOLD`` iterations without ``--debug`` is
rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the sibling catalogue importable whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ansatz_forms_catalogue as cat  # noqa: E402

logger = logging.getLogger("ablation_ansatz_forms")

SMOKE_TEST_NUM_ITERATIONS_THRESHOLD = 1000


# ===========================================================================
# Seeding helpers (master seed -> deterministic role-tagged per-role seeds)
# ===========================================================================

def derive_seed(master_seed: int, role: str) -> int:
    """Deterministically derive a per-role seed from the master seed.

    Uses a stable hash (blake2b) of ``"<master_seed>:<role>"`` so the mapping is
    independent of ``PYTHONHASHSEED`` and reproducible across machines.  The
    role tag is the *only* decorrelation key: two variants sharing the master
    seed get identical model initialisation and sampler trajectories.
    """
    digest = hashlib.blake2b(
        f"{master_seed}:{role}".encode(), digest_size=8
    ).hexdigest()
    return int(digest, 16) % (2**31 - 1)


# ===========================================================================
# Transient-CUDA retry (gpu_p13 array tasks occasionally land on a busy GPU)
# ===========================================================================

def cuda_retry(fn, *, attempts: int = 6, base_delay: float = 10.0):
    """Call ``fn()``, retrying on transient CUDA "device busy/unavailable" errors.

    On shared multi-GPU nodes an array task can touch a GPU that is momentarily
    busy, raising ``CUDA error: CUDA-capable device(s) is/are busy or
    unavailable`` at the first device access (model ``.to(device)`` or the first
    allocation).  The condition is transient, so we retry with a linear backoff
    before giving up; non-CUDA errors propagate immediately.
    """
    import time as _time

    import torch

    last_exc = None
    for k in range(attempts):
        try:
            return fn()
        except RuntimeError as exc:
            msg = str(exc).lower()
            transient = "cuda" in msg and (
                "busy" in msg or "unavailable" in msg or "initialization" in msg
            )
            if not transient:
                raise
            last_exc = exc
            delay = base_delay * (k + 1)
            logger.warning(
                "transient CUDA error (attempt %d/%d): %s; retrying in %.0fs",
                k + 1, attempts, exc, delay,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _time.sleep(delay)
    raise RuntimeError(
        f"CUDA still unavailable after {attempts} attempts"
    ) from last_exc


# ===========================================================================
# Problem assembly (operator, domain, terminal datum, exact reference)
# ===========================================================================

def build_problem(ic_name: str):
    """Return a dict of torch callables/scalars describing the IC problem.

    Keys: ``sigma``, ``T``, ``x_lo``, ``x_hi``, ``terminal_datum`` (g(x)),
    ``exact`` (u(x, t) or None), ``label`` (TeX formula for the figure textbox).
    """
    import torch  # local import keeps the catalogue/login path torch-free

    import math

    from learning_option_pricing.pde import (
        chen_mangasarian_max,
        heat_call_exact,
        heat_call_payoff,
        heat_propagate,
        heat_put_exact,
        heat_put_payoff,
        heat_sine_exact,
        heat_sine_terminal,
        heat_theta3_exact,
        heat_theta3_terminal,
        smooth_call_payoff,
        smooth_call_payoff_cm_time,
    )

    conf = cat.ic_by_name(ic_name)
    sigma = float(conf["sigma"])
    T = float(conf["T"])
    p = conf["params"]
    extension_fn = None  # default: hard forms extend the terminal datum directly

    if ic_name == "sine":
        c, f = float(p["c"]), float(p["f"])

        def terminal_datum(x):
            return heat_sine_terminal(x, c=c, f=f)

        def exact(x, t):
            return heat_sine_exact(x, t, T=T, sigma=sigma, c=c, f=f)

    elif ic_name == "theta3":
        n_modes = int(p["n_modes"])

        def terminal_datum(x):
            return heat_theta3_terminal(x, n_modes=n_modes)

        def exact(x, t):
            return heat_theta3_exact(x, t, T=T, sigma=sigma, n_modes=n_modes)

    elif ic_name == "call":
        K = float(conf["K"])
        beta = float(p["beta"])

        def terminal_datum(x):
            return smooth_call_payoff(x, K, beta=beta)

        def exact(x, t):
            return heat_call_exact(x, t, K=K, T=T, sigma=sigma)

    elif ic_name == "call_cm":
        K = float(conf["K"])
        eps0 = float(p["eps0"])

        # Terminal datum at t=T is the *true* payoff (no bias); the hard-form
        # extension is the time-dependent CM smoothing (exact at t=T, smooth
        # for t<T).
        def terminal_datum(x):
            return heat_call_payoff(x, K)

        def extension_fn(x, t):
            return smooth_call_payoff_cm_time(x, t, K=K, T=T, eps0=eps0)

        def exact(x, t):
            return heat_call_exact(x, t, K=K, T=T, sigma=sigma)

    elif ic_name == "bermudan_put":
        # Stage [0, t1]: framework T = t1 (the exercise date); the option matures
        # at T_option.  Terminal datum = smoothed max of the put payoff and the
        # European continuation; exact reference = Gaussian convolution of it.
        K = float(conf["K"])
        T_option = float(p["T_option"])
        eps = float(p["eps"])
        t1 = T  # framework terminal time == exercise date
        y_lo, y_hi = math.log(5.0), math.log(600.0)

        def continuation(x):
            return heat_put_exact(x, torch.full_like(x, t1), K=K, T=T_option, sigma=sigma)

        def terminal_datum(x):
            return chen_mangasarian_max(heat_put_payoff(x, K), continuation(x), eps)

        def exact(x, t):
            return heat_propagate(terminal_datum, x, t, t_terminal=t1, sigma=sigma,
                                  y_lo=y_lo, y_hi=y_hi)

    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown IC {ic_name!r}")

    return {
        "sigma": sigma,
        "T": T,
        "x_lo": float(conf["x_lo"]),
        "x_hi": float(conf["x_hi"]),
        "x_eval_lo": float(conf["x_eval_lo"]),
        "x_eval_hi": float(conf["x_eval_hi"]),
        "terminal_datum": terminal_datum,
        "extension_fn": extension_fn,
        "exact": exact,
        "label": conf["label"],
    }


def build_ansatz(variant: dict, problem: dict, hparams: dict, *, model_seed: int):
    """Construct the :class:`BlendedTerminalAnsatz` for a method variant."""
    import torch

    from learning_option_pricing.models.blended_ansatz import (
        BlendedTerminalAnsatz,
        make_blending,
    )
    from learning_option_pricing.models.resnet import ResNet

    torch.manual_seed(model_seed)
    net = ResNet(
        d_in=2,
        d_out=1,
        n=int(hparams["net_width"]),
        M=int(hparams["net_blocks"]),
        L=int(hparams["net_layers_per_block"]),
    )

    form = variant["form"]
    blending = None
    if form in ("hard_constant", "hard_blended"):
        blending = make_blending(
            variant["blending"], T=problem["T"], sigma=problem["sigma"]
        )

    # Affine input normalisation maps (x, t) into roughly [-1, 1]^2 so the tanh
    # ResNet sees well-scaled inputs; identical for every form.
    x_lo, x_hi, T = problem["x_lo"], problem["x_hi"], problem["T"]
    x_mid, x_half = 0.5 * (x_lo + x_hi), 0.5 * (x_hi - x_lo)

    def normalizer(xt):
        x = (xt[:, 0:1] - x_mid) / x_half
        t = 2.0 * xt[:, 1:2] / T - 1.0
        return torch.cat([x, t], dim=1)

    return BlendedTerminalAnsatz(
        net,
        problem["terminal_datum"],
        blending,
        form=form,
        normalizer=normalizer,
        extension_fn=problem.get("extension_fn"),
    )


# ===========================================================================
# Sampling
# ===========================================================================

def make_samplers(problem: dict, hparams: dict, *, sampler_seed: int, device):
    """Return interior / terminal / boundary collocation samplers."""
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(sampler_seed)
    x_lo, x_hi, T = problem["x_lo"], problem["x_hi"], problem["T"]
    n_int = int(hparams["n_interior"])
    n_tc = int(hparams["n_terminal"])
    n_bd = int(hparams["n_boundary"])

    def _u(n):
        return torch.rand(n, generator=gen)

    def sample_interior():
        x = (x_lo + (x_hi - x_lo) * _u(n_int)).to(device).requires_grad_(True)
        t = (T * _u(n_int)).to(device).requires_grad_(True)
        return x, t

    def sample_terminal():
        x = (x_lo + (x_hi - x_lo) * _u(n_tc)).to(device)
        t = torch.full((n_tc,), T, device=device)
        return x, t

    def sample_boundary():
        half = n_bd // 2
        xb = torch.cat([
            torch.full((half,), x_lo),
            torch.full((n_bd - half,), x_hi),
        ]).to(device)
        tb = (T * _u(n_bd)).to(device)
        return xb, tb

    return sample_interior, sample_terminal, sample_boundary


# ===========================================================================
# Training
# ===========================================================================

def train_variant(variant, problem, hparams, *, num_iterations, seed, device, log_every):
    """Train one variant; return (model, history dict)."""
    import torch

    from learning_option_pricing.models.blended_ansatz import residual_decomposition

    sigma = problem["sigma"]
    form = variant["form"]
    a = float(hparams["soft_pde_weight_a"])

    model_seed = derive_seed(seed, "model_init")
    sampler_seed = derive_seed(seed, "sampler")
    model = cuda_retry(
        lambda: build_ansatz(variant, problem, hparams, model_seed=model_seed).to(device)
    )
    sample_interior, sample_terminal, sample_boundary = make_samplers(
        problem, hparams, sampler_seed=sampler_seed, device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(hparams["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)

    history = {k: [] for k in (
        "iter", "loss", "loss_pde", "loss_tc", "boundary_error",
        "network_energy", "cross_term", "forcing_floor",
        "forcing_velocity", "forcing_diffusion", "grad_norm", "lr",
    )}
    best_loss = float("inf")
    best_state = None
    best_iter = -1

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "[%s/%s] training: form=%s blending=%s params=%d seeds(model=%d,sampler=%d)",
        problem["ic_name"], variant["name"], form, variant["blending"],
        n_params, model_seed, sampler_seed,
    )

    for it in range(1, num_iterations + 1):
        optimizer.zero_grad()

        # --- PDE residual + decomposition channels -----------------------
        x_f, t_f = sample_interior()
        decomp = residual_decomposition(model, x_f, t_f, sigma)
        loss_pde = decomp["loss"]

        # --- total objective per form (no spatial-boundary term) ---------
        # soft_pinn:  a * L_pde + (1 - a) * L_tc      (note's weighted form)
        # hard forms: L_pde only   (terminal exact by construction)
        # pure_nn:    L_pde only   (no terminal data -> non-identifiable control)
        loss_tc_for_loss = None
        if form == "soft_pinn":
            x_tc, t_tc = sample_terminal()
            with torch.no_grad():
                g_tc = problem["terminal_datum"](x_tc)
            u_tc = model(torch.stack([x_tc, t_tc], dim=1)).squeeze(-1)
            loss_tc_for_loss = ((u_tc - g_tc) ** 2).mean()
            loss = a * loss_pde + (1.0 - a) * loss_tc_for_loss
        else:
            loss = loss_pde

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e12)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss and torch.isfinite(loss).item():
            best_loss = loss_val
            best_iter = it
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Log densely over the first 100 iterations (the transient where the
        # network learns to cancel the forcing happens fast) then every
        # log_every; this makes iteration 1 and the early dynamics resolvable on
        # a log-iteration axis.
        if it <= 100 or it % log_every == 0 or it == num_iterations:
            # Diagnostics (never enter the loss): terminal mismatch of the trial
            # solution and spatial-boundary drift vs the exact reference.
            with torch.no_grad():
                x_tc, t_tc = sample_terminal()
                u_tc = model(torch.stack([x_tc, t_tc], dim=1)).squeeze(-1)
                tc_err = ((u_tc - problem["terminal_datum"](x_tc)) ** 2).mean().item()
                x_bd, t_bd = sample_boundary()
                u_bd = model(torch.stack([x_bd, t_bd], dim=1)).squeeze(-1)
                bd_err = ((u_bd - problem["exact"](x_bd, t_bd)) ** 2).mean().item()
            history["iter"].append(it)
            history["loss"].append(loss_val)
            history["loss_pde"].append(loss_pde.item())
            history["loss_tc"].append(tc_err)
            history["boundary_error"].append(bd_err)
            history["network_energy"].append(decomp["network_energy"].item())
            history["cross_term"].append(decomp["cross_term"].item())
            history["forcing_floor"].append(decomp["forcing_floor"].item())
            history["forcing_velocity"].append(decomp["forcing_velocity"].item())
            history["forcing_diffusion"].append(decomp["forcing_diffusion"].item())
            history["grad_norm"].append(float(grad_norm))
            history["lr"].append(scheduler.get_last_lr()[0])
            logger.info(
                "[%s] it=%d loss=%.3e (pde=%.3e | tc_diag=%.3e bd_diag=%.3e | "
                "netE=%.3e cross=%.3e floor=%.3e)",
                variant["name"], it, loss_val, loss_pde.item(), tc_err, bd_err,
                decomp["network_energy"].item(), decomp["cross_term"].item(),
                decomp["forcing_floor"].item(),
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("[%s] restored best state from iter %d (loss=%.3e)",
                    variant["name"], best_iter, best_loss)

    history["best_iter"] = best_iter
    history["best_loss"] = best_loss
    return model, history


# ===========================================================================
# Metrics (vs exact reference)
# ===========================================================================

def compute_metrics(model, problem, *, n_x=200, n_t=100):
    """Relative-L2 errors of the trained trial solution against the exact solution."""
    import numpy as np
    import torch

    device = next(model.parameters()).device
    # Metrics are evaluated on the inner window (buffer from the sampling edges),
    # since the PDE is posed on R and no lateral boundary condition is imposed.
    x_lo, x_hi = problem["x_eval_lo"], problem["x_eval_hi"]
    T = problem["T"]
    xs = torch.linspace(x_lo, x_hi, n_x, device=device)
    ts = torch.linspace(0.0, T, n_t, device=device)
    XX, TT = torch.meshgrid(xs, ts, indexing="ij")
    xt = torch.stack([XX.reshape(-1), TT.reshape(-1)], dim=1)
    with torch.no_grad():
        u_pred = model(xt).squeeze(-1)
        u_ref = problem["exact"](xt[:, 0], xt[:, 1])
    err = u_pred - u_ref
    rel_l2 = (err.norm() / u_ref.norm()).item()

    # error at t = 0 (the propagated-back slice)
    x0 = torch.linspace(x_lo, x_hi, n_x, device=device)
    t0 = torch.zeros(n_x, device=device)
    with torch.no_grad():
        u0 = model(torch.stack([x0, t0], dim=1)).squeeze(-1)
        u0_ref = problem["exact"](x0, t0)
    rel_l2_t0 = ((u0 - u0_ref).norm() / u0_ref.norm()).item()

    # terminal mismatch at t = T (how well the form enforces the datum)
    tT = torch.full((n_x,), T, device=device)
    with torch.no_grad():
        uT = model(torch.stack([x0, tT], dim=1)).squeeze(-1)
        gT = problem["terminal_datum"](x0)
    tc_l2 = ((uT - gT).norm() / (gT.norm() + 1e-12)).item()

    return {
        "rel_l2": rel_l2,
        "rel_l2_t0": rel_l2_t0,
        "tc_l2": tc_l2,
    }


def compute_slices(model, problem, *, n_x=300):
    """Tabulate predicted vs exact solution slices for torch-free replotting.

    Saves the trial solution, the exact reference and the terminal datum on the
    initial slice ``t = 0`` and the terminal slice ``t = T`` so every figure can
    be rebuilt from disk without reloading the model.
    """
    import numpy as np
    import torch

    device = next(model.parameters()).device
    x_lo, x_hi, T = problem["x_lo"], problem["x_hi"], problem["T"]
    x = torch.linspace(x_lo, x_hi, n_x, device=device)
    out = {
        "x": x.cpu().numpy(),
        "x_eval_lo": np.asarray([problem["x_eval_lo"]]),
        "x_eval_hi": np.asarray([problem["x_eval_hi"]]),
    }
    for tag, t_val in (("t0", 0.0), ("tT", T)):
        t = torch.full((n_x,), t_val, device=device)
        with torch.no_grad():
            u_pred = model(torch.stack([x, t], dim=1)).squeeze(-1)
            u_ref = problem["exact"](x, t)
        out[f"u_pred_{tag}"] = u_pred.cpu().numpy()
        out[f"u_ref_{tag}"] = u_ref.cpu().numpy()
    with torch.no_grad():
        out["g"] = problem["terminal_datum"](x).cpu().numpy()
    return out


# ===========================================================================
# Persistence
# ===========================================================================

def save_variant(vdir: Path, model, history, metrics, slices):
    import numpy as np
    import torch

    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), vdir / "models" / "model.pt")
    hist_arrays = {k: np.asarray(v) for k, v in history.items() if isinstance(v, list)}
    np.savez_compressed(vdir / "hist.npz", **hist_arrays)
    np.savez_compressed(vdir / "metrics.npz", **{k: np.asarray([v]) for k, v in metrics.items()})
    np.savez_compressed(vdir / "slices.npz", **slices)


def write_summary(path: Path, payload: dict):
    import yaml

    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False, width=float("inf"))


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ic", choices=cat.ic_names(), default="sine",
                   help="Initial (terminal) condition / problem.")
    p.add_argument("--variant", choices=cat.variant_names(), default=None,
                   help="Single method variant (array-task mode). Omit to run all.")
    p.add_argument("--seed", type=int, default=0, help="Master seed.")
    p.add_argument("--num-iterations", type=int, default=None,
                   help="Override the iteration budget.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--debug", action="store_true",
                   help="Mark as a smoke/test run (prefixes output with _debug_).")
    p.add_argument("--replot", type=str, default=None,
                   help="Regenerate comparison plots from an existing run DIR (no training).")
    p.add_argument("--init-only", action="store_true",
                   help="Create dir + metadata + one config YAML per array task, "
                        "print the absolute EXPDIR, then exit (login-node safe).")
    p.add_argument("--ablation-dir", type=str, default=None,
                   help="Shared parent output dir (array mode / explicit override).")
    p.add_argument("--config-dir", type=str, default=None,
                   help="Folder of per-task YAML configs (array-worker contract).")
    p.add_argument("--config-name", type=str, default=None,
                   help="Basename (no extension) of the YAML config to load.")
    # hyperparameter overrides
    p.add_argument("--n-interior", type=int, default=None)
    p.add_argument("--n-terminal", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    return p


def resolve_hparams(args) -> dict:
    h = dict(cat.DEFAULT_HPARAMS)
    if args.num_iterations is not None:
        h["num_iterations"] = args.num_iterations
    if args.n_interior is not None:
        h["n_interior"] = args.n_interior
    if args.n_terminal is not None:
        h["n_terminal"] = args.n_terminal
    if args.learning_rate is not None:
        h["learning_rate"] = args.learning_rate
    return h


def main(argv=None) -> int:
    assert cat.RUNNER_SCRIPT_STEM == Path(__file__).stem, (
        f"RUNNER_SCRIPT_STEM={cat.RUNNER_SCRIPT_STEM!r} != script stem "
        f"{Path(__file__).stem!r}; rename one so the data folder cannot drift."
    )
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    from learning_option_pricing.utils.run_context import script_data_dir

    if args.replot is not None:
        from _ansatz_forms_plots import replot
        replot(Path(args.replot))
        return 0

    # Array-worker contract: a per-task YAML fully specifies the task.
    is_worker = args.config_dir is not None and args.config_name is not None
    cfg_hparams: dict = {}
    if is_worker:
        import yaml
        cfg_path = Path(args.config_dir) / f"{args.config_name}.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        args.ic = cfg["ic"]
        args.variant = cfg["variant"]
        args.seed = int(cfg["seed"])
        args.debug = bool(cfg.get("debug", args.debug))
        args.ablation_dir = cfg["ablation_dir"]
        cfg_hparams = cfg.get("hparams", {})

    hparams = resolve_hparams(args)
    hparams.update(cfg_hparams)

    # Smoke-test guard: short runs must be flagged.
    if (hparams["num_iterations"] < SMOKE_TEST_NUM_ITERATIONS_THRESHOLD
            and not args.debug):
        raise SystemExit(
            f"--num-iterations {hparams['num_iterations']} is below the smoke-test "
            f"threshold ({SMOKE_TEST_NUM_ITERATIONS_THRESHOLD}); pass --debug to "
            f"flag this as exploratory, or raise the iteration count."
        )

    debug_prefix = "_debug_" if args.debug else ""
    if args.ablation_dir is not None:
        ablation_dir = Path(args.ablation_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%fZ")
        ablation_dir = (
            script_data_dir(__file__)
            / (f"{debug_prefix}{ts}_{args.ic}"
               f"_iters{hparams['num_iterations']}_seed{args.seed}")
        )
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Self-contained run log (per-task in array mode to avoid interleaving).
    log_name = f"ablation_{args.variant}.log" if is_worker else "ablation.log"
    log_path = ablation_dir / log_name
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter(
        "%(asctime)sZ %(levelname)s [%(name)s] %(message)s", "%Y-%m-%dT%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    label = cat.ic_by_name(args.ic)["label"]

    # metadata (written once, at init / local-run; workers must not race on it).
    # This path is torch-free so it is safe to run on a cluster login node.
    if not is_worker:
        write_summary(ablation_dir / "metadata.yaml", {
            "command": " ".join(sys.argv),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ic": args.ic,
            "seed": args.seed,
            "hparams": hparams,
            "label": label,
            "variants": cat.variant_names(),
        })

    if args.init_only:
        # Write one config YAML per array task, then print the absolute EXPDIR on
        # stdout (logs go to stderr) for the launcher to capture.  No torch
        # import on this path -> safe on the login node.
        import yaml
        configs_dir = ablation_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        for v in cat.METHOD_VARIANTS:
            task_cfg = {
                "ic": args.ic,
                "variant": v["name"],
                "seed": args.seed,
                "debug": args.debug,
                "ablation_dir": str(ablation_dir.resolve()),
                "hparams": hparams,
            }
            with open(configs_dir / f"{v['name']}.yaml", "w") as f:
                yaml.dump(task_cfg, f, sort_keys=False)
        logger.info("init-only: wrote %d task configs to %s",
                    len(cat.METHOD_VARIANTS), configs_dir)
        print(str(ablation_dir.resolve()))
        return 0

    # --- training path: torch is needed from here on ---------------------
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Probe the GPU once up front so a momentarily-busy device is waited out
    # before any training work, rather than failing the task mid-build.
    if device.type == "cuda":
        cuda_retry(lambda: torch.zeros(1, device=device))

    logger.info("=" * 72)
    logger.info("ANSATZ-FORM ABLATION (backward heat equation)")
    logger.info("=" * 72)
    logger.info("  command:   %s", " ".join(sys.argv))
    logger.info("  python:    %s", sys.version.split()[0])
    logger.info("  torch:     %s", torch.__version__)
    logger.info("  cuda:      %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("  gpu:       %s", torch.cuda.get_device_name(0))
    logger.info("  device:    %s", device)
    logger.info("  ic:        %s", args.ic)
    logger.info("  seed:      %d", args.seed)
    logger.info("  hparams:   %s", hparams)
    logger.info("  output:    %s", ablation_dir)
    logger.info("  log:       %s", log_path)

    problem = build_problem(args.ic)
    problem["ic_name"] = args.ic

    variants_to_run = (
        [cat.variant_by_name(args.variant)] if args.variant is not None
        else list(cat.METHOD_VARIANTS)
    )
    log_every = max(1, hparams["num_iterations"] // 50)

    summary: dict = {}
    for variant in variants_to_run:
        t0 = time.time()
        model, history = train_variant(
            variant, problem, hparams,
            num_iterations=hparams["num_iterations"],
            seed=args.seed, device=device, log_every=log_every,
        )
        metrics = compute_metrics(model, problem)
        slices = compute_slices(model, problem)
        elapsed = time.time() - t0
        vdir = ablation_dir / f"variant_{variant['name']}"
        save_variant(vdir, model, history, metrics, slices)
        logger.info("[%s] done in %.1fs | metrics=%s", variant["name"], elapsed, metrics)
        summary[variant["name"]] = {
            **metrics,
            "best_loss": history["best_loss"],
            "best_iter": history["best_iter"],
            "wall_time_s": elapsed,
        }

    # In single-variant (array) mode write a per-variant summary to avoid races.
    summary_path = (
        ablation_dir / f"summary_{args.variant}.yaml" if args.variant is not None
        else ablation_dir / "summary.yaml"
    )
    write_summary(summary_path, summary)
    logger.info("wrote %s", summary_path)

    # Build comparison plots immediately when all variants ran in-process.
    if args.variant is None:
        try:
            from _ansatz_forms_plots import replot
            replot(ablation_dir)
        except Exception as exc:  # plotting must never lose a trained run
            logger.warning("plotting failed (artefacts are saved): %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
