"""Torch-free catalogue + init helpers for ``ablation_singularity_logS.py``.

This module is intentionally **kept free of any torch import** so that

1. the ``--init-only`` fast-path at the top of ``ablation_singularity_logS.py``
   can run on a login node (or any cheap CPU) in a fraction of a second,
   instead of paying the multi-second ``import torch`` cost on Lustre, and

2. the generic SLURM array launcher
   (``bash_scripts/cluster/jeanzay/python/experiment_array_launcher.sh``) can
   delegate the per-task YAML generation to ``--init-only`` on the login
   node without loading torch either.

What's in here:
* The BSM / domain / sampling constants used by the variant catalogue.
* The colour palette and grid helpers used by the ablation modes.
* ``_PLOT_EXCLUDED_VARIANTS`` — single source of truth for variants that
  should be skipped by the comparison plots.
* ``_build_variants(mode)`` — the variant catalogue function (copied here
  verbatim — see the runtime sanity check in ``ablation_singularity_logS``
  that guards against drift from ``phase3_training``).
* ``data_root_for_mode(mode)`` — mode → top-level data folder routing.
* ``handle_init_only_cli()`` — the implementation of the ``--init-only``
  flag, lifted out of ``ablation_singularity_logS.main`` so it can run
  before torch is even imported.

Constants are hard-coded here (rather than imported from
``phase3_training``) because importing phase3_training would transitively
load torch and defeat the whole point of this module.  The runtime
assertion in ``ablation_singularity_logS`` checks the two stay in sync.
"""
from __future__ import annotations

import math

# ── BSM parameters (mirror of ``phase3_training``) ──────────────────────────
K     = 100.0
r     = 0.02
sigma = 0.25
T     = 1.0
q     = 0.0

# ── Domain in (x = ln S, t) coordinates ─────────────────────────────────────
S_LO,      S_HI      = 20.0,  160.0
S_EVAL_LO, S_EVAL_HI = 60.0,  140.0
X_LO      = math.log(S_LO)        # ≈ 3.00
X_HI      = math.log(S_HI)        # ≈ 5.08
X_EVAL_LO = math.log(S_EVAL_LO)   # ≈ 4.09
X_EVAL_HI = math.log(S_EVAL_HI)   # ≈ 4.94
X_ATM     = math.log(K)           # ≈ 4.61

# ── Default training set sizes (mirror of ``phase3_training``) ──────────────
N_TC = 1024
N_F  = 4 * N_TC

# ── Variant-catalogue helpers ───────────────────────────────────────────────
_EPS_GRID   = [0.005 * T, 0.01 * T, 0.02 * T, 0.05 * T, 0.10 * T]
_BETA_GRID  = [10, 50, 100, 500, 1000]
_IS_CONFIGS = [(2.0, 0.5), (5.0, 0.5), (10.0, 0.5)]
_COLORS     = ["tab:blue", "tab:orange", "tab:green", "tab:red",
               "tab:purple", "tab:brown", "tab:pink", "tab:gray",
               "tab:olive", "tab:cyan", "steelblue", "coral"]

# ── Variants excluded from the ablation pipeline ────────────────────────────
# Despite the historical name, this set now gates both *training* and
# *plotting*: variants listed here are skipped by the per-task YAML writer in
# ``handle_init_only_cli`` (so SLURM array launchers do not waste GPU on them)
# and filtered out of every comparison figure by ``_plot_comparison`` /
# ``_replot``.  Catalogue entries themselves are preserved so the
# hyperparameters and the rationale for the exclusion stay reviewable and
# can be re-activated by simply removing the name from this set.
_PLOT_EXCLUDED_VARIANTS: set[str] = {
    "vpinn_lbfgs_is_tau",  # importance-sampling τ→0 biased variant — removed
                           # from plots pending a fair comparison study
    "vpinn_lbfgs",         # stochastic-batch L-BFGS — superseded by
                           # vpinn_lbfgs_full_batch which is strictly better
                           # on all metrics (rel_L2, delta, gamma, GEI)
    "engd",                # strong-form ENGD — Gram matrix singular on this
                           # problem; needs more regularisation before it can
                           # be revived (kept in the catalogue for the future
                           # fix attempt, but excluded from active ablations)
    "vpinn_engd",          # weak-form ENGD — same singular-Gram issue as the
                           # strong-form counterpart; excluded for the same
                           # reason
}


# ── Mode → data folder routing ──────────────────────────────────────────────

# Name of the runner script that owns the modes below.  Hard-coded as a string
# (rather than imported from ``learning_option_pricing.utils.run_context``) so
# that this module remains torch-free — see the module docstring at the top of
# the file for why that matters.  The runtime assertion in
# ``ablation_singularity_logS.main`` checks the runner's ``__file__`` against
# this constant so the two cannot silently drift apart.
RUNNER_SCRIPT_STEM = "ablation_singularity_logS"


def data_root_for_mode(mode: str) -> str:
    """Top-level data directory for a given ``mode`` of the
    ``ablation_singularity_logS`` runner.

    Per the per-script data-folder convention in ``CLAUDE.md``, every run lands
    under ``data/<runner_stem>/``.  The ``mode`` then becomes a sub-folder so
    Hard-IC experiments and the boundary-singularity comparison cannot mix
    their ``--replot`` / ``--add-variant`` targets by accident:

        data/ablation_singularity_logS/hard-ic-ansatz-european-call/<ts>_..._
        data/ablation_singularity_logS/compare-boundary-singularity-european-call/<ts>_..._

    Returns:
        Path string (kept as ``str`` for byte-for-byte compatibility with
        callers that build paths via ``Path(...) / ts_tag``).
    """
    return f"data/{RUNNER_SCRIPT_STEM}/{mode}"


# ── Variant catalogue ───────────────────────────────────────────────────────

def _build_variants(mode: str) -> list[dict]:
    """Return the list of variant config dicts for the requested mode.

    Each entry is a plain ``dict`` of metadata and hyperparameters — no
    tensors, no torch references — so this function is safe to call from
    the init-only path.
    """
    # ── Color families ────────────────────────────────────────────────────────
    # Strong-form PINNs  → blue palette   (#0d47a1 dark … #42a5f5 light)
    # Weak-form VPINNs   → red/warm palette (#b71c1c dark … #ff7043 orange-red)
    # Analytical BS ref  → black (enforced in each plotting function)
    # All learned models share linestyle="-" so they form a unified visual group.
    # ──────────────────────────────────────────────────────────────────────────
    naive_cfg = dict(
        name="naive", label="Naive (control)",
        sampler_type="naive", payoff_type="exact",
        eps=0.0, beta=None, sigma_is=None, mix=0.0,
        # default_num_iterations: every variant declares its own natural
        # training budget so that the CLI no longer needs --num-iterations
        # for production runs.  Adam strong-form: 100_000 is the budget
        # the comparison-boundary-singularity ablation runs at (aligned
        # with the hard-IC ablation for cross-comparability); below
        # ~2_000 the loss is still in rapid descent.
        default_num_iterations=100000,
        color="#0d47a1", linestyle="-", linewidth=2.0,
    )

    if mode == "compare-boundary-singularity-european-call":
        return [
            naive_cfg,
            dict(name="truncated", label=r"$\varepsilon$-trunc. ($\varepsilon=1\%T$)",
                 sampler_type="truncated", payoff_type="exact",
                 eps=0.01*T, beta=None, sigma_is=None, mix=0.0,
                 default_num_iterations=100000,
                 color="#1976d2", linestyle="-", linewidth=2.0),
            dict(name="smooth", label=r"Smooth ($\beta=100$)",
                 sampler_type="naive", payoff_type="smooth",
                 eps=0.0, beta=100, sigma_is=None, mix=0.0,
                 default_num_iterations=100000,
                 color="#42a5f5", linestyle="-", linewidth=2.0),
            dict(name="vpinn", label="VPINN (weak form)",
                 sampler_type="vpinn", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 # n_tau=512: more temporal coverage (was 128 → 32x undersampled vs n_f=4096)
                 # lam_f=200: VPINN Lf is ~10x smaller than strong-form Lf; rescale so
                 #   PDE contributes ~50% of the gradient (was 24% with lam_f=20)
                 # eps=0.0: no temporal truncation — train on full [0,T] including singularity
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 default_num_iterations=100000,
                 color="#b71c1c", linestyle="-", linewidth=2.0),
            dict(name="vpinn_engd", label="VPINN + ENGD (nat. grad.)",
                 sampler_type="vpinn_engd", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # ENGD steps are ~100× more expensive per iteration than Adam;
                 # 1000 natural-gradient steps ≈ 20k Adam steps wall-clock.
                 default_num_iterations=1000,
                 color="#ff7043", linestyle="-", linewidth=2.0),
            dict(name="engd", label="Strong-form + ENGD (paper-faithful, lstsq)",
                 sampler_type="engd", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 # Paper-faithful: small network (129 params), fixed (N-2)^2=784
                 # interior + N-1=29 terminal points, lstsq solve (no Tikhonov).
                 # M/n_params ≈ 6 — same regime as Zeinhofer et al. ICML 2023.
                 n_grid=30,
                 tikhonov_rel=1e-6,
                 default_num_iterations=1000,
                 color="#7986cb", linestyle="-", linewidth=2.0),
            # Note: two failed variants explored during diagnostics —
            #   `engd_tc_dense` (N_tc=200 vs 29)        : marginal, same trap
            #   `engd_alt` (alternating G_F / G_TC)     : never reaches J^T r=0
            # Documented in documents/methodology/engd_singularity_diagnostic.md
            # and removed from the catalog.
            dict(name="vpinn_lbfgs", label="VPINN + L-BFGS (stoch. batch)",
                 sampler_type="vpinn_lbfgs", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # Excluded from comparison plots (see _PLOT_EXCLUDED_VARIANTS).
                 # Kept here for ad-hoc inspection.  500 outer L-BFGS steps is
                 # enough to expose the curvature-noise pathology of the
                 # stochastic-batch variant — running longer just wastes GPU.
                 default_num_iterations=500,
                 color="#e53935", linestyle="-", linewidth=2.0),
            dict(name="vpinn_lbfgs_is_tau",
                 label=r"VPINN + L-BFGS (biased $\tau\to 0$ sampling, $\alpha=0.3$)",
                 sampler_type="vpinn_lbfgs_is_tau", payoff_type="exact",
                 eps=0.001 * T, beta=None, sigma_is=None, mix=0.0,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # τ = T·U^(1/0.3) with U~U(0,1) — concentrates time samples near
                 # the maturity singularity. The estimator is intentionally biased
                 # (no IS correction) so that γ near τ=0 carries more weight.
                 is_tau_alpha=0.3,
                 default_num_iterations=500,
                 color="#ff7043", linestyle="-", linewidth=2.0),
            dict(name="vpinn_lbfgs_full_batch",
                 label="VPINN + L-BFGS (fixed full batch)",
                 sampler_type="vpinn_lbfgs_full_batch", payoff_type="exact",
                 eps=0.0, beta=None, sigma_is=None, mix=0.0,
                 # Same 512 time points drawn once before the loop and held fixed
                 # for the entire run.  The objective is deterministic across outer
                 # L-BFGS steps → secant condition is always valid → curvature
                 # history is reliable.  Compare against vpinn_lbfgs (stochastic)
                 # to isolate the effect of batch noise on the quasi-Newton update.
                 # Convergence observed at ~266 outer steps; 500 is a safe cap.
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 default_num_iterations=500,
                 color="#ff7043", linestyle="-", linewidth=2.0),
            dict(name="vpinn_lbfgs_is_tau_full_batch",
                 label=r"VPINN + L-BFGS (IS $\tau\to0$, fixed full batch)",
                 sampler_type="vpinn_lbfgs_is_tau_full_batch", payoff_type="exact",
                 eps=0.001 * T, beta=None, sigma_is=None, mix=0.0,
                 # Combines IS τ→0 biased sampling with a fixed batch:
                 # — IS τ→0 (α=0.3): concentrates the 512 time points near the
                 #   maturity singularity, improving γ resolution near τ=0.
                 # — Fixed batch: the same biased sample is reused for every outer
                 #   L-BFGS step → deterministic objective → valid secant condition.
                 # — Full convergence observed at ~266 steps on the uniform variant;
                 #   500 steps aligns with the uniform sibling's safe cap.
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 is_tau_alpha=0.3,
                 default_num_iterations=500,
                 color="#7b1fa2", linestyle="-", linewidth=2.0),
        ]

    if mode == "hard-ic-ansatz-european-call":
        # ──────────────────────────────────────────────────────────────────
        # Hard-IC ansatz experiment — mirrors the variant set of
        # "compare-boundary-singularity-european-call" but with the model
        # replaced by the hard-IC ansatz:
        #
        #     V(x, t) = ((T - t)/T) · NN(x, t) + g_2(x)
        #
        # where g_2 is the exact payoff (or its softplus smoothing for the
        # `smooth` analogue).  Since V(·, T) = g_2(·) bit-for-bit, the
        # L_ic / quadrature-IC penalty is zero by construction.  All
        # variants below carry lam_tc=0 to make the intent explicit (and to
        # short-circuit the IC computation for the L-BFGS variants where
        # the override is already plumbed through).
        # ──────────────────────────────────────────────────────────────────
        _hard_ic_common = dict(model_type="hard_ic_ansatz",
                               sigma_is=None, mix=0.0, lam_tc=0.0)
        return [
            # ── Strong-form variants ────────────────────────────────────
            dict(name="hard_ic_naive",
                 label="Hard-IC ansatz — Naïve (strong form)",
                 sampler_type="naive", payoff_type="exact",
                 eps=0.0, beta=None,
                 default_num_iterations=100000,
                 color="#0d47a1", linestyle="-", linewidth=2.0,
                 **_hard_ic_common),
            dict(name="hard_ic_truncated",
                 label=r"Hard-IC ansatz — $\varepsilon$-trunc. ($\varepsilon=1\%T$)",
                 sampler_type="truncated", payoff_type="exact",
                 eps=0.01 * T, beta=None,
                 default_num_iterations=100000,
                 color="#1976d2", linestyle="-", linewidth=2.0,
                 **_hard_ic_common),
            dict(name="hard_ic_smooth",
                 label=r"Hard-IC ansatz — Smooth $g_2$ ($\beta=100$)",
                 # The softplus is moved INTO the ansatz (g2) rather than the
                 # IC loss (which no longer exists): V(x,T) = softplus(e^x-K).
                 sampler_type="naive", payoff_type="smooth",
                 eps=0.0, beta=100,
                 default_num_iterations=100000,
                 color="#42a5f5", linestyle="-", linewidth=2.0,
                 **_hard_ic_common),
            # ── Weak-form (VPINN) variants ──────────────────────────────
            dict(name="hard_ic_vpinn",
                 label="Hard-IC ansatz — VPINN (weak form, Adam)",
                 sampler_type="vpinn", payoff_type="exact",
                 eps=0.0, beta=None,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 default_num_iterations=100000,
                 color="#b71c1c", linestyle="-", linewidth=2.0,
                 **_hard_ic_common),
            dict(name="hard_ic_vpinn_lbfgs_full_batch",
                 label="Hard-IC ansatz — VPINN + L-BFGS (uniform, fixed batch)",
                 sampler_type="vpinn_lbfgs_full_batch", payoff_type="exact",
                 eps=0.0, beta=None,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 # L-BFGS uniform converges around outer step ~266 on this
                 # problem; 500 is a safe ceiling.
                 default_num_iterations=500,
                 color="#ff7043", linestyle="-", linewidth=2.0,
                 **_hard_ic_common),
            dict(name="hard_ic_vpinn_lbfgs_is_tau_full_batch",
                 label=r"Hard-IC ansatz — VPINN + L-BFGS (IS $\tau\to0$, fixed batch)",
                 sampler_type="vpinn_lbfgs_is_tau_full_batch", payoff_type="exact",
                 eps=0.001 * T, beta=None,
                 n_tau=512, K_test=20, n_quad=100, lam_f=200.0,
                 is_tau_alpha=0.3,
                 default_num_iterations=500,
                 color="#7b1fa2", linestyle="-", linewidth=2.0,
                 **_hard_ic_common),
        ]

    if mode == "ablation-eps":
        variants = [naive_cfg]
        for i, eps in enumerate(_EPS_GRID):
            pct = int(round(eps / T * 100))
            variants.append(dict(
                name=f"trunc_{pct}pct", label=rf"$\varepsilon={pct}\%T$",
                sampler_type="truncated", payoff_type="exact",
                eps=eps, beta=None, sigma_is=None, mix=0.0,
                default_num_iterations=20000,
                color=_COLORS[i+1], linestyle="--", linewidth=1.8,
            ))
        return variants

    if mode == "ablation-beta":
        variants = [naive_cfg]
        for i, beta in enumerate(_BETA_GRID):
            variants.append(dict(
                name=f"smooth_b{beta}", label=rf"$\beta={beta}$",
                sampler_type="naive", payoff_type="smooth",
                eps=0.0, beta=beta, sigma_is=None, mix=0.0,
                default_num_iterations=20000,
                color=_COLORS[i+1], linestyle="-.", linewidth=1.8,
            ))
        return variants

    if mode == "ablation-is":
        variants = [
            naive_cfg,
            dict(name="trunc_1pct", label=r"Trunc. unif.",
                 sampler_type="truncated", payoff_type="exact",
                 eps=0.01*T, beta=None, sigma_is=None, mix=0.0,
                 default_num_iterations=20000,
                 color="tab:orange", linestyle="--", linewidth=2.0),
        ]
        for i, (sig, mix) in enumerate(_IS_CONFIGS):
            variants.append(dict(
                name=f"is_sig{int(sig)}_mix{int(mix*100)}",
                label=rf"IS $\sigma_x={sig}$, mix={mix}",
                sampler_type="importance", payoff_type="exact",
                eps=0.01*T, beta=None, sigma_is=sig, mix=mix,
                default_num_iterations=20000,
                color=_COLORS[i+2], linestyle=":", linewidth=1.8,
            ))
        return variants

    raise ValueError(f"Unknown mode: {mode!r}")


# ── Init-only fast path ─────────────────────────────────────────────────────

def handle_init_only_cli() -> None:
    """Torch-free implementation of ``--init-only``.

    Called from the top of ``ablation_singularity_logS.py`` *before* torch
    is imported.  Parses the relevant CLI flags directly with argparse,
    creates the timestamped ablation directory + per-variant subdirectories
    + metadata.yaml + empty summary.yaml, then prints the absolute path on
    stdout (last line) so a bash launcher can capture it with::

        EXPDIR=$(python ... --init-only --device cpu | tail -n1)

    The output is *byte-for-byte* compatible with what the in-function
    branch in ``main()`` used to produce; the in-function branch is kept
    as a safety net for programmatic callers that bypass the top-level
    fast path (e.g. ``runpy.run_module``).
    """
    import argparse
    import logging
    import sys
    from datetime import datetime
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--num-iterations", dest="num_iterations",
                        type=int, default=None,
                        help=("Global override for every variant's "
                              "default_num_iterations. Omit to let each "
                              "variant use its own natural budget."))
    parser.add_argument("--seed", dest="seed", type=int, default=0,
                        help=("Ablation-wide master seed (default 0). "
                              "Persisted to metadata.yaml so subsequent "
                              "--add-variant jobs reuse exactly this value."))
    parser.add_argument("--variant", default=None)
    parser.add_argument("--n-tc", dest="n_tc", type=int, default=None)
    parser.add_argument("--n-f",  dest="n_f",  type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--init-only", dest="init_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # ``parse_known_args`` so that flags we don't care about here
    # (e.g. --add-variant, --replot) don't trip the parser.
    args, _ = parser.parse_known_args()

    n_tc = args.n_tc if args.n_tc is not None else N_TC
    n_f  = args.n_f  if args.n_f  is not None else N_F

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    variant_suffix = f"_variant_{args.variant}" if args.variant else ""
    # Test / smoke runs prepend ``_debug_``.  Real runs start with a digit
    # (the timestamp) and the underscore sorts after digits in C/UTF-8
    # locale, so debug runs land in a contiguous block at the bottom of
    # ``ls`` — visually separated from real runs without manual filtering,
    # and trivially wipeable with
    #     find data -type d -name '_debug_*' -prune -exec rm -rf {} +
    debug_prefix = "_debug_" if args.debug else ""
    # The folder name only carries an explicit iteration count when the
    # user provided a global ``--num-iterations`` override; otherwise each
    # variant uses its own ``default_num_iterations`` and a single global
    # number in the path would be misleading.
    num_iterations_tag = (
        f"_num_iterations_{args.num_iterations}"
        if args.num_iterations is not None else ""
    )
    data_root = Path(data_root_for_mode(args.mode))
    ablation_dir = (
        data_root
        / f"{debug_prefix}{timestamp}_{args.mode}_logS{num_iterations_tag}{variant_suffix}"
    )
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "comparison").mkdir(exist_ok=True)

    variants = _build_variants(args.mode)
    if args.variant is not None:
        matching = [v for v in variants if v["name"] == args.variant]
        if not matching:
            raise SystemExit(
                f"--variant {args.variant!r} not found in mode {args.mode!r}. "
                f"Available: {[v['name'] for v in variants]}"
            )
        variants = matching
    for v in variants:
        for sub in ("training_metrics", "models"):
            (ablation_dir / f"variant_{v['name']}" / sub).mkdir(parents=True, exist_ok=True)

    log_file = ablation_dir / "ablation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info(
        f"--init-only (torch-free fast path): mode={args.mode} "
        f"num_iterations={args.num_iterations!r} "
        f"(None = per-variant default_num_iterations)"
    )
    logger.info(f"master_seed={args.seed}  (shared across every variant of this ablation)")
    logger.info(f"output: {ablation_dir}")

    # ``metadata.yaml`` — shared with the main-script version.  ``num_iterations``
    # is the user-supplied global override (or null when each variant should use
    # its own ``default_num_iterations``); the actual iteration count per variant
    # is recorded in that variant's hist.npz / log file.
    with open(ablation_dir / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "cmdline":         sys.argv,
            "mode":            args.mode,
            "num_iterations":  args.num_iterations,
            "master_seed":     args.seed,
            "coords":          "logS",
            "device":          args.device,
            "n_tc":            n_tc,
            "n_f":             n_f,
            "K":               K,  "r": r, "sigma": sigma, "T": T,
            "x_lo":            X_LO,      "x_hi":      X_HI,      "x_atm":      X_ATM,
            "x_eval_lo":       X_EVAL_LO, "x_eval_hi": X_EVAL_HI,
        }, f)

    # Empty ``summary.yaml`` (populated by subsequent ``--add-variant`` jobs).
    with open(ablation_dir / "summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"variants": []}, f, allow_unicode=True)

    # ``configs/`` — one YAML per active variant, consumed downstream by the
    # generic SLURM-array worker via ``--config-dir / --config-name``.  Writing
    # them here (instead of in the bash launcher) lets every orchestrator that
    # respects the experiment-launcher contract (i.e. "run --init-only on the
    # login node, then array-launch each YAML it produced under configs/")
    # work against this script without embedding catalogue-specific Python.
    configs_dir = ablation_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    n_configs_written = 0
    for v in variants:
        if v["name"] in _PLOT_EXCLUDED_VARIANTS:
            continue
        cfg = {
            "mode":            args.mode,
            "variant_name":    v["name"],
            "ablation_dir":    str(ablation_dir.resolve()),
            # ``null`` instructs the worker to use the variant's catalogue
            # ``default_num_iterations``; edit the YAML between init and
            # array submission to override on the cluster.
            "num_iterations":  None,
        }
        with open(configs_dir / f"{v['name']}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        n_configs_written += 1

    logger.info(
        "--init-only: created ablation dir, metadata.yaml, empty "
        f"summary.yaml, and {n_configs_written} per-variant configs under "
        f"{configs_dir.name}/.  No variant trained."
    )
    # Path on the *last* stdout line — bash launchers capture it via tail.
    print(str(ablation_dir.resolve()))
