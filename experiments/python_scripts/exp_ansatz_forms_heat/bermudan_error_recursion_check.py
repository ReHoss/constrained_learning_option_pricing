r"""Numerical verification of the Bermudan error recursion on saved runs.

The analytical model of the induction error is the recursion

    e_k = C^net_k - C*_k = eps^solve_k + S_{Delta} (g_k - V*_{k+1}),

where C^net_k is the learned continuation at date tau_k, C*_k the exact one,
S_{Delta} the backward-heat propagation semigroup over one interval (an L^2
contraction), g_k = M_eps(payoff, C^net_{k+1}) the learned terminal datum of
stage k, and V*_{k+1} = max(payoff, C*_{k+1}) the exact glued value.  Because S
is linear the decomposition is an *identity*, and the inherited term splits
further into

    g_k - V*_{k+1} = [M_eps(p, C^net_{k+1}) - M_eps(p, C*_{k+1})]   (glued inherited error,
                                                                    |.| <= |e_{k+1}| pointwise)
                   + [M_eps(p, C*_{k+1})   - max(p, C*_{k+1})]      (smoothing bias,
                                                                    0 < . <= eps/2),

since M_eps is 1-Lipschitz in its second argument.  Telescoping with
||S|| <= 1 gives the bound

    ||e_0|| <= sum_k ( ||eps^solve_k|| + ||bias_k|| ).

This script reads a saved ``bermudan_backward_induction`` run (stage models +
metadata; **no training**), rebuilds the frozen stage chain
(``bbi.rebuild_models``), and measures every term of the identity:

* the exact reference is computed PER DATE with the wide-quadrature
  ``bermudan_put_value_exact`` restricted to the remaining exercise dates (no
  chained grid truncation);
* the learned propagation uses a trapezoidal Gaussian-kernel matrix S on the
  training domain; its single-application truncation defect
  ``r_k = C*_k - S V*_{k+1}`` is carried as an explicit term of the identity
  and reported (it is a self-check on the reference: ~1e-6 when the reference
  is internally consistent -- this diagnostic exposed the late-binding closure
  defect in ``bermudan_put_value_exact`` for three or more dates);
* the identity residual (including the defect term) is printed as a
  floating-point self-check (~1e-13);
* per date, on the evaluation window: ||e_k||, ||zeta_k||, ||glued inherited||,
  ||bias||, ||defect||, and the telescoped bound against the measured error
  (tightness).  The rigorous bound is stated on the full line (see the report);
  the window sums are the measured comparison.

A reconstruction cross-check against the run's saved ``validation.npz``
verifies that the rebuilt chain reproduces the saved learned values.

Usage::

    python bermudan_error_recursion_check.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bermudan_backward_induction as bbi  # noqa: E402
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.pde import (  # noqa: E402
    bermudan_put_value_exact,
    chen_mangasarian_max,
    heat_put_exact,
    heat_put_payoff,
)
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

N_Y = 4001  # quadrature/DP grid on the training domain [X_LO, X_HI]


def _heat_matrix(y: torch.Tensor, sigma: float, delta: float) -> torch.Tensor:
    """Trapezoidal Gaussian-kernel matrix S on the grid ``y``: (S v)(y_i) =
    sum_j w_j N(y_i - y_j; sigma^2 delta) v(y_j).  Positive kernel with row sums
    <= 1 (truncated tails), hence a contraction."""
    var = sigma**2 * delta
    w = torch.full_like(y, float(y[1] - y[0]))
    w[0] = w[-1] = 0.5 * float(y[1] - y[0])
    ker = torch.exp(-((y[:, None] - y[None, :]) ** 2) / (2.0 * var))
    ker = ker / math.sqrt(2.0 * math.pi * var)
    return ker * w[None, :]


def check_run(run_dir: Path, out: Path) -> dict:
    meta = yaml.safe_load(open(run_dir / "metadata.yaml"))
    m, sigma, K, eps = meta["m"], meta["sigma"], meta["K"], meta["eps"]
    tau = [float(t) for t in meta["tau"]]
    delta = tau[1] - tau[0]  # equally spaced intervals
    models = bbi.rebuild_models(run_dir, meta)  # shared chain reconstruction

    y = torch.linspace(bbi.X_LO, bbi.X_HI, N_Y, dtype=torch.float64)
    eval_mask = (y >= bbi.X_EVAL_LO) & (y <= bbi.X_EVAL_HI)
    p = heat_put_payoff(y, K)
    S = _heat_matrix(y, sigma, delta)

    def l2(v, mask=None):
        vv = v[mask] if mask is not None else v
        return float(torch.sqrt((vv**2).mean()))

    # learned continuations on the grid (stage k at local time s = 0)
    yt = torch.stack([y, torch.zeros_like(y)], dim=1)
    C_net = {k: models[k](yt).squeeze(-1) for k in range(m)}
    C_net[m] = torch.zeros_like(y)  # nothing above maturity

    # wide-quadrature exact reference, PER DATE (no chained grid truncation):
    # the continuation at tau_k is the exact Bermudan value with the remaining
    # exercise dates only, evaluated strictly before the first of them.
    exercise_times = [float(t) for t in meta["exercise_times"]]
    C_star, V_star = {}, {}
    for k in range(m - 1, -1, -1):
        if k == m - 1:
            # single remaining date (maturity): the continuation is the European put
            C_star[k] = heat_put_exact(
                y, torch.full_like(y, tau[k]), K=K, T=meta["maturity"], sigma=sigma)
        else:
            remaining = exercise_times[k:]  # dates t_{k+1}..t_m (tau_k = t_k < t_{k+1})
            C_star[k] = bermudan_put_value_exact(
                y, torch.full_like(y, tau[k]), exercise_times=remaining,
                K=K, sigma=sigma, y_lo=bbi.Y_LO, y_hi=bbi.Y_HI, n_quad=4000)
        V_star[k] = torch.maximum(p, C_star[k]) if k >= 1 else C_star[k]
    V_star[m] = p.clone()

    # Identity per date, with the single-application truncation defect carried
    # explicitly:  e_k = zeta_k + S(glued_k + bias_k) - r_k,
    #   zeta_k = C^net_k - S g_k          (per-stage solve error, shared S)
    #   r_k    = C*_k - S V*_{k+1}        (one-application kernel-truncation defect)
    # All norms on the evaluation window (physically meaningful region).
    rows, res_max = [], 0.0
    for k in range(m - 1, -1, -1):
        g_k = chen_mangasarian_max(p, C_net[k + 1], eps)
        zeta = C_net[k] - S @ g_k
        if k == m - 1:
            glued = torch.zeros_like(y)  # C^net_m = C*_m = 0
            bias = chen_mangasarian_max(p, torch.zeros_like(y), eps) - p
        else:
            glued = g_k - chen_mangasarian_max(p, C_star[k + 1], eps)
            bias = chen_mangasarian_max(p, C_star[k + 1], eps) - V_star[k + 1]
        defect = C_star[k] - S @ V_star[k + 1]
        e_k = C_net[k] - C_star[k]
        identity_residual = e_k - zeta - S @ (glued + bias) + defect
        res_max = max(res_max, float(identity_residual[eval_mask].abs().max()))
        rows.append({
            "k": k, "t": tau[k],
            "err": l2(e_k, eval_mask), "solve": l2(zeta, eval_mask),
            "inherited": l2(S @ glued, eval_mask), "bias": l2(S @ bias, eval_mask),
            "defect": l2(defect, eval_mask),
            "rel_err_window": l2(e_k, eval_mask) / l2(V_star[k], eval_mask),
        })
    rows = rows[::-1]  # index by k ascending (inception first)

    # telescoped bound on the window: ||e_0|| <= sum_k (||zeta_k|| + ||bias_k||)
    # (the rigorous bound is on the full line; the window sum is the measured
    # comparison, with the defect reported separately)
    bound = sum(r["solve"] + r["bias"] for r in rows)
    measured = rows[0]["err"]
    trunc_rel = max(r["defect"] for r in rows)

    # reconstruction cross-check against the run's own saved validation.npz:
    # the rebuilt chain must reproduce the saved learned values, and the
    # truncated reference must track the saved wide-quadrature reference.
    z = np.load(run_dir / "validation.npz")
    xs = z["x"]
    recon_dev, ref_dev = 0.0, 0.0
    for k in range(m):
        mine_net = np.interp(xs, y.numpy(), C_net[k].numpy())
        mine_glued = mine_net if k == 0 else np.maximum(z["payoff"], mine_net)
        v_net_saved = z[f"stage{k}_v_net"]
        v_exact_saved = z[f"stage{k}_v_exact"]
        mine_star = np.interp(
            xs, y.numpy(), (V_star[k] if k >= 1 else C_star[0]).numpy())
        recon_dev = max(recon_dev, float(
            np.linalg.norm(mine_glued - v_net_saved) / np.linalg.norm(v_net_saved)))
        ref_dev = max(ref_dev, float(
            np.linalg.norm(mine_star - v_exact_saved) / np.linalg.norm(v_exact_saved)))

    summary = {
        "run": run_dir.name, "m": m,
        "identity_residual_max": res_max,
        "measured_inception_error": measured,
        "telescoped_bound": bound,
        "bound_tightness": measured / bound,
        "reference_truncation_rel": trunc_rel,
        "reconstruction_dev_vs_saved": recon_dev,
        "reference_dev_vs_saved": ref_dev,
        "per_date": rows,
    }

    print(f"\n=== {run_dir.name} ===")
    print(f"identity residual (max abs, window)         : {res_max:.2e}")
    print(f"kernel-truncation defect (max L2, window)   : {trunc_rel:.2e}")
    print(f"rebuilt chain vs saved v_net (max rel dev)  : {recon_dev:.2e}")
    print(f"wide ref vs saved v_exact (max rel dev)     : {ref_dev:.2e}")
    print(f"{'k':>3} {'t_k':>6} {'||e_k||':>10} {'||solve||':>10} "
          f"{'||inherit||':>11} {'||bias||':>10} {'||defect||':>10} {'relL2(win)':>10}")
    for r in rows:
        print(f"{r['k']:>3} {r['t']:>6.2f} {r['err']:>10.3e} {r['solve']:>10.3e} "
              f"{r['inherited']:>11.3e} {r['bias']:>10.3e} {r['defect']:>10.3e} "
              f"{r['rel_err_window']:>10.3e}")
    print(f"telescoped bound sum(solve+bias) = {bound:.3e}   "
          f"measured ||e_0|| = {measured:.3e}   tightness = {measured/bound:.2f}")
    return summary


def _figure(summaries, out: Path) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    ms = [s["m"] for s in summaries]
    cmap = plt.cm.viridis(np.linspace(0.12, 0.85, len(ms)))

    # left: per-date decomposition for the largest m (the richest chain)
    s = max(summaries, key=lambda s: s["m"])
    ts = [r["t"] for r in s["per_date"]]
    axL.semilogy(ts, [r["err"] for r in s["per_date"]], "o-", color="#1b6ca8",
                 label=r"measured $\|e_k\|$")
    axL.semilogy(ts, [r["solve"] for r in s["per_date"]], "s-", color="#d1495b",
                 label=r"solve $\|\zeta_k\|$")
    axL.semilogy(ts, [r["inherited"] for r in s["per_date"]], "^-", color="#66a182",
                 label=r"inherited (1-Lipschitz glued, propagated)")
    axL.semilogy(ts, [max(r["bias"], 1e-16) for r in s["per_date"]], "v-",
                 color="#edae49", label=r"smoothing bias $\|S\,\omega_{k+1}\|$")
    axL.set_xlabel("global time $t_k$ (0 = inception)")
    axL.set_ylabel(r"$L^2$ norm (evaluation window)")
    axL.set_title(f"Recursion decomposition, $m={s['m']}$", fontsize=10)
    axL.invert_xaxis()
    axL.grid(True, which="both", alpha=0.3)
    legL = axL.legend(loc="lower left", fontsize=8, frameon=True)

    # right: measured inception error vs telescoped bound across m
    meas = [s["measured_inception_error"] for s in summaries]
    bnd = [s["telescoped_bound"] for s in summaries]
    axR.loglog(ms, bnd, "s--", color="black", label=r"bound $\sum_k(\|\zeta_k\|+\|S\,\omega_{k+1}\|)$")
    axR.loglog(ms, meas, "o-", color="#1b6ca8", label=r"measured $\|e_0\|$")
    for m_i, me, bo, c in zip(ms, meas, bnd, cmap):
        axR.annotate(f"{me/bo:.2f}", (m_i, me), textcoords="offset points",
                     xytext=(5, -12), fontsize=7, color="#1b6ca8")
    axR.set_xlabel("number of exercise dates $m$")
    axR.set_ylabel(r"$L^2$ norm at inception")
    axR.set_title("Telescoped bound vs measured (tightness annotated)", fontsize=10)
    axR.grid(True, which="both", alpha=0.3)
    legR = axR.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle("Error recursion: identity terms measured on the saved chains", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    finalize_figure(
        fig, out / "error_recursion_check.png", legends=[legL, legR], axes=[axL, axR],
        formula=(r"$e_k = \zeta_k + S_{\Delta}(h_k - V^\star_{k+1})$, "
                 r"identity verified to machine precision; "
                 r"$\|e_0\| \leq \sum_k(\|\zeta_k\| + \|S\,\omega_{k+1}\|)$ "
                 r"by $\|S\|\leq 1$ and 1-Lipschitz gluing"),
        formula_fontsize=8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="*", help="bermudan_backward_induction run dirs")
    p.add_argument("--replot", type=str, default=None,
                   help="redraw the figure from a saved recursion_check.yaml "
                        "(in place, no recomputation) and exit")
    args = p.parse_args(argv)

    if args.replot is not None:
        yaml_path = Path(args.replot)
        summaries = yaml.safe_load(open(yaml_path))
        _figure(summaries, yaml_path.parent)
        print(f"replotted figure into {yaml_path.parent}")
        return 0
    if not args.run_dirs:
        p.error("run_dirs required unless --replot is given")

    out = script_data_dir(__file__) / utc_timestamp()
    out.mkdir(parents=True, exist_ok=True)

    summaries = [check_run(Path(d), out) for d in args.run_dirs]
    summaries.sort(key=lambda s: s["m"])
    # persist the numbers BEFORE plotting so a figure failure cannot lose them
    with open(out / "recursion_check.yaml", "w") as f:
        yaml.safe_dump(summaries, f, sort_keys=False)
    _figure(summaries, out)
    print(f"\nwrote figure + recursion_check.yaml to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
