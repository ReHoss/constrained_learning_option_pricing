r"""Variational check of the scalar-coefficient forcing floor.

The eigenfunction-characterisation proposition states, for a datum
g = sum_j c_j w_j expanded in an orthonormal eigenbasis of the generator
(L w_j = -rho_j w_j) and any scalar interpolation coefficient lam in
C^1([0,T]) with lam(T) = 1:

  * the extension forcing P(lam g) vanishes for some lam iff the spectral
    support of g is a singleton;
  * with two spectral components rho_a < rho_b the forcing norm is bounded
    below by  min(|c_a|,|c_b|) (rho_b - rho_a) kappa_2
              / (2 (1 + (rho_b - rho_a) kappa_1))  > 0.

This script tests the bound against the TRUE optimum over the whole scalar
family: the squared forcing norm

    J(lam) = sum_j c_j^2 int_0^T (lam'(t) - rho_j lam(t))^2 dt

is quadratic in lam, so its minimiser subject to lam(T) = 1 is obtained
exactly (up to grid resolution) by solving one banded linear system on a time
grid -- no training, no descent.  Checks:

  A. single mode: the minimum is numerically zero and the minimiser matches
     the exponential  e^{rho (t-T)}  (proposition part (i));
  B. two-mode sweep over f (spectral gap rho_f - rho_1): the analytic lower
     bound holds below the numerical optimum, which lies below the
     mode-1-matched coefficient's forcing (upper reference); ratios reported;
  C. the study's data: the optimal-coefficient floor for the two-mode sine
     instance (c = 0.5, f = 4, T = 0.1) and the theta_3 instance
     (c_n = 2 e^{-n^2}, T = 0.1), compared with the linear and
     matched-exponential coefficients of the ablation catalogue -- how much
     of the fixed-family floor is intrinsic to the scalar restriction.

Torch-free (numpy + scipy banded solve); figure + optimal_floor.yaml.
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
import yaml  # noqa: E402
from scipy.linalg import solve_banded  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

SIGMA = 1.0
N_T = 4001


def rho(n: float, sigma: float = SIGMA) -> float:
    return 0.5 * sigma**2 * (n * math.pi) ** 2


def optimal_coefficient(rates, weights, T: float, n_t: int = N_T):
    """Exact minimiser of J(lam) = sum_j w_j^2 int (lam' - rho_j lam)^2 dt with
    lam(T) = 1, by assembling the quadratic form on a uniform grid (midpoint
    elements) and solving the banded normal equations with the last node
    eliminated.  Returns (t, lam, J_min)."""
    t = np.linspace(0.0, T, n_t)
    dt = t[1] - t[0]
    # element i couples (lam_i, lam_{i+1}):  f_{j,i} = (lam_{i+1}-lam_i)/dt
    #                                              - rho_j (lam_i+lam_{i+1})/2
    # J = sum_j w_j^2 dt sum_i f_{j,i}^2  -> tridiagonal quadratic form.
    d_main = np.zeros(n_t)
    d_off = np.zeros(n_t - 1)
    for r_j, w_j in zip(rates, weights):
        a = -1.0 / dt - 0.5 * r_j   # coefficient of lam_i in f
        b = 1.0 / dt - 0.5 * r_j    # coefficient of lam_{i+1} in f
        c2 = w_j**2 * dt
        d_main[:-1] += c2 * a * a
        d_main[1:] += c2 * b * b
        d_off += c2 * a * b
    # eliminate lam_{n-1} = 1: minimise over lam_0..lam_{n-2}
    n_free = n_t - 1
    rhs = np.zeros(n_free)
    rhs[-1] = -d_off[-1] * 1.0
    ab = np.zeros((3, n_free))
    ab[0, 1:] = d_off[: n_free - 1]
    ab[1, :] = d_main[:n_free]
    ab[2, :-1] = d_off[: n_free - 1]
    lam_free = solve_banded((1, 1), ab, rhs)
    lam = np.concatenate([lam_free, [1.0]])
    # objective value
    J = 0.0
    for r_j, w_j in zip(rates, weights):
        f = (lam[1:] - lam[:-1]) / dt - r_j * 0.5 * (lam[1:] + lam[:-1])
        J += w_j**2 * dt * float(np.sum(f**2))
    return t, lam, J


def forcing_norm(lam, t, rates, weights):
    dt = t[1] - t[0]
    J = 0.0
    for r_j, w_j in zip(rates, weights):
        f = (lam[1:] - lam[:-1]) / dt - r_j * 0.5 * (lam[1:] + lam[:-1])
        J += w_j**2 * dt * float(np.sum(f**2))
    return math.sqrt(J)


def lower_bound(rho_a, rho_b, w_a, w_b, T):
    delta = rho_b - rho_a
    kappa2 = math.sqrt((1.0 - math.exp(-2 * rho_a * T)) / (2 * rho_a)) if rho_a > 0 else math.sqrt(T)
    kappa1 = (1.0 - math.exp(-rho_a * T)) / rho_a if rho_a > 0 else T
    return min(abs(w_a), abs(w_b)) * delta * kappa2 / (2.0 * (1.0 + delta * kappa1))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)
    out = script_data_dir(__file__) / utc_timestamp()
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # --- A. single mode: exact annihilation -------------------------------
    T = 1.0
    t, lam, J = optimal_coefficient([rho(1)], [1.0], T)
    matched = np.exp(rho(1) * (t - T))
    summary["A_single_mode_floor"] = math.sqrt(J)
    summary["A_matched_sup_dev"] = float(np.abs(lam - matched).max())
    print(f"A. single mode: optimal floor = {math.sqrt(J):.2e} (analytic 0), "
          f"sup|lam_opt - e^(rho(t-T))| = {summary['A_matched_sup_dev']:.2e}")

    # --- B. two-mode sweep: bound <= optimum <= matched --------------------
    c = 0.5
    rows = []
    for f in range(2, 13):
        rates, weights = [rho(1), rho(f)], [1.0, c]
        t, lam, J = optimal_coefficient(rates, weights, T)
        opt = math.sqrt(J)
        lb = lower_bound(rho(1), rho(f), 1.0, c, T)
        lam_m = np.exp(rho(1) * (t - T))
        up = forcing_norm(lam_m, t, rates, weights)
        rows.append({"f": f, "gap": rho(f) - rho(1), "lower_bound": lb,
                     "optimal": opt, "matched": up,
                     "bound_ok": bool(lb <= opt * (1 + 1e-9)),
                     "opt_over_bound": opt / lb, "matched_over_opt": up / opt})
        print(f"B. f={f:2d}: bound {lb:9.3e} <= optimal {opt:9.3e} "
              f"<= matched {up:9.3e}   opt/bound={opt/lb:6.1f}  matched/opt={up/opt:5.2f}")
    summary["B_two_mode_sweep"] = rows
    summary["B_bound_holds_everywhere"] = bool(all(r["bound_ok"] for r in rows))

    # --- C. the study's data (T = 0.1) -------------------------------------
    T01 = 0.1
    cases = {}
    # two-mode sine instance
    rates, weights = [rho(1), rho(4)], [1.0, 0.5]
    t, lam_o, J = optimal_coefficient(rates, weights, T01)
    lam_lin = t / T01
    lam_exp = np.exp(rho(1) * (t - T01))
    cases["sine_c05_f4"] = {
        "optimal": math.sqrt(J),
        "linear": forcing_norm(lam_lin, t, rates, weights),
        "matched_exp": forcing_norm(lam_exp, t, rates, weights),
        "lower_bound": lower_bound(rho(1), rho(4), 1.0, 0.5, T01),
    }
    # theta_3 instance (modes 1..6, coefficients 2 e^{-n^2})
    rates = [rho(n) for n in range(1, 7)]
    weights = [2.0 * math.exp(-(n**2)) for n in range(1, 7)]
    t, lam_o3, J3 = optimal_coefficient(rates, weights, T01)
    cases["theta3"] = {
        "optimal": math.sqrt(J3),
        "linear": forcing_norm(t / T01, t, rates, weights),
        "matched_exp": forcing_norm(np.exp(rho(1) * (t - T01)), t, rates, weights),
        "lower_bound": lower_bound(rho(1), rho(2), weights[0], weights[1], T01),
    }
    summary["C_study_data"] = cases
    for name, v in cases.items():
        print(f"C. {name}: bound {v['lower_bound']:.3e} <= optimal {v['optimal']:.3e}; "
              f"linear {v['linear']:.3e}, matched-exp {v['matched_exp']:.3e} "
              f"(matched/opt = {v['matched_exp']/v['optimal']:.2f}, "
              f"linear/opt = {v['linear']/v['optimal']:.2f})")

    with open(out / "optimal_floor.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)

    # --- figure -------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    gaps = [r["gap"] for r in rows]
    axL.loglog(gaps, [r["optimal"] for r in rows], "o-", color="#1b6ca8",
               label="optimal scalar coefficient (exact minimiser)")
    axL.loglog(gaps, [r["matched"] for r in rows], "s--", color="#66a182",
               label="mode-1-matched coefficient (upper reference)")
    axL.loglog(gaps, [r["lower_bound"] for r in rows], ":", color="black",
               label="proven lower bound")
    axL.set_xlabel(r"spectral gap $\varrho_f - \varrho_1$")
    axL.set_ylabel(r"forcing norm $\|P(\lambda g)\|$")
    axL.set_title("Two-mode floor vs spectral gap ($T=1$, $c=0.5$)", fontsize=10)
    axL.grid(True, which="both", alpha=0.3)
    legL = axL.legend(loc="upper left", fontsize=8, frameon=True)

    axR.plot(t, lam_o, "-", color="#1b6ca8", lw=1.6,
             label="optimal coefficient (sine instance)")
    axR.plot(t, np.exp(rho(1) * (t - T01)), "--", color="#66a182", lw=1.4,
             label=r"matched $e^{\varrho_1 (t-T)}$")
    axR.plot(t, t / T01, ":", color="grey", lw=1.2, label="linear $t/T$")
    axR.set_xlabel("$t$")
    axR.set_ylabel(r"$\lambda(t)$")
    axR.set_title("Optimal vs fixed coefficients ($T=0.1$)", fontsize=10)
    axR.grid(True, alpha=0.3)
    legR = axR.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle("Variational check of the scalar-coefficient forcing floor", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    finalize_figure(
        fig, out / "optimal_coefficient_floor.png", legends=[legL, legR], axes=[axL, axR],
        formula=(r"$J(\lambda)=\sum_j c_j^2\int_0^T(\lambda'-\varrho_j\lambda)^2 dt$, "
                 r"minimised exactly over $\lambda(T)=1$ (banded linear system); "
                 r"proven bound $\leq$ optimum $\leq$ matched coefficient"),
        formula_fontsize=8)
    print(f"\nwrote figure + optimal_floor.yaml to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
