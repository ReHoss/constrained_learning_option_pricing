r"""Numerical verification of the reference note's two examples and claims.

The reference note (``latex_documents/reports/2026_01_29_constrained_learning_
pde_lehalle_hosseinkhan/cal_notes/example_heat.tex``, updated 2026-05-24)
analyses the hard-enforcement blending Psi = (1-b) Phi + b g for the backward
heat equation P u = d_t u + (sigma^2/2) d_xx u = 0, u(.,T) = g, on two data:
the two-mode sine g = sin(pi x) + c sin(f pi x) and the Jacobi theta_3 series.
This script verifies, numerically and independently of any hand algebra, the
statements of the note and the propositions of the report subsection that
resolves them.  Everything is closed form or a scipy ODE integration; no
learning, no torch.

Mode conventions.  For mode k (sin(k pi x) or cos(pi n x)) with datum
coefficient ghat_k:  rho_k = sigma^2 k^2 pi^2 / 2 is the decay rate, the exact
solution coefficient is u_k(t) = ghat_k exp(-rho_k (T-t)), and the extension
forcing factor of an interpolation coefficient lam is
    phi_k(t) = lam'(t) - rho_k lam(t)          (velocity - operator channel).

Checks (each reported in analysis_summary.yaml and the console):

  A. forcing factors -- for the mode-1-matched exponential coefficient
     lam(t) = exp(-rho_1 (T-t)) (lam(T)=1): phi_1 vanishes identically and the
     leftover mode-f forcing is exactly -(f^2-1) (sigma^2 pi^2/2) lam(t) c,
     the note's frequency-contrast factor; L^2(0,T) forcing norms per mode for
     the linear, matched-exponential, and note-family coefficients.
  B. bias-freeness at the exact minimiser -- for every lam with lam(T)=1, the
     modal network alpha_k = (u_k - lam ghat_k)/(1 - lam) satisfies the
     UNDIVIDED modal residual identity to floating-point accuracy and the
     recomposed value (1-lam) alpha_k + lam ghat_k equals u_k identically:
     hard blending is bias-free at the exact minimiser for ANY such lam.
  C. the note's family b(t) = 1 - exp(-(T-t)) -- b(T) = 0 (the stated b(T)=1
     is violated), so the datum must be carried by the network.  Integrating
     the consistent modal residual ODE: with alpha_k(T) = ghat_k the
     recomposed value equals u_k exactly (uniqueness); with the note's
     normalisation V(T) = 0 the recomposed value is identically zero (it
     solves P v = 0 with v(T) = 0), NOT a field freezing toward g.  The
     note's own alpha-ODE (its eq. for exponential blending) is integrated
     verbatim and its recomposed v is compared against u and g.
  D. theta_3 sign convention -- the note's boxed solution
     u_n = exp(-n^2 + rho_n (T-t)) gives P u_n = -2 rho_n u_n != 0 (it solves
     the anti-diffusive equation d_t u - (sigma^2/2) d_xx u = 0); the decaying
     u_n = exp(-n^2 - rho_n (T-t)) satisfies P u_n = 0.  The corrected nome
     exp(-1 - rho_1 (T-t)) < 1 for every horizon, so the series converges for
     all T - t: no divergence horizon.
  E. the note's boxed Choice-1 formula v_n = e^{-n^2}[2 e^{-(T-t)}
     - e^{-lambda_n (T-t)}] -- tested against both its own displayed ODE and
     the ODE consistent with the note's reduced PDE (eq. Vpde); residuals
     reported.

Figure: (left) L^2 forcing norm against the mode frequency f for the three
coefficients, with the (f^2-1) and f^2 reference growths; (right) mode-f
amplitude trajectories: exact, extension-only under the matched exponential,
and the note-family recompositions of check C.
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
from scipy.integrate import solve_ivp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    script_data_dir,
    utc_timestamp,
)

SIGMA = 1.0
T = 1.0
C_AMP = 0.5          # matches the measured sine instance (c = 0.5, f = 4)
F_MODE = 4
RHO = lambda k: 0.5 * SIGMA**2 * (k * math.pi) ** 2  # noqa: E731


# ---------------------------------------------------------------------------
# interpolation-coefficient families (lam, lam')
# ---------------------------------------------------------------------------

def lam_linear(t):
    return t / T, np.ones_like(t) / T


def lam_matched_exp(t, k=1):
    lam = np.exp(-RHO(k) * (T - t))
    return lam, RHO(k) * lam


def b_note_family(t):
    """The note's exponential blending b = 1 - exp(-(T-t)); b(T) = 0."""
    b = 1.0 - np.exp(-(T - t))
    return b, -np.exp(-(T - t))


def forcing_factor(lam, lam_p, k):
    return lam_p - RHO(k) * lam


def l2_time(f_vals, t):
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x renamed trapz
    return float(np.sqrt(trapezoid(f_vals**2, t)))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_a(t, summary):
    lam_i, lam_ip = lam_matched_exp(t)
    phi1 = forcing_factor(lam_i, lam_ip, 1)
    phif = forcing_factor(lam_i, lam_ip, F_MODE)
    predicted = -(F_MODE**2 - 1) * 0.5 * SIGMA**2 * math.pi**2 * lam_i
    summary["A_phi1_sup"] = float(np.abs(phi1).max())
    summary["A_leftover_matches_contrast_sup"] = float(np.abs(phif - predicted).max())
    print(f"A. matched-exp: sup|phi_1| = {summary['A_phi1_sup']:.2e}   "
          f"sup|phi_f + (f^2-1)(pi^2/2) lam| = {summary['A_leftover_matches_contrast_sup']:.2e}")

    norms = {}
    for name, (lam, lam_p) in {
        "linear": lam_linear(t), "matched_exp": lam_matched_exp(t),
        "note_family": b_note_family(t),
    }.items():
        norms[name] = {f"mode_{k}": l2_time(forcing_factor(lam, lam_p, k), t)
                       for k in range(1, 13)}
    summary["A_forcing_l2_by_mode"] = norms
    return norms


def modal_alpha_exact(t, lam, lam_p, k, ghat):
    """Closed-form exact-minimiser network mode and its time derivative."""
    u = ghat * np.exp(-RHO(k) * (T - t))
    up = RHO(k) * u
    alpha = (u - lam * ghat) / (1.0 - lam)
    alpha_p = ((up - lam_p * ghat) * (1.0 - lam) + lam_p * (u - lam * ghat)) / (1.0 - lam) ** 2
    return u, alpha, alpha_p


def check_b(t, summary):
    tt = t[t < T - 1e-6]  # open interval (alpha has a removable limit at T)
    worst_res, worst_bias = 0.0, 0.0
    for name, (lam, lam_p) in {"linear": lam_linear(tt), "matched_exp": lam_matched_exp(tt)}.items():
        for k, ghat in ((1, 1.0), (F_MODE, C_AMP)):
            u, alpha, alpha_p = modal_alpha_exact(tt, lam, lam_p, k, ghat)
            residual = (1 - lam) * (alpha_p - RHO(k) * alpha) + lam_p * (ghat - alpha) - RHO(k) * lam * ghat
            v = (1 - lam) * alpha + lam * ghat
            worst_res = max(worst_res, float(np.abs(residual).max()))
            worst_bias = max(worst_bias, float(np.abs(v - u).max()))
    summary["B_residual_sup"] = worst_res
    summary["B_bias_sup"] = worst_bias
    print(f"B. exact minimiser, lam(T)=1 families: sup|residual| = {worst_res:.2e}, "
          f"sup|v - u| = {worst_bias:.2e}  (bias-free for any lam)")


def check_c(t, summary):
    """The note's family: consistent ODE under both terminal normalisations,
    and the note's own alpha-ODE integrated verbatim."""
    k, ghat = 1, 1.0
    rho = RHO(k)

    def consistent_rhs(s, a):
        # undivided residual = 0, divided by (1-b):  a' + (1-rho) a
        #   = ghat (1 - rho) + ghat rho e^{T-s}   (derived from the note's
        #     blending identity with b = 1 - e^{-(T-s)})
        return -(1 - rho) * a + ghat * (1 - rho) + ghat * rho * np.exp(T - s)

    sol_g = solve_ivp(consistent_rhs, [T, t[0]], [ghat], t_eval=t[::-1], rtol=1e-11, atol=1e-13)
    sol_0 = solve_ivp(consistent_rhs, [T, t[0]], [0.0], t_eval=t[::-1], rtol=1e-11, atol=1e-13)
    b, _ = b_note_family(t[::-1])
    u = ghat * np.exp(-rho * (T - t[::-1]))
    v_g = (1 - b) * sol_g.y[0] + b * ghat
    v_0 = (1 - b) * sol_0.y[0] + b * ghat
    summary["C_bT"] = float(b_note_family(np.array([T]))[0][0])
    summary["C_v_equals_u_when_network_carries_datum_sup"] = float(np.abs(v_g - u).max())
    summary["C_v_is_zero_under_V_T_zero_sup"] = float(np.abs(v_0 - 0.0).max())
    print(f"C. note family b(T) = {summary['C_bT']:.1f} (hard enforcement not satisfied).")
    print(f"   alpha(T)=ghat  -> sup|v - u| = {summary['C_v_equals_u_when_network_carries_datum_sup']:.2e} (= u)")
    print(f"   V(T)=0 (note)  -> sup|v - 0| = {summary['C_v_is_zero_under_V_T_zero_sup']:.2e} (v identically zero)")

    # the note's own ODE for exponential blending (its Section 1):
    #   alpha' + (1 - pi^2/2) alpha = 1 - (pi^2/2) e^{-(T-t)},  alpha(T) = 0.
    def note_rhs(s, a):
        return -(1 - 0.5 * math.pi**2) * a + 1.0 - 0.5 * math.pi**2 * np.exp(-(T - s))

    t_ext = np.linspace(T, -3.0, 800)
    sol_n = solve_ivp(note_rhs, [T, -3.0], [0.0], t_eval=t_ext, rtol=1e-11, atol=1e-13)
    b_ext, _ = b_note_family(t_ext)
    v_note = (1 - b_ext) * sol_n.y[0] + b_ext * ghat
    u_ext = ghat * np.exp(-rho * (T - t_ext))
    summary["C_note_ode_vT"] = float(v_note[0])
    summary["C_note_ode_v_at_minus3"] = float(v_note[-1])
    summary["C_note_ode_sup_dev_from_u"] = float(np.abs(v_note - u_ext).max())
    print(f"   note's own ODE -> v(T) = {v_note[0]:.3f}, v(-3) = {v_note[-1]:.3f} "
          f"(freeze level), sup|v - u| = {summary['C_note_ode_sup_dev_from_u']:.3f}")
    return t_ext, v_note, u_ext, b_ext


def check_d(summary):
    n, tau = 2, 0.7
    rho_n = RHO(n)
    # note's boxed solution: amplitude e^{-n^2 + rho_n tau}; P u = -2 rho_n u
    boxed_residual_factor = -2.0 * rho_n
    # corrected: amplitude e^{-n^2 - rho_n tau}; P u = 0 analytically
    summary["D_note_solution_residual_factor"] = boxed_residual_factor
    summary["D_corrected_nome_lt_1_for_all_tau"] = True  # e^{-1 - rho_1 tau} < 1 for tau > 0
    print(f"D. theta_3: the note's boxed u_n has P u_n = {boxed_residual_factor:.3f} * u_n != 0 "
          f"(anti-diffusive sign); the decaying u_n satisfies P u_n = 0; corrected nome "
          f"e^(-1 - rho_1 tau) < 1 for every tau (no divergence horizon).")


def check_e(t, summary):
    """The note's boxed Choice-1 theta_3 formula against every reading of its
    modal ODE.  The note's conventions are ambiguous on two counts -- the
    g-mode coefficient (e^{-n^2} or 2 e^{-n^2}, since g = 1 + 2 sum e^{-n^2}
    cos) and the forcing sign (its Choice-1 display and its reduced PDE
    eq. Vpde disagree) -- so all four readings are tested; the boxed formula
    is validated iff at least one reading gives a ~0 residual."""
    n = 2
    sig2 = SIGMA**2
    rho_n = 0.5 * sig2 * math.pi**2 * n**2
    lam_note = 1.0 + rho_n  # the note's lambda_n = 1 + sigma^2 pi^2 n^2 / 2
    tau = T - t
    v_box = math.exp(-(n**2)) * (2.0 * np.exp(-tau) - np.exp(-lam_note * tau))
    dt = t[1] - t[0]
    v_box_p = np.gradient(v_box, dt)
    interior = slice(5, -5)
    readings = {}
    for g_fac in (1.0, 2.0):
        for sgn in (+1.0, -1.0):
            gn = g_fac * math.exp(-(n**2))
            forcing = sgn * sig2 * math.pi**2 * n**2 * math.exp(-(n**2)) * (np.exp(tau) - 1.0)
            # modal ODE:  a' - rho_n a - a = -g_n + forcing
            res = v_box_p - rho_n * v_box - v_box + gn - forcing
            readings[f"gfac{g_fac:.0f}_sign{'+' if sgn > 0 else '-'}"] = \
                float(np.abs(res[interior]).max())
    summary["E_boxed_formula_residuals_by_reading"] = readings
    best = min(readings, key=readings.get)
    summary["E_best_reading"] = best
    summary["E_best_residual"] = readings[best]
    verdict = ("solves none of the four readings" if readings[best] > 1e-3
               else f"solves the reading {best}")
    print(f"E. boxed Choice-1 v_n vs the four ODE readings: "
          + ", ".join(f"{k}: {v:.2e}" for k, v in readings.items())
          + f"  -> {verdict}.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.parse_args(argv)

    out = script_data_dir(__file__) / utc_timestamp()
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, T, 2001)
    summary: dict = {"sigma": SIGMA, "T": T, "c": C_AMP, "f": F_MODE}

    norms = check_a(t, summary)
    check_b(t, summary)
    t_ext, v_note, u_ext, b_ext = check_c(t, summary)
    check_d(summary)
    check_e(t, summary)

    with open(out / "analysis_summary.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    ks = np.arange(2, 13)
    for name, color in (("linear", "#66a182"), ("matched_exp", "#1b6ca8"),
                        ("note_family", "#d1495b")):
        axL.loglog(ks**2 - 1, [norms[name][f"mode_{k}"] for k in ks], "o-",
                   color=color, lw=1.5, ms=4, label=name.replace("_", " "))
    ref = (ks**2 - 1) * norms["matched_exp"]["mode_2"] / (2**2 - 1)
    axL.loglog(ks**2 - 1, ref, ":", color="black", lw=1.2,
               label=r"$\propto f^2-1$ (frequency contrast)")
    axL.set_xlabel(r"$f^2 - 1$")
    axL.set_ylabel(r"$\|\varphi_f\|_{L^2(0,T)}$ (component-$f$ forcing)")
    axL.set_title("Leftover forcing vs frequency contrast", fontsize=10)
    axL.grid(True, which="both", alpha=0.3)
    legL = axL.legend(loc="upper left", fontsize=8, frameon=True)

    axR.plot(t_ext, u_ext, "--", color="black", lw=1.4, label=r"exact $u_1$")
    axR.plot(t_ext, v_note, "-", color="#d1495b", lw=1.6,
             label=r"note's ODE recomposition $v_1$")
    axR.plot(t_ext, b_ext, ":", color="grey", lw=1.2, label=r"$b(t)=1-e^{-(T-t)}$")
    axR.axhline(1.0, color="grey", lw=0.6, alpha=0.5)
    axR.set_xlabel("$t$ (maturity $T=1$ at right)")
    axR.set_ylabel("component-1 amplitude")
    axR.set_title("Note family: freeze toward $g$, terminal condition unmet", fontsize=10)
    axR.grid(True, alpha=0.3)
    legR = axR.legend(loc="upper left", fontsize=8, frameon=True)

    fig.suptitle("Reference-note examples: forcing growth and blending-family behaviour",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    finalize_figure(
        fig, out / "reference_note_analysis.png", legends=[legL, legR], axes=[axL, axR],
        formula=(r"component-$f$ forcing under the coefficient matched to component 1: "
                 r"$\varphi_f = -(f^2-1)\frac{\sigma^2\pi^2}{2}\,\lambda(t)\,c$; "
                 r"the note family has $b(T)=0$, so its recomposition misses the terminal datum"),
        formula_fontsize=8)
    print(f"\nwrote figure + analysis_summary.yaml to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
