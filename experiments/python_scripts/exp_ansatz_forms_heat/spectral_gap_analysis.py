r"""Spectral-gap analysis of the terminal-data extension forcing.

Tests the hypothesis: the best terminal-data extension is the one whose
operator image P Psi exhibits a spectral gap with respect to the
network-reachable forcings -- i.e. P Psi's energy sits *below* the frequency
cutoff at which the network can no longer cancel, so the uncancellable part
Pi_{S^perp} P Psi is small.

Method (per hard-form run, reusing the trained models):
  * rebuild the ansatz and load model.pt;
  * on a uniform spatial grid at several time slices, evaluate
        forcing      f = P Psi              (extension forcing),
        residual     r = P u_hat            (achieved residual = Pi_{S^perp} f),
    via the heat operator;
  * take the spatial real-FFT power spectra |f_k|^2 and |r_k|^2 (averaged over
    slices); the per-mode cancellation ratio |r_k|^2/|f_k|^2 reveals the network
    cutoff k* (low k cancelled -> ratio ~0; high k uncancelled -> ratio ~1);
  * metrics: uncancellable fraction ||r||^2/||f||^2 (= achieved L_pde / floor),
    and the high-band energy fraction of the forcing above k*.

Outputs (torch-free figures saved to data/spectral_gap_analysis/<timestamp>/):
  * spectra_by_ic.png        -- |f_k| vs |r_k| per IC, hard forms overlaid;
  * uncancellable_vs_error.png -- does Pi_{S^perp} energy predict rel L2
    (collapse across forms AND ICs, unlike the floor)?
  * summary_spectral.yaml.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ansatz_forms_catalogue as cat  # noqa: E402

HARD = ["hard_constant_linear", "hard_constant_exp",
        "hard_convex_linear", "hard_convex_exp"]
N_X = 256          # spatial grid (power of two for the FFT)
T_SLICES = (0.1, 0.3, 0.5, 0.7, 0.9)  # fractions of T


def _load_ansatz(run_dir: Path, variant_name: str):
    import yaml
    import torch
    import ablation_ansatz_forms as runner

    with open(run_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)
    ic = meta["ic"]
    hparams = meta["hparams"]
    problem = runner.build_problem(ic)
    variant = cat.variant_by_name(variant_name)
    ansatz = runner.build_ansatz(variant, problem, hparams, model_seed=0)
    state = torch.load(run_dir / f"variant_{variant_name}" / "models" / "model.pt",
                       map_location="cpu")
    ansatz.load_state_dict(state)
    ansatz.eval()
    return ansatz, problem


def _spectra(ansatz, problem):
    """Spatial power spectra of the forcing, its two channels, and the residual.

    Returns a dict with wavenumber ``k`` and the slice-averaged power spectra of:

    * ``f``    -- the total extension forcing :math:`\\mathcal{P}\\Psi`;
    * ``vel``  -- the interpolation-velocity channel :math:`\\partial_t\\Psi`
                  (:math:`\\lambda' g` for the convex form; zero for additive);
    * ``diff`` -- the diffusion channel :math:`\\tfrac{\\sigma^2}{2}\\partial_{xx}\\Psi`
                  (:math:`\\lambda\\tfrac{\\sigma^2}{2} g''`);
    * ``r``    -- the achieved residual :math:`\\mathcal{P}\\hat u`,

    plus their mean energies ``*_e`` (band-integrated power).  Separating the two
    forcing channels is what lets the spectrum *attribute* the uncancellable
    high-wavenumber remainder to the diffusion channel (the regularised payoff
    curvature) rather than the low-wavenumber velocity channel.
    """
    import numpy as np
    import torch

    from learning_option_pricing.pde.operators import heat_operator, heat_operator_parts

    sigma = problem["sigma"]
    T = problem["T"]
    x_lo, x_hi = problem["x_eval_lo"], problem["x_eval_hi"]
    x = torch.linspace(x_lo, x_hi, N_X, dtype=torch.float64)
    nfreq = N_X // 2 + 1
    pow_acc = {key: np.zeros(nfreq) for key in ("f", "vel", "diff", "r")}
    energy = {key: 0.0 for key in ("f", "vel", "diff", "r")}
    ansatz.double()

    def _accumulate(key, arr):
        pow_acc[key] += np.abs(np.fft.rfft(arr - arr.mean())) ** 2
        energy[key] += float((arr ** 2).mean())

    for frac in T_SLICES:
        xx = x.clone().requires_grad_(True)
        tt = torch.full_like(xx, frac * T).requires_grad_(True)
        xt = torch.stack([xx, tt], dim=1)
        # extension forcing f = P Psi, split into velocity + diffusion channels
        psi = ansatz.extension(xx.unsqueeze(-1), tt.unsqueeze(-1)).squeeze(-1)
        vel, diff = heat_operator_parts(psi, xx, tt, sigma)
        f = vel + diff
        # achieved residual r = P u_hat
        u = ansatz(xt).squeeze(-1)
        r = heat_operator(u, xx, tt, sigma)
        _accumulate("f", f.detach().numpy())
        _accumulate("vel", vel.detach().numpy())
        _accumulate("diff", diff.detach().numpy())
        _accumulate("r", r.detach().numpy())

    n = len(T_SLICES)
    k = np.arange(nfreq)
    out = {"k": k}
    for key in pow_acc:
        out[key + "_pow"] = pow_acc[key] / n
        out[key + "_e"] = energy[key] / n
    return out


def _cutoff(k, f_pow, r_pow):
    """Network cancellation cutoff k*: smallest k where |r_k|^2/|f_k|^2 >= 0.5
    among modes carrying non-negligible forcing power."""
    import numpy as np
    fp = np.clip(f_pow, 1e-30, None)
    ratio = r_pow / fp
    sig = f_pow > 0.01 * f_pow.max()  # ignore empty modes
    for ki in range(1, len(k)):
        if sig[ki] and ratio[ki] >= 0.5:
            return int(k[ki])
    return int(k[-1])


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    from learning_option_pricing.utils.run_context import script_data_dir, utc_timestamp
    from _figure_layout import finalize_figure

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None,
                   help="dir holding the *_iters*_seed* runs (default: ablation data dir)")
    p.add_argument("--seed", type=int, default=0, help="which seed's models to read")
    args = p.parse_args(argv)

    import glob
    data_root = (Path(args.data_root) if args.data_root
                 else script_data_dir(Path(__file__).parent / "ablation_ansatz_forms.py"))
    out = script_data_dir(__file__) / f"{utc_timestamp()}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    ics = cat.ic_names()
    colors = {v["name"]: v["color"] for v in cat.METHOD_VARIANTS}
    labels = {v["name"]: v["label"] for v in cat.METHOD_VARIANTS}
    results = {}  # (ic, variant) -> dict

    spectra = {}  # (ic, variant) -> (k, f_pow, r_pow)
    for ic in ics:
        runs = glob.glob(str(data_root / f"*_{ic}_iters*_seed{args.seed}"))
        if not runs:
            continue
        run_dir = Path(sorted(runs)[-1])
        # rel_l2 from summary
        summ = yaml.safe_load(open(run_dir / "summary.yaml"))
        for v in HARD:
            ansatz, problem = _load_ansatz(run_dir, v)
            sp = _spectra(ansatz, problem)
            k, f_pow, r_pow = sp["k"], sp["f_pow"], sp["r_pow"]
            f_e, r_e = sp["f_e"], sp["r_e"]
            kstar = _cutoff(k, f_pow, r_pow)
            high_frac = float(f_pow[k >= kstar].sum() / max(f_pow.sum(), 1e-30))
            results[(ic, v)] = {
                "rel_l2": float(summ[v]["rel_l2"]),
                "floor": float(f_e),
                "velocity": float(sp["vel_e"]),
                "diffusion": float(sp["diff_e"]),
                "uncancellable": float(r_e),
                "uncancellable_frac": float(r_e / max(f_e, 1e-30)),
                "k_star": kstar,
                "forcing_high_band_frac": high_frac,
            }
            spectra[(ic, v)] = sp
            print(f"{ic:8s} {v:22s} relL2={summ[v]['rel_l2']:.2e} "
                  f"floor={f_e:.2e} vel={sp['vel_e']:.2e} diff={sp['diff_e']:.2e} "
                  f"uncanc={r_e:.2e} k*={kstar} highfrac={high_frac:.2f}")

    # ---- Figure 1: spectra per IC (forcing solid, residual dashed) ----
    present_ics = [ic for ic in ics if any((ic, v) in spectra for v in HARD)]
    ncol = len(present_ics)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.2), squeeze=False)
    for ax, ic in zip(axes[0], present_ics):
        for v in HARD:
            if (ic, v) not in spectra:
                continue
            sp = spectra[(ic, v)]
            k, f_pow, r_pow = sp["k"], sp["f_pow"], sp["r_pow"]
            ax.loglog(k[1:], np.clip(f_pow[1:], 1e-30, None), "-",
                      color=colors[v], lw=1.3, label=labels[v])
            ax.loglog(k[1:], np.clip(r_pow[1:], 1e-30, None), "--",
                      color=colors[v], lw=1.0, alpha=0.8)
        ax.set_title(ic, fontsize=10)
        ax.set_xlabel("spatial wavenumber $k$")
        ax.set_ylabel(r"power $|\widehat{\cdot}_k|^2$")
        ax.grid(True, which="both", alpha=0.3)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles, labs, loc="lower center", ncol=4, fontsize=7,
                     frameon=True, bbox_to_anchor=(0.5, 0.08))
    fig.suptitle("Forcing spectrum $|\\widehat{\\mathcal{P}\\Psi}_k|^2$ (solid) vs "
                 "achieved-residual spectrum $|\\widehat{\\mathcal{P}\\hat u}_k|^2$ (dashed)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.22, 1, 0.95])
    finalize_figure(fig, out / "spectra_by_ic.png", legends=[leg], axes=list(axes[0]),
                    formula=(r"network cancels low $k$ (residual $\ll$ forcing); "
                             r"the surviving high-$k$ residual $=\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi$ "
                             r"sets the error"))

    # ---- Figure 1b: forcing-channel split (velocity vs diffusion) vs residual,
    #      for the convex form on call vs call_cm (the report's key comparison).
    #      Validates the attribution: velocity = low-k cancellable, diffusion =
    #      high-k uncancellable, achieved residual tracks the diffusion tail. ----
    split_form = "hard_convex_linear"
    split_ics = [ic for ic in ("call", "call_cm") if (ic, split_form) in spectra]
    if split_ics:
        # sharey/sharex so the two panels are on one common scale (directly
        # comparable); only the left panel carries the y-label and tick labels.
        figc, axesc = plt.subplots(1, len(split_ics), figsize=(5.8 * len(split_ics), 5.0),
                                   squeeze=False, sharex=True, sharey=True)
        for j, (ax, ic) in enumerate(zip(axesc[0], split_ics)):
            sp = spectra[(ic, split_form)]
            k = sp["k"]
            # General forcing channels of f = P Psi with Psi = lambda(t) g; the
            # legend states the operator-general form, the formula box specialises
            # it to the heat instance L = (sigma^2/2) d_xx.
            ax.loglog(k[1:], np.clip(sp["vel_pow"][1:], 1e-30, None), "-",
                      color="#1b6ca8", lw=1.4,
                      label=r"velocity $\partial_t\Psi=\lambda'(t)\,g$")
            ax.loglog(k[1:], np.clip(sp["diff_pow"][1:], 1e-30, None), "-",
                      color="#d1495b", lw=1.4,
                      label=r"operator channel $\lambda(t)\,\mathcal{L}g$")
            ax.loglog(k[1:], np.clip(sp["r_pow"][1:], 1e-30, None), "--",
                      color="black", lw=1.3,
                      label=r"residual $\mathcal{P}\hat u=\partial_t\hat u+\mathcal{L}\hat u$")
            note = {
                "call": "softplus, $\\beta=100$: operator channel is a sub-grid spike\n"
                        "(width $\\sim1/(\\beta K)\\sim10^{-4}\\ll$ grid $dx$) — aliased, not physical",
                "call_cm": "Chen--Mangasarian: band-limited, resolved",
            }.get(ic, "")
            ax.set_title(f"{ic}\n{note}", fontsize=8)
            ax.set_xlabel("spatial wavenumber $k$")
            if j == 0:
                ax.set_ylabel(r"power $|\widehat{\cdot}_k|^2$")
            ax.grid(True, which="both", alpha=0.3)
        handlesc, labsc = axesc[0, 0].get_legend_handles_labels()
        # legend sits clearly above the two-line formula box (anchored ~0.012 by
        # finalize_figure) so the two never overlap.
        legc = figc.legend(handlesc, labsc, loc="lower center", ncol=3, fontsize=8,
                           frameon=True, bbox_to_anchor=(0.5, 0.13))
        figc.suptitle(f"Forcing-channel spectra vs achieved residual "
                      f"({split_form}, convex form; common scale)", fontsize=11)
        figc.tight_layout(rect=[0, 0.30, 1, 0.94])
        finalize_figure(
            figc, out / "forcing_channels_spectra.png", legends=[legc],
            axes=list(axesc[0]),
            formula=(
                r"general: $f=\mathcal{P}\Psi$ with $\Psi=\lambda(t)\,g$  "
                r"$\Rightarrow$  velocity $\partial_t\Psi=\lambda'(t)\,g$,  "
                r"operator channel $\lambda(t)\,\mathcal{L}g$,  "
                r"residual $\mathcal{P}\hat u=\partial_t\hat u+\mathcal{L}\hat u$" "\n"
                r"heat instance $\mathcal{L}=\frac{\sigma^2}{2}\partial_{xx}$:  "
                r"operator channel $=\lambda(t)\,\frac{\sigma^2}{2}\,g''$;  "
                r"the network cancels the low-$k$ velocity, the residual tracks the "
                r"high-$k$ operator tail $\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi$"),
            formula_fontsize=8)

    # ---- Figure 2: does the uncancellable energy predict the error? ----
    markers = {"sine": "o", "theta3": "s", "call": "^", "call_cm": "D"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for (ic, v), d in results.items():
        ax.scatter(d["uncancellable"], d["rel_l2"], c=colors[v],
                   marker=markers.get(ic, "o"), s=90, edgecolors="black", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"achieved (uncancellable) residual $\|\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi\|^2=\mathcal{L}^\star$")
    ax.set_ylabel(r"relative $L^2$ error vs exact")
    ax.set_title("Uncancellable residual predicts accuracy (across forms AND ICs)")
    ax.grid(True, which="both", alpha=0.3)
    from matplotlib.lines import Line2D
    vh = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[v],
                 markeredgecolor="black", markersize=9, label=labels[v]) for v in HARD]
    ih = [Line2D([0], [0], marker=markers[ic], color="w", markerfacecolor="grey",
                 markeredgecolor="black", markersize=9, label=ic)
          for ic in markers if ic in present_ics]
    l1 = ax.legend(handles=vh, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   fontsize=7, frameon=True, title="form")
    ax.add_artist(l1)
    l2 = ax.legend(handles=ih, loc="lower left", bbox_to_anchor=(1.02, 0.0),
                   fontsize=7, frameon=True, title="IC")
    fig.tight_layout(rect=[0, 0.08, 0.78, 1])
    finalize_figure(fig, out / "uncancellable_vs_error.png", legends=[l1, l2], axes=[ax],
                    formula=(r"$\mathcal{L}^\star=\|\Pi_{\mathcal{S}^\perp}\mathcal{P}\Psi\|^2$ "
                             r"(achieved residual); hypothesis $\mathrm{rel}\,L^2\sim\sqrt{\mathcal{L}^\star}$"))

    with open(out / "summary_spectral.yaml", "w") as f:
        yaml.dump({f"{ic}/{v}": d for (ic, v), d in results.items()}, f,
                  default_flow_style=False, sort_keys=False)
    print(f"\nwrote figures + summary to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
