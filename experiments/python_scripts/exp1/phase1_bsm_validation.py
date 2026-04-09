"""Phase 1 — BSM mathematical components validation.

Validates core BSM functions from learning_option_pricing.pricing.terminal
with 8 mandatory plots and a scalar summary table.

Parameters: K=100, r=0.02, sigma=0.25, T=1, q=0 (Section 4.1.2).

Usage:
    python experiments/python_scripts/exp1/phase1_bsm_validation.py
"""
from __future__ import annotations

import math
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.pricing.terminal import (
    _d_tilde_0,
    _d_tilde_1,
    _d_tilde_2,
    _normal_cdf,
    black_scholes_put,
    european_put_ve1,
    european_put_ve2,
    g1_linear,
    g2_american_put,
    payoff_call,
    payoff_put,
)

# ---------------------------------------------------------------------------
# Parameters (Section 4.1.2)
# ---------------------------------------------------------------------------
K = 100.0
r = 0.02
sigma = 0.25
T = 1.0
q = 0.0  # no dividends

SQRT_2PI = math.sqrt(2.0 * math.pi)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("data/phase1_bsm_validation") / f"{timestamp}_K{K:.0f}_r{r}_sig{sigma}_T{T:.0f}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save metadata
metadata = {
    "command": " ".join(sys.argv),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "parameters": {
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "q": q,
    }
}
with open(OUT_DIR / "metadata.yaml", "w") as f:
    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

print(f"Saving plots to: {OUT_DIR}")


# ---------------------------------------------------------------------------
# Helper: convert tensors to numpy for plotting
# ---------------------------------------------------------------------------
def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ===================================================================
# Plot 1 — Payoff shape
# ===================================================================
def plot1_payoff():
    s = torch.linspace(60.0, 140.0, 500)
    phi_put = payoff_put(s, K)
    phi_call = payoff_call(s, K)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(to_np(s), to_np(phi_put), label=r"Put: $(K-s)^+$", linewidth=2)
    ax.plot(to_np(s), to_np(phi_call), label=r"Call: $(s-K)^+$", linewidth=2, linestyle="--")
    ax.axvline(K, color="gray", linestyle=":", alpha=0.6, label=f"K = {K:.0f}")
    ax.set_xlabel("s (underlying price)")
    ax.set_ylabel(r"$\Phi(s)$")
    ax.set_title(f"Plot 1 — Payoff functions (European/American Call and Put Options, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot1_payoff.png", dpi=150)
    plt.close(fig)
    print("[OK] Plot 1 — Payoff shape")


# ===================================================================
# Plot 2 — European put price surface (heatmap)
# ===================================================================
def plot2_european_put_surface():
    ns, nt = 200, 200
    s_vals = torch.linspace(60.0, 140.0, ns)
    t_vals = torch.linspace(0.0, T, nt)
    S, T_grid = torch.meshgrid(s_vals, t_vals, indexing="ij")
    tau_grid = T - T_grid

    Ve = black_scholes_put(S, K, r, sigma, tau_grid)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.pcolormesh(
        to_np(t_vals), to_np(s_vals), to_np(Ve), shading="auto", cmap="Blues"
    )
    fig.colorbar(im, ax=ax, label=r"$V^e(s, t)$")
    ax.set_xlabel("t")
    ax.set_ylabel("s")
    ax.set_title(rf"Plot 2 — European Put Option price $V^e(s, t)$ (K={K}, r={r}, $\sigma$={sigma}, T={T})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot2_european_put_surface.png", dpi=150)
    plt.close(fig)
    print("[OK] Plot 2 — European put price surface")


# ===================================================================
# Plot 3 — Terminal condition recovery
# ===================================================================
def plot3_terminal_recovery():
    s = torch.linspace(60.0, 140.0, 500)
    tau_zero = torch.zeros_like(s)
    phi = payoff_put(s, K)

    Ve_terminal = black_scholes_put(s, K, r, sigma, tau_zero)
    g2_terminal = g2_american_put(s, K, r, sigma, tau_zero)

    diff_Ve = to_np(Ve_terminal - phi)
    diff_g2 = to_np(g2_terminal - phi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(to_np(s), diff_Ve, linewidth=2)
    axes[0].set_title(r"$V^e(s, T) - \Phi(s)$")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("Difference")
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))

    axes[1].plot(to_np(s), diff_g2, linewidth=2, color="tab:orange")
    axes[1].set_title(r"$(V_1^e + V_2^e)(s, T) - \Phi(s)$")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("Difference")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style="scientific", axis="y", scilimits=(-3, 3))

    fig.suptitle(f"Plot 3 — Terminal condition recovery (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T}, should be ~0 everywhere)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot3_terminal_recovery.png", dpi=150)
    plt.close(fig)

    max_diff_Ve = float(torch.max(torch.abs(Ve_terminal - phi)))
    max_diff_g2 = float(torch.max(torch.abs(g2_terminal - phi)))
    print(f"[OK] Plot 3 — Terminal recovery: max|Ve-Phi|={max_diff_Ve:.2e}, max|g2-Phi|={max_diff_g2:.2e}")
    return max_diff_Ve, max_diff_g2


# ===================================================================
# Plot 4 — Singularity behavior near expiry (s=K, tau -> 0)
# ===================================================================
def plot4_singularity():
    tau_vals = torch.linspace(1e-6, 0.5, 1000)
    s_atm = torch.full_like(tau_vals, K)

    Ve_vals = black_scholes_put(s_atm, K, r, sigma, tau_vals)
    Ve1_vals = european_put_ve1(s_atm, K, r, sigma, tau_vals)
    Ve1_Ve2_vals = g2_american_put(s_atm, K, r, sigma, tau_vals)
    # Known at-the-money approximation: K * sigma * sqrt(tau) / sqrt(2*pi)
    atm_approx = K * sigma * torch.sqrt(tau_vals) / SQRT_2PI

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(to_np(tau_vals), to_np(Ve_vals), label=r"$V^e(K, t)$", linewidth=2)
    ax.plot(to_np(tau_vals), to_np(Ve1_vals), label=r"$V_1^e(K, t)$", linewidth=2, linestyle="--")
    ax.plot(to_np(tau_vals), to_np(Ve1_Ve2_vals), label=r"$V_1^e + V_2^e$", linewidth=2, linestyle="-.")
    ax.plot(to_np(tau_vals), to_np(atm_approx), label=r"ATM approx $K\sigma\sqrt{\tau}/\sqrt{2\pi}$",
            linewidth=2, linestyle=":", color="black")
    ax.set_xlabel(r"$\tau = T - t$")
    ax.set_ylabel("Option value at s=K")
    ax.set_title(rf"Plot 4 — Singularity behavior near expiry ($s = K = {K}$, European Put Option, r={r}, $\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot4_singularity.png", dpi=150)
    plt.close(fig)
    print("[OK] Plot 4 — Singularity behavior near expiry")


# ===================================================================
# Plot 5 — g2 vs Ve comparison at fixed t=0.5
# ===================================================================
def plot5_g2_vs_Ve():
    s = torch.linspace(60.0, 140.0, 500)
    tau_half = torch.full_like(s, T - 0.5)  # tau = 0.5

    Ve_vals = black_scholes_put(s, K, r, sigma, tau_half)
    g2_vals = g2_american_put(s, K, r, sigma, tau_half)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(to_np(s), to_np(Ve_vals), label=r"$V^e(s, t{=}0.5)$", linewidth=2)
    ax.plot(to_np(s), to_np(g2_vals), label=r"$g_2(s, t{=}0.5) = V_1^e + V_2^e$",
            linewidth=2, linestyle="--")
    ax.set_xlabel("s")
    ax.set_ylabel("Option value")
    ax.set_title(f"Plot 5 — $g_2$ vs $V^e$ at $t = 0.5$ (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot5_g2_vs_Ve.png", dpi=150)
    plt.close(fig)
    print("[OK] Plot 5 — g2 vs Ve comparison")


# ===================================================================
# Plot 6 — g1 shape
# ===================================================================
def plot6_g1():
    t_vals = torch.linspace(0.0, T, 200)
    g1_vals = g1_linear(T, t_vals)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(to_np(t_vals), to_np(g1_vals), linewidth=2)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$g_1(t) = T - t$")
    ax.set_title(f"Plot 6 — $g_1(s, t) = T - t$ (K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot6_g1.png", dpi=150)
    plt.close(fig)

    g1_at_0 = float(g1_linear(T, torch.tensor(0.0)))
    g1_at_T = float(g1_linear(T, torch.tensor(T)))
    print(f"[OK] Plot 6 — g1: g1(0)={g1_at_0:.4f}, g1(T)={g1_at_T:.4f}")


# ===================================================================
# Plot 7 — Residual smoothness check: Ve - g2 heatmap
# ===================================================================
def plot7_residual():
    ns, nt = 200, 200
    s_vals = torch.linspace(60.0, 140.0, ns)
    t_vals = torch.linspace(0.0, T, nt)
    S, T_grid = torch.meshgrid(s_vals, t_vals, indexing="ij")
    tau_grid = T - T_grid

    Ve = black_scholes_put(S, K, r, sigma, tau_grid)
    g2 = g2_american_put(S, K, r, sigma, tau_grid)
    residual = Ve - g2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Ve itself (for magnitude reference)
    im0 = axes[0].pcolormesh(to_np(t_vals), to_np(s_vals), to_np(Ve), shading="auto", cmap="Blues")
    fig.colorbar(im0, ax=axes[0], label=r"$V^e(s,t)$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("s")
    axes[0].set_title(r"$V^e(s, t)$")

    # Right: residual Ve - g2
    vmax = float(torch.max(torch.abs(residual)))
    im1 = axes[1].pcolormesh(
        to_np(t_vals), to_np(s_vals), to_np(residual),
        shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    )
    fig.colorbar(im1, ax=axes[1], label=r"$V^e - g_2$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("s")
    axes[1].set_title(r"$V^e(s,t) - g_2(s,t)$ (residual for NN to learn)")

    fig.suptitle(f"Plot 7 — Residual smoothness check (European Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot7_residual.png", dpi=150)
    plt.close(fig)
    print(f"[OK] Plot 7 — Residual: max|Ve-g2|={vmax:.4f}")


# ===================================================================
# Plot 8 — Tau epsilon guard check (NaN detection)
# ===================================================================
def plot8_tau_guard():
    tau_tiny = [1e-4, 1e-6, 1e-8]
    s = torch.linspace(60.0, 140.0, 500)

    fig, ax = plt.subplots(figsize=(7, 4))
    nan_detected = False

    for tau_val in tau_tiny:
        tau_t = torch.full_like(s, tau_val)
        Ve = black_scholes_put(s, K, r, sigma, tau_t)
        has_nan = bool(torch.isnan(Ve).any())
        has_inf = bool(torch.isinf(Ve).any())
        if has_nan or has_inf:
            nan_detected = True
            print(f"  [WARNING] NaN/Inf detected at tau={tau_val:.0e}!")
        ax.plot(to_np(s), to_np(Ve), label=rf"$\tau = {tau_val:.0e}$", linewidth=1.5)

    # Also plot exact payoff for reference
    phi = payoff_put(s, K)
    ax.plot(to_np(s), to_np(phi), label=r"Payoff $\Phi(s)$", linewidth=2, linestyle=":", color="black")

    ax.set_xlabel("s")
    ax.set_ylabel(r"$V^e(s, t)$")
    ax.set_title(rf"Plot 8 — $V^e$ near $\tau \to 0$ (epsilon guard check, European Put Option, K={K}, r={r}, $\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot8_tau_guard.png", dpi=150)
    plt.close(fig)

    if nan_detected:
        print("[WARN] Plot 8 — NaN/Inf detected in epsilon guard check!")
    else:
        print("[OK] Plot 8 — No NaN/Inf detected for tiny tau values")


# ===================================================================
# Summary table
# ===================================================================
def summary_table(max_diff_Ve: float, max_diff_g2: float):
    s_grid = torch.linspace(60.0, 140.0, 500)

    # Ve(K, t=0.5)
    s_K = torch.tensor([K])
    tau_05 = torch.tensor([0.5])
    Ve_at_K = float(black_scholes_put(s_K, K, r, sigma, tau_05))
    g2_at_K = float(g2_american_put(s_K, K, r, sigma, tau_05))

    print("\n" + "=" * 60)
    print("PHASE 1 — SCALAR CHECKS SUMMARY")
    print("=" * 60)
    print(f"  max |Ve(s, T) - Phi(s)|      = {max_diff_Ve:.2e}   (expect < 1e-6)")
    print(f"  max |g2(s, T) - Phi(s)|      = {max_diff_g2:.2e}   (expect < 1e-6)")
    print(f"  Ve(K=100, t=0.5)             = {Ve_at_K:.4f}   (expect ~6-7)")
    print(f"  g2(K=100, t=0.5)             = {g2_at_K:.4f}   (expect close to above)")
    print("=" * 60)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"Phase 1 BSM Validation — K={K}, r={r}, sigma={sigma}, T={T}, q={q}")
    print(f"PyTorch version: {torch.__version__}\n")

    plot1_payoff()
    plot2_european_put_surface()
    max_diff_Ve, max_diff_g2 = plot3_terminal_recovery()
    plot4_singularity()
    plot5_g2_vs_Ve()
    plot6_g1()
    plot7_residual()
    plot8_tau_guard()
    summary_table(max_diff_Ve, max_diff_g2)

    print(f"\nAll plots saved to: {OUT_DIR}")
