"""Phase 2 — ETCNN architecture validation.

Validates the ResNet backbone, ETCNN trial-solution wrapper, input
normalisation, and PINN baseline with 6 sanity checks and 6 plots.

Parameters: K=100, r=0.02, sigma=0.25, T=1, q=0 (Section 4.1.2).

Usage:
    python experiments/python_scripts/exp1/phase2_etcnn_architecture.py
"""
from __future__ import annotations

import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from learning_option_pricing.models.etcnn import (
    AmericanPutETCNN,
    InputNormalization,
    PINN,
)
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.pricing.terminal import (
    black_scholes_put,
    g1_linear,
    payoff_put,
)

# ---------------------------------------------------------------------------
# Parameters (Section 4.1.2)
# ---------------------------------------------------------------------------
K = 100.0
r = 0.02
sigma = 0.25
T = 1.0
q = 0.0

# Default hyperparameters (Section 3.2)
M = 4
L = 2
n = 50
d_in = 2
d_out = 1

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = (
    Path("data/phase2_etcnn_architecture")
    / f"{timestamp}_M{M}_L{L}_n{n}_K{K:.0f}"
)
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
    },
    "hyperparameters": {
        "M": M,
        "L": L,
        "n": n,
        "d_in": d_in,
        "d_out": d_out,
    }
}
with open(OUT_DIR / "metadata.yaml", "w") as f:
    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

print(f"Saving plots to: {OUT_DIR}")


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


# ===================================================================
# Build models
# ===================================================================
def build_models(seed: int = 42):
    """Build ETCNN and PINN with the same random seed."""
    torch.manual_seed(seed)
    etcnn = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=True)
    etcnn.eval()

    torch.manual_seed(seed)
    pinn = PINN(
        resnet=ResNet(d_in=d_in, d_out=d_out, n=n, M=M, L=L),
        normalizer=InputNormalization(K),
    )
    pinn.eval()

    return etcnn, pinn


# ===================================================================
# Check 1 — Parameter count
# ===================================================================
def check1_param_count(model: torch.nn.Module, name: str) -> int:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Check 1 — Parameter count ({name})")
    print(f"{'='*60}")
    print(f"  Total trainable parameters: {total:,}")

    # Manual cross-check for default config
    # Input layer:  d_in*n + n = 2*50 + 50 = 150
    # Each block:   L * (n*n + n) = 2 * (2500 + 50) = 5100
    # M blocks:     4 * 5100 = 20400
    # Output layer: n*d_out + d_out = 50*1 + 1 = 51
    # Total:        150 + 20400 + 51 = 20601
    expected = d_in * n + n + M * L * (n * n + n) + n * d_out + d_out
    print(f"  Expected (manual):           {expected:,}")
    if total == expected:
        print("  [OK] Parameter count matches manual calculation.")
    else:
        print(f"  [WARN] Mismatch! Got {total}, expected {expected}.")
    return total


# ===================================================================
# Check 2 — Output shape
# ===================================================================
def check2_output_shape(etcnn: torch.nn.Module):
    print(f"\n{'='*60}")
    print("Check 2 — Output shape")
    print(f"{'='*60}")
    s = torch.rand(256) * 80.0 + 60.0  # s in [60, 140]
    t = torch.rand(256) * T             # t in [0, T]
    x = torch.stack([s, t], dim=1)      # (256, 2)

    with torch.no_grad():
        out = etcnn(x)

    assert out.shape == (256, 1), f"Expected shape (256, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in output!"
    assert not torch.isinf(out).any(), "Inf detected in output!"
    print(f"  Output shape: {out.shape}  [OK]")
    print(f"  NaN: {torch.isnan(out).any().item()}  Inf: {torch.isinf(out).any().item()}  [OK]")


# ===================================================================
# Check 3 — Terminal condition satisfaction (CRITICAL)
# ===================================================================
def check3_terminal_condition(etcnn: torch.nn.Module) -> float:
    print(f"\n{'='*60}")
    print("Check 3 — Terminal condition satisfaction (CRITICAL)")
    print(f"{'='*60}")
    s = torch.linspace(60.0, 140.0, 1000)
    t = torch.full_like(s, T)
    x = torch.stack([s, t], dim=1)

    with torch.no_grad():
        u_tilde = etcnn(x).squeeze()
    phi = payoff_put(s, K)
    diff = u_tilde - phi
    max_err = float(torch.max(torch.abs(diff)))

    print(f"  max |u_tilde_NN(s, T) - Phi(s)| = {max_err:.2e}")

    if max_err < 1e-7:
        print("  [OK] Terminal condition satisfied to < 1e-7.")
    else:
        print("  [FAIL] Terminal condition NOT satisfied! Stopping.")
        sys.exit(1)
    return max_err


# ===================================================================
# Check 4 — g1 zeroing at terminal
# ===================================================================
def check4_g1_zeroing():
    print(f"\n{'='*60}")
    print("Check 4 — g1 zeroing at terminal")
    print(f"{'='*60}")
    s = torch.linspace(60.0, 140.0, 500)
    t_terminal = torch.full_like(s, T)
    g1_vals = g1_linear(T, t_terminal)
    max_g1 = float(torch.max(torch.abs(g1_vals)))
    print(f"  max |g1(s, T)| = {max_g1:.2e}")
    assert max_g1 < 1e-15, f"g1 not zero at terminal: {max_g1}"
    print("  [OK] g1(s, T) = 0 for all s — network output fully suppressed at t=T.")


# ===================================================================
# Plot 1 — Untrained output surface
# ===================================================================
def plot1_untrained_surface(etcnn: torch.nn.Module):
    ns, nt = 200, 200
    s_vals = torch.linspace(60.0, 140.0, ns)
    t_vals = torch.linspace(0.0, T, nt)
    S, T_grid = torch.meshgrid(s_vals, t_vals, indexing="ij")

    x = torch.stack([S.reshape(-1), T_grid.reshape(-1)], dim=1)
    with torch.no_grad():
        u_tilde = etcnn(x).reshape(ns, nt)

    tau_grid = T - T_grid
    Ve = black_scholes_put(S, K, r, sigma, tau_grid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].pcolormesh(
        to_np(t_vals), to_np(s_vals), to_np(u_tilde), shading="auto", cmap="Blues"
    )
    fig.colorbar(im0, ax=axes[0], label=r"$\tilde{u}_{NN}(s, t)$")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("s")
    axes[0].set_title(r"ETCNN output $\tilde{u}_{NN}(s, t)$ (untrained)")

    im1 = axes[1].pcolormesh(
        to_np(t_vals), to_np(s_vals), to_np(Ve), shading="auto", cmap="Blues"
    )
    fig.colorbar(im1, ax=axes[1], label=r"$V^e(s, t)$")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("s")
    axes[1].set_title(r"True European put $V^e(s, t)$")

    fig.suptitle(f"Plot 1 — Untrained ETCNN vs true solution (American Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T}, terminal edge should match)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot1_untrained_surface.png", dpi=150)
    plt.close(fig)
    print("[OK] Plot 1 — Untrained output surface")


# ===================================================================
# Plot 2 — Terminal edge overlay
# ===================================================================
def plot2_terminal_edge(etcnn: torch.nn.Module):
    s = torch.linspace(60.0, 140.0, 500)
    t_T = torch.full_like(s, T)
    x = torch.stack([s, t_T], dim=1)

    with torch.no_grad():
        u_tilde = etcnn(x).squeeze()
    phi = payoff_put(s, K)
    Ve_T = black_scholes_put(s, K, r, sigma, torch.zeros_like(s))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(to_np(s), to_np(u_tilde), label=r"$\tilde{u}_{NN}(s, T)$", linewidth=2)
    ax.plot(to_np(s), to_np(phi), label=r"$\Phi(s) = (K-s)^+$", linewidth=2, linestyle="--")
    ax.plot(to_np(s), to_np(Ve_T), label=r"$V^e(s, T)$", linewidth=2, linestyle=":")
    ax.axvline(K, color="gray", linestyle=":", alpha=0.5, label=f"K = {K:.0f}")
    ax.set_xlabel("s")
    ax.set_ylabel("Option value")
    ax.set_title(f"Plot 2 — Terminal edge: ETCNN, payoff, and $V^e$ at $t = T$ (American Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot2_terminal_edge.png", dpi=150)
    plt.close(fig)
    print("[OK] Plot 2 — Terminal edge overlay (all three should overlap)")


# ===================================================================
# Plot 3 — PINN vs ETCNN terminal comparison
# ===================================================================
def plot3_pinn_vs_etcnn(etcnn: torch.nn.Module, pinn: torch.nn.Module):
    s = torch.linspace(60.0, 140.0, 500)
    t_T = torch.full_like(s, T)
    x = torch.stack([s, t_T], dim=1)

    with torch.no_grad():
        u_etcnn = etcnn(x).squeeze()
        u_pinn = pinn(x).squeeze()
    phi = payoff_put(s, K)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(to_np(s), to_np(u_etcnn), label=r"$\tilde{u}_{NN}^{\mathrm{ETCNN}}(s, T)$", linewidth=2)
    axes[0].plot(to_np(s), to_np(phi), label=r"$\Phi(s)$", linewidth=2, linestyle="--")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel("Option value")
    axes[0].set_title("ETCNN at $t = T$ (matches payoff)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(to_np(s), to_np(u_pinn), label=r"$u_{NN}^{\mathrm{PINN}}(s, T)$", linewidth=2, color="tab:red")
    axes[1].plot(to_np(s), to_np(phi), label=r"$\Phi(s)$", linewidth=2, linestyle="--")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("Option value")
    axes[1].set_title("PINN at $t = T$ (does NOT match payoff)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Plot 3 — ETCNN vs PINN: terminal condition at initialization (American Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot3_pinn_vs_etcnn_terminal.png", dpi=150)
    plt.close(fig)

    pinn_err = float(torch.max(torch.abs(u_pinn - phi)))
    print(f"[OK] Plot 3 — PINN terminal error: max|u_PINN(s,T) - Phi(s)| = {pinn_err:.4f}")
    return pinn_err


# ===================================================================
# Plot 4 — Gradient flow check
# ===================================================================
def plot4_gradient_flow(etcnn: torch.nn.Module) -> bool:
    s = torch.rand(256) * 80.0 + 60.0
    t = torch.rand(256) * T
    x = torch.stack([s, t], dim=1)

    etcnn.train()
    out = etcnn(x)
    loss = out.mean()
    loss.backward()

    names = []
    norms = []
    for name, param in etcnn.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            names.append(name)
            norms.append(grad_norm)

    etcnn.eval()
    etcnn.zero_grad()

    norms_arr = np.array(norms)
    all_nonzero = bool(np.all(norms_arr > 1e-8))
    none_exploding = bool(np.all(norms_arr < 100.0))
    gradients_ok = all_nonzero and none_exploding

    # Shorten names for readability
    short_names = [n.replace("resnet.", "").replace(".weight", ".W").replace(".bias", ".b") for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(norms)), norms, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xticks(range(len(norms)))
    ax.set_xticklabels(short_names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Gradient norm")
    ax.set_title(f"Plot 4 — Gradient norms per parameter (dummy loss = mean of outputs, K={K}, r={r}, $\\sigma$={sigma}, T={T})")
    ax.set_yscale("log")
    ax.axhline(1e-8, color="red", linestyle="--", alpha=0.5, label="vanishing threshold (1e-8)")
    ax.axhline(100, color="orange", linestyle="--", alpha=0.5, label="exploding threshold (100)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot4_gradient_flow.png", dpi=150)
    plt.close(fig)

    print(f"[OK] Plot 4 — Gradient flow: min={norms_arr.min():.2e}, max={norms_arr.max():.2e}")
    print(f"  All > 1e-8: {all_nonzero}  |  All < 100: {none_exploding}  → {'OK' if gradients_ok else 'WARN'}")
    return gradients_ok


# ===================================================================
# Plot 5 — Input normalisation effect
# ===================================================================
def plot5_input_normalization():
    seed = 123
    torch.manual_seed(seed)
    net_norm = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=True)
    net_norm.eval()

    torch.manual_seed(seed)
    net_raw = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=False)
    net_raw.eval()

    s = torch.linspace(60.0, 140.0, 300)
    t_half = torch.full_like(s, 0.5)
    x = torch.stack([s, t_half], dim=1)

    with torch.no_grad():
        out_norm = net_norm(x).squeeze()
        out_raw = net_raw(x).squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(to_np(s), to_np(out_norm), linewidth=2, label="With normalisation (s/K)")
    axes[0].plot(to_np(s), to_np(out_raw), linewidth=2, linestyle="--", label="Without normalisation")
    axes[0].set_xlabel("s")
    axes[0].set_ylabel(r"$\tilde{u}_{NN}(s, t{=}0.5)$")
    axes[0].set_title("ETCNN output at $t = 0.5$")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    diff = out_norm - out_raw
    axes[1].plot(to_np(s), to_np(diff), linewidth=2, color="tab:orange")
    axes[1].set_xlabel("s")
    axes[1].set_ylabel("Difference")
    axes[1].set_title("Difference (normalised $-$ raw)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Plot 5 — Input normalisation effect (American Put Option, K={K}, r={r}, $\\sigma$={sigma}, T={T}, same seed, different input scaling)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot5_input_normalization.png", dpi=150)
    plt.close(fig)

    max_diff = float(torch.max(torch.abs(diff)))
    print(f"[OK] Plot 5 — Input normalisation: max difference = {max_diff:.4f} (confirms normalisation is active)")


# ===================================================================
# Plot 6 — Architecture summary (text)
# ===================================================================
def plot6_architecture_summary(etcnn: torch.nn.Module):
    print(f"\n{'='*60}")
    print("Plot 6 — Architecture summary")
    print(f"{'='*60}")

    print(f"\n  AmericanPutETCNN(K={K}, r={r}, sigma={sigma}, T={T})")
    print(f"  Input: (batch, {d_in}) = (s, t)")
    print(f"  InputNormalization: s → s/K (K={K})")
    print(f"  ResNet backbone:")
    print(f"    input_layer:  Linear({d_in}, {n}) + Tanh")
    for m in range(M):
        for l_idx in range(L):
            print(f"    block[{m}].layer[{l_idx}]: Linear({n}, {n}) + Tanh")
        print(f"    block[{m}]: + skip connection")
    print(f"    output_layer: Linear({n}, {d_out})")
    print(f"  ETCNN modification:")
    print(f"    u_tilde(s,t) = g1(s,t) * u_NN(s/K, t) + g2(s,t)")
    print(f"    g1(s,t) = T - t  (vanishes at t=T)")
    print(f"    g2(s,t) = V1^e(s,t) + V2^e(s,t)  (= payoff at t=T)")
    print(f"  Output: (batch, {d_out})")

    # Also try torchsummary-style print
    print(f"\n  --- PyTorch module structure ---")
    for name, module in etcnn.named_modules():
        if name:
            indent = "    " + "  " * name.count(".")
            print(f"{indent}{name}: {module.__class__.__name__}")

    # Parameter shapes
    print(f"\n  --- Parameter shapes ---")
    for name, p in etcnn.named_parameters():
        print(f"    {name}: {list(p.shape)}")

    print(f"{'='*60}")


# ===================================================================
# Final summary
# ===================================================================
def final_summary(
    total_params: int,
    terminal_err: float,
    gradients_ok: bool,
    pinn_terminal_err: float,
):
    print(f"\n{'='*60}")
    print("PHASE 2 — FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total trainable parameters:              {total_params:,}")
    print(f"  max |u_tilde_NN(s,T) - Phi(s)|:          {terminal_err:.2e}  {'[PASS]' if terminal_err < 1e-7 else '[FAIL]'}")
    print(f"  Gradients flow cleanly:                  {'Yes' if gradients_ok else 'No'}")
    print(f"  PINN terminal error (must be > 0):       {pinn_terminal_err:.4f}  {'[PASS]' if pinn_terminal_err > 0.01 else '[WARN]'}")
    print(f"{'='*60}")

    if terminal_err < 1e-7 and gradients_ok and pinn_terminal_err > 0.01:
        print("\n  All Phase 2 checks PASSED. Ready for Phase 3.")
    else:
        print("\n  Some checks failed — review above before proceeding.")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"Phase 2 ETCNN Architecture Validation")
    print(f"  K={K}, r={r}, sigma={sigma}, T={T}, q={q}")
    print(f"  M={M}, L={L}, n={n}, d_in={d_in}, d_out={d_out}")
    print(f"  PyTorch version: {torch.__version__}\n")

    etcnn, pinn = build_models()

    # Checks
    total_params = check1_param_count(etcnn, "ETCNN")
    check2_output_shape(etcnn)
    terminal_err = check3_terminal_condition(etcnn)
    check4_g1_zeroing()

    # Plots
    plot1_untrained_surface(etcnn)
    plot2_terminal_edge(etcnn)
    pinn_err = plot3_pinn_vs_etcnn(etcnn, pinn)
    gradients_ok = plot4_gradient_flow(etcnn)
    plot5_input_normalization()
    plot6_architecture_summary(etcnn)

    # Final summary
    final_summary(total_params, terminal_err, gradients_ok, pinn_err)

    print(f"\nAll plots saved to: {OUT_DIR}")
