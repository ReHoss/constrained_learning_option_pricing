"""Standalone diagnostic: visualize g2(s) and the ansatz backbone g2(s) + v(s, t).

For a given Bermudan experiment folder (produced by phase3_training.py with
--extraction), this script:

1. Loads etcnn_a.pt (Stage A model).
2. Recomputes the singularity extraction: s*, c, fictitious put v(s,t), residual g2(s).
3. Produces two figures:
   - Fig A  (plot_g2_shape.png)  : g2(s) alone — the PCHIP residual.
   - Fig B  (plot_g2_backbone.png): g2(s) + v(s, t) for several t ∈ [0, t1].

Usage::

    cd <project_root>
    python experiments/python_scripts/exp1/plot_g2_backbone.py \\
        data/phase3_training/20260422_204939_iters5000_5000_K100_extraction_g2-bs

The output PNGs are saved inside the experiment folder under ``diagnostics/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Make the project importable regardless of cwd
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))

from learning_option_pricing.models.etcnn import AmericanPutETCNN
from learning_option_pricing.pricing.interpolation import PchipInterpolator
from learning_option_pricing.pricing.singularity import (
    FictitiousEuropeanPut,
    build_singularity_extraction,
)
from learning_option_pricing.pricing.terminal import payoff_put

# ---------------------------------------------------------------------------
# Parse experiment directory
# ---------------------------------------------------------------------------
if len(sys.argv) < 2:
    # default to the specific experiment if not provided
    EXP_DIR = _ROOT / (
        "data/phase3_training/"
        "20260422_204939_iters5000_5000_K100_extraction_g2-bs"
    )
else:
    EXP_DIR = Path(sys.argv[1]).resolve()

if not EXP_DIR.exists():
    sys.exit(f"[ERROR] Experiment folder not found: {EXP_DIR}")

META = EXP_DIR / "metadata.yaml"
if not META.exists():
    sys.exit(f"[ERROR] metadata.yaml not found in {EXP_DIR}")

with open(META) as f:
    meta = yaml.safe_load(f)

params = meta["parameters"]
hp     = meta["hyperparameters"]
domain = meta["domain"]

K      = float(params["K"])
r      = float(params["r"])
sigma  = float(params["sigma"])
T      = float(params["T"])
t1     = float(params.get("t1", 0.5))
g2_type = str(hp.get("g2_type", "bs"))

S_LO   = float(domain.get("S_TRAIN_LO", 20.0))
S_HI   = float(domain.get("S_TRAIN_HI", 160.0))

MODEL_A = EXP_DIR / "models" / "etcnn_a.pt"
if not MODEL_A.exists():
    sys.exit(f"[ERROR] etcnn_a.pt not found in {EXP_DIR / 'models'}")

DEVICE = torch.device("cpu")

print(f"Experiment : {EXP_DIR.name}")
print(f"Parameters : K={K}, r={r}, sigma={sigma}, T={T}, t1={t1}, g2_type={g2_type}")

# ---------------------------------------------------------------------------
# Load Stage A model
# ---------------------------------------------------------------------------
etcnn_a = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=True, g2_type=g2_type)
etcnn_a.load_state_dict(torch.load(MODEL_A, map_location=DEVICE))
etcnn_a.eval()
print("Loaded etcnn_a.pt")

# ---------------------------------------------------------------------------
# Recompute singularity extraction
# ---------------------------------------------------------------------------
print("Recomputing singularity extraction ...")
s_star, c, fict_put, s_nodes, v_target, residual_cpu = build_singularity_extraction(
    etcnn_a, K=K, r=r, sigma=sigma, t1=t1,
    s_lo=S_LO, s_hi=S_HI, device=DEVICE,
)
print(f"  s* = {s_star:.4f},  c = {c:.6f}")

residual_interp = PchipInterpolator(s_nodes, residual_cpu)
fict_put.to(DEVICE)

# ---------------------------------------------------------------------------
# Evaluation grid for plots  [S_EVAL_LO, S_EVAL_HI]
# ---------------------------------------------------------------------------
S_EVAL_LO = float(domain.get("S_EVAL_LO", 60.0))
S_EVAL_HI = float(domain.get("S_EVAL_HI", 120.0))
s_plot = torch.linspace(S_EVAL_LO, S_EVAL_HI, 600)

def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

# ---------------------------------------------------------------------------
# Compute quantities
# ---------------------------------------------------------------------------
with torch.no_grad():
    # g2(s) from PCHIP interpolant
    g2_vals = residual_interp(s_plot)

    # v(s, t) for several t values
    t_values = [0.0, t1 / 4, t1 / 2, 3 * t1 / 4, t1]
    v_at = {}
    for t_val in t_values:
        t_tensor = torch.full_like(s_plot, t_val)
        if abs(t_val - t1) < 1e-9:
            # use exact closed form at maturity
            v_at[t_val] = fict_put.at_maturity(s_plot)
        else:
            v_at[t_val] = fict_put(s_plot, t_tensor)

    # V_Berm(s, t1) from Stage A (for reference)
    t1_vec = torch.full_like(s_plot, t1)
    x_t1 = torch.stack([s_plot, t1_vec], dim=1)
    hold_t1 = etcnn_a(x_t1).squeeze()
    phi_plot = payoff_put(s_plot, K)
    v_berm_t1 = torch.maximum(phi_plot, hold_t1)

    # v(s, t1) alone (the fictitious put at t1)
    v_t1 = fict_put.at_maturity(s_plot)

    # European BS put g2 from the European ETCNN ansatz (g2_type="bs" with tau=T-t)
    from learning_option_pricing.pricing.terminal import black_scholes_put
    tau_mid = T - t1 / 2
    g2_eur_anchor = black_scholes_put(s_plot, K, r, sigma, torch.tensor(tau_mid))

# ---------------------------------------------------------------------------
# Fig A — g2(s) shape
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(to_np(s_plot), to_np(g2_vals),
        label=r"$g_2(s) = V^{\mathrm{Berm}}_{\bar\theta}(s,t_1) - v(s,t_1)$ (PCHIP)",
        color="blue", linewidth=2)
ax.plot(to_np(s_plot), to_np(v_t1),
        label=rf"$v(s,t_1) = {c:.3f}\cdot(s^*-s)^+$",
        color="green", linewidth=1.8, linestyle="--")
ax.plot(to_np(s_plot), to_np(phi_plot),
        label=r"Payoff $\Phi(s) = (K-s)^+$",
        color="red", linewidth=1.5, linestyle=":")
ax.axvline(s_star, color="grey", linestyle=":", alpha=0.7,
           label=rf"$s^* = {s_star:.2f}$")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Asset price $s$")
ax.set_ylabel("Value")
ax.set_title(r"Residual $g_2(s)$ extracted at $t_1$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(to_np(s_plot), to_np(v_berm_t1),
        label=r"$V^{\mathrm{Berm}}_{\bar\theta}(s,t_1) = v(s,t_1) + g_2(s)$",
        color="black", linewidth=2, linestyle="-.")
ax.plot(to_np(s_plot), to_np(v_t1 + g2_vals),
        label=r"$v(s,t_1) + g_2(s)$ (PCHIP recomposition)",
        color="blue", linewidth=1.8, linestyle="--")
ax.plot(to_np(s_plot), to_np(g2_vals),
        label=r"$g_2(s)$ alone",
        color="steelblue", linewidth=1.5, linestyle=":")
ax.axvline(s_star, color="grey", linestyle=":", alpha=0.7,
           label=rf"$s^* = {s_star:.2f}$")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Asset price $s$")
ax.set_title(r"Decomposition: $V^{\mathrm{Berm}}_{\bar\theta}(s,t_1) = v(s,t_1) + g_2(s)$")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle(
    rf"Fig A — Shape of $g_2(s)$  |  K={K}, r={r}, $\sigma$={sigma}, $t_1$={t1}, "
    rf"$s^*={s_star:.2f}$, $c={c:.4f}$",
    fontsize=11,
)
fig.tight_layout()
out_A = EXP_DIR / "diagnostics" / "plot_g2_shape.png"
fig.savefig(out_A, dpi=150)
plt.close(fig)
print(f"[OK] Fig A saved: {out_A}")

# ---------------------------------------------------------------------------
# Fig B — g2(s) + v(s, t) for several t values (backbone of Stage B ansatz)
# ---------------------------------------------------------------------------
cmap = plt.get_cmap("viridis_r")
t_colors = {tv: cmap(i / (len(t_values) - 1)) for i, tv in enumerate(t_values)}

fig, ax = plt.subplots(figsize=(10, 6))

for t_val in t_values:
    backbone = g2_vals + v_at[t_val]
    at_t1 = abs(t_val - t1) < 1e-9
    if at_t1:
        label = r"$g_2(s)+v(s,t_1)=V^{\mathrm{Berm}}_{\bar\theta}(s,t_1)$"
    else:
        label = rf"$g_2(s)+v(s,{t_val:.3f})$"
    lw = 2.5 if at_t1 else 1.8
    ls = "-." if at_t1 else "-"
    ax.plot(to_np(s_plot), to_np(backbone),
            label=label, color=t_colors[t_val], linewidth=lw, linestyle=ls)

ax.plot(to_np(s_plot), to_np(phi_plot),
        label=r"Payoff $\Phi(s)=(K-s)^+$", color="red", linewidth=1.5, linestyle=":")
ax.axvline(s_star, color="grey", linestyle=":", alpha=0.7,
           label=rf"$s^* = {s_star:.2f}$")
ax.axhline(0, color="k", linewidth=0.5)
ax.set_xlabel("Asset price $s$")
ax.set_ylabel("Value")
ax.set_title(
    r"Backbone of Stage B ansatz: $g_2(s) + v(s,t)$  "
    r"(deterministic part, without NN correction)"
)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

fig.suptitle(
    rf"Fig B — $g_2(s) + v(s,t)$ across $t\in[0, t_1]$  |  "
    rf"K={K}, r={r}, $\sigma$={sigma}, $t_1$={t1}",
    fontsize=11,
)
fig.tight_layout()
out_B = EXP_DIR / "diagnostics" / "plot_g2_backbone.png"
fig.savefig(out_B, dpi=150)
plt.close(fig)
print(f"[OK] Fig B saved: {out_B}")

print("\nDone.")
