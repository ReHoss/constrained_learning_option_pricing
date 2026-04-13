import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.append(".")

from learning_option_pricing.models.etcnn import AmericanPutETCNN
from learning_option_pricing.pricing.singularity import build_singularity_extraction
from learning_option_pricing.pricing.interpolation import PchipInterpolator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data/phase3_training/20260413_131729_iters500_K100_extraction_g2-taylor")
    args = parser.parse_args()
    
    folder = Path(args.folder)
    model_a_path = folder / "models" / "etcnn_a.pt"
    
    if not model_a_path.exists():
        print(f"Error: {model_a_path} not found.")
        return
        
    # Hardcoded from metadata.yaml
    K = 100.0
    r = 0.02
    sigma = 0.25
    T = 1.0
    t1 = 0.5
    q = 0.0
    
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    
    print(f"Loading ETCNN_A from {model_a_path}...")
    # 1. Load ETCNN_A
    etcnn_a = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, normalize_input=True, g2_type="taylor")
    etcnn_a.load_state_dict(torch.load(model_a_path, map_location=device))
    etcnn_a.eval()
    
    print("Executing Stage B extraction at t1...")
    # 2. Execute Stage B extraction at t1
    s_lo, s_hi = 10.0, 170.0
    s_star, c_scale, fict_put, s_nodes, v_target, residual_cpu = build_singularity_extraction(
        etcnn_a, K, r, sigma, t1, s_lo=s_lo, s_hi=s_hi, device=device, n_grid=2000
    )
    
    # Isolate the PCHIP interpolant g2(s)
    g2_spline = PchipInterpolator(s_nodes, residual_cpu)
    
    print(f"Exercise boundary found at s* = {s_star:.4f}")
    
    # 3. Plot the Micro-Step (First Derivative) around s*
    s_micro = torch.linspace(80.0, 82.0, 5000, requires_grad=True, dtype=torch.float64)
    g2_micro = g2_spline(s_micro)
    
    # First derivative of g2
    dg2_ds = torch.autograd.grad(
        g2_micro, s_micro, 
        grad_outputs=torch.ones_like(g2_micro),
        create_graph=True
    )[0]
    
    # Second derivative of g2 on micro grid
    d2g2_ds2_micro = torch.autograd.grad(
        dg2_ds, s_micro,
        grad_outputs=torch.ones_like(dg2_ds),
        create_graph=False
    )[0]
    
    # Micro BSM residual
    bsm_residual_micro = 0.5 * sigma**2 * s_micro**2 * d2g2_ds2_micro + (r - q) * s_micro * dg2_ds - r * g2_micro
    r_spline_micro = bsm_residual_micro**2
    
    # 4. The Smoking Gun (Second Derivative / PDE Residual of g2)
    s_global = torch.linspace(60.0, 120.0, 2000, requires_grad=True, dtype=torch.float64)
    
    g2_global = g2_spline(s_global)
    
    # Calculate the spatial BSM operator:
    # L(g2) = 0.5 * sigma^2 * s^2 * d2g2_ds2 + (r-q) * s * dg2_ds - r * g2
    
    dg2_ds_global = torch.autograd.grad(
        g2_global, s_global,
        grad_outputs=torch.ones_like(g2_global),
        create_graph=True
    )[0]
    
    d2g2_ds2_global = torch.autograd.grad(
        dg2_ds_global, s_global,
        grad_outputs=torch.ones_like(dg2_ds_global),
        create_graph=False
    )[0]
    
    bsm_residual = 0.5 * sigma**2 * s_global**2 * d2g2_ds2_global + (r - q) * s_global * dg2_ds_global - r * g2_global
    r_spline = bsm_residual**2
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top Left: Micro-step of first derivative
    ax = axes[0, 0]
    ax.plot(s_micro.detach().numpy(), dg2_ds.detach().numpy(), label=r"$\partial_s g_2(s)$", color="blue")
    ax.axvline(s_star, color="red", linestyle="--", label=f"$s^* = {s_star:.4f}$")
    ax.set_title(r"Micro-Step in $\partial_s g_2(s)$ near $s^*$")
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$\partial_s g_2(s)$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top Right: Micro-step of second derivative
    ax = axes[0, 1]
    ax.plot(s_micro.detach().numpy(), d2g2_ds2_micro.detach().numpy(), label=r"$\partial_{ss} g_2(s)$", color="purple")
    ax.axvline(s_star, color="red", linestyle="--", label=f"$s^* = {s_star:.4f}$")
    ax.set_title(r"Micro-Step in $\partial_{ss} g_2(s)$ near $s^*$")
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$\partial_{ss} g_2(s)$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom Left: Global Squared BSM residual
    ax = axes[1, 0]
    ax.semilogy(s_global.detach().numpy(), r_spline.detach().numpy(), label=r"$R_{\mathrm{spline}}(s) = |\mathcal{L}(g_2(s))|^2$", color="red")
    ax.axvline(s_star, color="black", linestyle="--", label=f"$s^* = {s_star:.4f}$")
    ax.set_title(r"Global Squared BSM Residual of $g_2(s)$ alone")
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$|\mathcal{L}(g_2(s))|^2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom Right: Zoomed Squared BSM residual
    ax = axes[1, 1]
    ax.semilogy(s_micro.detach().numpy(), r_spline_micro.detach().numpy(), label=r"$R_{\mathrm{spline}}(s)$", color="darkred")
    ax.axvline(s_star, color="black", linestyle="--", label=f"$s^* = {s_star:.4f}$")
    ax.set_title(r"Zoomed Squared BSM Residual near $s^*$")
    ax.set_xlabel("$s$")
    ax.set_ylabel(r"$|\mathcal{L}(g_2(s))|^2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    out_path = folder / "diagnostics" / "plot_stage_b_artifact_proof.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved diagnostic plot to {out_path}")

if __name__ == "__main__":
    main()
