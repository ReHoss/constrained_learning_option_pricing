import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(".")

from learning_option_pricing.models.etcnn import AmericanPutETCNN
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.pricing.terminal import payoff_put
from learning_option_pricing.pricing.interpolation import PchipInterpolator

folder = Path("data/phase3_training/20260409_201206_iters2000_K100_interppchip")
model_path = folder / "etcnn_a.pt"

K = 100.0
r = 0.02
sigma = 0.25
T = 1.0
t1 = 0.5

# Initialize model
resnet = ResNet(d_in=2, d_out=1, n=50, M=4, L=2)
model = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, resnet=resnet)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Create grid
s_grid = torch.linspace(20, 160, 2000).view(-1, 1)
t_grid = torch.full_like(s_grid, t1)
x_grid = torch.cat([s_grid, t_grid], dim=1)

with torch.no_grad():
    hold_value = model(x_grid)
    exercise_value = payoff_put(s_grid, K)
    v_t1 = torch.max(exercise_value, hold_value)

s_grid_1d = s_grid.squeeze()
v_t1_1d = v_t1.squeeze()

interp = PchipInterpolator(s_grid_1d, v_t1_1d)

# Find s*
diff_ex = exercise_value - hold_value
sign_changes = torch.where(diff_ex[:-1] * diff_ex[1:] < 0)[0]
s_star = float(s_grid[sign_changes[0].item()]) if len(sign_changes) > 0 else 80.92

# Test 1: Interpolant Curvature
s_fine = torch.linspace(s_star - 1.0, 82.0, 1000)
s_wide = torch.linspace(s_star - 20.0, s_star + 20.0, 2000)
h = 1e-3

def raw_target(s_tensor):
    t_tensor = torch.full_like(s_tensor, t1)
    x_tensor = torch.cat([s_tensor.view(-1, 1), t_tensor.view(-1, 1)], dim=1)
    hold = model(x_tensor).squeeze()
    phi = payoff_put(s_tensor, K)
    return torch.maximum(phi, hold)

with torch.no_grad():
    # Fine grid
    v_plus = interp(s_fine + h)
    v_center = interp(s_fine)
    v_minus = interp(s_fine - h)
    gamma_spline = (v_plus - 2 * v_center + v_minus) / (h ** 2)

    v_raw_plus = raw_target(s_fine + h)
    v_raw_center = raw_target(s_fine)
    v_raw_minus = raw_target(s_fine - h)
    gamma_raw = (v_raw_plus - 2 * v_raw_center + v_raw_minus) / (h ** 2)

    # Wide grid
    v_wide_plus = interp(s_wide + h)
    v_wide_center = interp(s_wide)
    v_wide_minus = interp(s_wide - h)
    gamma_spline_wide = (v_wide_plus - 2 * v_wide_center + v_wide_minus) / (h ** 2)

    v_raw_wide_plus = raw_target(s_wide + h)
    v_raw_wide_center = raw_target(s_wide)
    v_raw_wide_minus = raw_target(s_wide - h)
    gamma_raw_wide = (v_raw_wide_plus - 2 * v_raw_wide_center + v_raw_wide_minus) / (h ** 2)

fig, axes = plt.subplots(3, 1, figsize=(9, 12))

# Top plot: Curvature (Wide view)
axes[0].plot(s_wide.numpy(), gamma_spline_wide.numpy(), label="Curvature of $\\mathcal{I}(s)$ (PCHIP)", color="purple", linewidth=2)
axes[0].plot(s_wide.numpy(), gamma_raw_wide.numpy(), label="Curvature of Raw Target $\\max(\\Phi, U_A)$", color="green", linewidth=2, linestyle="--")
if not np.isnan(s_star):
    axes[0].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
axes[0].set_title(f"Test 1: PCHIP Curvature around $s^*$ (Zoomed Out)\n$\\mathcal{{I}}(s)$ is the interpolated intermediate condition $V(s, t_1)$")
axes[0].set_xlabel("Asset Price $s$")
axes[0].set_ylabel("$[V(s+h) - 2V(s) + V(s-h)] / h^2$")
axes[0].set_ylim(-20, 20) # Constrain y-axis to see the noise better
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Middle plot: Curvature (Zoomed in)
axes[1].plot(s_fine.numpy(), gamma_spline.numpy(), label="Curvature of $\\mathcal{I}(s)$ (PCHIP)", color="purple", linewidth=2)
axes[1].plot(s_fine.numpy(), gamma_raw.numpy(), label="Curvature of Raw Target $\\max(\\Phi, U_A)$", color="green", linewidth=2, linestyle="--")
if not np.isnan(s_star):
    axes[1].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
axes[1].set_title(f"Test 1: PCHIP Curvature around $s^*$ (Zoomed In)")
axes[1].set_xlabel("Asset Price $s$")
axes[1].set_ylabel("$[V(s+h) - 2V(s) + V(s-h)] / h^2$")
axes[1].set_ylim(-20, 20) # Constrain y-axis to see the noise better
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Bottom plot: Function values (Zoomed in)
axes[2].plot(s_fine.numpy(), v_center.numpy(), label="$\\mathcal{I}(s)$ (PCHIP)", color="blue", linewidth=2)
axes[2].plot(s_fine.numpy(), v_raw_center.numpy(), label="Raw Target $\\max(\\Phi, U_A)$", color="orange", linewidth=2, linestyle="--")
if not np.isnan(s_star):
    axes[2].axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
axes[2].set_title(f"Test 1: Function values around $s^*$ (Zoomed In)")
axes[2].set_xlabel("Asset Price $s$")
axes[2].set_ylabel("Option Value")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

fig.tight_layout()

out_path = folder / "plotB1d_interpolant_gamma.png"
plt.savefig(out_path, dpi=150)
print(f"Saved custom plot to {out_path}")
