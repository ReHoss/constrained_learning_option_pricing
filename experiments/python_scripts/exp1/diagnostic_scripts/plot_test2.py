import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(".")

from learning_option_pricing.models.etcnn import AmericanPutETCNN, ETCNN, InputNormalization
from learning_option_pricing.models.resnet import ResNet
from learning_option_pricing.pricing.terminal import payoff_put, bsm_operator
from learning_option_pricing.pricing.interpolation import CubicSplineInterpolator

folder = Path("data/phase3_training/20260409_154609_iters20000_K100_interpcubic")
model_a_path = folder / "etcnn_a.pt"
model_b_path = folder / "etcnn_b.pt"

K = 100.0
r = 0.02
sigma = 0.25
T = 1.0
t1 = 0.5
q = 0.0

# 1. Recreate ETCNN_A to get the intermediate condition
resnet_a = ResNet(d_in=2, d_out=1, n=50, M=4, L=2)
etcnn_a = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, resnet=resnet_a)
etcnn_a.load_state_dict(torch.load(model_a_path, map_location="cpu"))
etcnn_a.eval()

s_grid = torch.linspace(10, 170, 2000).view(-1, 1)
t_grid = torch.full_like(s_grid, t1)
x_grid = torch.cat([s_grid, t_grid], dim=1)

with torch.no_grad():
    hold_value = etcnn_a(x_grid)
    exercise_value = payoff_put(s_grid, K)
    v_t1 = torch.max(exercise_value, hold_value)

s_grid_1d = s_grid.squeeze()
v_t1_1d = v_t1.squeeze()
interp = CubicSplineInterpolator(s_grid_1d, v_t1_1d)

# Find s*
diff_ex = exercise_value - hold_value
sign_changes = torch.where(diff_ex[:-1] * diff_ex[1:] < 0)[0]
s_star = float(s_grid[sign_changes[0].item()]) if len(sign_changes) > 0 else 80.92

# 2. Recreate ETCNN_B
def g1_b(s, t): return t1 - t
def g2_b(s, t): return interp(s)

resnet_b = ResNet(d_in=2, d_out=1, n=50, M=4, L=2)
etcnn_b = ETCNN(resnet=resnet_b, g1=g1_b, g2=g2_b, normalizer=InputNormalization(K))
etcnn_b.load_state_dict(torch.load(model_b_path, map_location="cpu"))
etcnn_b.eval()

# 3. Test II: Spatial Distribution of the PDE Residual at t1-
s_eval = torch.linspace(60.0, 120.0, 1000)
s_eval.requires_grad_(True)
t_eval = torch.full_like(s_eval, t1 - 1e-4)
t_eval.requires_grad_(True)

x_eval = torch.stack([s_eval, t_eval], dim=1)
u_b = etcnn_b(x_eval).squeeze()

residual = bsm_operator(u_b, s_eval, t_eval, r, q, sigma)
R_s = residual ** 2

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(s_eval.detach().numpy(), R_s.detach().numpy(), label="PDE Residual $R(s)$", color="darkorange", linewidth=2)
ax.axvline(s_star, color="red", linestyle=":", alpha=0.7, label=f"Exercise boundary $s^* \\approx {s_star:.1f}$")
ax.set_title(f"Test II: Spatial Distribution of PDE Residual at $t_1^-$")
ax.set_xlabel("Asset Price $s$")
ax.set_ylabel("$R(s) = |\\mathcal{F}(U_B)(s, t_1^-)|^2$")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = folder / "plotB9_test2_pde_residual.png"
plt.savefig(out_path, dpi=150)
print(f"Saved custom plot to {out_path}")
