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

# Interpolator expects 1D tensors
s_grid_1d = s_grid.squeeze()
v_t1_1d = v_t1.squeeze()

interp = PchipInterpolator(s_grid_1d, v_t1_1d)
s_plot = torch.linspace(60, 140, 500)
v_interp = interp(s_plot)

plt.figure(figsize=(8, 5))
plt.plot(s_plot.numpy(), v_interp.numpy(), label="Interpolated $V(s, t_1)$", color="blue", linewidth=2)
plt.plot(s_plot.numpy(), payoff_put(s_plot, K).numpy(), label="Payoff $\Phi(s)$", color="red", linestyle="--")
plt.axvline(x=K, color="grey", linestyle=":", label="Strike K")
plt.title(f"Interpolated Function at $t_1 = {t1}$")
plt.xlabel("Asset Price $s$")
plt.ylabel("Option Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

out_path = folder / "plot_interpolated_t1_custom.png"
plt.savefig(out_path, dpi=150)
print(f"Saved custom plot to {out_path}")
