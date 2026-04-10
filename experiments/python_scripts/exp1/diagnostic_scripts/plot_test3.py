import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(".")

from learning_option_pricing.models.etcnn import AmericanPutETCNN
from learning_option_pricing.models.resnet import ResNet

folder = Path("data/phase3_training/20260409_201206_iters2000_K100_interppchip")
model_path = folder / "etcnn_a.pt"

K = 100.0
r = 0.02
sigma = 0.25
T = 1.0

# Initialize model
resnet = ResNet(d_in=2, d_out=1, n=50, M=4, L=2)
model = AmericanPutETCNN(K=K, r=r, sigma=sigma, T=T, resnet=resnet)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

weights = []
layer_names = []

# Input layer (Sequential: Linear -> Tanh)
w = model.resnet.input_layer[0].weight.detach().numpy().flatten()
weights.append(w)
layer_names.append("Input")

# Blocks
for i, block in enumerate(model.resnet.blocks):
    # block.layers is a Sequential of [Linear, Tanh, Linear, Tanh]
    for j, layer in enumerate(block.layers):
        if isinstance(layer, torch.nn.Linear):
            w = layer.weight.detach().numpy().flatten()
            weights.append(w)
            layer_names.append(f"B{i} L{j//2}")

# Output layer
w = model.resnet.output_layer.weight.detach().numpy().flatten()
weights.append(w)
layer_names.append("Output")

fig, ax = plt.subplots(figsize=(12, 6))
parts = ax.violinplot(weights, showmeans=True, showextrema=True)

for pc in parts['bodies']:
    pc.set_facecolor('purple')
    pc.set_edgecolor('black')
    pc.set_alpha(0.5)

ax.set_xticks(np.arange(1, len(layer_names) + 1))
ax.set_xticklabels(layer_names, rotation=45, ha="right")
ax.set_ylabel("Weight Value ($w$)")
ax.set_title("Test III: Neuron Weight Magnitudes in ETCNN_A\nLarge weights ($|w| > 1$) amplify high-frequency noise via $w^2$ in the second derivative")
ax.grid(True, alpha=0.3)
# Add horizontal lines at -1 and 1 to highlight large weights
ax.axhline(1.0, color='red', linestyle=':', alpha=0.5)
ax.axhline(-1.0, color='red', linestyle=':', alpha=0.5)

fig.tight_layout()

out_path = folder / "plotB10_test3_weight_distribution.png"
plt.savefig(out_path, dpi=150)
print(f"Saved custom plot to {out_path}")
