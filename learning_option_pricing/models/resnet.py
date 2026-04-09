"""ResNet building block used as the backbone inside ETCNN.

Reference: He et al. (2016) — Deep Residual Learning for Image Recognition.
           Zhang, Guo, Lu (2026) — ETCNN, Section 3.2.

Architecture summary
--------------------
- M residual blocks, each with L fully-connected layers of width n.
- Skip connection: output of block m = last-layer pre-activation + block input.
- Activation: tanh (following the ETCNN paper).
- Final linear readout: W_out @ g^(M+1,0) + b_out.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Single residual block (L layers, width n, tanh activations).

    Forward pass:
        h = x
        for each layer:
            h = tanh(W_l @ h + b_l)
        return h + x   (residual skip connection)
    """

    def __init__(self, n: int, L: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(L):
            layers.append(nn.Linear(n, n))
            layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class ResNet(nn.Module):
    """ResNet backbone: M blocks, each of L layers with n neurons.

    Architecture (Fig. 2):
        input_layer:  Linear(d_in, n) + tanh
        blocks:       M x ResidualBlock(n, L)
        output_layer: Linear(n, d_out)  (no activation)

    Args:
        d_in:   Input dimension (number of features).
        d_out:  Output dimension.
        n:      Width of each hidden layer.
        M:      Number of residual blocks.
        L:      Number of layers per block.
    """

    def __init__(
        self,
        d_in: int = 2,
        d_out: int = 1,
        n: int = 50,
        M: int = 4,
        L: int = 2,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n = n
        self.M = M
        self.L = L

        self.input_layer = nn.Sequential(nn.Linear(d_in, n), nn.Tanh())
        self.blocks = nn.Sequential(*[ResidualBlock(n, L) for _ in range(M)])
        self.output_layer = nn.Linear(n, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.blocks(h)
        return self.output_layer(h)
