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

# TODO: implement ResNet


class ResidualBlock:
    """Single residual block (L layers, width n, tanh activations)."""

    # TODO


class ResNet:
    """ResNet backbone: M blocks, each of L layers with n neurons.

    Args:
        d_in:   Input dimension (number of features).
        d_out:  Output dimension.
        n:      Width of each hidden layer.
        M:      Number of residual blocks.
        L:      Number of layers per block.
    """

    # TODO
