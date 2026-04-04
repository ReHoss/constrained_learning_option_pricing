"""Exp 1 — Single-asset American put option (ETCNN baseline).

Reproduces the single-asset American put experiment from:
    Zhang, Guo, Lu (2026) — ETCNN, Section 4.1.

Parameters
----------
K=100, r=0.02, T=1, σ=0.25  (Table 3 in the paper).

Steps
-----
1. Generate reference solution via binomial tree (N=4000).
2. Train ETCNN with g2 = V1^e + V2^e and input normalisation.
3. Evaluate: relative L2 error, MAE on [60, 120] × [0, T].
4. Plot: price surface, error heatmap, free boundary, training curves.
"""
from __future__ import annotations

# TODO: implement experiment
