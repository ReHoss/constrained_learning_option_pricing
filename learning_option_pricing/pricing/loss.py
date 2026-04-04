"""Linear complementarity loss terms for BSM equations (American / Bermuda options).

Reference: Zhang, Guo, Lu (2026) — ETCNN, Section 2.2.2.

The composite loss for American options is

    L(theta) = lambda_bs * L_bs(theta)   # penalises F(V) > 0
             + lambda_tv * L_tv(theta)   # penalises TV(V) < 0  (time value >= 0)
             + lambda_eq * L_eq(theta)   # complementarity equality F*TV = 0
             + lambda_tc * L_tc(theta)   # terminal condition (zero for ETCNN)

ETCNN eliminates lambda_tc by construction; only lambda_bs, lambda_tv, lambda_eq remain.
"""
from __future__ import annotations

import torch


def loss_bs(F_u: torch.Tensor) -> torch.Tensor:
    """BSM operator penalty: penalise F(u) > 0.

    L_bs = mean( max(F(u), 0)^2 )

    Args:
        F_u: Batch of F(u_theta) values.
    """
    return torch.mean(torch.clamp(F_u, min=0.0) ** 2)


def loss_tv(TV_u: torch.Tensor) -> torch.Tensor:
    """Time-value penalty: penalise TV(u) < 0.

    L_tv = mean( max(-TV(u), 0)^2 )

    Args:
        TV_u: Batch of TV(u_theta) = u_theta - Phi values.
    """
    return torch.mean(torch.clamp(-TV_u, min=0.0) ** 2)


def loss_eq(F_u: torch.Tensor, TV_u: torch.Tensor) -> torch.Tensor:
    """Complementarity equality: F(u) * TV(u) = 0.

    L_eq = mean( (F(u) * TV(u))^2 )

    Args:
        F_u:  Batch of F(u_theta) values.
        TV_u: Batch of TV(u_theta) values.
    """
    return torch.mean((F_u * TV_u) ** 2)


def composite_loss(
    F_u: torch.Tensor,
    TV_u: torch.Tensor,
    *,
    lam_bs: float = 1.0,
    lam_tv: float = 1.0,
    lam_eq: float = 1.0,
) -> torch.Tensor:
    """Weighted sum of the three complementarity loss terms.

    Args:
        F_u:    Batch of F(u_theta) values.
        TV_u:   Batch of TV(u_theta) values.
        lam_bs: Weight for L_bs.
        lam_tv: Weight for L_tv.
        lam_eq: Weight for L_eq.

    Returns:
        Scalar total loss.
    """
    return lam_bs * loss_bs(F_u) + lam_tv * loss_tv(TV_u) + lam_eq * loss_eq(F_u, TV_u)
