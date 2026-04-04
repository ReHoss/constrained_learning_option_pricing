"""Linear complementarity loss terms for BSM equations (American / Bermuda options).

Reference: Zhang, Guo, Lu (2026) — ETCNN, Section 2.2.2.

The composite loss for American options is

    L(θ) = λ_bs * L_bs(θ)   # penalises F(V) > 0
          + λ_tv * L_tv(θ)   # penalises TV(V) < 0  (time value ≥ 0)
          + λ_eq * L_eq(θ)   # complementarity equality F·TV = 0
          + λ_tc * L_tc(θ)   # terminal condition (zero for ETCNN)

ETCNN eliminates λ_tc by construction; only λ_bs, λ_tv, λ_eq remain.
"""
from __future__ import annotations

# TODO: implement loss terms


def loss_bs(F_u):
    """BSM operator penalty: penalise F(u) > 0.

    L_bs = mean( max(F(u), 0)^2 )

    Args:
        F_u: Batch of F(u_θ) values.
    """
    # TODO
    raise NotImplementedError


def loss_tv(TV_u):
    """Time-value penalty: penalise TV(u) < 0.

    L_tv = mean( max(-TV(u), 0)^2 )

    Args:
        TV_u: Batch of TV(u_θ) = u_θ - Φ values.
    """
    # TODO
    raise NotImplementedError


def loss_eq(F_u, TV_u):
    """Complementarity equality: F(u) · TV(u) = 0.

    L_eq = mean( (F(u) · TV(u))^2 )

    Args:
        F_u:  Batch of F(u_θ) values.
        TV_u: Batch of TV(u_θ) values.
    """
    # TODO
    raise NotImplementedError


def composite_loss(F_u, TV_u, *, lam_bs: float = 1.0, lam_tv: float = 1.0, lam_eq: float = 1.0):
    """Weighted sum of the three complementarity loss terms.

    Args:
        F_u:    Batch of F(u_θ) values.
        TV_u:   Batch of TV(u_θ) values.
        lam_bs: Weight for L_bs.
        lam_tv: Weight for L_tv.
        lam_eq: Weight for L_eq.

    Returns:
        Scalar total loss.
    """
    # TODO
    raise NotImplementedError
