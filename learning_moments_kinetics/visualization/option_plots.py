"""Plotting utilities for ETCNN option pricing experiments.

Typical figures:
    - Option price surface V(s, t) vs reference.
    - Pointwise error heatmap |V_pred - V_true|.
    - Free boundary S*(t) comparison (ETCNN vs binomial tree).
    - Training dynamics (total loss, L_bs, L_tv, L_eq vs epoch).
"""
from __future__ import annotations

# TODO: implement plotting functions


def plot_price_surface(s_grid, t_grid, V, *, title: str = "Option price V(s,t)"):
    """2-D heatmap of the option price surface.

    Args:
        s_grid: 1-D array of underlying prices.
        t_grid: 1-D array of times.
        V:      2-D array of option prices, shape (len(s_grid), len(t_grid)).
        title:  Figure title.
    """
    # TODO
    raise NotImplementedError


def plot_error_heatmap(s_grid, t_grid, error, *, title: str = "Pointwise error"):
    """2-D heatmap of |V_pred - V_true|.

    Args:
        s_grid: 1-D array of underlying prices.
        t_grid: 1-D array of times.
        error:  2-D array of absolute errors.
        title:  Figure title.
    """
    # TODO
    raise NotImplementedError


def plot_free_boundary(t_grid, S_star_ref, S_star_pred, *, label_ref: str = "BT (N=4000)"):
    """Compare predicted free boundary S*(t) against a reference.

    Args:
        t_grid:      1-D array of times.
        S_star_ref:  Reference free boundary (binomial tree).
        S_star_pred: Predicted free boundary (ETCNN).
        label_ref:   Legend label for the reference.
    """
    # TODO
    raise NotImplementedError


def plot_training_curves(history: dict):
    """Plot loss components vs training epoch.

    Args:
        history: Dict with keys 'loss', 'loss_bs', 'loss_tv', 'loss_eq',
                 each mapping to a list of per-epoch values.
    """
    # TODO
    raise NotImplementedError
