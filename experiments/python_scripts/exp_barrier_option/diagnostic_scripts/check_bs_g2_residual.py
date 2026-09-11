r"""Check that the Black-Scholes g2 mode annihilates the interior PDE residual.

Reference: working note "A rigorous statement of exact-constraint learning at
a conflicting constraint corner: the knock-out barrier option" (S. Ouaissi
internship, 2026-06-24), Section 4, and the module docstring of
``learning_option_pricing/pricing/barrier.py``.

The corner-regularised extension of Definition 5 has three mutually exclusive
terminal-function modes for its payoff-like factor: the raw discontinuous
payoff, the Chen-Mangasarian smoothed payoff (constant or time-graded
bandwidth), and the exact Black-Scholes European put price :math:`V^e(s,t)`.
Unlike the first two, :math:`V^e` already solves the Black-Scholes operator
:math:`\mathcal L^{BS}V^e=0` everywhere it is smooth, so the corner-regularised
extension built from it,

.. math::

    h_\varepsilon^{BS}(s,t) = \zeta\!\left(\frac{s-B}{\varepsilon}\right) V^e(s,t),

is expected to satisfy :math:`\mathcal L^{BS}h_\varepsilon^{BS} \approx 0`
away from the corner layer :math:`\{s-B\le\varepsilon\}`, where the cutoff
:math:`\zeta` is exactly 1 and its derivatives exactly 0. Inside the corner
layer, :math:`\zeta` itself is not constant, so the product rule generates a
nonzero forcing term even though :math:`V^e` solves the operator everywhere.

This script evaluates, by autograd applied to ``g2`` alone (never to a trained
network ``Phi_theta``), the interior residual

.. math::

    \mathcal F(g_2)(s,t) = \partial_t g_2 + \frac12\sigma^2s^2\partial_{ss}g_2
        + rs\,\partial_s g_2 - r g_2,

for all four terminal-function modes of
:mod:`learning_option_pricing.pricing.barrier`
(:func:`make_corner_regularised_extension`,
:func:`make_corner_regularised_extension_with_smoothed_payoff` with
``grading="constant"`` and ``grading="time_graded"``, and
:func:`make_corner_regularised_extension_with_black_scholes_payoff`), on two
regions: a main grid away from the corner layer, where the Black-Scholes mode
is expected to vanish to floating-point precision, and a band immediately
above the barrier, spanning the corner layer, where the cutoff :math:`\zeta`
is expected to contribute a nonzero residual for every mode including the
Black-Scholes one.

Pure closed-form evaluation and autograd -- no network, no training, no
collocation sampling.

Usage:
    python3 experiments/python_scripts/exp_barrier_option/diagnostic_scripts/\
check_bs_g2_residual.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from learning_option_pricing.pricing.barrier import (  # noqa: E402
    make_corner_regularised_extension,
    make_corner_regularised_extension_with_black_scholes_payoff,
    make_corner_regularised_extension_with_smoothed_payoff,
)
from learning_option_pricing.utils.run_context import find_repo_root  # noqa: E402

torch.set_default_dtype(torch.float64)

# Pilot contract (K=1, B=0.6, r=0.03, sigma=0.3, T=1) and the corner-layer
# bandwidth requested for this check ("eps_coin" in the note's notation,
# "epsilon" as the parameter name of every make_corner_regularised_extension*
# builder), shared identically across the four modes so the comparison is a
# like-for-like one.
STRIKE_K = 1.0
BARRIER_B = 0.6
RISK_FREE_RATE = 0.03
VOLATILITY_SIGMA = 0.3
MATURITY_T = 1.0
CORNER_LAYER_EPSILON = 0.1

# Chen-Mangasarian smoothing bandwidth (independent parameter from the
# corner-layer epsilon above; not specified by the task, so the pilot
# script's own default is reused -- see DEFAULT_EPS0 in pilot_down_and_out_put.py).
MANGASARIAN_EPS0 = 0.05

N_S_POINTS = 200
N_T_POINTS = 100
S_RANGE_MAIN = (0.7, 2.0)
S_RANGE_BAND = (0.6, 0.7)
T_RANGE = (0.0, 0.99)


def pde_residual(
    g2_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    s_values: torch.Tensor,
    t_values: torch.Tensor,
) -> torch.Tensor:
    r"""Evaluate :math:`\mathcal F(g_2)` on the ``(s,t)`` grid by autograd on ``g2`` alone.

    ``s_values``/``t_values`` are ordinary (non-leaf) grids; they are cloned,
    detached, and turned into fresh leaf tensors with ``requires_grad_(True)``
    before being passed to ``g2_fn``, so only ``g2_fn`` itself is
    differentiated -- no network, no ``g1``, no composite ansatz is involved
    anywhere in this computation.

    The Black-Scholes operator formula duplicates
    :func:`~learning_option_pricing.pricing.terminal.bsm_operator` rather than
    calling it, because that shared operator assumes both ``s`` and ``t``
    genuinely appear in the graph of ``V`` and calls ``torch.autograd.grad``
    without ``allow_unused``; the raw-payoff mode's ``g2`` is, by
    construction (see the module docstring of ``barrier.py``), exactly
    time-independent, so ``t`` never enters its computational graph and the
    shared operator's ``torch.autograd.grad(V, (t,), ...)`` call raises. Using
    ``allow_unused=True`` here and substituting an exact zero for an unused
    partial derivative is the correct mathematical statement for that mode
    (:math:`\partial_t g_2\equiv0`), not an approximation.

    Args:
        g2_fn: One of the four ``h_eps*`` callables built by the
            ``make_corner_regularised_extension*`` functions.
        s_values: 1-D grid of underlying-price values.
        t_values: 1-D grid of calendar-time values.

    Returns:
        :math:`\mathcal F(g_2)(s,t)`, flattened over the ``(s,t)`` grid,
        detached from the autograd graph.
    """
    ss, tt = torch.meshgrid(s_values, t_values, indexing="ij")
    s_flat = ss.reshape(-1).clone().detach().requires_grad_(True)
    t_flat = tt.reshape(-1).clone().detach().requires_grad_(True)
    g2_values = g2_fn(s_flat, t_flat)

    (grad_s,) = torch.autograd.grad(
        g2_values, (s_flat,), grad_outputs=torch.ones_like(g2_values), create_graph=True, allow_unused=True,
    )
    if grad_s is None:
        grad_s = torch.zeros_like(s_flat)
        grad_ss = torch.zeros_like(s_flat)
    else:
        (grad_ss,) = torch.autograd.grad(
            grad_s, (s_flat,), grad_outputs=torch.ones_like(grad_s), create_graph=True, allow_unused=True,
        )
        if grad_ss is None:
            grad_ss = torch.zeros_like(s_flat)

    (grad_t,) = torch.autograd.grad(
        g2_values, (t_flat,), grad_outputs=torch.ones_like(g2_values), create_graph=True, allow_unused=True,
    )
    if grad_t is None:
        grad_t = torch.zeros_like(t_flat)

    residual = (
        grad_t
        + 0.5 * VOLATILITY_SIGMA**2 * s_flat**2 * grad_ss
        + RISK_FREE_RATE * s_flat * grad_s
        - RISK_FREE_RATE * g2_values
    )
    return residual.detach()


def build_modes() -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    """The four mutually exclusive ``g2`` terminal-function modes, same corner-layer epsilon."""
    return {
        "raw": make_corner_regularised_extension(
            STRIKE_K, BARRIER_B, CORNER_LAYER_EPSILON,
        ),
        "mangasarian_constant": make_corner_regularised_extension_with_smoothed_payoff(
            STRIKE_K, BARRIER_B, CORNER_LAYER_EPSILON, MATURITY_T, MANGASARIAN_EPS0, grading="constant",
        ),
        "mangasarian_time_graded": make_corner_regularised_extension_with_smoothed_payoff(
            STRIKE_K, BARRIER_B, CORNER_LAYER_EPSILON, MATURITY_T, MANGASARIAN_EPS0, grading="time_graded",
        ),
        "black_scholes": make_corner_regularised_extension_with_black_scholes_payoff(
            STRIKE_K, BARRIER_B, CORNER_LAYER_EPSILON, RISK_FREE_RATE, VOLATILITY_SIGMA, MATURITY_T,
        ),
    }


def render_table(rows: list[dict]) -> str:
    header = f"| {'Mode':<24} | {'Region':<38} | {'N points':>8} | {'max|F(g2)|':>14} | {'mean|F(g2)|':>14} |"
    sep = "|" + "-" * (len(header) - 2) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['mode']:<24} | {row['region']:<38} | {row['n_points']:>8d} | "
            f"{row['max_abs_residual']:>14.6e} | {row['mean_abs_residual']:>14.6e} |"
        )
    return "\n".join(lines)


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Default dtype: {torch.get_default_dtype()}")
    print(
        f"Contract: K={STRIKE_K:g}, B={BARRIER_B:g}, r={RISK_FREE_RATE:g}, "
        f"sigma={VOLATILITY_SIGMA:g}, T={MATURITY_T:g}, corner-layer epsilon={CORNER_LAYER_EPSILON:g}"
    )
    print(f"Mangasarian smoothing bandwidth eps0={MANGASARIAN_EPS0:g} (independent parameter)")
    print()

    modes = build_modes()
    s_main = torch.linspace(S_RANGE_MAIN[0], S_RANGE_MAIN[1], N_S_POINTS, dtype=torch.float64)
    s_band = torch.linspace(S_RANGE_BAND[0], S_RANGE_BAND[1], N_S_POINTS, dtype=torch.float64)
    t_grid = torch.linspace(T_RANGE[0], T_RANGE[1], N_T_POINTS, dtype=torch.float64)

    regions = (
        (f"main, s in [{S_RANGE_MAIN[0]:g}, {S_RANGE_MAIN[1]:g}] (away from barrier)", s_main),
        (f"band, s in [{S_RANGE_BAND[0]:g}, {S_RANGE_BAND[1]:g}] (corner layer)", s_band),
    )

    rows = []
    for mode_name, g2_fn in modes.items():
        for region_name, s_values in regions:
            residual = pde_residual(g2_fn, s_values, t_grid)
            rows.append({
                "mode": mode_name,
                "region": region_name,
                "n_points": residual.numel(),
                "max_abs_residual": float(residual.abs().max()),
                "mean_abs_residual": float(residual.abs().mean()),
            })

    print(render_table(rows))

    out_dir = find_repo_root(Path(__file__)) / "data" / "bs_g2_residual"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "residual_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "region", "n_points", "max_abs_residual", "mean_abs_residual"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written -> {csv_path}")


if __name__ == "__main__":
    main()
