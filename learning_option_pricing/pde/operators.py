"""Differential operators for the boundary-constrained learning experiments.

The backward heat operator is the parabolic counterpart of the Black--Scholes
operator in :func:`learning_option_pricing.pricing.terminal.bsm_operator`,
written in the same terminal-value (forward-time) convention: the data are
prescribed at the terminal time ``t = T`` and the solution is propagated
backwards.  Concretely, for the heat equation

    P u = d u / d t + (sigma^2 / 2) d^2 u / d x^2 = 0,   u(x, T) = g(x),

the operator returns the pointwise residual ``P u``.  Multiplying ``P u`` by a
test function and integrating recovers the weak form; squaring and averaging
recovers the strong-form residual loss used in the experiments.
"""
from __future__ import annotations

import torch


def heat_operator(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    r"""Backward heat operator ``P u = du/dt + (sigma^2/2) d^2u/dx^2``.

    The diffusion coefficient is written as ``sigma^2 / 2`` to match the
    convention of ``cal_notes/example_heat.tex`` (where ``sigma = 1`` gives the
    ``1/2 d_xx`` form) and of the Black--Scholes operator in
    :func:`learning_option_pricing.pricing.terminal.bsm_operator`.

    ``u`` must be connected to ``x`` and ``t`` through the autograd graph (both
    created with ``requires_grad=True``).  A field that happens not to depend on
    one of the coordinates (e.g. a time-constant terminal-data extension
    :math:`\Psi = g(x)`) is handled gracefully: the corresponding derivative is
    taken to be zero rather than raising an "unused tensor" error.

    Args:
        u:     Field values, shape ``(N,)`` (or broadcast-compatible).  Must be
               a differentiable function of ``x`` and/or ``t``.
        x:     Spatial coordinate, shape ``(N,)``.
        t:     Time coordinate, shape ``(N,)``.
        sigma: Diffusion scale; the diffusion coefficient is ``sigma^2 / 2``.

    Returns:
        The residual ``P u`` of shape ``(N,)``.
    """
    time_part, diffusion_part = heat_operator_parts(u, x, t, sigma)
    return time_part + diffusion_part


def heat_operator_parts(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the two additive parts of :func:`heat_operator`.

    ``P u = (du/dt) + (sigma^2/2) d^2u/dx^2`` is split into

    * the **time part** ``du/dt`` and
    * the **diffusion part** ``(sigma^2/2) d^2u/dx^2``,

    whose sum is :func:`heat_operator`.  Applied to a terminal-data extension
    :math:`\Psi`, the time part is the *blending-velocity* forcing
    (:math:`\lambda' g` for :math:`\Psi = \lambda g`) and the diffusion part is
    the *damped-diffusion* forcing (:math:`\lambda \tfrac{\sigma^2}{2} g''`),
    letting the forcing floor be attributed by mechanism.  Fields independent of
    ``x`` or ``t`` contribute a zero part (handled gracefully).

    Returns:
        ``(time_part, diffusion_part)``, each of shape ``(N,)``.
    """
    (grad_u_t,) = torch.autograd.grad(
        u, (t,), grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True
    )
    (grad_u_x,) = torch.autograd.grad(
        u, (x,), grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True
    )

    time_part = grad_u_t if grad_u_t is not None else torch.zeros_like(t)
    if grad_u_x is None:
        diffusion_part = torch.zeros_like(x)
    else:
        (grad_u_xx,) = torch.autograd.grad(
            grad_u_x,
            (x,),
            grad_outputs=torch.ones_like(grad_u_x),
            create_graph=True,
            allow_unused=True,
        )
        grad_u_xx = grad_u_xx if grad_u_xx is not None else torch.zeros_like(x)
        diffusion_part = 0.5 * sigma**2 * grad_u_xx

    return time_part, diffusion_part
