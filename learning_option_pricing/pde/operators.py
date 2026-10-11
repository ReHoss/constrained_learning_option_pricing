r"""Differential operators for the boundary-constrained learning experiments.

The backward residual operator is the parabolic counterpart of the
Black--Scholes operator in
:func:`learning_option_pricing.pricing.terminal.bsm_operator`, written in the
same terminal-value (forward-time) convention: the data are prescribed at the
terminal time ``t = T`` and the solution is propagated backwards.  For the
constant-coefficient spatial generator
:math:`A = c_2\,\partial_{xx} + c_1\,\partial_x + c_0` the residual operator is

.. math::

    P u = \partial_t u + c_2\,\partial_{xx} u + c_1\,\partial_x u + c_0\, u,

returned pointwise by :func:`constant_coefficient_operator`.  Multiplying
``P u`` by a test function and integrating recovers the weak form; squaring
and averaging recovers the strong-form residual loss used in the experiments.

The backward heat operator :func:`heat_operator` is the special case
``coefficients = {2: 0.5 * sigma**2}`` and delegates to the generic operator;
its two-part split :func:`heat_operator_parts` is preserved for backward
compatibility with the existing experiments.
"""
from __future__ import annotations

import torch

# Differential orders the constant-coefficient operator supports.  Orders 0-2
# are the advection-diffusion-reaction generators of stage 2; orders 3 and 4
# (added 2026-09-29 for the fourth-order cell) and 5 and 6 (added 2026-10-11
# for the sixth-order cell) are summed into a single ``higher_order``
# channel, so the channels of an order <= 2 generator are unchanged bitwise.
SUPPORTED_DIFFERENTIAL_ORDERS = (0, 1, 2, 3, 4, 5, 6)
MAXIMAL_DIFFERENTIAL_ORDER = max(SUPPORTED_DIFFERENTIAL_ORDERS)


def _validated_operator_coefficients(
    coefficients: dict[int, float],
) -> dict[int, float]:
    """Validate and normalise the coefficient mapping of the generic operator.

    A channel is produced exactly for the orders **present** in the mapping
    (an explicit zero coefficient still produces its — zero — channel);
    orders outside :data:`SUPPORTED_DIFFERENTIAL_ORDERS` raise.

    Raises:
        ValueError: If an order is not an integer in
            :data:`SUPPORTED_DIFFERENTIAL_ORDERS`.
    """
    normalised: dict = {}
    for order, coefficient in coefficients.items():
        if int(order) != order or int(order) not in SUPPORTED_DIFFERENTIAL_ORDERS:
            raise ValueError(
                "constant-coefficient operator orders must belong to "
                f"{SUPPORTED_DIFFERENTIAL_ORDERS}, received order {order!r}"
            )
        # A callable coefficient c(x) (variable-coefficient cells, added
        # 2026-09-29) is kept as is and evaluated at the spatial coordinate;
        # a constant is converted to float, so its channels are unchanged.
        normalised[int(order)] = coefficient if callable(coefficient) else float(coefficient)
    return normalised


def coefficient_values(coefficient, coord: torch.Tensor):
    """A float coefficient as is, or a callable coefficient evaluated at ``coord``."""
    return coefficient(coord) if callable(coefficient) else coefficient


def constant_coefficient_operator(
    field: torch.Tensor,
    coord: torch.Tensor,
    t: torch.Tensor,
    coefficients: dict[int, float],
) -> torch.Tensor:
    r"""Backward residual operator
    :math:`P u = \partial_t u + \sum_j c_j\,\partial_x^j u` by autograd.

    ``field`` must be connected to ``coord`` and ``t`` through the autograd
    graph (both created with ``requires_grad=True``).  A field that happens
    not to depend on one of the coordinates (e.g. a time-constant
    terminal-data extension :math:`\Psi = g(x)`) is handled gracefully: the
    corresponding derivative is taken to be zero rather than raising an
    "unused tensor" error.  Only the channels of the orders **present** in
    ``coefficients`` are added, so the heat special case reproduces the
    historical two-term sum bitwise.

    Args:
        field:        Field values, shape ``(N,)`` (or broadcast-compatible).
        coord:        Spatial coordinate, shape ``(N,)``.
        t:            Time coordinate, shape ``(N,)``.
        coefficients: Mapping from differential order to real coefficient,
                      e.g. ``{2: nu, 1: mu, 0: r0}``; orders restricted to
                      :data:`SUPPORTED_DIFFERENTIAL_ORDERS`.

    Returns:
        The residual ``P u`` of shape ``(N,)``.

    Raises:
        ValueError: If ``coefficients`` contains an unsupported order.
    """
    normalised_coefficients = _validated_operator_coefficients(coefficients)
    parts = constant_coefficient_operator_parts(
        field, coord, t, normalised_coefficients
    )
    operator_values = parts["velocity"]
    if 2 in normalised_coefficients:
        operator_values = operator_values + parts["diffusion"]
    if 1 in normalised_coefficients:
        operator_values = operator_values + parts["advection"]
    if any(order >= 3 for order in normalised_coefficients):
        operator_values = operator_values + parts["higher_order"]
    if 0 in normalised_coefficients:
        operator_values = operator_values + parts["reaction"]
    return operator_values


def constant_coefficient_operator_parts(
    field: torch.Tensor,
    coord: torch.Tensor,
    t: torch.Tensor,
    coefficients: dict[int, float],
) -> dict[str, torch.Tensor]:
    r"""Return the additive channels of :func:`constant_coefficient_operator`.

    The residual splits into

    * ``velocity``  — :math:`\partial_t u` (always computed);
    * ``diffusion`` — :math:`c_2\,\partial_{xx} u`;
    * ``advection`` — :math:`c_1\,\partial_x u`;
    * ``reaction``  — :math:`c_0\, u`;
    * ``higher_order`` — :math:`\sum_{3 \le j \le 6} c_j\,\partial_x^j u`.

    All five keys are always present in the returned mapping; a channel whose
    order is absent from ``coefficients`` is a zero tensor.  Applied to a
    terminal-data extension :math:`\Psi`, the velocity channel is the
    *interpolation-velocity* forcing and the remaining channels attribute the
    forcing floor by mechanism.  Fields independent of ``coord`` or ``t``
    contribute a zero channel (handled gracefully through
    ``allow_unused=True``).  Autograd uses ``create_graph=True`` so the
    channels remain differentiable.

    Args:
        field:        Field values, shape ``(N,)`` (or broadcast-compatible).
        coord:        Spatial coordinate, shape ``(N,)``.
        t:            Time coordinate, shape ``(N,)``.
        coefficients: Mapping from differential order to real coefficient.

    Returns:
        ``{"velocity": ..., "diffusion": ..., "advection": ...,
        "reaction": ...}``, each of shape ``(N,)``.

    Raises:
        ValueError: If ``coefficients`` contains an unsupported order.
    """
    normalised_coefficients = _validated_operator_coefficients(coefficients)

    (grad_field_t,) = torch.autograd.grad(
        field,
        (t,),
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        allow_unused=True,
    )
    velocity_channel = (
        grad_field_t if grad_field_t is not None else torch.zeros_like(t)
    )

    needs_first_space_derivative = any(
        order >= 1 for order in normalised_coefficients
    )
    grad_field_x = None
    if needs_first_space_derivative:
        (grad_field_x,) = torch.autograd.grad(
            field,
            (coord,),
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            allow_unused=True,
        )
    first_space_derivative = (
        grad_field_x if grad_field_x is not None else torch.zeros_like(coord)
    )

    needs_second_space_derivative = any(
        order >= 2 for order in normalised_coefficients
    )
    if needs_second_space_derivative and grad_field_x is not None:
        (grad_field_xx,) = torch.autograd.grad(
            grad_field_x,
            (coord,),
            grad_outputs=torch.ones_like(grad_field_x),
            create_graph=True,
            allow_unused=True,
        )
        second_space_derivative = (
            grad_field_xx if grad_field_xx is not None else torch.zeros_like(coord)
        )
    else:
        second_space_derivative = torch.zeros_like(coord)

    # Space derivatives of orders 3 to 6, each only when a coefficient of that
    # order or higher is present (repeated autograd of the previous derivative).
    higher_space_derivatives = {}
    previous_derivative = second_space_derivative
    for order in range(3, MAXIMAL_DIFFERENTIAL_ORDER + 1):
        if not any(
            o in normalised_coefficients
            for o in range(order, MAXIMAL_DIFFERENTIAL_ORDER + 1)
        ):
            break
        if previous_derivative.requires_grad:
            (next_derivative,) = torch.autograd.grad(
                previous_derivative,
                (coord,),
                grad_outputs=torch.ones_like(previous_derivative),
                create_graph=True,
                allow_unused=True,
            )
        else:
            next_derivative = None
        previous_derivative = (
            next_derivative if next_derivative is not None else torch.zeros_like(coord)
        )
        higher_space_derivatives[order] = previous_derivative
    higher_order_channel = torch.zeros_like(coord)
    for order, derivative in higher_space_derivatives.items():
        if order in normalised_coefficients:
            higher_order_channel = (
                higher_order_channel
                + coefficient_values(normalised_coefficients[order], coord) * derivative
            )

    diffusion_channel = (
        coefficient_values(normalised_coefficients[2], coord) * second_space_derivative
        if 2 in normalised_coefficients
        else torch.zeros_like(coord)
    )
    advection_channel = (
        coefficient_values(normalised_coefficients[1], coord) * first_space_derivative
        if 1 in normalised_coefficients
        else torch.zeros_like(coord)
    )
    reaction_channel = (
        coefficient_values(normalised_coefficients[0], coord) * field
        if 0 in normalised_coefficients
        else torch.zeros_like(field)
    )

    return {
        "velocity": velocity_channel,
        "diffusion": diffusion_channel,
        "advection": advection_channel,
        "reaction": reaction_channel,
        "higher_order": higher_order_channel,
    }


def heat_operator(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    r"""Backward heat operator ``P u = du/dt + (sigma^2/2) d^2u/dx^2``.

    The special case ``coefficients = {2: 0.5 * sigma**2}`` of
    :func:`constant_coefficient_operator` (bitwise identical to the
    historical two-term implementation, since only the velocity and diffusion
    channels are summed).  The diffusion coefficient is written as
    ``sigma^2 / 2`` to match the convention of ``cal_notes/example_heat.tex``
    (where ``sigma = 1`` gives the ``1/2 d_xx`` form) and of the
    Black--Scholes operator in
    :func:`learning_option_pricing.pricing.terminal.bsm_operator`.

    Args:
        u:     Field values, shape ``(N,)`` (or broadcast-compatible).  Must be
               a differentiable function of ``x`` and/or ``t``.
        x:     Spatial coordinate, shape ``(N,)``.
        t:     Time coordinate, shape ``(N,)``.
        sigma: Diffusion scale; the diffusion coefficient is ``sigma^2 / 2``.

    Returns:
        The residual ``P u`` of shape ``(N,)``.
    """
    return constant_coefficient_operator(u, x, t, {2: 0.5 * sigma**2})


def heat_operator_parts(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the two additive parts of :func:`heat_operator`.

    ``P u = (du/dt) + (sigma^2/2) d^2u/dx^2`` is split into

    * the **time part** ``du/dt`` (the ``velocity`` channel) and
    * the **diffusion part** ``(sigma^2/2) d^2u/dx^2`` (the ``diffusion``
      channel),

    whose sum is :func:`heat_operator`.  Applied to a terminal-data extension
    :math:`\Psi`, the time part is the *interpolation-velocity* forcing
    (:math:`\lambda' g` for :math:`\Psi = \lambda g`) and the diffusion part is
    the *damped-diffusion* forcing (:math:`\lambda \tfrac{\sigma^2}{2} g''`),
    letting the forcing floor be attributed by mechanism.  Fields independent of
    ``x`` or ``t`` contribute a zero part (handled gracefully).

    Returns:
        ``(time_part, diffusion_part)``, each of shape ``(N,)``.
    """
    parts = constant_coefficient_operator_parts(u, x, t, {2: 0.5 * sigma**2})
    return parts["velocity"], parts["diffusion"]
