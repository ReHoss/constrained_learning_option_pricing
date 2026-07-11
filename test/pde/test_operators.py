r"""Tests for the constant-coefficient residual operator and its channels.

Covered here (specification Section 6.2): the parts-sum identity of
:func:`constant_coefficient_operator_parts` against
:func:`constant_coefficient_operator` and against a manual autograd assembly,
the bitwise equality of the heat special case with the historical two-term
implementation, the graceful zero channels for fields independent of one
coordinate, and the rejection of unsupported differential orders.
"""
from __future__ import annotations

import pytest
import torch

from learning_option_pricing.pde.operators import (
    constant_coefficient_operator,
    constant_coefficient_operator_parts,
    heat_operator,
    heat_operator_parts,
)


def _collocation_batch(size=64):
    generator = torch.Generator().manual_seed(0)
    x = torch.rand(size, generator=generator, dtype=torch.float64).requires_grad_(
        True
    )
    t = torch.rand(size, generator=generator, dtype=torch.float64).requires_grad_(
        True
    )
    return x, t


def _smooth_test_field(x, t):
    """A smooth field depending on both coordinates."""
    return torch.sin(2.0 * x) * torch.exp(-t) + x**2 * t


def test_parts_sum_identity():
    coefficients = {2: 0.35, 1: -0.7, 0: 0.2}
    x, t = _collocation_batch()
    u = _smooth_test_field(x, t)
    parts = constant_coefficient_operator_parts(u, x, t, coefficients)
    assert set(parts) == {"velocity", "diffusion", "advection", "reaction"}

    x2, t2 = _collocation_batch()
    u2 = _smooth_test_field(x2, t2)
    operator_values = constant_coefficient_operator(u2, x2, t2, coefficients)
    parts_sum = (
        parts["velocity"] + parts["diffusion"] + parts["advection"] + parts["reaction"]
    )
    assert torch.allclose(operator_values, parts_sum, rtol=0.0, atol=0.0)


def test_parts_match_manual_autograd_assembly():
    coefficients = {2: 0.35, 1: -0.7, 0: 0.2}
    x, t = _collocation_batch()
    u = _smooth_test_field(x, t)
    parts = constant_coefficient_operator_parts(u, x, t, coefficients)

    (grad_u_t,) = torch.autograd.grad(
        u, (t,), grad_outputs=torch.ones_like(u), create_graph=True
    )
    (grad_u_x,) = torch.autograd.grad(
        u, (x,), grad_outputs=torch.ones_like(u), create_graph=True
    )
    (grad_u_xx,) = torch.autograd.grad(
        grad_u_x, (x,), grad_outputs=torch.ones_like(grad_u_x), create_graph=True
    )
    assert torch.allclose(parts["velocity"], grad_u_t, atol=1e-12)
    assert torch.allclose(parts["diffusion"], 0.35 * grad_u_xx, atol=1e-12)
    assert torch.allclose(parts["advection"], -0.7 * grad_u_x, atol=1e-12)
    assert torch.allclose(parts["reaction"], 0.2 * u, atol=1e-12)


def test_heat_special_case_bitwise_equality():
    # The heat operator is the special case {2: 0.5 sigma^2}; only the
    # velocity and diffusion channels are summed, so the result must be
    # bitwise identical to the historical two-term implementation.
    sigma = 0.8
    x, t = _collocation_batch()
    u = _smooth_test_field(x, t)
    heat_values = heat_operator(u, x, t, sigma)

    x2, t2 = _collocation_batch()
    u2 = _smooth_test_field(x2, t2)
    (grad_u_t,) = torch.autograd.grad(
        u2, (t2,), grad_outputs=torch.ones_like(u2), create_graph=True
    )
    (grad_u_x,) = torch.autograd.grad(
        u2, (x2,), grad_outputs=torch.ones_like(u2), create_graph=True
    )
    (grad_u_xx,) = torch.autograd.grad(
        grad_u_x, (x2,), grad_outputs=torch.ones_like(grad_u_x), create_graph=True
    )
    historical_values = grad_u_t + 0.5 * sigma**2 * grad_u_xx
    assert torch.equal(heat_values, historical_values)

    time_part, diffusion_part = heat_operator_parts(u, x, t, sigma)
    assert torch.equal(time_part + diffusion_part, heat_values)
    assert torch.equal(time_part, grad_u_t)
    assert torch.equal(diffusion_part, 0.5 * sigma**2 * grad_u_xx)


def test_fields_independent_of_one_coordinate_give_zero_channels():
    coefficients = {2: 0.35, 1: -0.7, 0: 0.2}
    x, t = _collocation_batch()

    time_constant_field = torch.sin(x)  # no t dependence
    parts = constant_coefficient_operator_parts(
        time_constant_field, x, t, coefficients
    )
    assert torch.all(parts["velocity"] == 0.0)

    space_constant_field = torch.exp(-t)  # no x dependence
    parts = constant_coefficient_operator_parts(
        space_constant_field, x, t, coefficients
    )
    assert torch.all(parts["diffusion"] == 0.0)
    assert torch.all(parts["advection"] == 0.0)


def test_unsupported_differential_order_raises():
    x, t = _collocation_batch()
    u = _smooth_test_field(x, t)
    with pytest.raises(ValueError):
        constant_coefficient_operator(u, x, t, {4: -0.05, 1: 1.3})
    with pytest.raises(ValueError):
        constant_coefficient_operator_parts(u, x, t, {3: 1.0})
