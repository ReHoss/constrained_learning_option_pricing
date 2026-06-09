"""Tests for the backward heat operator and its closed-form references.

The defining property checked here is that each analytical reference solution
annihilates the heat operator, ``P u = d_t u + (sigma^2/2) d_xx u = 0``, on the
interior, and recovers its terminal datum at ``t = T``.
"""
from __future__ import annotations

import math

import torch

from learning_option_pricing.pde import (
    heat_call_exact,
    heat_call_payoff,
    heat_operator,
    heat_sine_exact,
    heat_sine_terminal,
    heat_theta3_exact,
    heat_theta3_terminal,
    smooth_call_payoff,
    smooth_call_payoff_cm_time,
)


def _residual_of_exact(exact_fn, x_lo, x_hi, T, sigma, *, n=200):
    """Return the max |P u| of an exact solution sampled on the interior."""
    x = torch.linspace(x_lo, x_hi, n, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.0, 0.9 * T, n, dtype=torch.float64, requires_grad=True)
    u = exact_fn(x, t)
    residual = heat_operator(u, x, t, sigma)
    return residual.abs().max().item()


def test_sine_exact_solves_heat_equation():
    T, sigma, c, f = 0.2, 1.0, 0.5, 4.0

    def exact(x, t):
        return heat_sine_exact(x, t, T=T, sigma=sigma, c=c, f=f)

    assert _residual_of_exact(exact, 0.0, 1.0, T, sigma) < 1e-8


def test_sine_terminal_recovered():
    T, sigma, c, f = 0.2, 1.0, 0.5, 4.0
    x = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
    t = torch.full_like(x, T)
    got = heat_sine_exact(x, t, T=T, sigma=sigma, c=c, f=f)
    want = heat_sine_terminal(x, c=c, f=f)
    assert torch.allclose(got, want, atol=1e-10)


def test_sine_homogeneous_dirichlet():
    # Integer f keeps u(t, 0) = u(t, 1) = 0 for all t.
    T, sigma, c, f = 0.2, 1.0, 0.5, 4.0
    t = torch.linspace(0.0, T, 20, dtype=torch.float64)
    for x_b in (0.0, 1.0):
        xb = torch.full_like(t, x_b)
        u_b = heat_sine_exact(xb, t, T=T, sigma=sigma, c=c, f=f)
        assert u_b.abs().max().item() < 1e-10


def test_theta3_exact_solves_heat_equation():
    sigma = 1.0
    T = 0.15  # below 2/(sigma^2 pi^2) ~ 0.2026

    def exact(x, t):
        return heat_theta3_exact(x, t, T=T, sigma=sigma)

    assert _residual_of_exact(exact, -1.0, 1.0, T, sigma) < 1e-7


def test_theta3_terminal_recovered():
    sigma, T = 1.0, 0.15
    x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64)
    t = torch.full_like(x, T)
    got = heat_theta3_exact(x, t, T=T, sigma=sigma)
    want = heat_theta3_terminal(x)
    assert torch.allclose(got, want, atol=1e-10)


def test_theta3_homogeneous_neumann():
    # Cosine series => d_x u vanishes at x = +-1 for all t.
    sigma, T = 1.0, 0.15
    t = torch.linspace(0.0, T, 10, dtype=torch.float64)
    for x_b in (-1.0, 1.0):
        xb = torch.full_like(t, x_b).requires_grad_(True)
        tb = t.clone().requires_grad_(True)
        u_b = heat_theta3_exact(xb, tb, T=T, sigma=sigma)
        (u_x,) = torch.autograd.grad(u_b.sum(), xb)
        assert u_x.abs().max().item() < 1e-8


def test_call_exact_solves_heat_equation():
    K, T, sigma = 100.0, 1.0, 0.25

    def exact(x, t):
        return heat_call_exact(x, t, K=K, T=T, sigma=sigma)

    assert _residual_of_exact(exact, math.log(60.0), math.log(160.0), T, sigma) < 1e-6


def test_call_exact_recovers_payoff_near_terminal():
    K, T, sigma = 100.0, 1.0, 0.25
    x = torch.linspace(math.log(60.0), math.log(160.0), 80, dtype=torch.float64)
    t = torch.full_like(x, T - 1e-6)
    got = heat_call_exact(x, t, K=K, T=T, sigma=sigma)
    want = heat_call_payoff(x, K)
    assert torch.allclose(got, want, atol=1e-2)


def test_smooth_call_converges_to_payoff():
    K = 100.0
    x = torch.linspace(math.log(60.0), math.log(160.0), 80, dtype=torch.float64)
    payoff = heat_call_payoff(x, K)
    err_coarse = (smooth_call_payoff(x, K, beta=10.0) - payoff).abs().max()
    err_fine = (smooth_call_payoff(x, K, beta=1000.0) - payoff).abs().max()
    assert err_fine < err_coarse


def test_cm_time_smoothing_exact_at_terminal():
    # Vanishing bandwidth at t=T => exact payoff (no terminal bias).
    K, T, eps0 = 100.0, 1.0, 10.0
    x = torch.linspace(math.log(60.0), math.log(160.0), 80, dtype=torch.float64)
    tT = torch.full_like(x, T)
    got = smooth_call_payoff_cm_time(x, tT, K=K, T=T, eps0=eps0)
    assert torch.allclose(got, heat_call_payoff(x, K), atol=1e-10)


def test_cm_time_smoothing_smooth_interior():
    # For t<T the kink at x=ln K is rounded: value strictly above the payoff.
    K, T, eps0 = 100.0, 1.0, 10.0
    x0 = torch.tensor([math.log(K)], dtype=torch.float64)
    for t_val in (0.0, 0.5):
        t = torch.full_like(x0, t_val)
        val = smooth_call_payoff_cm_time(x0, t, K=K, T=T, eps0=eps0)
        eps = eps0 * (T - t_val) / T
        # at the kink (e^x = K): value = eps/2 > 0 = payoff
        assert torch.allclose(val, torch.tensor([eps / 2], dtype=torch.float64), atol=1e-8)
        assert val.item() > 0.0
