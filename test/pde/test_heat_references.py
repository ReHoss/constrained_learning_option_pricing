"""Tests for the backward heat operator and its closed-form references.

The defining property checked here is that each analytical reference solution
annihilates the heat operator, ``P u = d_t u + (sigma^2/2) d_xx u = 0``, on the
interior, and recovers its terminal datum at ``t = T``.
"""
from __future__ import annotations

import math

import torch

from learning_option_pricing.pde import (
    chen_mangasarian_max,
    heat_call_exact,
    heat_call_payoff,
    heat_operator,
    heat_operator_parts,
    heat_propagate,
    heat_put_exact,
    heat_put_payoff,
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


def test_heat_operator_parts_sum_and_values():
    # u = sin(pi x) * exp(a t): d_t u = a u, (sigma^2/2) d_xx u = -(sigma^2/2) pi^2 u.
    sigma, a = 0.7, 1.3
    x = torch.linspace(0.1, 0.9, 50, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.0, 1.0, 50, dtype=torch.float64, requires_grad=True)
    u = torch.sin(math.pi * x) * torch.exp(a * t)
    time_part, diff_part = heat_operator_parts(u, x, t, sigma)
    # parts sum to the full operator
    assert torch.allclose(time_part + diff_part, heat_operator(u, x, t, sigma), atol=1e-10)
    # analytic values
    assert torch.allclose(time_part, a * u, atol=1e-8)
    assert torch.allclose(diff_part, -0.5 * sigma**2 * math.pi**2 * u, atol=1e-8)


def test_put_exact_solves_heat_equation():
    K, T, sigma = 100.0, 1.0, 0.25

    def exact(x, t):
        return heat_put_exact(x, t, K=K, T=T, sigma=sigma)

    assert _residual_of_exact(exact, math.log(40.0), math.log(200.0), T, sigma) < 1e-6


def test_put_exact_recovers_payoff_near_terminal():
    K, T, sigma = 100.0, 1.0, 0.25
    x = torch.linspace(math.log(40.0), math.log(200.0), 80, dtype=torch.float64)
    t = torch.full_like(x, T - 1e-6)
    assert torch.allclose(heat_put_exact(x, t, K=K, T=T, sigma=sigma),
                          heat_put_payoff(x, K), atol=1e-2)


def test_heat_propagate_matches_closed_form_put():
    # numerical Gaussian convolution of the put payoff == analytic European put
    K, T, sigma = 100.0, 1.0, 0.25

    def payoff(y):
        return heat_put_payoff(y, K)

    x = torch.linspace(math.log(60.0), math.log(140.0), 50, dtype=torch.float64)
    t = torch.full_like(x, 0.5)
    num = heat_propagate(payoff, x, t, t_terminal=T, sigma=sigma,
                         y_lo=math.log(5.0), y_hi=math.log(600.0), n_quad=4000)
    cf = heat_put_exact(x, t, K=K, T=T, sigma=sigma)
    assert (num - cf).norm() / cf.norm() < 1e-4


def test_heat_propagate_returns_g_at_terminal():
    # at tau -> 0 the convolution must return g(x) exactly (no delta-kernel garbage)
    K, sigma, T = 100.0, 0.25, 0.5

    def g(y):
        return heat_put_payoff(y, K)

    x = torch.linspace(math.log(40.0), math.log(160.0), 60, dtype=torch.float64)
    t = torch.full_like(x, T)  # tau = 0
    assert torch.allclose(
        heat_propagate(g, x, t, t_terminal=T, sigma=sigma,
                       y_lo=math.log(5.0), y_hi=math.log(600.0)),
        g(x), atol=1e-10)


def test_heat_propagate_evaluates_datum_once_for_a_full_interval():
    # Regression guard against an O(2^m) blow-up in the chained Bermudan reference.
    # For a full inter-exercise interval (tau above the grid-resolvable floor) the
    # datum must be evaluated exactly once, on the quadrature grid.  The earlier
    # torch.where(raw_tau < floor, g(x), conv) form evaluated g(x) eagerly as well;
    # since the datum is the recursive value at the stage above, that second call
    # re-descends the whole chain at every level, doubling the work per level and
    # holding an O(n_quad^2) kernel live at every depth.  Counting the calls pins
    # the single evaluation and hence the linear-in-m cost.
    K, sigma, T = 100.0, 0.25, 1.0
    calls = {"n": 0}

    def counting_datum(y):
        calls["n"] += 1
        return heat_put_payoff(y, K)

    x = torch.linspace(math.log(60.0), math.log(140.0), 50, dtype=torch.float64)
    t = torch.full_like(x, 0.5)  # tau = 0.5, well above the grid floor
    heat_propagate(counting_datum, x, t, t_terminal=T, sigma=sigma,
                   y_lo=math.log(5.0), y_hi=math.log(600.0), n_quad=4000)
    assert calls["n"] == 1


def test_chen_mangasarian_max_converges():
    a = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float64)
    b = torch.tensor([4.0, 2.0, 3.0], dtype=torch.float64)
    coarse = (chen_mangasarian_max(a, b, 1.0) - torch.maximum(a, b)).abs().max()
    fine = (chen_mangasarian_max(a, b, 0.01) - torch.maximum(a, b)).abs().max()
    assert fine < coarse and fine < 1e-2


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


# ---------------------------------------------------------------------------
# Exact Bermudan-put backward induction (multi-stage reference)
# ---------------------------------------------------------------------------

def _berm(x, t, exercise_times, K=100.0, sigma=0.25):
    from learning_option_pricing.pde import bermudan_put_value_exact
    import math
    return bermudan_put_value_exact(
        x, t, exercise_times=exercise_times, K=K, sigma=sigma,
        y_lo=math.log(5.0), y_hi=math.log(600.0), n_quad=4000)


def test_bermudan_m2_continuation_at_t1_equals_european():
    # For two exercise dates {t1, T}, the continuation at t1 is the European put,
    # so V(t1) = max(payoff, European(t1)).  The convolution must reproduce the
    # closed-form European value to grid accuracy.
    K, sigma, t1, T = 100.0, 0.25, 0.5, 1.0
    x = torch.linspace(math.log(50.0), math.log(150.0), 80, dtype=torch.float64)
    got = _berm(x, torch.full_like(x, t1), [t1, T], K=K, sigma=sigma)
    european = heat_put_exact(x, torch.full_like(x, t1), K=K, T=T, sigma=sigma)
    want = torch.maximum(heat_put_payoff(x, K), european)
    assert torch.allclose(got, want, atol=2e-3), (got - want).abs().max().item()


def test_bermudan_value_at_maturity_is_payoff():
    K, sigma, t1, T = 100.0, 0.25, 0.5, 1.0
    x = torch.linspace(math.log(50.0), math.log(150.0), 60, dtype=torch.float64)
    got = _berm(x, torch.full_like(x, T), [t1, T], K=K, sigma=sigma)
    assert torch.allclose(got, heat_put_payoff(x, K), atol=1e-10)


def test_bermudan_dominates_european():
    # Early-exercise premium is non-negative: Bermudan >= European at inception.
    K, sigma, t1, T = 100.0, 0.25, 0.5, 1.0
    x = torch.linspace(math.log(50.0), math.log(150.0), 60, dtype=torch.float64)
    berm = _berm(x, torch.zeros_like(x), [t1, T], K=K, sigma=sigma)
    european = heat_put_exact(x, torch.zeros_like(x), K=K, T=T, sigma=sigma)
    assert (berm - european).min().item() > -2e-3  # >= 0 up to grid tol


def test_bermudan_value_monotone_in_exercise_dates():
    # More exercise opportunities cannot decrease the value: m=3 >= m=2 at t=0.
    K, sigma, T = 100.0, 0.25, 1.0
    x = torch.linspace(math.log(50.0), math.log(150.0), 50, dtype=torch.float64)
    v2 = _berm(x, torch.zeros_like(x), [0.5, T], K=K, sigma=sigma)
    v3 = _berm(x, torch.zeros_like(x), [1.0 / 3, 2.0 / 3, T], K=K, sigma=sigma)
    assert (v3 - v2).min().item() > -2e-3


def test_bermudan_intermediate_date_reduces_to_european_continuation():
    # Regression test for the late-binding closure defect: at the LAST
    # intermediate date t_2 of a three-date chain the continuation is the
    # European put over [t_2, T], so
    #     V(t_2, .) = max(payoff, European(t_2, .)).
    # Before the fix, the stage callable propagated the payoff over the elapsed
    # time T - t_1 instead of T - t_2 (the loop variable was captured by
    # reference), which fails this identity by O(1e-1).
    K, sigma, T = 100.0, 0.25, 1.0
    t1, t2 = 1.0 / 3, 2.0 / 3
    x = torch.linspace(math.log(50.0), math.log(150.0), 60, dtype=torch.float64)
    got = _berm(x, torch.full_like(x, t2), [t1, t2, T], K=K, sigma=sigma)
    european = heat_put_exact(x, torch.full_like(x, t2), K=K, T=T, sigma=sigma)
    want = torch.maximum(heat_put_payoff(x, K), european)
    assert torch.allclose(got, want, atol=2e-3), (got - want).abs().max().item()


def test_bermudan_dynamic_program_tower_property():
    # DP self-consistency on a four-date chain: the value at t_1 with the full
    # remaining date list must equal the max-glued one-interval propagation of
    # the value at t_2 with its own remaining list,
    #     V(t_1, .) = max(payoff, S_{t_2-t_1} V(t_2, .)).
    # This is the identity the induction relies on stage by stage.
    K, sigma, T = 100.0, 0.25, 1.0
    t1, t2, t3 = 0.25, 0.5, 0.75
    x = torch.linspace(math.log(60.0), math.log(140.0), 40, dtype=torch.float64)
    lhs = _berm(x, torch.full_like(x, t1), [t1, t2, t3, T], K=K, sigma=sigma)

    def v_t2(y):
        return _berm(y, torch.full_like(y, t2), [t2, t3, T], K=K, sigma=sigma)

    cont = heat_propagate(v_t2, x, torch.full_like(x, t1), t_terminal=t2,
                          sigma=sigma, y_lo=math.log(5.0), y_hi=math.log(600.0),
                          n_quad=1500)
    rhs = torch.maximum(heat_put_payoff(x, K), cont)
    assert torch.allclose(lhs, rhs, atol=5e-3), (lhs - rhs).abs().max().item()
