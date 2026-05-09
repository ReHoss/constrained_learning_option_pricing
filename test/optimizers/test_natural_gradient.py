"""Unit tests for the ENGD optimizer (Empirical Natural Gradient Descent).

These tests validate the *correctness* of each building block:

1. ``test_bsm_scalar_matches_classical_operator``
       The functional ``_bsm_scalar`` (uses ``torch.func.grad``) must produce
       the same value as the existing ``bsm_operator`` (uses
       ``torch.autograd.grad``) for the same input.

2. ``test_jacobian_matches_autograd``
       ``compute_jacobians`` must produce the same Jacobian rows as a manual
       per-sample autograd computation.

3. ``test_cg_solves_linear_system``
       ``solve_cg`` must produce the same solution as
       ``torch.linalg.solve(G, g)`` on a small explicit Gram matrix.

4. ``test_line_search_picks_best_step``
       ``grid_line_search`` returns the step that minimises the loss on a
       handcrafted quadratic landscape.

5. ``test_engd_step_decreases_loss_on_fixed_batch``
       Smoke test: a single ENGD step on a fixed collocation batch must not
       *increase* the loss (line search guarantees non-increase).

Run with::

    pytest -xvs test/optimizers/test_natural_gradient.py
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from learning_option_pricing.optimizers.natural_gradient import (
    ENGDOptimizer,
    _bsm_scalar,
    _gram_matvec,
    compute_jacobians,
    flat_grad,
    flat_params,
    grid_line_search,
    set_flat_params,
    solve_cg,
)
from learning_option_pricing.pricing.terminal import bsm_operator, payoff_put


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
    """Minimal 2-input/1-output MLP — small parameter count for fast tests."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        return self.fc3(h)


@pytest.fixture(autouse=True)
def _double_precision():
    """Run all tests in float64 — matrix operations need it for tight tols."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyMLP(hidden=6)


@pytest.fixture
def bsm_params():
    return dict(r=0.02, q=0.0, sigma=0.25)


# ===========================================================================
# Test 1 — Functional BSM residual matches the classical bsm_operator
# ===========================================================================


def test_bsm_scalar_matches_classical_operator(model, bsm_params):
    """The functional residual must agree with the autograd-based one."""
    s_val, t_val = 95.0, 0.3

    # --- functional path (used by jacrev) ---
    params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}
    s = torch.tensor(s_val)
    t = torch.tensor(t_val)
    F_func = _bsm_scalar(params_dict, model, s, t, **bsm_params).item()

    # --- classical path (used by the loss) ---
    s_g = torch.tensor([s_val], requires_grad=True)
    t_g = torch.tensor([t_val], requires_grad=True)
    x = torch.stack([s_g, t_g], dim=1)
    V = model(x).squeeze()
    F_cls = bsm_operator(V, s_g, t_g, **bsm_params).item()

    assert math.isclose(F_func, F_cls, rel_tol=1e-9, abs_tol=1e-12), (
        f"_bsm_scalar={F_func}  vs  bsm_operator={F_cls}"
    )


# ===========================================================================
# Test 2 — Per-sample Jacobians match a manual autograd computation
# ===========================================================================


def test_jacobian_matches_autograd(model, bsm_params):
    """`compute_jacobians` ≡ per-point autograd of the BSM residual."""
    torch.manual_seed(1)
    n = 5
    s_pts = torch.tensor([60.0, 80.0, 100.0, 120.0, 140.0])
    t_pts = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    s_tc = torch.tensor([70.0, 110.0])
    t_tc = torch.tensor([1.0, 1.0])

    J_F, J_TC = compute_jacobians(model, s_pts, t_pts, s_tc, t_tc, **bsm_params)
    n_params = sum(p.numel() for p in model.parameters())
    assert J_F.shape == (n, n_params)
    assert J_TC.shape == (2, n_params)

    # --- Manual reference: differentiate one point at a time ---
    for i in range(n):
        # Reset params (we only differentiate, so .data unchanged)
        s_g = s_pts[i].clone().requires_grad_(True)
        t_g = t_pts[i].clone().requires_grad_(True)
        x = torch.stack([s_g, t_g]).unsqueeze(0)
        V = model(x).squeeze()
        F = bsm_operator(V, s_g, t_g, **bsm_params)

        # Gradient of the scalar F w.r.t. all params
        grads = torch.autograd.grad(F, list(model.parameters()), create_graph=False)
        ref = torch.cat([g.flatten() for g in grads])

        diff = (J_F[i] - ref).norm() / (ref.norm() + 1e-30)
        assert diff < 1e-9, (
            f"Row {i} mismatch: rel diff = {diff:.3e}\n"
            f"  J_F norm = {J_F[i].norm().item():.3e}\n"
            f"  ref norm = {ref.norm().item():.3e}"
        )

    # --- Same check for terminal Jacobian rows ---
    for j in range(2):
        x = torch.stack([s_tc[j], t_tc[j]]).unsqueeze(0)
        V = model(x).squeeze()
        grads = torch.autograd.grad(V, list(model.parameters()), create_graph=False)
        ref = torch.cat([g.flatten() for g in grads])

        diff = (J_TC[j] - ref).norm() / (ref.norm() + 1e-30)
        assert diff < 1e-9, f"TC row {j}: rel diff = {diff:.3e}"


# ===========================================================================
# Test 3 — CG matches the direct linear solver on a known Gram matrix
# ===========================================================================


def test_cg_solves_linear_system():
    """CG must solve Gx = g for an SPD G, matching a direct solve."""
    torch.manual_seed(2)
    n_params = 30
    n_gram = 12

    J_F = torch.randn(n_gram, n_params)
    J_TC = torch.randn(n_gram // 2, n_params)
    g = torch.randn(n_params)

    lam_f, lam_tc, reg = 2.5, 1.0, 1e-3

    # --- Solve via direct linear algebra ---
    G = (
        (lam_f / J_F.shape[0]) * (J_F.T @ J_F)
        + (lam_tc / J_TC.shape[0]) * (J_TC.T @ J_TC)
        + reg * torch.eye(n_params)
    )
    delta_direct = torch.linalg.solve(G, g)

    # --- Solve via CG ---
    delta_cg = solve_cg(g, J_F, J_TC, lam_f, lam_tc, reg, n_iters=200, tol=1e-14)

    rel_err = (delta_direct - delta_cg).norm() / delta_direct.norm()
    assert rel_err < 1e-8, f"CG vs direct: rel error = {rel_err:.3e}"


def test_cg_converges_in_at_most_n_steps():
    """CG on an SPD system converges in at most n steps in exact arithmetic."""
    torch.manual_seed(3)
    n_params = 15
    J_F = torch.randn(20, n_params)
    J_TC = torch.zeros(0, n_params)  # disable terminal contribution
    g = torch.randn(n_params)
    reg = 1e-6

    delta_full = solve_cg(g, J_F, J_TC, 1.0, 0.0, reg, n_iters=n_params + 5, tol=0.0)

    # Reference solve
    G = (1.0 / 20) * (J_F.T @ J_F) + reg * torch.eye(n_params)
    delta_direct = torch.linalg.solve(G, g)
    rel_err = (delta_full - delta_direct).norm() / delta_direct.norm()
    assert rel_err < 1e-6, f"After n+5 CG steps, rel err = {rel_err:.3e}"


def test_cg_handles_zero_terminal_jacobian():
    """When N_TC = 0 (J_TC has zero rows), the CG must still work
    (e.g. for ETCNN where the terminal Gram contribution vanishes)."""
    torch.manual_seed(4)
    n_params = 20
    J_F = torch.randn(15, n_params)
    J_TC = torch.zeros(0, n_params)
    g = torch.randn(n_params)

    # _gram_matvec divides by J_TC.shape[0] = 0 — must handle gracefully.
    # We test with lam_tc = 0 to bypass.
    delta = solve_cg(g, J_F, J_TC, lam_f=1.0, lam_tc=0.0, reg=1e-3, n_iters=50)
    assert torch.isfinite(delta).all()


# ===========================================================================
# Test 4 — Grid line search finds the (approximate) minimum
# ===========================================================================


def test_line_search_picks_best_step():
    """On a quadratic loss along the search direction, grid line search
    must select a step close to the true minimiser."""
    torch.manual_seed(5)

    class Linear1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([2.0]))  # initial value

        def forward(self, x):
            return self.w * x

    model = Linear1D()
    direction = torch.tensor([1.0])  # nat_grad = 1

    # Loss = (w - 1.5)^2 → minimum at w=1.5 → optimal alpha = 2.0 - 1.5 = 0.5
    def loss_fn():
        return (model.w - 1.5).pow(2).sum()

    # The grid is alpha_0 * 0.5^k. With alpha_0=1, the candidates are
    # 1, 0.5, 0.25, ..., so 0.5 is in the grid exactly.
    alpha = grid_line_search(model, loss_fn, direction, n_steps=15, step_max=1.0)
    assert math.isclose(alpha, 0.5, rel_tol=1e-9), f"Expected 0.5, got {alpha}"


def test_line_search_restores_params_on_exit():
    """grid_line_search must leave the model parameters unchanged on return."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([7.0, -3.0]))

        def forward(self):
            return self.w

    m = Tiny()
    init = m.w.detach().clone()
    direction = torch.ones_like(init)
    grid_line_search(m, lambda: m.w.pow(2).sum(), direction, n_steps=10)
    assert torch.allclose(m.w.detach(), init), "Parameters were not restored"


# ===========================================================================
# Test 5 — End-to-end smoke test of ENGDOptimizer.step()
# ===========================================================================


def test_engd_step_decreases_loss_on_fixed_batch(model, bsm_params):
    """A single ENGD step on a *fixed* batch must not increase the loss
    (the line search guarantees non-increase, modulo numerical noise)."""
    K = 100.0
    torch.manual_seed(6)

    s_f = (torch.rand(40) * 80 + 30).requires_grad_(True)
    t_f = (torch.rand(40) * 0.95).requires_grad_(True)
    s_tc = torch.rand(20) * 80 + 30
    t_tc = torch.full((20,), 1.0)
    phi = payoff_put(s_tc, K)

    def loss(model_):
        s = s_f.detach().clone().requires_grad_(True)
        t = t_f.detach().clone().requires_grad_(True)
        V = model_(torch.stack([s, t], dim=1)).squeeze()
        F = bsm_operator(V, s, t, **bsm_params)
        V_tc = model_(torch.stack([s_tc, t_tc], dim=1)).squeeze()
        return 5.0 * F.pow(2).mean() + 1.0 * (V_tc - phi).pow(2).mean()

    L0 = loss(model).item()

    engd = ENGDOptimizer(
        model, **bsm_params, lam_f=5.0, lam_tc=1.0, reg=1e-4,
        cg_iters=40, ls_steps=25,
    )

    model.zero_grad()
    loss(model).backward()
    g = flat_grad(model)

    info = engd.step(
        g,
        s_f.detach()[:20],
        t_f.detach()[:20],
        s_tc[:10],
        t_tc[:10],
        lambda: loss(model),
    )

    L1 = loss(model).item()
    assert L1 <= L0 + 1e-9, f"Loss increased after ENGD step: {L0} -> {L1}"
    # Convergence diagnostic must be small (a healthy CG)
    assert info["cg_residual_norm"] < 1.0, info


def test_helpers_round_trip(model):
    """flat_params / set_flat_params must be inverses."""
    flat0 = flat_params(model).clone()
    flat_perturbed = flat0 + 0.5
    set_flat_params(model, flat_perturbed)
    flat1 = flat_params(model)
    assert torch.allclose(flat1, flat_perturbed)
    set_flat_params(model, flat0)
    flat2 = flat_params(model)
    assert torch.allclose(flat2, flat0)


def test_flat_grad_matches_concatenated_grads(model, bsm_params):
    """flat_grad(model) must equal torch.cat([p.grad.flatten() ...])."""
    x = torch.tensor([[100.0, 0.5]], requires_grad=False)
    L = model(x).sum()
    L.backward()
    expected = torch.cat([p.grad.flatten() for p in model.parameters()])
    got = flat_grad(model)
    assert torch.allclose(got, expected)
