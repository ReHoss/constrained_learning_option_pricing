"""Unit tests for the VPINN ENGD optimizer.

The critical correctness check (test 1) compares the *functional* weak
residual computation (composable with ``torch.func.jacrev``) against the
existing :class:`VPINNLoss` forward — they must agree to machine
precision because they implement the same Galerkin formula.

Test 2 cross-checks the per-row Jacobian rows of ``vpinn_jacobian``
against a manual ``torch.autograd.grad`` computation.

Test 3 is an end-to-end smoke test: a single ``VPINNENGDOptimizer.step()``
on a fixed batch must not increase the loss.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from learning_option_pricing.optimizers.natural_gradient_vpinn import (
    VPINNENGDOptimizer,
    _vpinn_residuals,
    vpinn_jacobian,
)
from learning_option_pricing.optimizers.natural_gradient import (
    flat_grad,
)
from learning_option_pricing.vpinn import VPINNLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class TinyMLP(nn.Module):
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
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture
def vpinn_loss():
    return VPINNLoss(
        sigma=0.25, r=0.02, q=0.0,
        x_max=1.5, K_test=6, n_quad=20,
        dtype=torch.float64,
    )


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TinyMLP(hidden=8)


# ===========================================================================
# Test 1 — _vpinn_residuals matches VPINNLoss forward (sum of squared residuals)
# ===========================================================================


def test_functional_residuals_match_classical_loss(vpinn_loss, model):
    """The functional residual vector squared and meaned must equal the
    scalar loss returned by the existing VPINNLoss.forward."""
    torch.manual_seed(1)
    tau_batch = torch.rand(4) * 1.0  # 4 time points

    # --- functional path ---
    params_dict = {k: v.detach().clone() for k, v in model.named_parameters()}
    R_flat = _vpinn_residuals(
        params_dict, model, tau_batch,
        vpinn_loss.x_nodes, vpinn_loss.phi_w, vpinn_loss.dphi_w,
        vpinn_loss.sigma, vpinn_loss.mu, vpinn_loss.r,
    )
    L_func = R_flat.pow(2).mean().item()

    # --- classical path ---
    L_cls = vpinn_loss(model, tau_batch).item()

    assert math.isclose(L_func, L_cls, rel_tol=1e-9, abs_tol=1e-12), (
        f"functional={L_func}  vs  VPINNLoss={L_cls}  (rel diff {abs(L_func-L_cls)/max(abs(L_cls), 1e-30):.3e})"
    )


# ===========================================================================
# Test 2 — Jacobian rows match a manual autograd computation
# ===========================================================================


def test_vpinn_jacobian_matches_autograd(vpinn_loss, model):
    """vpinn_jacobian(...) must produce the same rows as autograd applied to
    each scalar residual R_{i,k} individually."""
    torch.manual_seed(2)
    tau_batch = torch.tensor([0.1, 0.5, 0.9])
    K = vpinn_loss.phi_w.shape[0]
    n_meas = tau_batch.shape[0] * K
    n_params = sum(p.numel() for p in model.parameters())

    J_func = vpinn_jacobian(model, vpinn_loss, tau_batch)
    assert J_func.shape == (n_meas, n_params)

    # --- Reference path: classical autograd on individual residuals ---
    # Recompute R[i, k] using the existing VPINNLoss machinery and
    # backprop on each scalar residual.
    N_q = vpinn_loss.x_nodes.shape[0]
    N_tau = tau_batch.shape[0]
    tau_rep = tau_batch.unsqueeze(1).expand(N_tau, N_q).reshape(-1, 1)
    x_rep = vpinn_loss.x_nodes.unsqueeze(0).expand(N_tau, N_q).reshape(-1, 1)
    tau_rep = tau_rep.detach().requires_grad_(True)
    x_rep = x_rep.detach().requires_grad_(True)
    u = model(torch.cat([tau_rep, x_rep], dim=1))
    du_dtau, du_dx = torch.autograd.grad(
        u, [tau_rep, x_rep], grad_outputs=torch.ones_like(u), create_graph=True
    )
    u_vals = u.squeeze(1).reshape(N_tau, N_q)
    u_tau_ = du_dtau.squeeze(1).reshape(N_tau, N_q)
    u_x_ = du_dx.squeeze(1).reshape(N_tau, N_q)
    f_phi = u_tau_ - vpinn_loss.mu * u_x_ + vpinn_loss.r * u_vals
    f_dphi = (vpinn_loss.sigma**2 / 2.0) * u_x_
    R = f_phi @ vpinn_loss.phi_w.T + f_dphi @ vpinn_loss.dphi_w.T  # (N_tau, K)

    # Spot-check 4 rows of the Jacobian
    indices = [(0, 0), (0, K - 1), (1, 2), (N_tau - 1, K // 2)]
    for (i, k) in indices:
        # Recompute the graph for each row (autograd consumes it)
        tau_rep = tau_batch.unsqueeze(1).expand(N_tau, N_q).reshape(-1, 1)
        x_rep = vpinn_loss.x_nodes.unsqueeze(0).expand(N_tau, N_q).reshape(-1, 1)
        tau_rep = tau_rep.detach().requires_grad_(True)
        x_rep = x_rep.detach().requires_grad_(True)
        u = model(torch.cat([tau_rep, x_rep], dim=1))
        du_dtau, du_dx = torch.autograd.grad(
            u, [tau_rep, x_rep], grad_outputs=torch.ones_like(u), create_graph=True
        )
        u_vals = u.squeeze(1).reshape(N_tau, N_q)
        u_tau_ = du_dtau.squeeze(1).reshape(N_tau, N_q)
        u_x_ = du_dx.squeeze(1).reshape(N_tau, N_q)
        f_phi = u_tau_ - vpinn_loss.mu * u_x_ + vpinn_loss.r * u_vals
        f_dphi = (vpinn_loss.sigma**2 / 2.0) * u_x_
        R_ = f_phi @ vpinn_loss.phi_w.T + f_dphi @ vpinn_loss.dphi_w.T

        R_ik = R_[i, k]
        grads = torch.autograd.grad(R_ik, list(model.parameters()))
        ref_row = torch.cat([g.flatten() for g in grads])
        idx_flat = i * K + k
        diff = (J_func[idx_flat] - ref_row).norm() / (ref_row.norm() + 1e-30)
        assert diff < 1e-9, (
            f"Row ({i},{k}) [flat {idx_flat}] mismatch: rel diff = {diff:.3e}"
        )


# ===========================================================================
# Test 3 — VPINN ENGD step does not increase loss on fixed batch
# ===========================================================================


def test_vpinn_engd_step_does_not_increase_loss(vpinn_loss, model):
    """A single VPINN ENGD step on a fixed batch must not increase the loss."""
    torch.manual_seed(3)
    tau_batch = torch.linspace(0.05, 0.95, 8)

    def loss():
        return vpinn_loss(model, tau_batch)

    L0 = loss().item()
    model.zero_grad()
    loss().backward()
    g = flat_grad(model)

    engd = VPINNENGDOptimizer(
        model, vpinn_loss, reg=1e-3, cg_iters=80, ls_steps=25,
    )
    info = engd.step(g, tau_batch, loss)
    L1 = loss().item()

    assert L1 <= L0 + 1e-9, f"Loss increased: {L0} -> {L1}"
    assert info["cg_residual_norm"] < 1.0, info
    assert math.isfinite(info["step_size"])
