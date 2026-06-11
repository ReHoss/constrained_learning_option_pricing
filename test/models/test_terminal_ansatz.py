"""Tests for the four-form blended terminal ansatz.

Verified properties:
    - blending lambda(T) = 1 for both kinds, so the network prefactor vanishes;
    - hard forms recover the terminal datum at t = T regardless of the network;
    - soft / pure forms equal the bare network;
    - the residual decomposition identity P u_hat = R_theta + P Psi holds.
"""
from __future__ import annotations

import math

import torch

from learning_option_pricing.models.blended_ansatz import (
    BlendedTerminalAnsatz,
    make_blending,
    residual_decomposition,
)
from learning_option_pricing.models.resnet import ResNet


def _build(form, *, T=0.2, sigma=1.0, blending_kind="linear"):
    torch.manual_seed(0)
    net = ResNet(d_in=2, d_out=1, n=16, M=2, L=2).double()

    def terminal_datum(x):
        return torch.sin(math.pi * x)

    blending = make_blending(blending_kind, T=T, sigma=sigma)
    return BlendedTerminalAnsatz(
        net, terminal_datum, blending, form=form
    )


def test_blending_terminal_value_is_one():
    T = 0.2
    t_T = torch.tensor([T], dtype=torch.float64)
    for kind in ("linear", "exponential"):
        lam = make_blending(kind, T=T, sigma=1.0)
        assert torch.allclose(lam(t_T), torch.ones_like(t_T), atol=1e-12)


def test_hard_forms_recover_terminal_datum():
    T = 0.2
    x = torch.linspace(0.0, 1.0, 40, dtype=torch.float64).unsqueeze(-1)
    t = torch.full_like(x, T)
    xt = torch.cat([x, t], dim=1)
    want = torch.sin(math.pi * x)
    for form in ("hard_constant", "hard_blended"):
        ansatz = _build(form, T=T)
        got = ansatz(xt)
        assert torch.allclose(got, want, atol=1e-10), form


def test_soft_forms_equal_bare_network():
    T = 0.2
    x = torch.linspace(0.0, 1.0, 20, dtype=torch.float64).unsqueeze(-1)
    t = torch.linspace(0.0, T, 20, dtype=torch.float64).unsqueeze(-1)
    xt = torch.cat([x, t], dim=1)
    for form in ("soft_pinn", "pure_nn"):
        ansatz = _build(form, T=T)
        assert torch.allclose(ansatz(xt), ansatz.free_network(xt), atol=1e-12), form


def test_residual_decomposition_identity():
    # P u_hat (full forward, by autograd) must equal R_theta + P Psi
    # (the decomposition) pointwise, for both hard forms and both blendings.
    from learning_option_pricing.pde.operators import heat_operator

    T, sigma = 0.2, 1.0
    for form in ("hard_constant", "hard_blended"):
        for kind in ("linear", "exponential"):
            ansatz = _build(form, T=T, sigma=sigma, blending_kind=kind)
            x = torch.linspace(0.05, 0.95, 60, dtype=torch.float64, requires_grad=True)
            t = torch.linspace(0.01, 0.99 * T, 60, dtype=torch.float64, requires_grad=True)
            xt = torch.stack([x, t], dim=1)
            u = ansatz(xt).squeeze(-1)
            full_residual = heat_operator(u, x, t, sigma)

            decomp = residual_decomposition(ansatz, x, t, sigma)
            assert torch.allclose(
                full_residual, decomp["residual"], atol=1e-8
            ), (form, kind)
            assert torch.allclose(
                decomp["residual"],
                decomp["network_contribution"] + decomp["extension_forcing"],
                atol=1e-10,
            ), (form, kind)


def test_soft_decomposition_has_zero_forcing():
    T, sigma = 0.2, 1.0
    ansatz = _build("soft_pinn", T=T, sigma=sigma)
    x = torch.linspace(0.05, 0.95, 30, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.01, 0.99 * T, 30, dtype=torch.float64, requires_grad=True)
    decomp = residual_decomposition(ansatz, x, t, sigma)
    assert decomp["forcing_floor"].item() == 0.0
    assert torch.allclose(decomp["residual"], decomp["network_contribution"])
