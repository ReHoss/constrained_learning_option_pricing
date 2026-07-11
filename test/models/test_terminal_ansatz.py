"""Tests for the four-form terminal ansatz.

Verified properties:
    - interpolation coefficient lambda(T) = 1 for both kinds, so the network prefactor vanishes;
    - hard forms recover the terminal datum at t = T regardless of the network;
    - soft / pure forms equal the bare network;
    - the residual decomposition identity P u_hat = R_theta + P Psi holds.
"""
from __future__ import annotations

import math

import torch

from learning_option_pricing.models.terminal_ansatz import (
    TerminalAnsatz,
    make_interpolation_coefficient,
    residual_decomposition,
)
from learning_option_pricing.models.resnet import ResNet


def _build(form, *, T=0.2, sigma=1.0, interp_kind="linear"):
    torch.manual_seed(0)
    net = ResNet(d_in=2, d_out=1, n=16, M=2, L=2).double()

    def terminal_datum(x):
        return torch.sin(math.pi * x)

    interp_coeff = make_interpolation_coefficient(interp_kind, T=T, sigma=sigma)
    return TerminalAnsatz(
        net, terminal_datum, interp_coeff, form=form
    )


def test_interpolation_coefficient_terminal_value_is_one():
    T = 0.2
    t_T = torch.tensor([T], dtype=torch.float64)
    for kind in ("linear", "exponential"):
        lam = make_interpolation_coefficient(kind, T=T, sigma=1.0)
        assert torch.allclose(lam(t_T), torch.ones_like(t_T), atol=1e-12)


def test_hard_forms_recover_terminal_datum():
    T = 0.2
    x = torch.linspace(0.0, 1.0, 40, dtype=torch.float64).unsqueeze(-1)
    t = torch.full_like(x, T)
    xt = torch.cat([x, t], dim=1)
    want = torch.sin(math.pi * x)
    for form in ("hard_constant", "hard_convex"):
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
    # (the decomposition) pointwise, for both hard forms and both interpolation coefficients.
    from learning_option_pricing.pde.operators import heat_operator

    T, sigma = 0.2, 1.0
    for form in ("hard_constant", "hard_convex"):
        for kind in ("linear", "exponential"):
            ansatz = _build(form, T=T, sigma=sigma, interp_kind=kind)
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


# ---------------------------------------------------------------------------
# Stage-2 additions: generator coefficients, analytic-derivative bypass,
# startup cross-check (specification Section 1.4 items 3 and 5)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402  (kept close to the stage-2 test block)

from learning_option_pricing.models.terminal_ansatz import (  # noqa: E402
    cross_check_extension_forcing_analytic_versus_autograd,
)
from learning_option_pricing.pde import (  # noqa: E402
    bandlimited_bernoulli_cosine_coefficients,
    build_split_diffusion_extension_field,
    exact_solution_field,
)

# Stage-2 generator G1 (advection–diffusion–reaction).
GENERATOR_G1 = {2: 0.7, 1: 1.3, 0: -0.4}
CIRCLE_TERMINAL_TIME = 1.0


def _build_periodic_hard_constant_ansatz(*, with_derivative_bypass):
    """Hard-constant ansatz on the circle with a split-diffusion extension."""
    torch.manual_seed(0)
    net = ResNet(d_in=2, d_out=1, n=16, M=2, L=2).double()
    extension_field = build_split_diffusion_extension_field(
        GENERATOR_G1,
        bandlimited_bernoulli_cosine_coefficients(8),
        terminal_time=CIRCLE_TERMINAL_TIME,
    )
    interp_coeff = make_interpolation_coefficient(
        "linear", T=CIRCLE_TERMINAL_TIME
    )
    return TerminalAnsatz(
        net,
        None,
        interp_coeff,
        form="hard_constant",
        extension_fn=extension_field.field,
        extension_derivative_fns=(
            extension_field.derivative_callables()
            if with_derivative_bypass
            else None
        ),
    ), extension_field


def _circle_batch(size=128):
    generator = torch.Generator().manual_seed(1)
    x = (
        2.0 * math.pi * torch.rand(size, generator=generator, dtype=torch.float64)
    ).requires_grad_(True)
    t = (
        CIRCLE_TERMINAL_TIME
        * torch.rand(size, generator=generator, dtype=torch.float64)
    ).requires_grad_(True)
    return x, t


def test_sigma_route_reproduces_generator_coefficient_route():
    # sigma alone must be equivalent to {2: 0.5 sigma^2, 1: 0, 0: 0}, with
    # identically zero advection and reaction channels (backward compatible).
    T, sigma = 0.2, 1.0
    ansatz = _build("hard_constant", T=T, sigma=sigma)
    x = torch.linspace(0.05, 0.95, 40, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.01, 0.99 * T, 40, dtype=torch.float64, requires_grad=True)
    decomp_sigma = residual_decomposition(ansatz, x, t, sigma)

    x2 = x.detach().clone().requires_grad_(True)
    t2 = t.detach().clone().requires_grad_(True)
    decomp_generator = residual_decomposition(
        ansatz,
        x2,
        t2,
        generator_coefficients={2: 0.5 * sigma**2, 1: 0.0, 0: 0.0},
    )
    for channel in ("loss", "network_energy", "cross_term", "forcing_floor"):
        assert torch.allclose(
            decomp_sigma[channel], decomp_generator[channel], atol=1e-12
        ), channel
    assert decomp_sigma["forcing_advection"].item() == 0.0
    assert decomp_sigma["forcing_reaction"].item() == 0.0


def test_exactly_one_of_sigma_and_generator_coefficients():
    T = 0.2
    ansatz = _build("hard_constant", T=T)
    x = torch.linspace(0.05, 0.95, 10, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.01, 0.19, 10, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError):
        residual_decomposition(ansatz, x, t)
    with pytest.raises(ValueError):
        residual_decomposition(
            ansatz, x, t, 1.0, generator_coefficients={2: 0.5}
        )


def test_analytic_bypass_matches_autograd_route():
    ansatz_bypass, _ = _build_periodic_hard_constant_ansatz(
        with_derivative_bypass=True
    )
    ansatz_autograd, _ = _build_periodic_hard_constant_ansatz(
        with_derivative_bypass=False
    )
    x, t = _circle_batch()
    decomp_bypass = residual_decomposition(
        ansatz_bypass, x, t, generator_coefficients=GENERATOR_G1
    )
    x2 = x.detach().clone().requires_grad_(True)
    t2 = t.detach().clone().requires_grad_(True)
    decomp_autograd = residual_decomposition(
        ansatz_autograd, x2, t2, generator_coefficients=GENERATOR_G1
    )

    for channel in (
        "extension_forcing",
        "forcing_floor",
        "forcing_velocity",
        "forcing_diffusion",
        "forcing_advection",
        "forcing_reaction",
        "loss",
    ):
        deviation = (
            decomp_bypass[channel].detach() - decomp_autograd[channel].detach()
        )
        scale = torch.linalg.vector_norm(decomp_autograd[channel].detach())
        assert (
            torch.linalg.vector_norm(deviation) <= 1e-10 * max(scale.item(), 1.0)
        ), channel
    # The bypass assembles the theta-independent forcing outside the autograd
    # graph (torch.no_grad()): no gradient may flow through it.
    assert not decomp_bypass["extension_forcing"].requires_grad
    assert decomp_autograd["extension_forcing"].requires_grad


def test_extension_derivative_fns_validation():
    torch.manual_seed(0)
    net = ResNet(d_in=2, d_out=1, n=16, M=2, L=2).double()
    extension_field = build_split_diffusion_extension_field(
        GENERATOR_G1,
        bandlimited_bernoulli_cosine_coefficients(4),
        terminal_time=CIRCLE_TERMINAL_TIME,
    )
    interp_coeff = make_interpolation_coefficient(
        "linear", T=CIRCLE_TERMINAL_TIME
    )
    # Any form other than hard_constant raises.
    with pytest.raises(ValueError):
        TerminalAnsatz(
            net,
            None,
            interp_coeff,
            form="hard_convex",
            extension_fn=extension_field.field,
            extension_derivative_fns=extension_field.derivative_callables(),
        )
    # The key set must be exactly {"dt", "dx", "dxx"}.
    with pytest.raises(ValueError):
        TerminalAnsatz(
            net,
            None,
            interp_coeff,
            form="hard_constant",
            extension_fn=extension_field.field,
            extension_derivative_fns={"dt": extension_field.time_derivative},
        )


def test_cross_check_passes_for_correct_derivatives():
    ansatz, _ = _build_periodic_hard_constant_ansatz(with_derivative_bypass=True)
    x, t = _circle_batch()
    measured_deviation = cross_check_extension_forcing_analytic_versus_autograd(
        ansatz, x, t, generator_coefficients=GENERATOR_G1
    )
    assert measured_deviation <= 1e-10


def test_cross_check_passes_for_zero_forcing_exact_solution():
    # The exact-solution extension has identically vanishing forcing; the
    # cross-check must not divide round-off by round-off.
    torch.manual_seed(0)
    net = ResNet(d_in=2, d_out=1, n=16, M=2, L=2).double()
    extension_field = exact_solution_field(
        GENERATOR_G1,
        bandlimited_bernoulli_cosine_coefficients(8),
        terminal_time=CIRCLE_TERMINAL_TIME,
    )
    ansatz = TerminalAnsatz(
        net,
        None,
        make_interpolation_coefficient("linear", T=CIRCLE_TERMINAL_TIME),
        form="hard_constant",
        extension_fn=extension_field.field,
        extension_derivative_fns=extension_field.derivative_callables(),
    )
    x, t = _circle_batch()
    measured_deviation = cross_check_extension_forcing_analytic_versus_autograd(
        ansatz, x, t, generator_coefficients=GENERATOR_G1
    )
    assert measured_deviation <= 1e-10


def test_cross_check_aborts_on_wrong_derivatives():
    torch.manual_seed(0)
    net = ResNet(d_in=2, d_out=1, n=16, M=2, L=2).double()
    extension_field = build_split_diffusion_extension_field(
        GENERATOR_G1,
        bandlimited_bernoulli_cosine_coefficients(8),
        terminal_time=CIRCLE_TERMINAL_TIME,
    )

    def wrong_time_derivative(coord, time):
        return 1.5 * extension_field.time_derivative(coord, time)

    ansatz = TerminalAnsatz(
        net,
        None,
        make_interpolation_coefficient("linear", T=CIRCLE_TERMINAL_TIME),
        form="hard_constant",
        extension_fn=extension_field.field,
        extension_derivative_fns={
            "dt": wrong_time_derivative,
            "dx": extension_field.space_derivative,
            "dxx": extension_field.second_space_derivative,
        },
    )
    x, t = _circle_batch()
    with pytest.raises(RuntimeError):
        cross_check_extension_forcing_analytic_versus_autograd(
            ansatz, x, t, generator_coefficients=GENERATOR_G1
        )


def test_cross_check_requires_derivative_fns():
    ansatz, _ = _build_periodic_hard_constant_ansatz(
        with_derivative_bypass=False
    )
    x, t = _circle_batch()
    with pytest.raises(ValueError):
        cross_check_extension_forcing_analytic_versus_autograd(
            ansatz, x, t, generator_coefficients=GENERATOR_G1
        )
