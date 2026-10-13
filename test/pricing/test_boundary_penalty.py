"""Tests for the penalty baseline P of the barrier comparison (pricing/boundary_penalty.py).

The contract is the barrier pilot's (K = 1, B = 0.6, s_inf = 3, sigma = 0.3, r = 0.03, T = 1).
Monte Carlo tolerances are multiples of the estimated standard error, with fixed generator
seeds, so the tests are deterministic.
"""
import math

import pytest
import torch

from learning_option_pricing.pricing.boundary_penalty import (
    BarrierBoundaryPenaltySampling,
    FarFieldDatumLift,
    LogUniformMixtureOnInterval,
    boundary_constraint_violations,
    far_field_distance_factor,
    weighted_boundary_penalties,
)
from learning_option_pricing.pricing.terminal import black_scholes_put


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_weights_are_bounded_and_equal_to_the_uniform_bound_below_the_log_uniform_support():
    law = LogUniformMixtureOnInterval(length=2.4, smallest_value=1e-6, log_uniform_fraction=0.5)
    x, weights = law.sample(10_000, _generator(0))
    assert x.shape == weights.shape == (10_000,)
    assert float(x.min()) > 0.0 and float(x.max()) < 2.4
    assert float(weights.min()) > 0.0 and float(weights.max()) <= 2.0 + 1e-12
    below_support = x < 1e-6
    if below_support.any():
        assert torch.allclose(weights[below_support], torch.full_like(weights[below_support], 2.0))


@pytest.mark.parametrize("test_function", [lambda x: x, lambda x: torch.exp(-x), lambda x: torch.sqrt(x)])
def test_weighted_mean_is_an_unbiased_estimator_of_the_normalised_integral(test_function):
    length = 2.4
    law = LogUniformMixtureOnInterval(length=length, smallest_value=1e-6, log_uniform_fraction=0.5)
    estimates = []
    for seed in range(200):
        x, weights = law.sample(4096, _generator(seed))
        estimates.append(float(torch.mean(weights * test_function(x))))
    estimates = torch.tensor(estimates, dtype=torch.float64)
    fine_grid = torch.linspace(0.0, length, 2_000_001, dtype=torch.float64)
    exact = float(torch.trapezoid(test_function(fine_grid), fine_grid)) / length
    standard_error = float(estimates.std()) / math.sqrt(len(estimates))
    assert abs(float(estimates.mean()) - exact) < 4.0 * standard_error


def test_sampling_is_a_deterministic_function_of_the_generator_and_places_points_near_the_corner():
    sampling = BarrierBoundaryPenaltySampling(barrier=0.6, truncation_price=3.0, maturity=1.0)
    first = sampling.sample(1024, _generator(7), torch.float32)
    second = sampling.sample(1024, _generator(7), torch.float32)
    for key in first:
        assert torch.equal(first[key], second[key])
    assert torch.all(first["terminal_t"] == 1.0) and torch.all(first["barrier_s"] == 0.6)
    # Half of each face batch is log-uniform on [1e-6, l]: about a quarter of the batch below 1e-3 in distance.
    distance_to_barrier = first["terminal_s"].double() - 0.6
    assert int((distance_to_barrier < 1e-3).sum()) > 200
    assert int(((1.0 - first["barrier_t"].double()) < 1e-3).sum()) > 200


def test_far_field_factor_and_lift_reproduce_the_far_datum_and_leave_the_barrier_free():
    lift = FarFieldDatumLift(K=1.0, r=0.03, sigma=0.3, T=1.0, s_inf=3.0, terminal_profile="black_scholes")
    t = torch.linspace(0.0, 0.9, 10, dtype=torch.float64)
    s = torch.full_like(t, 3.0)
    assert torch.allclose(lift(s, t), black_scholes_put(s, 1.0, 0.03, 0.3, 1.0 - t))
    # Constant in s.
    assert torch.allclose(lift(torch.full_like(t, 0.7), t), lift(s, t))
    assert torch.allclose(far_field_distance_factor(s, 0.6, 3.0), torch.zeros_like(s))
    assert torch.allclose(far_field_distance_factor(torch.full_like(t, 0.6), 0.6, 3.0), torch.ones_like(s))
    raw_lift = FarFieldDatumLift(K=1.0, r=0.03, sigma=0.3, T=1.0, s_inf=3.0, terminal_profile="raw")
    assert torch.equal(raw_lift(s, t), torch.zeros_like(s))


class _ExactTerminalPayoffModel(torch.nn.Module):
    """Phi(s, t) = (K - s)^+: satisfies the terminal condition exactly, violates the barrier one by K - B."""

    def __init__(self) -> None:
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 - x[:, 0:1], min=0.0) + 0.0 * self.unused


def test_penalties_and_violations_of_a_model_equal_to_the_payoff():
    model = _ExactTerminalPayoffModel()
    sampling = BarrierBoundaryPenaltySampling(barrier=0.6, truncation_price=3.0, maturity=1.0)
    batch = sampling.sample(2048, _generator(3), torch.float64)
    terminal_penalty, barrier_penalty = weighted_boundary_penalties(model, batch, K=1.0)
    assert float(terminal_penalty) == 0.0
    # Phi(B, t)^2 = 0.16 for every t, and the weights have mean 1 in expectation.
    assert float(barrier_penalty) == pytest.approx(0.16, rel=0.05)
    violations = boundary_constraint_violations(model, K=1.0, B=0.6, T=1.0, s_inf=3.0)
    assert violations["terminal_condition_max_abs_violation"] == 0.0
    assert violations["barrier_condition_max_abs_violation"] == pytest.approx(0.4)
    assert violations["barrier_condition_max_abs_violation_outside_corner"] == pytest.approx(0.4)


def test_invalid_laws_are_refused():
    with pytest.raises(ValueError):
        LogUniformMixtureOnInterval(length=1.0, smallest_value=2.0)
    with pytest.raises(ValueError):
        LogUniformMixtureOnInterval(length=1.0, smallest_value=1e-3, log_uniform_fraction=1.0)
    with pytest.raises(ValueError):
        FarFieldDatumLift(K=1.0, r=0.03, sigma=0.3, T=1.0, s_inf=3.0, terminal_profile="split")
