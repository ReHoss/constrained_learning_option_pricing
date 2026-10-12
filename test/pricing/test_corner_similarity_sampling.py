"""Tests for the corner importance sampler (pricing/corner_similarity_sampling.py).

The contract is the barrier pilot's (B = 0.6, s_inf = 3, sigma = 0.3, T = 1). Every Monte Carlo
tolerance is a multiple of the estimated standard error, with a fixed generator seed, so the
tests are deterministic.
"""
import math

import pytest
import torch

from learning_option_pricing.pricing.corner_similarity_sampling import CornerSimilaritySampling

PILOT_SAMPLER = CornerSimilaritySampling(barrier=0.6, truncation_price=3.0, maturity=1.0, volatility=0.3)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_corner_density_integrates_to_one():
    """Midpoint quadrature of q in the coordinates (y, ln tau), y = ln(s/B): ds dtau = s tau dy dln(tau)."""
    sampler = PILOT_SAMPLER
    number_of_log_time_cells, number_of_log_price_cells = 4000, 4000
    log_tau_edges = torch.linspace(math.log(sampler.smallest_time_to_maturity), math.log(sampler.maturity),
                                   number_of_log_time_cells + 1, dtype=torch.float64)
    y_edges = torch.linspace(0.0, math.log(sampler.truncation_price / sampler.barrier), number_of_log_price_cells + 1,
                             dtype=torch.float64)
    log_tau = 0.5 * (log_tau_edges[1:] + log_tau_edges[:-1])
    y = 0.5 * (y_edges[1:] + y_edges[:-1])
    tau_grid, y_grid = torch.meshgrid(torch.exp(log_tau), y, indexing="ij")
    s_grid = sampler.barrier * torch.exp(y_grid)
    integrand = sampler.corner_density(s_grid, tau_grid) * s_grid * tau_grid
    integral = float(integrand.sum() * (log_tau_edges[1] - log_tau_edges[0]) * (y_edges[1] - y_edges[0]))
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_acceptance_probability_matches_the_rejection_rate():
    sampler = PILOT_SAMPLER
    number_of_draws = 1_000_000
    _, _, number_of_rejected_draws = sampler.sample_corner_component(number_of_draws, _generator(0))
    # First-pass rejections dominate; the empirical acceptance is draws / (draws + rejections).
    empirical_acceptance = number_of_draws / (number_of_draws + number_of_rejected_draws)
    standard_error = math.sqrt(empirical_acceptance * (1.0 - empirical_acceptance) / number_of_draws)
    assert abs(empirical_acceptance - sampler.acceptance_probability()) < 5.0 * standard_error + 1e-6


def test_acceptance_probability_is_one_when_no_draw_can_leave_the_domain():
    sampler = CornerSimilaritySampling(barrier=0.6, truncation_price=3.0, maturity=1.0, volatility=0.3,
                                       similarity_variable_max=1.0)  # xi_max sigma sqrt(2T) = 0.42 < ln 5
    assert sampler.acceptance_probability() == pytest.approx(1.0, abs=1e-14)


def test_corner_draws_lie_in_the_support():
    sampler = PILOT_SAMPLER
    s, tau, _ = sampler.sample_corner_component(100_000, _generator(1))
    xi = torch.log(s / sampler.barrier) / (sampler.volatility * torch.sqrt(2.0 * tau))
    assert bool((s >= sampler.barrier).all()) and bool((s <= sampler.truncation_price).all())
    assert bool((tau >= sampler.smallest_time_to_maturity).all()) and bool((tau <= sampler.maturity).all())
    assert bool((xi <= sampler.similarity_variable_max + 1e-12).all())


def test_mixture_batch_sizes_weights_and_determinism():
    sampler = PILOT_SAMPLER
    s, t, weights, statistics = sampler.sample_mixture(4096, _generator(2), torch.float32)
    assert s.shape == t.shape == weights.shape == (4096,)
    assert statistics["number_of_corner_points"] == 2048 and statistics["number_of_uniform_points"] == 2048
    assert float(weights.max()) <= 1.0 / (1.0 - sampler.effective_corner_fraction(4096)) + 1e-6
    assert float(weights.min()) > 0.0
    s_again, t_again, weights_again, _ = sampler.sample_mixture(4096, _generator(2), torch.float32)
    assert torch.equal(s, s_again) and torch.equal(t, t_again) and torch.equal(weights, weights_again)


@pytest.mark.parametrize("test_function_name", ["constant", "inverse_square_root_of_time_to_maturity"])
def test_weighted_mixture_mean_is_an_unbiased_estimator_of_the_uniform_mean(test_function_name):
    """E_m[w f] = E_u[f] for f = 1 (mean weight 1) and f = tau^{-1/2} (uniform mean 2/sqrt(T) = 2),
    the latter singular at the corner time, where the mixture concentrates its points."""
    sampler = PILOT_SAMPLER
    generator = _generator(3)
    estimates = []
    for _ in range(200):
        s, t, weights, _ = sampler.sample_mixture(4096, generator, torch.float64)
        tau = sampler.maturity - t
        values = torch.ones_like(tau) if test_function_name == "constant" else tau.rsqrt()
        estimates.append(float((weights * values).mean()))
    estimates = torch.tensor(estimates, dtype=torch.float64)
    expected = 1.0 if test_function_name == "constant" else 2.0 / math.sqrt(sampler.maturity)
    standard_error = float(estimates.std() / math.sqrt(len(estimates)))
    assert abs(float(estimates.mean()) - expected) < 5.0 * standard_error


@pytest.mark.parametrize("arguments", [
    {"corner_fraction": 0.0}, {"corner_fraction": 1.0}, {"smallest_time_to_maturity": 0.0},
    {"smallest_time_to_maturity": 1.0}, {"similarity_variable_max": 0.0},
])
def test_invalid_configurations_are_rejected(arguments):
    with pytest.raises(ValueError):
        CornerSimilaritySampling(barrier=0.6, truncation_price=3.0, maturity=1.0, volatility=0.3, **arguments)
