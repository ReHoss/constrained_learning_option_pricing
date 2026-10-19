"""Tests for the l1-triangle control law of the corner importance sampling
(pricing/corner_triangle_sampling.py).

The contract is the barrier pilot's (B = 0.6, s_inf = 3, T = 1), triangle size delta = 0.1, corner
fraction 0.5. Every Monte Carlo tolerance is a multiple of the estimated standard error, with a fixed
generator seed, so the tests are deterministic.
"""
import math

import pytest
import torch

from learning_option_pricing.pricing.corner_triangle_sampling import CornerTriangleSampling

PILOT_TRIANGLE_SAMPLER = CornerTriangleSampling(barrier=0.6, truncation_price=3.0, maturity=1.0, triangle_size=0.1)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def test_corner_draws_are_uniform_on_the_triangle():
    """Support, and the two first moments of the uniform law on {x, tau >= 0, x + tau <= delta}:
    E x = E tau = delta/3, Var x = Var tau = delta^2/18."""
    sampler = PILOT_TRIANGLE_SAMPLER
    delta = sampler.triangle_size
    s, tau, _ = sampler.sample_corner_component(400_000, _generator(0), torch.float64)
    distance_to_barrier = s - sampler.barrier
    assert bool((distance_to_barrier >= 0.0).all()) and bool((tau > 0.0).all())
    assert bool((distance_to_barrier + tau <= delta + 1e-15).all())
    standard_error_of_the_mean = math.sqrt(delta**2 / 18.0 / s.numel())
    for coordinate in (distance_to_barrier, tau):
        assert abs(float(coordinate.mean()) - delta / 3.0) < 5.0 * standard_error_of_the_mean
        assert float(coordinate.var()) == pytest.approx(delta**2 / 18.0, rel=1e-2)


def test_weights_take_the_two_closed_form_values():
    """w = 1/(1 - p) outside the triangle, 1/((1 - p) + p (2/delta^2)/u) inside, with u = 1/((s_inf - B) T)."""
    sampler = PILOT_TRIANGLE_SAMPLER
    s, t, weights, statistics = sampler.sample_mixture(4096, _generator(1), torch.float64)
    p = sampler.effective_corner_fraction(4096)
    u = 1.0 / ((sampler.truncation_price - sampler.barrier) * sampler.maturity)
    inside = (s - sampler.barrier) + (sampler.maturity - t) <= sampler.triangle_size
    expected = torch.where(inside, torch.full_like(s, 1.0 / ((1.0 - p) + p * (2.0 / sampler.triangle_size**2) / u)),
                           torch.full_like(s, 1.0 / (1.0 - p)))
    assert torch.allclose(weights, expected, rtol=1e-12, atol=0.0)
    assert statistics["number_of_corner_points"] == 2048 and statistics["number_of_uniform_points"] == 2048


def test_no_float32_draw_lies_on_the_maturity_slice():
    """In float32, t = T - tau rounds to T for tau < 2^-25; such draws are redrawn, so the network
    never sees tau = 0, where the similarity variable is not defined."""
    sampler = PILOT_TRIANGLE_SAMPLER
    generator = _generator(2)
    total_redraws = 0
    for _ in range(50):
        _, t, _, statistics = sampler.sample_mixture(65_536, generator, torch.float32)
        assert bool((t < torch.tensor(sampler.maturity, dtype=torch.float32)).all())
        total_redraws += statistics["number_of_rejected_corner_draws"]
    # 50 x 32768 corner draws at a probability of at most 6.0e-7 each: about 1 expected.
    assert total_redraws < 20


def test_mixture_is_deterministic_and_draws_the_uniform_group_as_the_plain_sampler():
    """Same generator -> same batch; the uniform group is the first draw of the generator, s then t,
    exactly as the pilot's uniform sampler."""
    sampler = PILOT_TRIANGLE_SAMPLER
    s, t, weights, _ = sampler.sample_mixture(4096, _generator(3), torch.float32)
    s_again, t_again, weights_again, _ = sampler.sample_mixture(4096, _generator(3), torch.float32)
    assert torch.equal(s, s_again) and torch.equal(t, t_again) and torch.equal(weights, weights_again)
    generator = _generator(3)
    s_plain = torch.rand(2048, generator=generator) * (sampler.truncation_price - sampler.barrier) + sampler.barrier
    t_plain = torch.rand(2048, generator=generator) * sampler.maturity
    assert torch.equal(s[:2048], s_plain) and torch.equal(t[:2048], t_plain)


def _uniform_mean_of_the_triangle_singular_function(sampler: CornerTriangleSampling) -> float:
    r"""int_Omega u 1_T tau^{-1/2} = u int_0^delta (delta - tau) tau^{-1/2} dtau = u (4/3) delta^{3/2}."""
    u = 1.0 / ((sampler.truncation_price - sampler.barrier) * sampler.maturity)
    return u * 4.0 / 3.0 * sampler.triangle_size**1.5


@pytest.mark.parametrize("test_function_name", [
    "constant", "inverse_square_root_of_time_to_maturity", "triangle_indicator",
    "inverse_square_root_of_time_to_maturity_on_the_triangle",
])
def test_weighted_mixture_mean_is_an_unbiased_estimator_of_the_uniform_mean(test_function_name):
    """E_m[w f] = E_u[f] for four test functions with closed-form uniform means: f = 1 (mean weight 1),
    f = tau^{-1/2} (2/sqrt(T)), f = 1_T (u delta^2/2) and f = 1_T tau^{-1/2} (u (4/3) delta^{3/2}), the last
    two supported where the mixture concentrates its points."""
    sampler = PILOT_TRIANGLE_SAMPLER
    generator = _generator(4)
    u = 1.0 / ((sampler.truncation_price - sampler.barrier) * sampler.maturity)
    expected = {
        "constant": 1.0,
        "inverse_square_root_of_time_to_maturity": 2.0 / math.sqrt(sampler.maturity),
        "triangle_indicator": u * sampler.triangle_area(),
        "inverse_square_root_of_time_to_maturity_on_the_triangle": _uniform_mean_of_the_triangle_singular_function(sampler),
    }[test_function_name]
    estimates = []
    for _ in range(200):
        s, t, weights, _ = sampler.sample_mixture(4096, generator, torch.float64)
        tau = sampler.maturity - t
        inside = ((s - sampler.barrier) + tau <= sampler.triangle_size).to(torch.float64)
        values = {
            "constant": torch.ones_like(tau),
            "inverse_square_root_of_time_to_maturity": tau.rsqrt(),
            "triangle_indicator": inside,
            "inverse_square_root_of_time_to_maturity_on_the_triangle": inside * tau.rsqrt(),
        }[test_function_name]
        estimates.append(float((weights * values).mean()))
    estimates = torch.tensor(estimates, dtype=torch.float64)
    standard_error = float(estimates.std() / math.sqrt(len(estimates)))
    assert abs(float(estimates.mean()) - expected) < 5.0 * standard_error


@pytest.mark.parametrize("arguments", [
    {"corner_fraction": 0.0}, {"corner_fraction": 1.0}, {"triangle_size": 0.0}, {"triangle_size": 1.5},
])
def test_invalid_configurations_are_rejected(arguments):
    with pytest.raises(ValueError):
        CornerTriangleSampling(**({"barrier": 0.6, "truncation_price": 3.0, "maturity": 1.0} | arguments))
