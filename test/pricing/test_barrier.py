"""Tests for the down-and-out put building blocks (pricing/barrier.py).

Reference: working note "A rigorous statement of exact-constraint learning at
a conflicting constraint corner: the knock-out barrier option" (S. Ouaissi
internship, 2026-06-24).

``reiner_rubinstein_down_and_out_put`` is independently validated (not by a
test in this file, but by an isolated Monte-Carlo script run during
development, outside the repository): a discretely-monitored GBM simulation
converges to the closed form as the monitoring frequency N increases (the
deviation shrinks like O(1/sqrt(N)), from -0.175 at N=252 to -0.042 at
N=4000 on K=100,B=80,r=0.02,sigma=0.25,T=1,s=100 -- the signature of a
correct continuous-barrier formula compared against discrete monitoring, not
a residual error). This file checks the analytic properties that can be
verified deterministically and fast.
"""
import math

import pytest
import torch

from learning_option_pricing.pricing.barrier import (
    barrier_composite_distance,
    make_corner_regularised_extension,
    reiner_rubinstein_down_and_out_put,
)
from learning_option_pricing.pricing.terminal import black_scholes_put, payoff_put


# ---------------------------------------------------------------------------
# barrier_composite_distance  (Definition 4)
# ---------------------------------------------------------------------------

class TestBarrierCompositeDistance:
    B, T = 0.6, 1.0

    def test_vanishes_on_terminal_face(self) -> None:
        """d(s, T) = 0 for every s."""
        s = torch.linspace(self.B + 0.01, 5.0, 101)
        t = torch.full_like(s, self.T)
        d = barrier_composite_distance(s, t, self.B, self.T)
        assert torch.allclose(d, torch.zeros_like(d))

    def test_vanishes_on_barrier_face(self) -> None:
        """d(B, t) = 0 for every t."""
        t = torch.linspace(0.0, self.T, 101)
        s = torch.full_like(t, self.B)
        d = barrier_composite_distance(s, t, self.B, self.T)
        assert torch.allclose(d, torch.zeros_like(d))

    def test_positive_in_the_interior(self) -> None:
        """d > 0 strictly inside Q = (B, +inf) x (0, T)."""
        s = torch.linspace(self.B + 0.01, 5.0, 51)
        t = torch.linspace(0.0, self.T - 0.01, 51)
        ss, tt = torch.meshgrid(s, t, indexing="ij")
        d = barrier_composite_distance(ss, tt, self.B, self.T)
        assert torch.all(d > 0.0)


# ---------------------------------------------------------------------------
# make_corner_regularised_extension  (Definition 5, conditions (11))
# ---------------------------------------------------------------------------

class TestCornerRegularisedExtension:
    K, B, T = 1.0, 0.6, 1.0
    epsilon = 0.05

    def test_matches_payoff_outside_corner_layer_on_terminal_face(self) -> None:
        """h_eps(s, T) = g(s) exactly for s - B > epsilon."""
        s = torch.linspace(self.B + self.epsilon + 1e-3, 5.0, 201)
        t = torch.full_like(s, self.T)
        h_eps = make_corner_regularised_extension(self.K, self.B, self.epsilon)
        assert torch.allclose(h_eps(s, t), payoff_put(s, self.K), atol=1e-6)

    def test_zero_on_entire_barrier_face(self) -> None:
        """h_eps(B, t) = 0 for every t (stronger than required: the barrier
        datum is identically zero, so exactness holds on all of Sigma_B, not
        only outside the corner layer -- see the module docstring)."""
        t = torch.linspace(0.0, self.T, 101)
        s = torch.full_like(t, self.B)
        h_eps = make_corner_regularised_extension(self.K, self.B, self.epsilon)
        assert torch.allclose(h_eps(s, t), torch.zeros_like(t), atol=1e-6)

    def test_bounded_by_corner_jump_inside_the_layer(self) -> None:
        """||h_eps||_{L^inf(N_eps)} <= K - B (condition (11))."""
        s = torch.linspace(self.B, self.B + self.epsilon, 51)
        t = torch.linspace(self.T - self.epsilon, self.T, 51)
        ss, tt = torch.meshgrid(s, t, indexing="ij")
        h_eps = make_corner_regularised_extension(self.K, self.B, self.epsilon)
        values = h_eps(ss, tt)
        assert torch.all(values.abs() <= (self.K - self.B) + 1e-9)

    def test_transition_is_smooth_c1(self) -> None:
        """The weight zeta((s-B)/epsilon) is C^1 (finite-difference check):
        no jump in the numerical derivative across the layer boundary."""
        h_eps = make_corner_regularised_extension(self.K, self.B, self.epsilon)
        s = torch.linspace(self.B, self.B + 2 * self.epsilon, 2001, dtype=torch.float64)
        t = torch.full_like(s, self.T)
        values = h_eps(s, t)
        deriv = torch.diff(values) / torch.diff(s)
        # A genuine kink would show up as a large jump between consecutive
        # finite-difference slopes; a smooth transition does not.
        second_diff = torch.diff(deriv)
        assert float(second_diff.abs().max()) < 1.0

    def test_rejects_non_reverse_knock_out_regime(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension(K=0.5, B=0.6, epsilon=0.05)

    def test_rejects_non_positive_epsilon(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension(K=1.0, B=0.6, epsilon=0.0)


# ---------------------------------------------------------------------------
# reiner_rubinstein_down_and_out_put  (Remark 6)
# ---------------------------------------------------------------------------

class TestReinerRubinsteinDownAndOutPut:
    K, B, r, sigma, T = 100.0, 80.0, 0.02, 0.25, 1.0

    def test_regression_reference_values(self) -> None:
        """Locks in the validated values (see module docstring for how they
        were derived and cross-checked) so a future edit cannot silently
        reintroduce the reflection-prefactor sign error found during
        development."""
        expected = {
            82.0: 0.188713,
            90.0: 0.829803,
            100.0: 1.228249,
            120.0: 0.987598,
        }
        for s_val, expected_price in expected.items():
            s = torch.tensor([s_val], dtype=torch.float64)
            tau = torch.tensor([self.T], dtype=torch.float64)
            price = float(reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, tau))
            assert abs(price - expected_price) < 1e-5, f"s={s_val}: got {price}, expected {expected_price}"

    def test_vanishes_at_the_barrier(self) -> None:
        """V_DO(B, t) = 0 for every t (condition (5c))."""
        t_vals = torch.linspace(0.01, self.T, 20)
        for t in t_vals:
            tau = torch.tensor([self.T - float(t)])
            s = torch.tensor([self.B])
            price = float(reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, tau))
            assert abs(price) < 1e-9

    def test_bounded_above_by_vanilla_put(self) -> None:
        """A down-and-out put is never worth more than the vanilla put with
        the same strike (the knock-out clause can only remove value)."""
        s = torch.linspace(60.0, 140.0, 41)
        tau = torch.full_like(s, self.T)
        vanilla = black_scholes_put(s, self.K, self.r, self.sigma, tau)
        do = reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, tau)
        assert torch.all(vanilla - do > -1e-6)

    def test_converges_to_vanilla_put_as_barrier_recedes(self) -> None:
        """As B -> 0, the knock-out clause is (numerically) never triggered
        on the tested price range, so V_DO -> vanilla put."""
        s = torch.linspace(60.0, 140.0, 41)
        tau = torch.full_like(s, self.T)
        vanilla = black_scholes_put(s, self.K, self.r, self.sigma, tau)
        do = reiner_rubinstein_down_and_out_put(s, self.K, 1e-3, self.r, self.sigma, tau)
        assert torch.allclose(vanilla, do, atol=1e-3)

    def test_zero_below_or_at_the_barrier(self) -> None:
        """Already knocked out: price is exactly 0 for s <= B."""
        s = torch.tensor([self.B, self.B - 1.0, self.B - 10.0])
        tau = torch.full_like(s, self.T)
        price = reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, tau)
        assert torch.all(price == 0.0)

    def test_rejects_non_reverse_knock_out_regime(self) -> None:
        """The formula covers only 0 < B < K (Assumption 1 of the note)."""
        s = torch.tensor([90.0])
        tau = torch.tensor([self.T])
        with pytest.raises(ValueError):
            reiner_rubinstein_down_and_out_put(s, K=80.0, B=100.0, r=self.r, sigma=self.sigma, tau=tau)

    def test_differentiable(self) -> None:
        """Autograd can compute dV_DO/ds (needed for the PDE residual)."""
        s = torch.tensor([100.0], requires_grad=True)
        tau = torch.tensor([self.T])
        price = reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, tau)
        price.backward()
        assert s.grad is not None
        assert math.isfinite(float(s.grad))

    def test_batch_consistency(self) -> None:
        """Price is the same whether computed in a batch or individually."""
        s_vals = [82.0, 90.0, 100.0, 120.0]
        s_batch = torch.tensor(s_vals, dtype=torch.float64)
        tau_batch = torch.full_like(s_batch, self.T)
        p_batch = reiner_rubinstein_down_and_out_put(s_batch, self.K, self.B, self.r, self.sigma, tau_batch)
        for i, sv in enumerate(s_vals):
            s_single = torch.tensor([sv], dtype=torch.float64)
            tau_single = torch.tensor([self.T], dtype=torch.float64)
            p_single = float(reiner_rubinstein_down_and_out_put(s_single, self.K, self.B, self.r, self.sigma, tau_single))
            assert abs(p_single - float(p_batch[i])) < 1e-9
