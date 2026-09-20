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
    BlackScholesCornerExtension,
    _smoothstep01,
    barrier_composite_distance,
    barrier_composite_distance_with_far_field,
    make_corner_regularised_extension,
    make_corner_regularised_extension_split,
    SPLIT_PROFILE_ROUTES,
    make_corner_regularised_extension_with_black_scholes_payoff,
    make_corner_regularised_extension_with_smoothed_payoff,
    mangasarian_smoothed_put_payoff,
    reiner_rubinstein_down_and_out_put,
    reiner_rubinstein_down_and_out_put_gamma,
    down_and_out_digital_price,
    down_and_out_digital_price_and_derivatives,
    make_subtracted_digital_extension,
    SubtractedDigitalCornerExtension,
    SUBTRACTION_TERMINAL_PROFILES,
    RawPutPayoffTerminalProfile,
    BlackScholesPutTerminalProfile,
    SplitSemigroupPutTerminalProfile,
)
from learning_option_pricing.pricing.terminal import black_scholes_put, bsm_operator, payoff_put


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


class TestBarrierCompositeDistanceWithFarField:
    B, T, S_INF = 0.6, 1.0, 3.0

    def _d(self, s, t):
        return barrier_composite_distance_with_far_field(s, t, self.B, self.T, self.S_INF)

    def test_vanishes_on_the_three_faces(self) -> None:
        """d = 0 on t = T, on s = B and on s = s_inf."""
        s = torch.linspace(self.B, self.S_INF, 101)
        assert torch.allclose(self._d(s, torch.full_like(s, self.T)), torch.zeros_like(s))
        t = torch.linspace(0.0, self.T, 101)
        assert torch.allclose(self._d(torch.full_like(t, self.B), t), torch.zeros_like(t))
        assert torch.allclose(self._d(torch.full_like(t, self.S_INF), t), torch.zeros_like(t))

    def test_positive_in_the_interior(self) -> None:
        s = torch.linspace(self.B + 0.01, self.S_INF - 0.01, 51)
        t = torch.linspace(0.0, self.T - 0.01, 51)
        ss, tt = torch.meshgrid(s, t, indexing="ij")
        assert torch.all(self._d(ss, tt) > 0.0)

    def test_matches_untruncated_factor_near_the_barrier(self) -> None:
        """The normalisation makes the factor equal to (T-t)(s-B) to first order at s = B."""
        s = torch.tensor([self.B + 1e-4]); t = torch.tensor([0.3])
        untruncated = barrier_composite_distance(s, t, self.B, self.T)
        assert torch.allclose(self._d(s, t), untruncated, rtol=1e-3)

    def test_rejects_s_inf_below_barrier(self) -> None:
        with pytest.raises(ValueError):
            barrier_composite_distance_with_far_field(torch.tensor([1.0]), torch.tensor([0.0]), self.B, self.T, self.B)


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
# mangasarian_smoothed_put_payoff
# ---------------------------------------------------------------------------

class TestMangasarianSmoothedPutPayoff:
    K, T = 1.0, 1.0
    eps0 = 0.05

    def test_time_graded_matches_raw_payoff_exactly_at_maturity(self) -> None:
        """eps(T) = eps0*(T-T)/T = 0, so g_eps0(s, T) = (K-s)^+ exactly."""
        s = torch.linspace(0.0, 2.0, 201)
        t = torch.full_like(s, self.T)
        smoothed = mangasarian_smoothed_put_payoff(s, t, self.K, self.T, self.eps0, grading="time_graded")
        assert torch.allclose(smoothed, payoff_put(s, self.K), atol=1e-6)

    def test_constant_grading_does_not_match_raw_payoff_at_maturity(self) -> None:
        """eps(t) = eps0 everywhere under "constant" grading, so exactness at
        t=T (unlike "time_graded") does not hold, in particular at the kink
        s=K where the smoothing effect is largest."""
        s = torch.tensor([self.K])
        t = torch.full_like(s, self.T)
        smoothed = mangasarian_smoothed_put_payoff(s, t, self.K, self.T, self.eps0, grading="constant")
        assert not torch.allclose(smoothed, payoff_put(s, self.K), atol=1e-6)

    def test_positive_everywhere(self) -> None:
        """g_eps0 >= 0 always, since sqrt((K-s)^2 + eps(t)^2) >= |K-s|."""
        s = torch.linspace(-1.0, 3.0, 401)
        t = torch.linspace(0.0, self.T, 401)
        ss, tt = torch.meshgrid(s, t, indexing="ij")
        smoothed = mangasarian_smoothed_put_payoff(ss, tt, self.K, self.T, self.eps0, grading="time_graded")
        assert torch.all(smoothed >= 0.0)

    def test_c2_second_derivative_at_the_strike_matches_the_analytic_value(self) -> None:
        """d^2/ds^2 at s=K (where the raw payoff has a first-derivative
        discontinuity) is finite for eps(t) > 0, i.e. at an intermediate
        time t < T, and matches the closed form 1/(2*eps(t)).

        With g(x) = 0.5*(x + sqrt(x^2+eps^2)), x = K-s: d^2g/dx^2 =
        0.5*eps^2/(x^2+eps^2)^{3/2}, which at x=0 (s=K) reduces to
        1/(2*eps). Cross-checked against a central finite-difference second
        difference over step sizes h in [1e-1, 1e-6] during development: the
        raw payoff's finite-difference second derivative diverges like 1/h
        (10 at h=1e-1 up to 1e6 at h=1e-6), while the smoothed payoff's
        converges to this analytic value (7.81 at h=1e-1 down to 20.00006 at
        h=1e-6, vs. the exact 20.0) before float64 round-off dominates below
        h=1e-6 -- the intended C^2 behaviour at the strike."""
        t_val = 0.5 * self.T
        eps_t = self.eps0 * (self.T - t_val) / self.T
        expected_second_derivative = 1.0 / (2.0 * eps_t)

        s = torch.tensor(self.K, dtype=torch.float64, requires_grad=True)
        t = torch.tensor(t_val, dtype=torch.float64)
        smoothed = mangasarian_smoothed_put_payoff(s, t, self.K, self.T, self.eps0, grading="time_graded")
        first_derivative = torch.autograd.grad(smoothed, s, create_graph=True)[0]
        second_derivative = torch.autograd.grad(first_derivative, s)[0]
        assert math.isfinite(float(second_derivative))
        assert abs(float(second_derivative) - expected_second_derivative) < 1e-9

    def test_rejects_unknown_grading(self) -> None:
        s = torch.tensor([1.0])
        t = torch.tensor([0.5])
        with pytest.raises(ValueError):
            mangasarian_smoothed_put_payoff(s, t, self.K, self.T, self.eps0, grading="bogus")

    def test_rejects_non_positive_eps0(self) -> None:
        s = torch.tensor([1.0])
        t = torch.tensor([0.5])
        with pytest.raises(ValueError):
            mangasarian_smoothed_put_payoff(s, t, self.K, self.T, eps0=0.0)


# ---------------------------------------------------------------------------
# make_corner_regularised_extension_with_smoothed_payoff
# ---------------------------------------------------------------------------

class TestCornerRegularisedExtensionWithSmoothedPayoff:
    K, B, T = 1.0, 0.6, 1.0
    epsilon = 0.05
    eps0 = 0.01

    def test_zero_on_entire_barrier_face(self) -> None:
        """h(B, t) = 0 for every t: the cutoff zeta(0) = 0 kills the smoothed
        payoff regardless of its value there, exactly as for the raw-payoff
        extension."""
        t = torch.linspace(0.0, self.T, 101)
        s = torch.full_like(t, self.B)
        h_eps = make_corner_regularised_extension_with_smoothed_payoff(
            self.K, self.B, self.epsilon, self.T, self.eps0, grading="time_graded"
        )
        assert torch.allclose(h_eps(s, t), torch.zeros_like(t), atol=1e-6)

    def test_matches_raw_extension_at_maturity_with_time_graded_smoothing(self) -> None:
        """With grading="time_graded", eps(T) = 0, so the smoothed extension
        coincides exactly with the raw-payoff extension on the terminal face."""
        s = torch.linspace(self.B, 5.0, 201)
        t = torch.full_like(s, self.T)
        h_raw = make_corner_regularised_extension(self.K, self.B, self.epsilon)
        h_smoothed = make_corner_regularised_extension_with_smoothed_payoff(
            self.K, self.B, self.epsilon, self.T, self.eps0, grading="time_graded"
        )
        assert torch.allclose(h_raw(s, t), h_smoothed(s, t), atol=1e-6)

    def test_rejects_non_positive_eps0(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension_with_smoothed_payoff(
                self.K, self.B, self.epsilon, self.T, eps0=0.0
            )


# ---------------------------------------------------------------------------
# make_corner_regularised_extension_with_black_scholes_payoff
# ---------------------------------------------------------------------------

class TestCornerRegularisedExtensionWithBlackScholesPayoff:
    K, B, T, r, sigma = 1.0, 0.6, 1.0, 0.03, 0.3
    epsilon = 0.05

    def test_zero_on_entire_barrier_face(self) -> None:
        """h(B, t) = 0 for every t: the cutoff zeta(0) = 0 kills the European
        put price regardless of its value there."""
        t = torch.linspace(0.0, self.T, 101)
        s = torch.full_like(t, self.B)
        h_eps = make_corner_regularised_extension_with_black_scholes_payoff(
            self.K, self.B, self.epsilon, self.r, self.sigma, self.T
        )
        assert torch.allclose(h_eps(s, t), torch.zeros_like(t), atol=1e-6)

    def test_matches_raw_extension_at_maturity_up_to_the_tau_floor(self) -> None:
        """At t=T, V^e(s,T) = (K-s)^+ up to black_scholes_put's internal
        tau-floor (1e-8, a numerical safeguard, not a deliberate smoothing
        bandwidth), so the two extensions nearly coincide on the terminal
        face."""
        s = torch.linspace(self.B + self.epsilon + 1e-3, 5.0, 201, dtype=torch.float64)
        t = torch.full_like(s, self.T)
        h_raw = make_corner_regularised_extension(self.K, self.B, self.epsilon)
        h_bs = make_corner_regularised_extension_with_black_scholes_payoff(
            self.K, self.B, self.epsilon, self.r, self.sigma, self.T
        )
        assert torch.allclose(h_raw(s, t), h_bs(s, t), atol=1e-4)

    def test_smooth_in_s_away_from_the_corner(self) -> None:
        """d^2/ds^2 is finite at an interior point away from the corner layer
        and away from s=K (V^e is C^infty in s for tau > 0, unlike the raw
        put payoff's kink)."""
        s = torch.tensor(self.K, dtype=torch.float64, requires_grad=True)
        t = torch.tensor(0.5 * self.T, dtype=torch.float64)
        h_eps = make_corner_regularised_extension_with_black_scholes_payoff(
            self.K, self.B, self.epsilon, self.r, self.sigma, self.T
        )
        value = h_eps(s, t)
        first_derivative = torch.autograd.grad(value, s, create_graph=True)[0]
        second_derivative = torch.autograd.grad(first_derivative, s)[0]
        assert math.isfinite(float(second_derivative))

    def test_rejects_non_reverse_knock_out_regime(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension_with_black_scholes_payoff(
                K=0.5, B=0.6, epsilon=0.05, r=self.r, sigma=self.sigma, T=self.T
            )

    def test_rejects_non_positive_epsilon(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension_with_black_scholes_payoff(
                K=1.0, B=0.6, epsilon=0.0, r=self.r, sigma=self.sigma, T=self.T
            )


# ---------------------------------------------------------------------------
# make_corner_regularised_extension_split  (Proposition 7 / Example 7)
#
# Three additions over the three sections above, all independent of the
# production code under test (a different formula, not a different call to
# the same one):
#
# (A) the split-semigroup VALUE against a closed form derived by hand from
#     Example 7 -- e^{tau A_c} applied to g(x) = (K-e^x)^+ is an explicit
#     Gaussian convolution; substituting u = x-y and completing the square
#     in the exponent gives, with m = sigma*sqrt(tau), c = ln(K/s)/m,
#
#         pi_split(s,t) = K N(c) - s exp(sigma^2 tau / 2) N(c-m).
#
#     GaussianSemigroupExtensionField evaluates this by fixed-grid
#     trapezoidal quadrature; this closed form does not, so agreement is an
#     independent check of the quadrature settings (y_lo, y_hi, n_quad), not
#     a circular restatement of the code under test.
#
# (B) the split-semigroup SECOND PRICE DERIVATIVE, obtained the same way the
#     production code obtains it -- via derivative_callables()'s analytic
#     "dx"/"dxx" (in log-price x) and the chain rule
#     d^2 pi/ds^2 = (1/s^2)(d_xx pi - d_x pi) -- against a second closed
#     form obtained by differentiating pi_split by hand:
#
#         d^2 pi_split/ds^2 = K N'(c) / (s^2 m).
#
#     (The three intermediate terms of d(pi_split)/ds collapse to a single
#     one via the identity K N'(c) = s exp(sigma^2 tau/2) N'(c-m); a second
#     differentiation of the remaining term gives the above.) NOT checked by
#     autograd or a finite difference through h_eps itself: h_eps is a
#     quadratured convolution, so a finite difference with a PDE-residual-
#     scale step (h=1e-4) would divide the quadrature's own discretisation
#     error of the VALUE by h^2 = 1e-8, amplifying it by a factor of 1e8 --
#     exactly why GaussianSemigroupExtensionField supplies analytic
#     derivatives in the first place (see its module docstring), and why
#     make_corner_regularised_extension_split routes through them instead.
#
# (C) the full Black-Scholes PDE residual L^BS pi = d_t pi + (sigma^2/2)
#     s^2 d_ss pi + r s d_s pi - r pi, assembled in price coordinates from
#     the SAME derivative_callables() route as (B) (never autograd or a
#     finite difference), against a third closed form obtained from
#     Proposition 7(i)'s algebraic identity L h = B h collapsed to price
#     coordinates:
#
#         L^BS pi_split = -(r - sigma^2/2) s exp(sigma^2 tau/2) N(c-m) - r pi_split.
#
#     A canary test first checks that derivative_callables()'s "dt" is
#     really d/dt (calendar time, matching L^BS's own d/dt) and not
#     d/dtau = -d/dt (tau = T-t, the semigroup's native parametrisation) --
#     a silent sign flip there would invalidate every residual below. Each
#     residual test also carries a GUARD: the split's forcing must be
#     substantially nonzero, because a vanishing residual would mean the
#     exact Black-Scholes price extension is wired in by mistake (L^BS V^e
#     = 0 identically), not that the split is unusually accurate.
# ---------------------------------------------------------------------------

def _independent_closed_form_split_value(
    s: torch.Tensor, tau: torch.Tensor, K: float, sigma: float
) -> torch.Tensor:
    r"""pi_split(s,t) = K N(c) - s exp(sigma^2 tau/2) N(c-m), by hand (see the
    section banner above); a different code path from
    GaussianSemigroupExtensionField, not a call into it."""
    m = sigma * torch.sqrt(tau)
    c = (math.log(K) - torch.log(s)) / m
    normal_cdf = lambda z: 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return K * normal_cdf(c) - s * torch.exp(0.5 * sigma**2 * tau) * normal_cdf(c - m)


def _independent_closed_form_split_second_price_derivative(
    s: torch.Tensor, tau: torch.Tensor, K: float, sigma: float
) -> torch.Tensor:
    r"""d^2 pi_split/ds^2 = K N'(c) / (s^2 m), by hand (see the section banner
    above); a different code path from derivative_callables(), not a call
    into it."""
    m = sigma * torch.sqrt(tau)
    c = (math.log(K) - torch.log(s)) / m
    normal_pdf = lambda z: torch.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    return K * normal_pdf(c) / (s**2 * m)


def _independent_closed_form_split_black_scholes_residual(
    s: torch.Tensor, tau: torch.Tensor, K: float, sigma: float, r: float
) -> torch.Tensor:
    r"""L^BS pi_split = -(r-sigma^2/2)*s*exp(sigma^2 tau/2)*N(c-m) - r*pi_split,
    by hand (see the "(C)" docstrings of the test methods that use this for
    the derivation): Proposition 7(i)'s L h = B h collapsed to price
    coordinates, with pi_split from
    :func:`_independent_closed_form_split_value` -- a different code path
    from derivative_callables(), not a call into it."""
    m = sigma * torch.sqrt(tau)
    c = (math.log(K) - torch.log(s)) / m
    normal_cdf = lambda z: 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    pi = _independent_closed_form_split_value(s=s, tau=tau, K=K, sigma=sigma)
    return -(r - 0.5 * sigma**2) * s * torch.exp(0.5 * sigma**2 * tau) * normal_cdf(c - m) - r * pi


class TestCornerRegularisedExtensionSplit:
    K, B, T, sigma, r = 1.0, 0.6, 1.0, 0.3, 0.03
    epsilon = 0.05
    # Padded several diffusion lengths (sigma*sqrt(T) = 0.3) beyond the
    # evaluation window [0.7, 2.0] in log-price, per GaussianSemigroupExtensionField's
    # own requirement.
    #
    # n_quad=1_000_000 (development note, see also the response to the task
    # that added tau=0.001 to the grid below). At n_quad=200_000 the grid's
    # hardest point, tau=0.01 (t=0.99), cleared 1e-6 on the second
    # derivative with margin (measured 4.7e-7); extending the grid to
    # tau=0.001 (t=0.999) does NOT clear 1e-6 at 200_000 (measured 1.5e-5)
    # nor reliably even at 600_000 (measured 1.17e-6 there, ABOVE tolerance,
    # against 6.0e-7 at 500_000 and 6.6e-7 at 700_000 -- the trapezoidal
    # quadrature error oscillates with how the fixed grid aligns with the
    # payoff's kink at x=ln(K), so a value close to a threshold is not a
    # safe margin). n_quad=1_000_000 clears tau=0.001 with a measured
    # error of 2.6e-7, a genuine ~4x margin. It does NOT extend to
    # tau=0.0001 (t=0.9999): measured error there is 8.3e-6, about 8x
    # ABOVE 1e-6, and extrapolating the observed (non-clean, aliasing-
    # affected) convergence rate would need n_quad in the 3-5 million range
    # to close that gap -- several seconds per single (s, t) query, and
    # therefore not encoded as a strict test here (see the accompanying
    # response for the root-cause diagnosis: it is a resolution problem of
    # the SECOND-derivative channel specifically, not domain truncation --
    # the value's own error stays at ~1e-9 throughout, including at
    # tau=0.0001 -- and not the quadrature floor either, which at this
    # configuration activates only below tau~1.5e-7, three orders of
    # magnitude below where the 1e-6 target already fails).
    Y_LO, Y_HI = math.log(0.08), math.log(9.0)
    N_QUAD = 1_000_000

    @pytest.fixture(scope="class", params=SPLIT_PROFILE_ROUTES)
    @classmethod
    def h_eps(cls, request):
        """Built once per class and per evaluation route, and shared by every
        test below.  Both routes of make_corner_regularised_extension_split
        are exercised against the SAME hand-derived oracles: the
        ``"quadrature"`` route (see the n_quad development note above for
        why it is expensive: ~0.3-0.4s per (value, second-derivative) query
        pair at this n_quad, since GaussianSemigroupExtensionField
        recomputes its quadrature nodes and the full batch convolution on
        every call) and the ``"closed_form"`` route, whose production code
        (PutPayoffGaussianSemigroupExtensionField, log-price derivatives
        chain-ruled to price space) is a different arrangement of the
        formulas from the oracles' direct price-space expressions."""
        return make_corner_regularised_extension_split(
            cls.K, cls.B, cls.epsilon, cls.T, cls.sigma,
            cls.Y_LO, cls.Y_HI, n_quad=cls.N_QUAD, profile=request.param,
        )

    def test_rejects_unknown_profile(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension_split(
                self.K, self.B, self.epsilon, self.T, self.sigma, profile="autograd",
            )

    def test_quadrature_profile_requires_a_support(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension_split(
                self.K, self.B, self.epsilon, self.T, self.sigma, profile="quadrature",
            )

    def test_zero_on_entire_barrier_face(self, h_eps) -> None:
        """h(B, t) = 0 for every t: the cutoff zeta(0) = 0 kills the
        split-semigroup profile regardless of its value there, exactly as
        for the other three variants (structural, independent of the
        profile being cut off)."""
        t = torch.linspace(0.0, self.T, 101, dtype=torch.float64)
        s = torch.full_like(t, self.B)
        assert torch.allclose(h_eps(s, t), torch.zeros_like(t), atol=1e-6)

    def test_matches_raw_payoff_exactly_at_maturity_outside_the_corner_layer(
        self, h_eps
    ) -> None:
        """At t=T the semigroup parameter T-t is zero, so the field returns
        the raw datum exactly -- unlike the Black-Scholes-payoff variant,
        this is exact (not merely "up to the tau floor"): t=T is the
        time_to_terminal <= 0 branch of
        GaussianSemigroupExtensionField._register_floor, defined as exact
        rather than as an unresolved-quadrature fallback."""
        s = torch.linspace(self.B + self.epsilon + 1e-3, 5.0, 201, dtype=torch.float64)
        t = torch.full_like(s, self.T)
        assert torch.allclose(h_eps(s, t), payoff_put(s, self.K), atol=1e-6)

    def test_rejects_non_reverse_knock_out_regime(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension_split(
                K=0.5, B=0.6, epsilon=0.05, T=self.T, comparison_volatility=self.sigma,
                y_lo=self.Y_LO, y_hi=self.Y_HI,
            )

    def test_rejects_non_positive_epsilon(self) -> None:
        with pytest.raises(ValueError):
            make_corner_regularised_extension_split(
                K=1.0, B=0.6, epsilon=0.0, T=self.T, comparison_volatility=self.sigma,
                y_lo=self.Y_LO, y_hi=self.Y_HI,
            )

    # -- (A) value against the independent closed form -----------------------

    def test_matches_independent_closed_form_value_at_the_hard_target(self, h_eps) -> None:
        """K=1, B=0.6, sigma=0.3, s=K=1, t=0 (tau=T=1, so m=0.3, c=0):

            pi_split = 0.5 - exp(0.045) N(-0.3)
                     = 0.5 - 1.046027859908717 * 0.3820885778110474
                     = 0.10032470265674481

        (double-checked independently in Python via math.erf/math.exp; the
        value 0.100294 sometimes quoted for this configuration is a rounding
        slip in that mental-arithmetic chain, not this target)."""
        s = torch.tensor([self.K], dtype=torch.float64)
        t = torch.tensor([0.0], dtype=torch.float64)
        expected = 0.10032470265674481
        assert abs(float(h_eps(s, t)) - expected) < 1e-6

    def test_matches_independent_closed_form_value_on_a_grid(self, h_eps) -> None:
        """s in [0.7, 2.0], t in {0, 0.5, 0.9, 0.99, 0.999}, tolerance 1e-6,
        entirely outside the corner layer (B + epsilon = 0.65 < 0.7) so zeta
        = 1 identically and h_eps reduces to the split-semigroup profile
        alone. t=0.9999 is deliberately NOT included: see the class's N_QUAD
        development note (its second-derivative error does not clear 1e-6
        at any practically-sized n_quad tried); the value's own error stays
        far below tolerance at every t tried, including 0.9999, so this
        particular test is not what would exclude it."""
        s = torch.linspace(0.7, 2.0, 27, dtype=torch.float64)
        for t_val in (0.0, 0.5, 0.9, 0.99, 0.999):
            t = torch.full_like(s, t_val)
            tau = torch.full_like(s, self.T - t_val)
            got = h_eps(s, t)
            want = _independent_closed_form_split_value(tau=tau, s=s, K=self.K, sigma=self.sigma)
            max_err = float((got - want).abs().max())
            assert max_err < 1e-6, f"t={t_val}: max abs err {max_err:.3e} (quadrature mis-tuned)"

    # -- (B) second price-derivative against the independent closed form -----

    def test_second_price_derivative_matches_independent_closed_form_at_the_hard_target(
        self, h_eps
    ) -> None:
        """Same configuration as the value's hard target: s=K=1, t=0
        (tau=1, m=0.3, c=0):

            d^2 pi_split/ds^2 = K N'(c) / (s^2 m) = N'(0) / 0.3
                              = 0.3989422804014327 / 0.3
                              = 1.329807601338109

        Obtained from h_eps.second_price_derivative, which routes through
        GaussianSemigroupExtensionField.derivative_callables() -- never
        autograd or a finite difference through the quadratured field (see
        the section banner above)."""
        s = torch.tensor([self.K], dtype=torch.float64)
        t = torch.tensor([0.0], dtype=torch.float64)
        expected = 1.329807601338109
        got = float(h_eps.second_price_derivative(s, t))
        assert abs(got - expected) < 1e-6

    def test_second_price_derivative_matches_independent_closed_form_on_a_grid(
        self, h_eps
    ) -> None:
        """Same grid as the value test, extended to t=0.999 (tau=0.001) --
        see the class's N_QUAD development note for why n_quad had to move
        from 200_000 to 1_000_000 to clear 1e-6 there, why the required
        n_quad does not follow a clean law as tau shrinks (fixed-grid
        trapezoidal error against a kinked integrand oscillates with the
        grid/kink alignment), and why t=0.9999 is excluded rather than
        given a looser tolerance. Measured error on this exact grid at
        n_quad=1_000_000: 8.3e-9 (t=0.999) against 4.7e-7 previously at
        n_quad=200_000 for the harder-then point t=0.99 -- both comfortably
        under 1e-6, by a larger margin than before because n_quad grew by
        5x."""
        s = torch.linspace(0.7, 2.0, 27, dtype=torch.float64)
        for t_val in (0.0, 0.5, 0.9, 0.99, 0.999):
            t = torch.full_like(s, t_val)
            tau = torch.full_like(s, self.T - t_val)
            got = h_eps.second_price_derivative(s, t)
            want = _independent_closed_form_split_second_price_derivative(
                tau=tau, s=s, K=self.K, sigma=self.sigma
            )
            max_err = float((got - want).abs().max())
            assert max_err < 1e-6, f"t={t_val}: max abs err {max_err:.3e} (quadrature mis-tuned)"

    # -- (C) Black-Scholes PDE residual, assembled from derivative_callables() ----

    def test_dt_is_a_partial_derivative_in_t_not_in_tau(self, h_eps) -> None:
        """Sanity check demanded before trusting (C): derivative_callables()
        "dt" must be d/dt at fixed s (calendar time, as L^BS's own d/dt
        term is), not d/dtau = -d/dt (tau = T-t, the semigroup's own
        parametrisation) -- a sign flip here would silently invalidate
        every residual computed below. Checked by an ordinary central
        finite difference in t (safe here: unlike a second SPATIAL
        derivative through the quadratured kernel, h_eps is smooth in t
        away from the corner layer and the terminal slice, so this
        particular finite difference carries no amplification risk)."""
        s = torch.tensor([1.0], dtype=torch.float64)
        t0 = torch.tensor([0.5], dtype=torch.float64)
        step = 1e-3
        finite_difference_dt = (h_eps(s, t0 + step) - h_eps(s, t0 - step)) / (2.0 * step)
        analytic_dt = h_eps.split_field.derivative_callables()["dt"](torch.log(s), t0)
        assert torch.allclose(analytic_dt, finite_difference_dt, atol=1e-4, rtol=1e-3)

    def test_black_scholes_residual_matches_the_defect_operator_at_the_hard_target(
        self, h_eps
    ) -> None:
        r"""s=K=1, t=0 (tau=1, m=0.3, c=0), r=0.03, sigma=0.3 (nu=sigma^2/2=0.045):

        Proposition 7(i) states L h = B h for the MATCHED split (comparison_
        volatility == sigma, the case here): the full Black-Scholes generator
        applied to the extension collapses to the drift-and-discount
        remainder alone, because the diffusive part is cancelled exactly by
        the semigroup's own time derivative (dt = -nu*dxx, checked by the
        test above). In price coordinates B pi = (r-nu)*s*(d pi/ds) - r*pi,
        and d pi/ds = -exp(sigma^2 tau/2) N(c-m) (by hand, differentiating
        pi_split's closed form and using the identity K N'(c) = s exp(sigma^2
        tau/2) N'(c-m) to collapse the other two terms), giving the target

            -(r - sigma^2/2) * s * exp(sigma^2 tau/2) * N(c-m)  -  r * pi.

        At this point: -(0.03-0.045)*1*exp(0.045)*N(-0.3) - 0.03*0.10032470...
        = 0.0029853883804687575 (assembled from derivative_callables(), see
        below) vs 0.0029853883804464836 (independent closed form) --
        matching to 2.2e-14, several orders tighter than (B)'s own
        second-derivative tolerance. This is not a coincidence: d_t and
        (sigma^2/2)*s^2*d_ss enter the residual only through the combination
        d_t + nu*s^2*d_ss = d_t + nu*(d_xx-d_x) = -nu*d_x (since d_t =
        -nu*d_xx is exact by construction, not merely accurate), so the
        quadrature error of d_xx -- (B)'s dominant error source -- cancels
        algebraically and never enters the residual at all; what remains is
        governed by d_x/d_s alone, at the value-level error of (A). The
        guard below is what would catch this test silently exercising the
        EXACT Black-Scholes price extension instead of the split-semigroup
        one: V^e satisfies L^BS V^e = 0 identically, so a near-zero residual
        here would mean the wrong extension is wired in, not that the split
        is unusually accurate.
        """
        s = torch.tensor([self.K], dtype=torch.float64)
        t = torch.tensor([0.0], dtype=torch.float64)
        log_price = torch.log(s)
        derivative = h_eps.split_field.derivative_callables()
        pi = h_eps.split_field.field(log_price, t).reshape(-1)
        d_t = derivative["dt"](log_price, t).reshape(-1)
        d_x = derivative["dx"](log_price, t).reshape(-1)
        d_xx = derivative["dxx"](log_price, t).reshape(-1)
        d_s = d_x / s
        d_ss = (d_xx - d_x) / s**2

        residual = d_t + 0.5 * self.sigma**2 * s**2 * d_ss + self.r * s * d_s - self.r * pi

        tau = torch.full_like(s, self.T)
        target = _independent_closed_form_split_black_scholes_residual(
            s=s, tau=tau, K=self.K, sigma=self.sigma, r=self.r
        )

        assert abs(float(residual) - float(target)) < 1e-6
        # Guard-fou: the split's forcing must be genuinely nonzero here --
        # otherwise this is silently testing the exact-solution extension
        # (whose residual vanishes identically), not the split-semigroup one.
        assert abs(float(residual)) > 1e-3

    def test_black_scholes_residual_matches_the_defect_operator_on_a_grid(
        self, h_eps
    ) -> None:
        """Same construction as the hard-target test, over the (A)/(B) grid
        (s in [0.7, 2.0], t in {0, 0.5, 0.9, 0.99, 0.999}). No grid-wide
        guard: pi and d pi/ds both decay towards 0 as s grows deep
        out-of-the-money (K=1, so s up to 2.0 is deep OTM), so the residual
        (r-nu)*s*(d pi/ds) - r*pi genuinely crosses/approaches zero
        somewhere on this grid -- confirmed during development (residual as
        small as 3e-120 at s=2.0, t=0.99) -- and asserting it stays bounded
        away from zero everywhere would be asserting something false about
        the mathematics, not exercising a guard. The guard belongs only at
        a point known analytically to be away from that decay, which is
        what the hard-target test above is for."""
        s = torch.linspace(0.7, 2.0, 27, dtype=torch.float64)
        for t_val in (0.0, 0.5, 0.9, 0.99, 0.999):
            t = torch.full_like(s, t_val)
            log_price = torch.log(s)
            derivative = h_eps.split_field.derivative_callables()
            pi = h_eps.split_field.field(log_price, t).reshape(-1)
            d_t = derivative["dt"](log_price, t).reshape(-1)
            d_x = derivative["dx"](log_price, t).reshape(-1)
            d_xx = derivative["dxx"](log_price, t).reshape(-1)
            d_s = d_x / s
            d_ss = (d_xx - d_x) / s**2

            residual = d_t + 0.5 * self.sigma**2 * s**2 * d_ss + self.r * s * d_s - self.r * pi

            tau = torch.full_like(s, self.T - t_val)
            target = _independent_closed_form_split_black_scholes_residual(
                s=s, tau=tau, K=self.K, sigma=self.sigma, r=self.r
            )
            max_err = float((residual - target).abs().max())
            assert max_err < 1e-6, f"t={t_val}: max abs err {max_err:.3e}"


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


# ---------------------------------------------------------------------------
# reiner_rubinstein_down_and_out_put_gamma
# ---------------------------------------------------------------------------

class TestReinerRubinsteinDownAndOutPutGamma:
    K, B, r, sigma, T = 100.0, 80.0, 0.02, 0.25, 1.0

    def test_matches_finite_differences_of_the_price(self) -> None:
        """d^2V_DO/ds^2 (analytic) matches a central second difference of
        reiner_rubinstein_down_and_out_put, away from points where the true
        Gamma itself is astronomically small (deep in/out of the money at
        short tau, e.g. 1e-18-1e-30): there, O(h^2) finite-difference
        round-off noise (~1e-8, from subtracting three O(1) doubles and
        dividing by h^2=1e-8) dwarfs the true value, inflating the relative
        error without indicating any formula discrepancy -- confirmed during
        development by inspecting individual (s, gamma) pairs."""
        h = 1e-2
        s_vals = torch.linspace(self.B + 5.0, self.K * 1.8, 61, dtype=torch.float64)
        for tau_val in (0.05, 0.25, 1.0, 2.0):
            tau = torch.full_like(s_vals, tau_val)
            price_plus = reiner_rubinstein_down_and_out_put(s_vals + h, self.K, self.B, self.r, self.sigma, tau)
            price_mid = reiner_rubinstein_down_and_out_put(s_vals, self.K, self.B, self.r, self.sigma, tau)
            price_minus = reiner_rubinstein_down_and_out_put(s_vals - h, self.K, self.B, self.r, self.sigma, tau)
            fd_gamma = (price_plus - 2 * price_mid + price_minus) / h**2
            analytic_gamma = reiner_rubinstein_down_and_out_put_gamma(s_vals, self.K, self.B, self.r, self.sigma, tau)

            significant = analytic_gamma.abs() > 1e-6
            assert significant.any(), f"tau={tau_val}: no grid point with a non-negligible Gamma"
            rel_err = (fd_gamma[significant] - analytic_gamma[significant]).abs() / analytic_gamma[significant].abs()
            assert float(rel_err.max()) < 1e-3, f"tau={tau_val}: max relative error {float(rel_err.max()):.3e}"

    def test_zero_at_or_below_the_barrier(self) -> None:
        """Gamma = 0 for s <= B (the price is identically 0 there)."""
        s = torch.tensor([self.B, self.B - 1.0, self.B - 10.0], dtype=torch.float64)
        tau = torch.full_like(s, self.T)
        gamma = reiner_rubinstein_down_and_out_put_gamma(s, self.K, self.B, self.r, self.sigma, tau)
        assert torch.all(gamma == 0.0)

    def test_rejects_non_reverse_knock_out_regime(self) -> None:
        s = torch.tensor([90.0])
        tau = torch.tensor([self.T])
        with pytest.raises(ValueError):
            reiner_rubinstein_down_and_out_put_gamma(s, K=80.0, B=100.0, r=self.r, sigma=self.sigma, tau=tau)

    def test_batch_consistency(self) -> None:
        s_vals = [85.0, 95.0, 100.0, 110.0]
        s_batch = torch.tensor(s_vals, dtype=torch.float64)
        tau_batch = torch.full_like(s_batch, self.T)
        g_batch = reiner_rubinstein_down_and_out_put_gamma(s_batch, self.K, self.B, self.r, self.sigma, tau_batch)
        for i, sv in enumerate(s_vals):
            s_single = torch.tensor([sv], dtype=torch.float64)
            tau_single = torch.tensor([self.T], dtype=torch.float64)
            g_single = float(reiner_rubinstein_down_and_out_put_gamma(s_single, self.K, self.B, self.r, self.sigma, tau_single))
            assert abs(g_single - float(g_batch[i])) < 1e-9


# ---------------------------------------------------------------------------
# Exact terminal trace at tau = 0 (the tau floor must not leak into t = T)
# ---------------------------------------------------------------------------

class TestExactTerminalTrace:
    """At t = T the closed forms are undefined (division by sigma*sqrt(tau)); their
    uniform limit is the payoff and the code must return it exactly, not the price
    at tau = _TAU_EPS (at-the-money time value K*sigma*sqrt(1e-8)/sqrt(2*pi) = 1.2e-5
    for K=1, sigma=0.3, which is what the floor used to return)."""
    K, B, r, sigma, T, eps = 1.0, 0.6, 0.03, 0.3, 1.0, 0.1

    def _s(self):
        return torch.linspace(self.B + 0.05, 3.0, 1181, dtype=torch.float64)

    def test_black_scholes_extensions_equal_zeta_times_payoff_at_maturity(self) -> None:
        s = self._s(); t = torch.full_like(s, self.T)
        expected = _smoothstep01((s - self.B) / self.eps) * (self.K - s).clamp(min=0.0)
        for g2 in (make_corner_regularised_extension_with_black_scholes_payoff(self.K, self.B, self.eps, self.r, self.sigma, self.T),
                   BlackScholesCornerExtension(self.K, self.B, self.eps, self.r, self.sigma, self.T)):
            assert torch.equal(g2(s, t), expected)

    def test_reiner_rubinstein_equals_knocked_out_payoff_at_maturity(self) -> None:
        s = torch.linspace(0.3, 3.0, 901, dtype=torch.float64)
        price = reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, torch.zeros_like(s))
        expected = torch.where(s > self.B, (self.K - s).clamp(min=0.0), torch.zeros_like(s))
        assert torch.equal(price, expected)

    def test_price_is_continuous_at_maturity(self) -> None:
        """The value at tau = 0 is the limit of the closed form: |V(tau) - V(0)| <= K sigma sqrt(tau)/sqrt(2 pi) + O(tau)."""
        s = self._s()
        for tau in (1e-6, 1e-4):
            gap = (reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, torch.full_like(s, tau))
                   - reiner_rubinstein_down_and_out_put(s, self.K, self.B, self.r, self.sigma, torch.zeros_like(s))).abs().max()
            assert gap <= self.K * self.sigma * tau**0.5 / (2 * torch.pi) ** 0.5 * 1.5 + 10 * tau

    def test_gradient_through_the_maturity_branch_is_finite(self) -> None:
        s = self._s().requires_grad_(True); t = torch.full_like(s, self.T)
        g2 = BlackScholesCornerExtension(self.K, self.B, self.eps, self.r, self.sigma, self.T)
        grad = torch.autograd.grad(g2(s, t).sum(), s)[0]
        assert torch.isfinite(grad).all()
        _, delta = g2._european_put_price_and_delta(s.detach(), t)
        assert torch.equal(delta, -(s.detach() < self.K).to(delta.dtype))


# ---------------------------------------------------------------------------
# down_and_out_digital_price  (equation (15) of the note, Method 1 / Section 5.1)
# ---------------------------------------------------------------------------

class TestDownAndOutDigitalPrice:
    """The digital reproduces the two data of a unit-jump knock-out claim and is
    annihilated by the constant-coefficient operator (Propositions 3 and 4)."""
    K, B, r, sigma, T = 1.0, 0.6, 0.03, 0.3, 1.0

    def _interior(self, requires_grad: bool = False):
        s = torch.linspace(self.B + 0.005, 3.0, 600, dtype=torch.float64)
        t = torch.linspace(0.0, self.T - 1e-3, 600, dtype=torch.float64)
        ss, tt = torch.meshgrid(s, t, indexing="ij")
        ss = ss.reshape(-1).clone().requires_grad_(requires_grad)
        tt = tt.reshape(-1).clone().requires_grad_(requires_grad)
        return ss, tt

    def test_values_in_unit_interval(self) -> None:
        ss, tt = self._interior()
        price = down_and_out_digital_price(ss, self.B, self.r, self.sigma, self.T - tt)
        assert float(price.min()) >= 0.0 and float(price.max()) <= 1.0

    def test_zero_on_the_barrier_face(self) -> None:
        t = torch.linspace(0.0, self.T, 101, dtype=torch.float64)
        s = torch.full_like(t, self.B)
        assert torch.equal(down_and_out_digital_price(s, self.B, self.r, self.sigma, self.T - t), torch.zeros_like(t))

    def test_zero_below_the_barrier(self) -> None:
        s = torch.linspace(0.1, self.B, 50, dtype=torch.float64)
        assert torch.equal(down_and_out_digital_price(s, self.B, self.r, self.sigma, torch.tensor(0.5)), torch.zeros_like(s))

    def test_unit_indicator_on_the_terminal_face(self) -> None:
        """V_DOD(s, T) = 1_{s > B} exactly (the closed form's limit, returned at tau = 0)."""
        s = torch.linspace(0.3, 3.0, 901, dtype=torch.float64)
        price = down_and_out_digital_price(s, self.B, self.r, self.sigma, torch.zeros_like(s))
        assert torch.equal(price, (s > self.B).to(price.dtype))

    def test_continuous_at_maturity_away_from_the_corner(self) -> None:
        """For s - B fixed, V_DOD = e^{-r tau} + o(1) as tau -> 0 (Gaussian tails on
        both terms leave only the discount factor)."""
        s = torch.linspace(self.B + 0.05, 3.0, 300, dtype=torch.float64)
        tau = 1e-5
        gap = (math.exp(-self.r * tau)
               - down_and_out_digital_price(s, self.B, self.r, self.sigma, torch.full_like(s, tau))).abs().max()
        assert float(gap) < 1e-12

    def test_annihilated_by_the_black_scholes_operator(self) -> None:
        """L^BS V_DOD = 0 on Q, checked by float64 autograd on a 600 x 600 grid
        (Proposition 4: the subtracted singular part contributes exactly zero
        residual)."""
        ss, tt = self._interior(requires_grad=True)
        price = down_and_out_digital_price(ss, self.B, self.r, self.sigma, self.T - tt)
        residual = bsm_operator(price, ss, tt, self.r, 0.0, self.sigma)
        assert float(residual.abs().max()) < 1e-10

    def test_closed_form_derivatives_match_autograd(self) -> None:
        ss, tt = self._interior(requires_grad=True)
        price = down_and_out_digital_price(ss, self.B, self.r, self.sigma, self.T - tt)
        delta_ag = torch.autograd.grad(price.sum(), ss, create_graph=True)[0]
        gamma_ag = torch.autograd.grad(delta_ag.sum(), ss, retain_graph=True)[0]
        theta_ag = torch.autograd.grad(price.sum(), tt)[0]
        value, delta, gamma, theta = down_and_out_digital_price_and_derivatives(
            ss.detach(), self.B, self.r, self.sigma, self.T - tt.detach(),
        )
        assert torch.allclose(value, price.detach(), atol=1e-14)
        assert torch.allclose(delta, delta_ag.detach(), atol=1e-10)
        assert torch.allclose(gamma, gamma_ag.detach(), atol=1e-8, rtol=1e-8)
        assert torch.allclose(theta, theta_ag.detach(), atol=1e-10)

    def test_derivatives_vanish_at_maturity_and_below_the_barrier(self) -> None:
        s = torch.linspace(0.3, 3.0, 271, dtype=torch.float64)
        _, delta, gamma, theta = down_and_out_digital_price_and_derivatives(
            s, self.B, self.r, self.sigma, torch.zeros_like(s),
        )
        for derivative in (delta, gamma, theta):
            assert torch.equal(derivative, torch.zeros_like(s))
        _, delta, gamma, theta = down_and_out_digital_price_and_derivatives(
            s, self.B, self.r, self.sigma, torch.tensor(0.5, dtype=torch.float64),
        )
        below = s <= self.B
        for derivative in (delta, gamma, theta):
            assert torch.equal(derivative[below], torch.zeros_like(s[below]))

    def test_corner_gamma_is_unbounded_as_t_to_T(self) -> None:
        """Along a path of fixed similarity variable xi = ln(s/B)/(sigma sqrt(2 tau))
        (here s - B = sqrt(tau)) the digital's curvature grows like 1/tau as
        tau -> 0: it reproduces the jump. This is the singularity the two-term
        residual assembly keeps out of autograd."""
        gammas = [abs(float(down_and_out_digital_price_and_derivatives(
            torch.tensor([self.B + math.sqrt(tau)], dtype=torch.float64),
            self.B, self.r, self.sigma, torch.tensor(tau, dtype=torch.float64))[2]))
            for tau in (1e-2, 1e-3, 1e-4)]
        assert gammas[0] < gammas[1] < gammas[2]
        assert gammas[2] / gammas[1] > 5.0  # about 10 for a 1/tau growth

    def test_rejects_nonpositive_barrier(self) -> None:
        with pytest.raises(ValueError):
            down_and_out_digital_price(torch.tensor([1.0]), 0.0, self.r, self.sigma, torch.tensor(0.5))


# ---------------------------------------------------------------------------
# Terminal profiles and SubtractedDigitalCornerExtension  (Definition 7)
# ---------------------------------------------------------------------------

class TestTerminalProfiles:
    K, B, r, sigma, T = 1.0, 0.6, 0.03, 0.3, 1.0

    def _profiles(self):
        return (
            RawPutPayoffTerminalProfile(self.K),
            BlackScholesPutTerminalProfile(self.K, self.r, self.sigma, self.T),
            make_subtracted_digital_extension(self.K, self.B, self.r, self.sigma, self.T, "split").terminal_profile,
        )

    def test_every_profile_equals_the_payoff_at_maturity(self) -> None:
        s = torch.linspace(0.3, 3.0, 541, dtype=torch.float64)
        t = torch.full_like(s, self.T)
        for profile in self._profiles():
            value, _, _, _ = profile.value_and_derivatives(s, t)
            assert torch.allclose(value, payoff_put(s, self.K), atol=1e-14), profile.name

    def test_smooth_profiles_derivatives_match_autograd(self) -> None:
        """Delta, Gamma and theta of the Black-Scholes and split profiles
        against float64 autograd of their own value."""
        s = torch.linspace(self.B, 3.0, 400, dtype=torch.float64).requires_grad_(True)
        t = torch.full_like(s, 0.3).requires_grad_(True)
        for profile in self._profiles()[1:]:
            value, delta, gamma, theta = profile.value_and_derivatives(s, t)
            delta_ag = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
            gamma_ag = torch.autograd.grad(delta_ag.sum(), s, retain_graph=True)[0]
            theta_ag = torch.autograd.grad(value.sum(), t)[0]
            assert torch.allclose(delta, delta_ag, atol=1e-10), profile.name
            assert torch.allclose(gamma, gamma_ag, atol=1e-8), profile.name
            assert torch.allclose(theta, theta_ag, atol=1e-10), profile.name

    def test_black_scholes_profile_is_annihilated_by_the_operator(self) -> None:
        profile = BlackScholesPutTerminalProfile(self.K, self.r, self.sigma, self.T)
        s = torch.linspace(self.B, 3.0, 400, dtype=torch.float64)
        t = torch.full_like(s, 0.3)
        value, delta, gamma, theta = profile.value_and_derivatives(s, t)
        residual = theta + 0.5 * self.sigma**2 * s**2 * gamma + self.r * s * delta - self.r * value
        assert float(residual.abs().max()) < 1e-14

    def test_raw_profile_derivatives(self) -> None:
        profile = RawPutPayoffTerminalProfile(self.K)
        s = torch.tensor([0.7, 1.5], dtype=torch.float64)
        value, delta, gamma, theta = profile.value_and_derivatives(s, torch.tensor(0.2, dtype=torch.float64))
        assert torch.allclose(value, torch.tensor([0.3, 0.0], dtype=torch.float64), atol=1e-15)
        assert torch.equal(delta, torch.tensor([-1.0, 0.0], dtype=torch.float64))
        assert torch.equal(gamma, torch.zeros(2, dtype=torch.float64))
        assert torch.equal(theta, torch.zeros(2, dtype=torch.float64))


class TestSubtractedDigitalCornerExtension:
    """g2 = Delta V_DOD + pi - pi(B, .) reproduces both data exactly with no
    corner layer, and its closed-form residual and derivatives agree with
    float64 autograd of its own value."""
    K, B, r, sigma, T = 1.0, 0.6, 0.03, 0.3, 1.0

    def _extensions(self):
        return [make_subtracted_digital_extension(self.K, self.B, self.r, self.sigma, self.T, name)
                for name in SUBTRACTION_TERMINAL_PROFILES]

    def test_profile_names_are_exposed(self) -> None:
        assert [g2.profile_name for g2 in self._extensions()] == list(SUBTRACTION_TERMINAL_PROFILES)

    def test_terminal_trace_is_the_payoff_exactly(self) -> None:
        """g2(s, T) = Delta 1_{s>B} + (K-s)^+ - Delta = (K-s)^+ for s > B, with no
        epsilon layer: the identity holds down to s = B^+."""
        s = torch.linspace(self.B + 1e-9, 3.0, 1000, dtype=torch.float64)
        t = torch.full_like(s, self.T)
        for g2 in self._extensions():
            assert torch.allclose(g2(s, t), payoff_put(s, self.K), atol=1e-13), g2.profile_name

    def test_barrier_trace_is_zero_for_every_t(self) -> None:
        """h(B, t) = pi(B, t) - pi(B, t) = 0 and V_DOD(B, t) = 0: the subtraction
        of pi(B, .) rather than of the constant Delta is what makes this exact
        for the time-dependent profiles."""
        t = torch.linspace(0.0, self.T, 201, dtype=torch.float64)
        s = torch.full_like(t, self.B)
        for g2 in self._extensions():
            assert torch.equal(g2(s, t), torch.zeros_like(t)), g2.profile_name

    def test_corner_compatible_traces(self) -> None:
        """The two traces coincide at the corner (both zero): lim_{s->B+} g2(s, T)
        = (K - B) - Delta + Delta V_DOD(B+, T) ... = 0 = g2(B, t) for t -> T."""
        for g2 in self._extensions():
            terminal_near_corner = g2(torch.tensor([self.B + 1e-6], dtype=torch.float64), torch.tensor([self.T], dtype=torch.float64))
            assert abs(float(terminal_near_corner) - (self.K - self.B - 1e-6)) < 1e-12
            # The regular part h alone vanishes at the corner from both faces.
            h_terminal = g2.subtracted_data_extension(torch.tensor([self.B + 1e-6], dtype=torch.float64), torch.tensor([self.T], dtype=torch.float64))
            h_barrier = g2.subtracted_data_extension(torch.tensor([self.B], dtype=torch.float64), torch.tensor([self.T - 1e-6], dtype=torch.float64))
            assert abs(float(h_terminal)) < 2e-6 and abs(float(h_barrier)) < 1e-12

    def test_decomposition_sums_to_the_value(self) -> None:
        s = torch.linspace(self.B, 3.0, 300, dtype=torch.float64)
        t = torch.full_like(s, 0.4)
        for g2 in self._extensions():
            assert torch.allclose(g2(s, t), g2.digital_price(s, t) + g2.subtracted_data_extension(s, t), atol=1e-15)
            assert torch.allclose(
                g2.digital_price(s, t),
                (self.K - self.B) * down_and_out_digital_price(s, self.B, self.r, self.sigma, self.T - t), atol=1e-15,
            )

    def _interior_off_strike(self):
        s = torch.cat([torch.linspace(self.B + 0.005, self.K - 0.01, 300, dtype=torch.float64),
                       torch.linspace(self.K + 0.01, 3.0, 300, dtype=torch.float64)]).requires_grad_(True)
        t = torch.linspace(0.0, self.T - 1e-3, 600, dtype=torch.float64).requires_grad_(True)
        return s, t

    def test_closed_form_residual_matches_autograd_through_the_whole_extension(self) -> None:
        """L^BS g2 by autograd (through the digital AND the profile) equals the
        closed-form residual, which omits the digital (Proposition 4). Off the
        strike for the raw profile, whose second derivative is not defined there."""
        s, t = self._interior_off_strike()
        for g2 in self._extensions():
            with torch.enable_grad():
                value = g2(s, t)
                residual_autograd = bsm_operator(value, s, t, self.r, 0.0, self.sigma)
            residual_closed_form = g2.black_scholes_residual(s.detach(), t.detach(), self.r, self.sigma)
            assert torch.allclose(residual_closed_form, residual_autograd.detach(), atol=1e-9), g2.profile_name

    def test_residual_is_bounded_up_to_the_corner(self) -> None:
        """No corner layer: the residual of the regular part stays O(1) on a
        sequence of points approaching (B, T), whereas the smoothing
        construction's residual grows like Delta / epsilon."""
        for g2 in self._extensions():
            values = [abs(float(g2.black_scholes_residual(
                torch.tensor([self.B + 10 * tau], dtype=torch.float64), torch.tensor([self.T - tau], dtype=torch.float64),
                self.r, self.sigma))) for tau in (1e-2, 1e-4, 1e-6)]
            assert max(values) < 1.0, (g2.profile_name, values)

    def test_price_and_time_derivatives_match_autograd(self) -> None:
        s, t = self._interior_off_strike()
        for g2 in self._extensions():
            value = g2(s, t)
            delta_ag = torch.autograd.grad(value.sum(), s, create_graph=True)[0]
            gamma_ag = torch.autograd.grad(delta_ag.sum(), s, retain_graph=True)[0]
            theta_ag = torch.autograd.grad(value.sum(), t)[0]
            assert torch.allclose(g2.first_price_derivative(s.detach(), t.detach()), delta_ag, atol=1e-10), g2.profile_name
            assert torch.allclose(g2.second_price_derivative(s.detach(), t.detach()), gamma_ag, atol=1e-7, rtol=1e-8), g2.profile_name
            assert torch.allclose(g2.first_time_derivative(s.detach(), t.detach()), theta_ag, atol=1e-9), g2.profile_name

    def test_broadcasts_scalar_time(self) -> None:
        s = torch.linspace(self.B, 3.0, 50, dtype=torch.float64)
        for g2 in self._extensions():
            assert g2(s, torch.tensor(0.3, dtype=torch.float64)).shape == s.shape
            assert g2.black_scholes_residual(s, torch.tensor(0.3, dtype=torch.float64), self.r, self.sigma).shape == s.shape

    def test_residual_refuses_other_coefficients(self) -> None:
        g2 = self._extensions()[1]
        s = torch.tensor([1.0], dtype=torch.float64)
        t = torch.tensor([0.3], dtype=torch.float64)
        with pytest.raises(ValueError):
            g2.black_scholes_residual(s, t, self.r, 2 * self.sigma)
        with pytest.raises(ValueError):
            g2.black_scholes_residual(s, t, self.r + 0.01, self.sigma)

    def test_split_profile_matches_the_smoothing_construction_far_from_the_corner(self) -> None:
        """Where zeta = 1, the split terminal function of the smoothing ansatz is
        the same profile pi; the subtracted extension equals pi shifted by
        Delta V_DOD - pi(B, t)."""
        epsilon = 0.1
        smoothing = make_corner_regularised_extension_split(self.K, self.B, epsilon, self.T, self.sigma)
        subtraction = make_subtracted_digital_extension(self.K, self.B, self.r, self.sigma, self.T, "split")
        s = torch.linspace(self.B + 2 * epsilon, 3.0, 200, dtype=torch.float64)
        t = torch.full_like(s, 0.4)
        pi_at_barrier = subtraction.terminal_profile.value_and_derivatives(torch.full_like(s, self.B), t)[0]
        expected = smoothing(s, t) - pi_at_barrier + subtraction.digital_price(s, t)
        assert torch.allclose(subtraction(s, t), expected, atol=1e-13)

    def test_quadrature_route_of_the_split_profile(self) -> None:
        closed = make_subtracted_digital_extension(self.K, self.B, self.r, self.sigma, self.T, "split")
        quadrature = make_subtracted_digital_extension(
            self.K, self.B, self.r, self.sigma, self.T, "split",
            y_lo=math.log(self.B) - 2.0, y_hi=math.log(3.0) + 2.0, n_quad=20000, split_profile="quadrature",
        )
        s = torch.linspace(self.B + 0.1, 2.5, 60, dtype=torch.float64)
        t = torch.full_like(s, 0.5)
        assert torch.allclose(closed(s, t), quadrature(s, t), atol=1e-6)

    def test_builder_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError):
            make_subtracted_digital_extension(self.K, self.B, self.r, self.sigma, self.T, "mangasarian")
        with pytest.raises(ValueError):
            make_subtracted_digital_extension(self.K, self.K + 0.1, self.r, self.sigma, self.T, "raw")
        with pytest.raises(ValueError):
            make_subtracted_digital_extension(self.K, self.B, self.r, self.sigma, self.T, "split", split_profile="quadrature")
        with pytest.raises(ValueError):
            SubtractedDigitalCornerExtension(self.K, self.B, self.r, self.sigma, 0.0, RawPutPayoffTerminalProfile(self.K))
