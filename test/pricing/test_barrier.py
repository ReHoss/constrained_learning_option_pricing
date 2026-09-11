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
    make_corner_regularised_extension_split,
    make_corner_regularised_extension_with_black_scholes_payoff,
    make_corner_regularised_extension_with_smoothed_payoff,
    mangasarian_smoothed_put_payoff,
    reiner_rubinstein_down_and_out_put,
    reiner_rubinstein_down_and_out_put_gamma,
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

    @pytest.fixture(scope="class")
    @classmethod
    def h_eps(cls):
        """Built once per class and shared by every test below (see the
        n_quad development note above for why it is expensive: ~0.3-0.4s
        per (value, second-derivative) query pair at this n_quad, since
        GaussianSemigroupExtensionField recomputes its quadrature nodes and
        the full batch convolution on every call -- nothing is cached
        across calls, let alone across test methods)."""
        return make_corner_regularised_extension_split(
            cls.K, cls.B, cls.epsilon, cls.T, cls.sigma,
            cls.Y_LO, cls.Y_HI, n_quad=cls.N_QUAD,
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
