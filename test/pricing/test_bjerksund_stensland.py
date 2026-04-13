"""Tests for the Bjerksund-Stensland (2002) American option approximation.

Validates the one-step flat boundary put approximation against the tables
in the paper:

    P. Bjerksund and G. Stensland, *Closed form valuation of American
    options*, Discussion paper 2002/09, NHH (2002).

The paper reports values for the FLAT boundary approximation (one-step,
columns labelled 'p' in the tables). We compare against those.
"""
import math

import pytest
import torch

from learning_option_pricing.pricing.bjerksund_stensland import (
    bs2002_exercise_boundary,
    bs2002_put,
)


# ---------------------------------------------------------------------------
# Table 3: K=100, b=r (no dividends), flat boundary put prices 'p'
# ---------------------------------------------------------------------------
# Parameters: b=r=0.08, sigma=0.20, T=0.25
TABLE3_CASES_A = [
    # (S, expected_p)
    (80, 20.00),
    (90, 10.01),
    (100, 3.16),
    (110, 0.65),
    (120, 0.09),
]

# Parameters: b=r=0.12, sigma=0.20, T=0.25
TABLE3_CASES_B = [
    (80, 20.00),
    (90, 10.00),
    (100, 2.86),
    (110, 0.54),
    (120, 0.07),
]

# Parameters: b=r=0.08, sigma=0.40, T=0.25
TABLE3_CASES_C = [
    (80, 20.28),
    (90, 12.48),
    (100, 7.04),
    (110, 3.66),
    (120, 1.77),
]

# Parameters: b=r=0.08, sigma=0.20, T=0.50
TABLE3_CASES_D = [
    (80, 20.00),
    (90, 10.24),
    (100, 4.11),
    (110, 1.37),
    (120, 0.39),
]


class TestTable3NoDividends:
    """Table 3: b=r (no dividends), put approximation."""

    K = 100.0

    @pytest.mark.parametrize("S,expected", TABLE3_CASES_A)
    def test_r008_s020_T025(self, S: float, expected: float) -> None:
        s = torch.tensor([float(S)])
        tau = torch.tensor([0.25])
        p = float(bs2002_put(s, self.K, 0.08, 0.20, tau, q=0.0))
        assert abs(p - expected) < 0.12, f"S={S}: got {p:.4f}, expected {expected}"

    @pytest.mark.parametrize("S,expected", TABLE3_CASES_B)
    def test_r012_s020_T025(self, S: float, expected: float) -> None:
        s = torch.tensor([float(S)])
        tau = torch.tensor([0.25])
        p = float(bs2002_put(s, self.K, 0.12, 0.20, tau, q=0.0))
        assert abs(p - expected) < 0.12, f"S={S}: got {p:.4f}, expected {expected}"

    @pytest.mark.parametrize("S,expected", TABLE3_CASES_C)
    def test_r008_s040_T025(self, S: float, expected: float) -> None:
        s = torch.tensor([float(S)])
        tau = torch.tensor([0.25])
        p = float(bs2002_put(s, self.K, 0.08, 0.40, tau, q=0.0))
        assert abs(p - expected) < 0.12, f"S={S}: got {p:.4f}, expected {expected}"

    @pytest.mark.parametrize("S,expected", TABLE3_CASES_D)
    def test_r008_s020_T050(self, S: float, expected: float) -> None:
        s = torch.tensor([float(S)])
        tau = torch.tensor([0.50])
        p = float(bs2002_put(s, self.K, 0.08, 0.20, tau, q=0.0))
        assert abs(p - expected) < 0.12, f"S={S}: got {p:.4f}, expected {expected}"


# ---------------------------------------------------------------------------
# Table 1: K=100, b=-0.04, put prices 'p' (flat boundary)
# ---------------------------------------------------------------------------
TABLE1_CASES_A = [
    # r=0.08, sigma=0.20, T=0.25 — b=-0.04 means q=r-b=0.12
    (80, 20.41),
    (90, 11.25),
    (100, 4.40),
    (110, 1.12),
    (120, 0.18),
]

TABLE1_CASES_B = [
    # r=0.08, sigma=0.20, T=0.50 — b=-0.04 means q=0.12
    (80, 20.95),
    (90, 12.63),
    (100, 6.37),
    (110, 2.65),
    (120, 0.92),
]


class TestTable1WithDividends:
    """Table 1: b=-0.04, put approximation."""

    K = 100.0
    b = -0.04

    @pytest.mark.parametrize("S,expected", TABLE1_CASES_A)
    def test_r008_s020_T025(self, S: float, expected: float) -> None:
        r = 0.08
        q = r - self.b  # q = 0.12
        s = torch.tensor([float(S)])
        tau = torch.tensor([0.25])
        p = float(bs2002_put(s, self.K, r, 0.20, tau, q=q))
        assert abs(p - expected) < 0.12, f"S={S}: got {p:.4f}, expected {expected}"

    @pytest.mark.parametrize("S,expected", TABLE1_CASES_B)
    def test_r008_s020_T050(self, S: float, expected: float) -> None:
        r = 0.08
        q = r - self.b
        s = torch.tensor([float(S)])
        tau = torch.tensor([0.50])
        p = float(bs2002_put(s, self.K, r, 0.20, tau, q=q))
        assert abs(p - expected) < 0.12, f"S={S}: got {p:.4f}, expected {expected}"


# ---------------------------------------------------------------------------
# Table 2: K=100, b=0.04, put prices 'p' (flat boundary)
# ---------------------------------------------------------------------------
TABLE2_CASES_A = [
    # r=0.08, sigma=0.20, T=0.25 — b=0.04 means q=r-b=0.04
    (80, 20.00),
    (90, 10.19),
    (100, 3.51),
    (110, 0.78),
    (120, 0.11),
]


class TestTable2:
    """Table 2: b=0.04, put approximation."""

    K = 100.0
    b = 0.04

    @pytest.mark.parametrize("S,expected", TABLE2_CASES_A)
    def test_r008_s020_T025(self, S: float, expected: float) -> None:
        r = 0.08
        q = r - self.b
        s = torch.tensor([float(S)])
        tau = torch.tensor([0.25])
        p = float(bs2002_put(s, self.K, r, 0.20, tau, q=q))
        assert abs(p - expected) < 0.12, f"S={S}: got {p:.4f}, expected {expected}"


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestStructural:
    """Test structural properties of the BS-2002 put approximation."""

    K = 100.0
    r = 0.02
    sigma = 0.25

    def test_terminal_condition(self) -> None:
        """g2(s, T) = (K - s)+ when tau -> 0."""
        s = torch.linspace(50.0, 150.0, 101)
        tau = torch.full_like(s, 1e-7)
        g2 = bs2002_put(s, self.K, self.r, self.sigma, tau)
        payoff = torch.clamp(self.K - s, min=0.0)
        assert torch.allclose(g2, payoff, atol=0.5), (
            f"Terminal condition error: max |g2 - payoff| = "
            f"{float(torch.max(torch.abs(g2 - payoff))):.4e}"
        )

    def test_put_geq_european(self) -> None:
        """American put >= European put (early exercise premium >= 0)."""
        from learning_option_pricing.pricing.terminal import black_scholes_put
        s = torch.linspace(60.0, 140.0, 81)
        tau = torch.full_like(s, 1.0)
        am = bs2002_put(s, self.K, self.r, self.sigma, tau)
        eu = black_scholes_put(s, self.K, self.r, self.sigma, tau)
        violations = float(torch.sum(am < eu - 0.01))
        assert violations == 0, (
            f"American < European in {int(violations)} points"
        )

    def test_put_geq_intrinsic(self) -> None:
        """American put >= intrinsic value (K - s)+."""
        s = torch.linspace(50.0, 150.0, 101)
        tau = torch.full_like(s, 1.0)
        am = bs2002_put(s, self.K, self.r, self.sigma, tau)
        payoff = torch.clamp(self.K - s, min=0.0)
        violations = float(torch.sum(am < payoff - 0.01))
        assert violations == 0, (
            f"American < intrinsic in {int(violations)} points"
        )

    def test_exercise_boundary_below_strike(self) -> None:
        """Put exercise boundary s* < K."""
        s_star = bs2002_exercise_boundary(self.K, self.r, self.sigma, 1.0)
        assert 0 < s_star < self.K, (
            f"Exercise boundary s*={s_star:.2f} not in (0, K={self.K})"
        )

    def test_exercise_boundary_monotone_in_tau(self) -> None:
        """Exercise boundary decreases as tau increases (more time = lower s*)."""
        taus = [0.1, 0.5, 1.0, 2.0]
        boundaries = [
            bs2002_exercise_boundary(self.K, self.r, self.sigma, t) for t in taus
        ]
        for i in range(len(boundaries) - 1):
            assert boundaries[i] >= boundaries[i + 1] - 0.5, (
                f"Boundary not monotone: s*(tau={taus[i]})={boundaries[i]:.2f} "
                f"> s*(tau={taus[i+1]})={boundaries[i+1]:.2f}"
            )

    def test_batch_consistency(self) -> None:
        """Price is the same whether computed in a batch or individually."""
        s_vals = [80.0, 90.0, 100.0, 110.0, 120.0]
        tau_val = 1.0

        # Batch
        s_batch = torch.tensor(s_vals)
        tau_batch = torch.full_like(s_batch, tau_val)
        p_batch = bs2002_put(s_batch, self.K, self.r, self.sigma, tau_batch)

        # Individual
        for i, sv in enumerate(s_vals):
            s_single = torch.tensor([sv])
            tau_single = torch.tensor([tau_val])
            p_single = float(bs2002_put(s_single, self.K, self.r, self.sigma, tau_single))
            assert abs(p_single - float(p_batch[i])) < 1e-5, (
                f"Batch mismatch at S={sv}: single={p_single:.6f}, "
                f"batch={float(p_batch[i]):.6f}"
            )

    def test_differentiable(self) -> None:
        """Autograd can compute dP/ds through the BS-2002 formula."""
        s = torch.tensor([100.0], requires_grad=True)
        tau = torch.tensor([1.0])
        p = bs2002_put(s, self.K, self.r, self.sigma, tau)
        p.backward()
        assert s.grad is not None
        delta = float(s.grad)
        assert -1.0 < delta < 0.0, f"Delta={delta:.4f} out of range (-1, 0)"
