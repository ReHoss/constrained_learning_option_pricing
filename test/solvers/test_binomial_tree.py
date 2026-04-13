"""Tests for the CRR binomial tree reference solver.

Validation strategy
-------------------
European put
    Convergence to Black-Scholes closed form (ground truth).
    Put-call parity: C - P = S - K * exp(-rT)  (q = 0).

American put
    Price >= European put  (early-exercise premium is non-negative).
    Price >= intrinsic value  max(K - S, 0).

American call (q = 0)
    Price == European call  (Merton's no-early-exercise theorem for q = 0).

Bermudan put
    exercise_dates = [T] only  =>  equals European put.
    exercise_dates ⊂ {t1, T}  =>  price in [European, American].
    More exercise dates  =>  higher price  (monotonicity).
"""
from __future__ import annotations

import math

import pytest

from learning_option_pricing.solvers.binomial_tree import (
    american_call_binomial_tree,
    american_put_binomial_tree,
    bermuda_put_binomial_tree,
    european_put_binomial_tree,
)

# ---------------------------------------------------------------------------
# Black-Scholes closed-form formulas (used as ground truth)
# ---------------------------------------------------------------------------


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def bs_call(S: float, K: float, r: float, sigma: float, T: float, q: float = 0.0) -> float:
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------

# ATM, moderate vol — canonical sanity-check regime
BASE = dict(S=100.0, K=100.0, r=0.05, sigma=0.20, T=1.0)

N_FAST = 500   # quick convergence checks
N_REF = 2000   # tighter checks (used in experiments)

TOL_FAST = 1e-2   # absolute price tolerance at N_FAST
TOL_REF = 3e-3    # absolute price tolerance at N_REF


# ---------------------------------------------------------------------------
# European put
# ---------------------------------------------------------------------------


class TestEuropeanPut:
    def test_convergence_to_black_scholes_atm(self):
        """ATM European put converges to BS price."""
        bt = european_put_binomial_tree(**BASE, N=N_REF)
        bs = bs_put(**BASE)
        assert abs(bt - bs) < TOL_REF, f"BT={bt:.6f}  BS={bs:.6f}  diff={bt-bs:.2e}"

    @pytest.mark.parametrize("S", [70.0, 100.0, 130.0])
    def test_convergence_across_moneyness(self, S):
        bt = european_put_binomial_tree(S=S, **{k: v for k, v in BASE.items() if k != "S"}, N=N_FAST)
        bs = bs_put(S=S, **{k: v for k, v in BASE.items() if k != "S"})
        assert abs(bt - bs) < TOL_FAST, f"S={S}  BT={bt:.6f}  BS={bs:.6f}"

    def test_put_call_parity(self):
        """C - P = S - K*exp(-rT) for European with q=0."""
        S, K, r, T = BASE["S"], BASE["K"], BASE["r"], BASE["T"]
        put = european_put_binomial_tree(**BASE, N=N_FAST)
        call = bs_call(**BASE)
        assert abs(call - put - (S - K * math.exp(-r * T))) < TOL_FAST

    def test_positive(self):
        assert european_put_binomial_tree(**BASE, N=N_FAST) > 0.0

    def test_monotone_in_strike(self):
        """Higher strike => higher put price."""
        p_lo = european_put_binomial_tree(S=100.0, K=90.0, r=0.05, sigma=0.20, T=1.0, N=N_FAST)
        p_hi = european_put_binomial_tree(S=100.0, K=110.0, r=0.05, sigma=0.20, T=1.0, N=N_FAST)
        assert p_hi > p_lo


# ---------------------------------------------------------------------------
# American put
# ---------------------------------------------------------------------------


class TestAmericanPut:
    def test_above_european(self):
        """American put >= European put (early-exercise premium >= 0)."""
        amer = american_put_binomial_tree(**BASE, N=N_FAST)
        euro = european_put_binomial_tree(**BASE, N=N_FAST)
        assert amer >= euro - 1e-9, f"American={amer:.6f} < European={euro:.6f}"

    def test_above_intrinsic(self):
        """American put >= intrinsic value max(K - S, 0)."""
        S, K = BASE["S"], BASE["K"]
        amer = american_put_binomial_tree(**BASE, N=N_FAST)
        assert amer >= max(K - S, 0.0) - 1e-9

    def test_deep_itm_close_to_intrinsic(self):
        """Deep ITM American put is close to its intrinsic value K - S."""
        S, K = 60.0, 100.0
        amer = american_put_binomial_tree(S=S, K=K, r=0.05, sigma=0.20, T=1.0, N=N_FAST)
        intrinsic = K - S
        assert abs(amer - intrinsic) < 2.0  # within $2 of intrinsic for deep ITM

    def test_early_exercise_premium_positive_for_deep_itm(self):
        """Early exercise premium is measurably positive for deep ITM."""
        kwargs = dict(S=60.0, K=100.0, r=0.05, sigma=0.20, T=1.0)
        amer = american_put_binomial_tree(**kwargs, N=N_FAST)
        euro = european_put_binomial_tree(**kwargs, N=N_FAST)
        assert amer > euro + 1e-4, f"Expected positive premium; amer={amer:.4f} euro={euro:.4f}"


# ---------------------------------------------------------------------------
# American call
# ---------------------------------------------------------------------------


class TestAmericanCall:
    def test_equals_european_no_dividends(self):
        """American call = European call for q=0 (Merton's no-early-exercise theorem)."""
        amer = american_call_binomial_tree(**BASE, N=N_FAST, q=0.0)
        bs = bs_call(**BASE)
        assert abs(amer - bs) < TOL_FAST, f"American call={amer:.6f}  BS call={bs:.6f}"

    def test_above_intrinsic(self):
        S, K = BASE["S"], BASE["K"]
        call = american_call_binomial_tree(**BASE, N=N_FAST)
        assert call >= max(S - K, 0.0) - 1e-9

    def test_positive(self):
        assert american_call_binomial_tree(**BASE, N=N_FAST) > 0.0


# ---------------------------------------------------------------------------
# Bermudan put
# ---------------------------------------------------------------------------


class TestBermudaPut:
    def test_terminal_only_equals_european(self):
        """Bermudan with only T as exercise date equals European put.

        The only allowed exercise is at maturity, which is identical to a
        European option.
        """
        euro = european_put_binomial_tree(**BASE, N=N_FAST)
        berm = bermuda_put_binomial_tree(**BASE, exercise_dates=[BASE["T"]], N=N_FAST)
        assert abs(berm - euro) < TOL_FAST, f"Bermudan={berm:.6f}  European={euro:.6f}"

    def test_between_european_and_american(self):
        """European <= Bermudan(t1=0.5) <= American."""
        euro = european_put_binomial_tree(**BASE, N=N_FAST)
        amer = american_put_binomial_tree(**BASE, N=N_FAST)
        berm = bermuda_put_binomial_tree(**BASE, exercise_dates=[0.5], N=N_FAST)
        assert euro - 1e-9 <= berm, f"Bermudan={berm:.6f} < European={euro:.6f}"
        assert berm <= amer + 1e-9, f"Bermudan={berm:.6f} > American={amer:.6f}"

    def test_monotone_in_number_of_exercise_dates(self):
        """Adding exercise dates cannot decrease the price."""
        p1 = bermuda_put_binomial_tree(**BASE, exercise_dates=[0.5], N=N_FAST)
        p2 = bermuda_put_binomial_tree(**BASE, exercise_dates=[0.25, 0.5, 0.75], N=N_FAST)
        assert p2 >= p1 - 1e-9, f"More dates gave lower price: {p2:.6f} < {p1:.6f}"

    def test_converges_to_american_with_many_dates(self):
        """Dense exercise schedule converges toward the American price."""
        T = BASE["T"]
        amer = american_put_binomial_tree(**BASE, N=N_FAST)
        dense_dates = [i * T / 20 for i in range(1, 21)]
        berm = bermuda_put_binomial_tree(**BASE, exercise_dates=dense_dates, N=N_FAST)
        # Should be within 2 % of American price
        assert abs(berm - amer) / amer < 0.02, f"Bermudan={berm:.4f}  American={amer:.4f}"

    @pytest.mark.parametrize("t1", [0.25, 0.5, 0.75])
    def test_single_early_date_in_bounds(self, t1):
        """Bermudan(t1) is in [European, American] for several t1 values."""
        euro = european_put_binomial_tree(**BASE, N=N_FAST)
        amer = american_put_binomial_tree(**BASE, N=N_FAST)
        berm = bermuda_put_binomial_tree(**BASE, exercise_dates=[t1], N=N_FAST)
        assert euro - 1e-9 <= berm <= amer + 1e-9
