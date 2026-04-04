"""Binomial tree (Cox-Ross-Rubinstein) reference solver for American options.

Used to generate benchmark solutions for accuracy evaluation.

Reference:
    Cox, Ross, Rubinstein (1979). Option pricing: a simplified approach.
    J. Financ. Econ. 7(3), 229-263.
"""
from __future__ import annotations


def american_put_binomial_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 4000,
    q: float = 0.0,
) -> float:
    """Price a single-asset American put via the CRR binomial tree.

    Args:
        S:      Current underlying price.
        K:      Strike price.
        r:      Risk-free rate (annualised).
        sigma:  Volatility (annualised).
        T:      Time to maturity (years).
        N:      Number of time steps (default 4000 for reference solution).
        q:      Continuous dividend yield (default 0).

    Returns:
        Option price at (S, t=0).
    """
    # TODO
    raise NotImplementedError


def american_call_binomial_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 4000,
    q: float = 0.0,
) -> float:
    """Price a single-asset American call with continuous dividend via CRR.

    Args: same as american_put_binomial_tree.
    """
    # TODO
    raise NotImplementedError


def bermuda_put_binomial_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    exercise_dates: list[float],
    N: int = 4000,
) -> float:
    """Price a single-asset Bermuda put (exercise only at prescribed dates).

    Args:
        S:               Current underlying price.
        K:               Strike price.
        r:               Risk-free rate.
        sigma:           Volatility.
        T:               Maturity.
        exercise_dates:  Sorted list of allowed exercise times in (0, T].
        N:               Number of time steps.

    Returns:
        Option price at (S, t=0).
    """
    # TODO
    raise NotImplementedError
