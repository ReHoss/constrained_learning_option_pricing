"""Binomial tree (Cox-Ross-Rubinstein) reference solver for American options.

Used to generate benchmark solutions for accuracy evaluation.

Reference:
    Cox, Ross, Rubinstein (1979). Option pricing: a simplified approach.
    J. Financ. Econ. 7(3), 229-263.
"""
from __future__ import annotations

import math

import numpy as np


def european_put_binomial_tree(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int = 4000,
    q: float = 0.0,
) -> float:
    """Price a European put via the CRR binomial tree.

    Args:
        S:      Current underlying price.
        K:      Strike price.
        r:      Risk-free rate (annualised).
        sigma:  Volatility (annualised).
        T:      Time to maturity (years).
        N:      Number of time steps.
        q:      Continuous dividend yield.

    Returns:
        Option price at (S, t=0).
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-(r - q) * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)

    # Terminal payoff
    prices = S * u ** np.arange(N, -1, -1) * d ** np.arange(0, N + 1)
    values = np.maximum(K - prices, 0.0)

    # Backward induction — no early exercise
    for _ in range(N):
        values = math.exp(-r * dt) * (p * values[:-1] + (1 - p) * values[1:])

    return float(values[0])


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
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-(r - q) * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)

    # Terminal payoff
    prices = S * u ** np.arange(N, -1, -1) * d ** np.arange(0, N + 1)
    values = np.maximum(K - prices, 0.0)

    # Backward induction with early exercise
    for i in range(N - 1, -1, -1):
        prices = S * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
        hold = math.exp(-r * dt) * (p * values[:-1] + (1 - p) * values[1:])
        exercise = np.maximum(K - prices, 0.0)
        values = np.maximum(hold, exercise)

    return float(values[0])


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
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)

    # Terminal payoff
    prices = S * u ** np.arange(N, -1, -1) * d ** np.arange(0, N + 1)
    values = np.maximum(prices - K, 0.0)

    # Backward induction with early exercise
    for i in range(N - 1, -1, -1):
        prices = S * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
        hold = math.exp(-r * dt) * (p * values[:-1] + (1 - p) * values[1:])
        exercise = np.maximum(prices - K, 0.0)
        values = np.maximum(hold, exercise)

    return float(values[0])


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
                         T is always included as an exercise date.
        N:               Number of time steps.

    Returns:
        Option price at (S, t=0).
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Map exercise dates to nearest time-step indices
    exercise_steps = set()
    for td in exercise_dates:
        step = int(round(td / dt))
        step = min(max(step, 0), N)
        exercise_steps.add(step)
    exercise_steps.add(N)  # terminal is always exercisable

    # Terminal payoff
    prices = S * u ** np.arange(N, -1, -1) * d ** np.arange(0, N + 1)
    values = np.maximum(K - prices, 0.0)

    # Backward induction — early exercise only at prescribed steps
    for i in range(N - 1, -1, -1):
        prices = S * u ** np.arange(i, -1, -1) * d ** np.arange(0, i + 1)
        hold = math.exp(-r * dt) * (p * values[:-1] + (1 - p) * values[1:])
        if i in exercise_steps:
            exercise = np.maximum(K - prices, 0.0)
            values = np.maximum(hold, exercise)
        else:
            values = hold

    return float(values[0])
