"""Exact Terminal Condition Neural Network (ETCNN) for option pricing.

Reference: Zhang, Guo, Lu (2026) — ETCNN, Section 3.

The trial solution is constructed as

    ũ_NN(s, t) = g1(s, t) * u_NN(s, t) + g2(s, t)

where
    g1(s, T) = 0           (vanishes at the terminal/exercise date)
    g2(s, T) = Φ(s)        (matches the payoff exactly)

so ũ_NN automatically satisfies the terminal condition regardless of u_NN.

g2 is additionally designed to capture the near-terminal singularities of the
true solution (√τ behaviour near t = T and non-differentiability at s = K).

Subclasses implement specific option types (American put, American call with
dividends, Bermuda, multi-asset, …).
"""
from __future__ import annotations

# TODO: implement ETCNN


class ETCNN:
    """Base ETCNN wrapper around a ResNet backbone.

    Args:
        resnet:     Trained or untrained ResNet backbone.
        g1:         Callable (s, t) -> scalar that vanishes at terminal dates.
        g2:         Callable (s, t) -> scalar that satisfies the payoff condition.
    """

    # TODO


class AmericanPutETCNN(ETCNN):
    """ETCNN for a single-asset American put option (no dividends).

    g2 = V1^e(s, t) + V2^e(s, t)  (first-order Taylor expansion of the
    European put price — see Zhang, Guo, Lu 2026, Section 4.1.1).
    """

    # TODO


class BermudaETCNN(ETCNN):
    """ETCNN for a single-asset Bermuda option with discrete exercise dates.

    TODO: define g1 and g2 for the piecewise-in-time setting.
    """

    # TODO
