"""Tests for the CRR binomial tree reference solver.

Key properties to verify:
    - European call price converges to Black-Scholes as N -> ∞.
    - American put price >= European put price everywhere.
    - Bermuda put price lies in [European price, American price].
    - Put-call parity holds for European options.
"""

# TODO: implement tests
