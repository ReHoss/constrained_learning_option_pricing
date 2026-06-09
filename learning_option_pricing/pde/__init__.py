"""Partial-differential-equation operators and closed-form references.

This subpackage collects the differential operators (e.g. the backward heat
operator) and the analytical reference solutions used by the boundary-constrained
learning experiments.  It is deliberately kept separate from
:mod:`learning_option_pricing.pricing`, which carries the finance-specific
Black--Scholes machinery, so that the heat-equation study can be imported and
unit-tested in isolation.
"""

from learning_option_pricing.pde.operators import heat_operator
from learning_option_pricing.pde.heat_references import (
    heat_call_exact,
    heat_call_payoff,
    heat_sine_exact,
    heat_sine_terminal,
    heat_theta3_exact,
    heat_theta3_terminal,
    smooth_call_payoff,
    smooth_call_payoff_cm_time,
)

__all__ = [
    "heat_operator",
    "heat_call_exact",
    "heat_call_payoff",
    "heat_sine_exact",
    "heat_sine_terminal",
    "heat_theta3_exact",
    "heat_theta3_terminal",
    "smooth_call_payoff",
    "smooth_call_payoff_cm_time",
]
