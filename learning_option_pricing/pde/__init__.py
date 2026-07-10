"""Partial-differential-equation operators and closed-form references.

This subpackage collects the differential operators (e.g. the backward heat
operator) and the analytical reference solutions used by the boundary-constrained
learning experiments.  It is deliberately kept separate from
:mod:`learning_option_pricing.pricing`, which contains the finance-specific
Black--Scholes machinery, so that the heat-equation study can be imported and
unit-tested in isolation.
"""

from learning_option_pricing.pde.operators import heat_operator, heat_operator_parts
from learning_option_pricing.pde.periodic_spectral_toolbox import (
    ConstantCoefficientGenerator,
    GeneratorSplit,
    PeriodisedBernoulliDatum,
    SquareWaveDatum,
    advection_diffusion_reaction,
    biharmonic_advection_reaction,
    black_scholes_log_price,
    operator_channel_floor,
    predicted_floor_exponent,
    predicted_operator_channel_floor_constant,
    symmetric_wavenumber_band,
    synthesise_datum_on_grid,
)
from learning_option_pricing.pde.terminal_data_extensions import (
    ConstantInTimeExtension,
    ConvexRawExtension,
    ExactSolutionExtension,
    GradedGaussianExtension,
    SplitSemigroupExtension,
    TerminalDataExtension,
    exponential_time_integral_factor,
    total_strip_forcing_squared,
)
from learning_option_pricing.pde.heat_references import (
    bermudan_put_value_exact,
    chen_mangasarian_max,
    heat_call_exact,
    heat_call_payoff,
    heat_propagate,
    heat_put_exact,
    heat_put_payoff,
    heat_sine_exact,
    heat_sine_terminal,
    heat_theta3_exact,
    heat_theta3_terminal,
    smooth_call_payoff,
    smooth_call_payoff_cm_time,
)

__all__ = [
    "heat_operator",
    "heat_operator_parts",
    "ConstantCoefficientGenerator",
    "GeneratorSplit",
    "PeriodisedBernoulliDatum",
    "SquareWaveDatum",
    "advection_diffusion_reaction",
    "biharmonic_advection_reaction",
    "black_scholes_log_price",
    "operator_channel_floor",
    "predicted_floor_exponent",
    "predicted_operator_channel_floor_constant",
    "symmetric_wavenumber_band",
    "synthesise_datum_on_grid",
    "ConstantInTimeExtension",
    "ConvexRawExtension",
    "ExactSolutionExtension",
    "GradedGaussianExtension",
    "SplitSemigroupExtension",
    "TerminalDataExtension",
    "exponential_time_integral_factor",
    "total_strip_forcing_squared",
    "bermudan_put_value_exact",
    "chen_mangasarian_max",
    "heat_call_exact",
    "heat_call_payoff",
    "heat_propagate",
    "heat_put_exact",
    "heat_put_payoff",
    "heat_sine_exact",
    "heat_sine_terminal",
    "heat_theta3_exact",
    "heat_theta3_terminal",
    "smooth_call_payoff",
    "smooth_call_payoff_cm_time",
]
