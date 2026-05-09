"""VPINN components for the Black-Scholes PDE (weak / variational formulation)."""
from .loss import VPINNLoss
from .quadrature import GaussLegendreQuadrature
from .test_functions import SinusoidalTestFunctions


__all__ = ["GaussLegendreQuadrature", "SinusoidalTestFunctions", "VPINNLoss"]
