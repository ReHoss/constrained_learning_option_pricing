"""Optimizers for PINN training.

* :class:`ENGDOptimizer` — Empirical Natural Gradient for *strong-form* PINNs
  (per-point PDE residual as the measurement function).
* :class:`VPINNENGDOptimizer` — Empirical Natural Gradient for *variational*
  PINNs (weak-form residuals against test functions as the measurements).
"""

from .natural_gradient import (
    ENGDOptimizer,
    compute_jacobians,
    flat_grad,
    flat_params,
    grid_line_search,
    measurement_jacobian,
    measurement_jacobian_fwd,
    set_flat_params,
    solve_cg,
)
from .natural_gradient_vpinn import (
    VPINNENGDOptimizer,
    vpinn_jacobian,
)

__all__ = [
    # strong-form
    "ENGDOptimizer",
    "compute_jacobians",
    # variational
    "VPINNENGDOptimizer",
    "vpinn_jacobian",
    # generic primitives
    "solve_cg",
    "grid_line_search",
    "measurement_jacobian",
    "measurement_jacobian_fwd",
    "flat_grad",
    "flat_params",
    "set_flat_params",
]
