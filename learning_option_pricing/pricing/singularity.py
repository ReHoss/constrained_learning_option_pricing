"""Singularity extraction ansatz for Bermudan option pricing at exercise dates.

At an exercise date $t_1$, the intermediate terminal condition

$$V^{\\mathrm{Berm}}_\\theta(s, t_1) = \\max(\\Phi(s), \\tilde{u}^{(A)}_\\theta(s, t_1))$$

has a $C^0$ non-differentiable kink at the optimal exercise boundary $s^*$.
The first spatial derivative jumps discontinuously, and the second derivative
(Gamma) contains a Dirac delta singularity.  Forcing a standard neural network
to learn this boundary causes the PDE residual to explode.

**Solution — Singularity Extraction Ansatz:**

Decompose the solution on $[0, t_1]$ as

$$U_B(s, t) = v(s, t) + \\tilde{u}_\\theta(s, t)$$

where $v(s, t) = c \\cdot P^{\\text{BS}}(s, s^*, r, \\sigma, t_1 - t)$ is a
*fictitious European put* (strike $s^*$, maturity $t_1$) scaled to cancel the
derivative jump.  Because the BSM operator $\\mathcal{L}$ is linear and
$\\mathcal{L}v = 0$ (exact BS solution), the PDE residual reduces to
$\\mathcal{L}\\tilde{u}_\\theta = 0$, and the network only needs to learn
the $C^1$-smooth residual.

**Scaling constant:**

At $t = t_1$, the derivative jump in $V^{\\mathrm{Berm}}_\\theta$ at $s^*$ is

$$\\Delta = \\frac{\\partial U_A}{\\partial s}\\Big|_{s^*} - (-1)
         = \\frac{\\partial U_A}{\\partial s}\\Big|_{s^*} + 1$$

(for a put, $\\Phi'(s) = -1$ in the exercise region).  The fictitious put
contributes a jump of $c$, so setting $c = \\Delta$ cancels the singularity.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from learning_option_pricing.pricing.terminal import black_scholes_put, payoff_put

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exercise boundary detection
# ---------------------------------------------------------------------------

def find_exercise_boundary(
    model: nn.Module,
    K: float,
    t1: float,
    s_lo: float,
    s_hi: float,
    device: torch.device,
    n_grid: int = 2000,
    bisection_tol: float = 1e-6,
    bisection_iters: int = 60,
) -> float:
    """Find exercise boundary $s^*$ at $t_1$ via grid search + bisection.

    $s^*$ is defined as the asset price where the payoff $\\Phi(s)$ equals
    the hold (continuation) value $U_A(s, t_1)$.  For a put, exercise is
    optimal for $s < s^*$.

    Args:
        model:          Trained Stage A network (ETCNN_A).
        K:              Strike price.
        t1:             Exercise date.
        s_lo, s_hi:     Search range for $s^*$.
        device:         Torch device.
        n_grid:         Initial grid resolution for sign-change detection.
        bisection_tol:  Convergence tolerance for bisection.
        bisection_iters: Maximum bisection iterations.

    Returns:
        s_star: Exercise boundary price (to within *bisection_tol*).

    Raises:
        ValueError: If no sign change is found in ``[s_lo, s_hi]``.
    """
    s_grid = torch.linspace(s_lo, s_hi, n_grid)
    t1_grid = torch.full_like(s_grid, t1)
    x = torch.stack([s_grid, t1_grid], dim=1).to(device)

    with torch.no_grad():
        hold = model(x).cpu().squeeze()
    exercise = payoff_put(s_grid, K)
    diff = exercise - hold  # positive in exercise region, negative in hold

    sign_changes = torch.where(diff[:-1] * diff[1:] < 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(
            f"No exercise boundary found in [{s_lo}, {s_hi}].  "
            "The network may not have converged, or the search range is too narrow."
        )

    # First crossing (exercise→hold transition for a put)
    idx = sign_changes[0].item()
    a, b = float(s_grid[idx]), float(s_grid[idx + 1])

    def _diff_at(s_val: float) -> float:
        s_t = torch.tensor([[s_val, t1]], dtype=torch.float32, device=device)
        with torch.no_grad():
            h = model(s_t).item()
        return max(K - s_val, 0.0) - h

    for _ in range(bisection_iters):
        if abs(b - a) < bisection_tol:
            break
        mid = 0.5 * (a + b)
        if _diff_at(a) * _diff_at(mid) < 0:
            b = mid
        else:
            a = mid

    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Scaling constant c
# ---------------------------------------------------------------------------

def compute_hold_delta_at_boundary(
    model: nn.Module,
    s_star: float,
    t1: float,
    device: torch.device,
) -> float:
    r"""Compute $\partial U_A / \partial s$ at $s^*$ using autograd.

    Args:
        model:  Trained Stage A network.
        s_star: Exercise boundary.
        t1:     Exercise date.
        device: Torch device.

    Returns:
        delta_A: Spatial derivative of the hold value at $s^*$.
    """
    s = torch.tensor([s_star], dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor([t1], dtype=torch.float32, device=device)
    x = torch.stack([s, t], dim=1)
    u = model(x).squeeze()
    (du_ds,) = torch.autograd.grad(u, s, create_graph=False)
    return du_ds.item()


def compute_scaling_constant(
    model: nn.Module,
    s_star: float,
    t1: float,
    device: torch.device,
) -> float:
    r"""Compute the scaling constant $c$ for the fictitious European put.

    For a put option the derivative jump at $s^*$ in $V^{\mathrm{Berm}}_{\bar{\theta}}$ is

    $$c = \frac{\partial U_A}{\partial s}\Big|_{s^*} + 1$$

    because $\Phi'(s) = -1$ for $s < s^*$ (exercise region) and the
    fictitious put $(s^* - s)^+$ has a derivative jump of magnitude 1.

    Args:
        model:  Trained Stage A network.
        s_star: Exercise boundary.
        t1:     Exercise date.
        device: Torch device.

    Returns:
        c: Scaling constant (typically $0 < c < 1$ for a put).
    """
    delta_A = compute_hold_delta_at_boundary(model, s_star, t1, device)
    c = delta_A + 1.0
    logger.info(
        f"  Singularity extraction: delta_A = {delta_A:.6f}, c = {c:.6f}"
    )
    return c


# ---------------------------------------------------------------------------
# Fictitious European put  v(s, t)
# ---------------------------------------------------------------------------

class FictitiousEuropeanPut(nn.Module):
    r"""Analytical singularity-absorbing function.

    $$v(s, t) = c \cdot P^{\text{BS}}(s,\; s^*,\; r,\; \sigma,\; t_1 - t)$$

    This is an exact solution to the BSM PDE ($\mathcal{L}v = 0$), designed
    to absorb the $C^0$ derivative kink in the Bermudan intermediate
    condition at the exercise boundary $s^*$.

    At maturity $t = t_1$: $v(s, t_1) = c \cdot (s^* - s)^+$.

    The constants ``c`` and ``s_star`` are registered as buffers so they
    survive ``.to(device)`` and ``.state_dict()`` serialisation.

    Args:
        c:      Scaling constant (cancels derivative jump).
        s_star: Exercise boundary (fictitious strike).
        r:      Risk-free rate.
        sigma:  Volatility.
        t1:     Exercise date (fictitious maturity).
    """

    def __init__(
        self, c: float, s_star: float, r: float, sigma: float, t1: float,
    ) -> None:
        super().__init__()
        self.register_buffer("_c", torch.tensor(c, dtype=torch.float32))
        self.register_buffer("_s_star", torch.tensor(s_star, dtype=torch.float32))
        self.r = r
        self.sigma = sigma
        self.t1 = t1

    # -- public read-only properties --
    @property
    def c(self) -> float:
        return float(self._c)

    @property
    def s_star(self) -> float:
        return float(self._s_star)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""Evaluate $v(s, t) = c \cdot P^{\text{BS}}(s, s^*, r, \sigma, t_1 - t)$.

        Args:
            s: Asset prices, shape ``(N, 1)`` or ``(N,)``.
            t: Times, shape ``(N, 1)`` or ``(N,)``.

        Returns:
            $v(s, t)$, same shape as *s*.
        """
        tau = self.t1 - t
        # Floor to a tiny positive number to keep BS formula stable
        tau = torch.clamp(tau, min=1e-10)
        return self._c * black_scholes_put(s, self.s_star, self.r, self.sigma, tau)

    def at_maturity(self, s: torch.Tensor) -> torch.Tensor:
        r"""Evaluate $v(s, t_1) = c \cdot (s^* - s)^+$ exactly (closed form)."""
        return self._c * torch.clamp(self._s_star - s, min=0.0)

    def __repr__(self) -> str:
        return (
            f"FictitiousEuropeanPut(c={self.c:.6f}, s_star={self.s_star:.2f}, "
            f"r={self.r}, sigma={self.sigma}, t1={self.t1})"
        )


# ---------------------------------------------------------------------------
# Orchestrator: build everything for Stage B
# ---------------------------------------------------------------------------

def build_singularity_extraction(
    model: nn.Module,
    K: float,
    r: float,
    sigma: float,
    t1: float,
    s_lo: float,
    s_hi: float,
    device: torch.device,
    n_grid: int = 2000,
) -> tuple[float, float, FictitiousEuropeanPut, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Orchestrate the full singularity extraction at an exercise date.

    1. Find the exercise boundary $s^*$ via bisection.
    2. Compute the scaling constant $c = \partial U_A/\partial s|_{s^*} + 1$.
    3. Build the fictitious European put $v(s, t)$.
    4. Evaluate $V^{\mathrm{Berm}}_{\bar{\theta}}$ and the $C^1$ residual on a dense grid.

    Args:
        model:   Trained Stage A network (ETCNN_A).
        K:       Contract strike price.
        r:       Risk-free rate.
        sigma:   Volatility.
        t1:      Exercise date.
        s_lo:    Lower bound of asset-price grid.
        s_hi:    Upper bound of asset-price grid.
        device:  Torch device.
        n_grid:  Number of grid points for tabulation.

    Returns:
        Tuple of ``(s_star, c, fictitious_put, s_nodes, v_target, residual)``:

        - **s_star** — exercise boundary price.
        - **c** — scaling constant.
        - **fictitious_put** — :class:`FictitiousEuropeanPut` module.
        - **s_nodes** — 1-D tensor of asset prices (CPU), shape ``(n_grid,)``.
        - **v_target** — $V^{\mathrm{Berm}}_{\bar{\theta}}(s, t_1)$ values (CPU).
        - **residual** — $C^1$ residual $V^{\mathrm{Berm}}_{\bar{\theta}} - v|_{t_1}$ (CPU).
    """
    logger.info("  Singularity extraction: finding exercise boundary ...")
    s_star = find_exercise_boundary(model, K, t1, s_lo, s_hi, device, n_grid)
    logger.info(f"  Exercise boundary: s* = {s_star:.6f}")

    c = compute_scaling_constant(model, s_star, t1, device)

    fict_put = FictitiousEuropeanPut(c, s_star, r, sigma, t1)
    logger.info(f"  {fict_put}")

    # --- Tabulate V_target and residual on a dense grid ---
    s_nodes = torch.linspace(s_lo, s_hi, n_grid)
    t1_tensor = torch.full_like(s_nodes, t1)
    x = torch.stack([s_nodes, t1_tensor], dim=1).to(device)

    with torch.no_grad():
        hold = model(x).cpu().squeeze()
    exercise = payoff_put(s_nodes, K)
    v_target = torch.maximum(exercise, hold)

    # v(s, t1) = c * max(s* - s, 0)  — exact closed form at maturity
    v_at_t1 = fict_put.at_maturity(s_nodes)
    residual = v_target - v_at_t1

    # --- Verify C^1 smoothness at s* ---
    _verify_c1_smoothness(s_nodes, residual, s_star)

    return s_star, c, fict_put, s_nodes.detach().cpu(), v_target.detach().cpu(), residual.detach().cpu()


def _verify_c1_smoothness(
    s: torch.Tensor, residual: torch.Tensor, s_star: float, h: float = 1e-3,
) -> None:
    """Log a diagnostic check that the residual is C^1 at s*."""
    # Find the index closest to s*
    idx = torch.argmin(torch.abs(s - s_star)).item()
    if idx < 2 or idx >= len(s) - 2:
        return

    ds = float(s[1] - s[0])
    # Finite-difference first derivative from left and right
    deriv_left = float((residual[idx] - residual[idx - 1]) / ds)
    deriv_right = float((residual[idx + 1] - residual[idx]) / ds)
    jump = abs(deriv_right - deriv_left)

    # Finite-difference second derivative (curvature)
    gamma = float((residual[idx + 1] - 2 * residual[idx] + residual[idx - 1]) / ds**2)

    logger.info(
        f"  C^1 verification at s*={s_star:.2f}: "
        f"dg2/ds(left)={deriv_left:.6f}, dg2/ds(right)={deriv_right:.6f}, "
        f"|jump|={jump:.2e}, gamma={gamma:.4f}"
    )
