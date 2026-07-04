"""Closed-form terminal data and exact solutions for the backward heat equation.

These references back the three initial (terminal) conditions of the
ansatz-form study described in
``latex_documents/reports/2026_01_29_constrained_learning_pde_lehalle_hosseinkhan/``:

1. **Two-mode sine** (``cal_notes/example_heat.tex`` Section 1).
   Terminal datum :math:`g(x) = \\sin(\\pi x) + c\\,\\sin(f\\pi x)` on
   :math:`[0, 1]`; the exact backward-heat solution keeps the sine modes with
   time-decaying amplitudes and satisfies homogeneous Dirichlet conditions
   :math:`u(t, 0) = u(t, 1) = 0`.

2. **Jacobi theta_3** (``cal_notes/example_heat.tex`` Section 2).
   Terminal datum :math:`g(x) = \\vartheta_3(x/2, e^{-1}) = 1 + 2\\sum_n
   e^{-n^2}\\cos(\\pi n x)` on :math:`[-1, 1]`; the exact solution is a
   theta_3 with a shifted nome.  **Sign convention:** under the terminal-value
   (backward) operator :math:`\\partial_t u + (\\sigma^2/2)\\partial_{xx}u = 0`
   used throughout this repository — the same convention as the Black--Scholes
   operator and the sine reference above — the mode amplitudes *decay* as ``t``
   decreases (the problem is well posed, equivalent to a forward heat equation
   in :math:`\\tau = T - t`).  The source note writes a growing-amplitude
   solution; that form actually solves :math:`\\partial_t u - (\\sigma^2/2)
   \\partial_{xx}u = 0` and is therefore inconsistent with the stated operator.
   We implement the operator-consistent decaying solution.

3. **Smoothed call payoff** (already implemented in the singularity study,
   re-expressed here under the *pure* heat operator).  Terminal datum is a
   smooth approximation of :math:`(e^x - K)^+` in the log-price coordinate
   :math:`x = \\ln S`; the exact pure-heat solution is a Black--Scholes call
   with zero rate and zero dividend, i.e. the Gaussian convolution of the
   payoff.

All functions are vectorised over batches of ``x`` and return tensors on the
same device/dtype as their input.  The diffusion coefficient is ``sigma^2 / 2``
to match :func:`learning_option_pricing.pde.operators.heat_operator`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_PI = math.pi


# ---------------------------------------------------------------------------
# Standard normal CDF (autograd-compatible, mirrors pricing.terminal)
# ---------------------------------------------------------------------------

def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * torch.erfc(-x / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# IC1 — two-mode sine
# ---------------------------------------------------------------------------

def heat_sine_terminal(
    x: torch.Tensor,
    *,
    c: float = 1.0,
    f: float = 4.0,
) -> torch.Tensor:
    r"""Two-mode sine terminal datum ``g(x) = sin(pi x) + c sin(f pi x)``.

    Args:
        x: Spatial coordinate (any shape).
        c: Amplitude of the high-frequency mode.
        f: Frequency multiplier of the high-frequency mode (integer-valued
           keeps homogeneous Dirichlet conditions exact on ``[0, 1]``).
    """
    return torch.sin(_PI * x) + c * torch.sin(f * _PI * x)


def heat_sine_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    T: float,
    sigma: float = 1.0,
    c: float = 1.0,
    f: float = 4.0,
) -> torch.Tensor:
    r"""Exact backward-heat solution for the two-mode sine terminal datum.

    .. math::

        u(t, x) = e^{\frac{\sigma^2 \pi^2}{2}(t - T)} \sin(\pi x)
                + c\, e^{\frac{\sigma^2 f^2 \pi^2}{2}(t - T)} \sin(f \pi x).

    Each mode decays as ``t`` decreases from ``T``; the high-frequency mode
    decays faster, which is the well-posed (forward-stable) direction.

    Args:
        x:     Spatial coordinate.
        t:     Time coordinate (broadcast-compatible with ``x``).
        T:     Terminal time.
        sigma: Diffusion scale (coefficient ``sigma^2 / 2``).
        c:     High-frequency amplitude.
        f:     High-frequency multiplier.
    """
    decay_1 = torch.exp(0.5 * sigma**2 * _PI**2 * (t - T))
    decay_f = torch.exp(0.5 * sigma**2 * (f * _PI) ** 2 * (t - T))
    return decay_1 * torch.sin(_PI * x) + c * decay_f * torch.sin(f * _PI * x)


# ---------------------------------------------------------------------------
# IC2 — Jacobi theta_3
# ---------------------------------------------------------------------------

def heat_theta3_terminal(
    x: torch.Tensor,
    *,
    n_modes: int = 6,
) -> torch.Tensor:
    r"""Jacobi theta_3 terminal datum ``g(x) = 1 + 2 sum_{n>=1} e^{-n^2} cos(pi n x)``.

    The series is truncated at ``n_modes`` terms; ``e^{-n^2}`` decays
    super-exponentially, so ``n_modes = 6`` is already machine-accurate
    (``e^{-49} ~ 5e-22``).

    Args:
        x:       Spatial coordinate.
        n_modes: Number of cosine modes retained.
    """
    out = torch.ones_like(x)
    for n in range(1, n_modes + 1):
        out = out + 2.0 * math.exp(-(n**2)) * torch.cos(_PI * n * x)
    return out


def heat_theta3_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    T: float,
    sigma: float = 1.0,
    n_modes: int = 6,
) -> torch.Tensor:
    r"""Exact backward-heat solution for the theta_3 terminal datum.

    .. math::

        u(t, x) = 1 + 2 \sum_{n \ge 1}
            e^{-n^2 + \frac{\sigma^2 \pi^2 n^2}{2}(t - T)} \cos(\pi n x).

    Under the terminal-value (backward) convention the mode amplitudes decay as
    ``t`` decreases from ``T``, so the solution is bounded for every ``T`` and
    no well-posedness horizon constraint is needed (cf. the module docstring on
    the sign convention).  The constant mode (amplitude ``1``) is stationary.

    Args:
        x:       Spatial coordinate.
        t:       Time coordinate.
        T:       Terminal time.
        sigma:   Diffusion scale.
        n_modes: Number of cosine modes retained.
    """
    out = torch.ones_like(x)
    for n in range(1, n_modes + 1):
        exponent = -(n**2) + 0.5 * sigma**2 * _PI**2 * n**2 * (t - T)
        out = out + 2.0 * torch.exp(exponent) * torch.cos(_PI * n * x)
    return out


# ---------------------------------------------------------------------------
# IC3 — smoothed call payoff under the pure heat operator (x = ln S)
# ---------------------------------------------------------------------------

def heat_call_payoff(x: torch.Tensor, K: float) -> torch.Tensor:
    """Exact call payoff in log-price coordinates: ``(e^x - K)^+``."""
    return torch.clamp(torch.exp(x) - K, min=0.0)


def smooth_call_payoff(x: torch.Tensor, K: float, *, beta: float) -> torch.Tensor:
    r"""Softplus smoothing of the call payoff ``(e^x - K)^+`` in ``x = ln S``.

    .. math::

        g_\beta(x) = \frac{1}{\beta}\log\!\bigl(1 + e^{\beta (e^x - K)}\bigr)
                     - \frac{\log 2}{\beta},

    which is ``C^\infty`` and converges uniformly to ``(e^x - K)^+`` as
    ``beta -> infinity`` (the ``-log 2 / beta`` shift makes the smoothing pass
    through the origin of the first-derivative discontinuity).  This matches the construction used in
    ``exp_singularity_european_call``.

    Args:
        x:    Log-price coordinate.
        K:    Strike.
        beta: Sharpness; larger ``beta`` is a tighter approximation.
    """
    log2 = math.log(2.0)
    return F.softplus(torch.exp(x) - K, beta=beta) - log2 / beta


def heat_put_payoff(x: torch.Tensor, K: float) -> torch.Tensor:
    """Exact put payoff in log-price coordinates: ``(K - e^x)^+``."""
    return torch.clamp(K - torch.exp(x), min=0.0)


def heat_put_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    K: float,
    T: float,
    sigma: float,
) -> torch.Tensor:
    r"""Exact pure-heat solution with the put payoff ``(K - e^x)^+`` at ``t = T``.

    Under ``d_t u + (sigma^2/2) d_xx u = 0`` the solution is the Gaussian
    convolution of the terminal payoff — a zero-rate Black--Scholes put with the
    no-drift convexity correction:

    .. math::

        u(t, x) = K\,N(d) - e^{x + \sigma^2 \tau / 2}\,N(d - \sigma\sqrt{\tau}),
        \qquad \tau = T - t,\quad d = \frac{\ln K - x}{\sigma\sqrt{\tau}}.

    At ``tau = 0`` it reduces to the payoff.  This is the European *continuation*
    value used in the Bermudan gluing.
    """
    tau = torch.clamp(T - t, min=1e-12)
    sqrt_tau = sigma * torch.sqrt(tau)
    d = (math.log(K) - x) / sqrt_tau
    return K * _normal_cdf(d) - torch.exp(x + 0.5 * sigma**2 * tau) * _normal_cdf(d - sqrt_tau)


def chen_mangasarian_max(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    r"""Smooth Chen--Mangasarian approximation of ``max(a, b)``.

    .. math::

        M_\varepsilon(a, b) = \tfrac12\Bigl(a + b + \sqrt{(a-b)^2 + \varepsilon^2}\Bigr),

    which is ``C^\infty`` and converges uniformly to ``max(a, b)`` as
    ``eps -> 0`` (bias bounded by ``eps/2``).  Used to glue the exercise payoff
    and the continuation value at a Bermudan exercise date.
    """
    return 0.5 * (a + b + torch.sqrt((a - b) ** 2 + eps**2))


def heat_propagate(
    g_fn,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    t_terminal: float,
    sigma: float,
    y_lo: float,
    y_hi: float,
    n_quad: int = 2000,
) -> torch.Tensor:
    r"""Exact backward-heat solution with a generic terminal datum ``g`` at
    ``t_terminal``, by numerical Gaussian convolution.

    Solves ``d_t u + (sigma^2/2) d_xx u = 0`` with ``u(\cdot, t_terminal) = g``:
    the solution is the heat kernel applied to ``g`` over ``tau = t_terminal - t``,

    .. math::

        u(t, x) = \int_{\mathbb{R}} \frac{1}{\sigma\sqrt{2\pi\tau}}
            \exp\!\Bigl(-\frac{(x-y)^2}{2\sigma^2\tau}\Bigr)\, g(y)\, dy,

    approximated by a fixed trapezoidal grid ``y in [y_lo, y_hi]`` with
    ``n_quad`` nodes.  Used as the machine-precision reference for the trained
    Bermudan stage (no binomial tree needed).  At ``tau -> 0`` it returns ``g(x)``.

    Args:
        g_fn:       Callable ``g(y) -> Tensor`` (the terminal datum, e.g. the
                    smoothed max of payoff and continuation).
        x, t:       Query points (1-D tensors, broadcast-compatible).
        t_terminal: Terminal time of the stage.
        sigma:      Diffusion scale.
        y_lo, y_hi: Quadrature support (must comfortably cover where ``g`` and the
                    Gaussian mass are supported).
        n_quad:     Number of quadrature nodes.
    """
    y = torch.linspace(y_lo, y_hi, n_quad, dtype=x.dtype, device=x.device)
    dy = (y_hi - y_lo) / (n_quad - 1)
    g_y = g_fn(y)  # (n_quad,)
    raw_tau = t_terminal - t
    # Below the grid-resolvable scale the Gaussian kernel collapses to a delta the
    # fixed quadrature cannot represent; there the heat solution is just g(x)
    # (the tau -> 0 limit u(., t_terminal) = g).  Require sigma*sqrt(tau) >= 5 dy.
    tau_floor = (5.0 * dy / sigma) ** 2
    tau = torch.clamp(raw_tau, min=tau_floor).unsqueeze(-1)  # (N, 1)
    xc = x.unsqueeze(-1)  # (N, 1)
    var = sigma**2 * tau
    kernel = torch.exp(-((xc - y.unsqueeze(0)) ** 2) / (2.0 * var)) / torch.sqrt(2.0 * math.pi * var)
    conv = (kernel * g_y.unsqueeze(0)).sum(dim=-1) * dy
    return torch.where(raw_tau < tau_floor, g_fn(x), conv)


def bermudan_put_value_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    exercise_times,
    K: float,
    sigma: float,
    y_lo: float,
    y_hi: float,
    n_quad: int = 4000,
) -> torch.Tensor:
    r"""Exact Bermudan-put value by backward induction with exact propagation.

    The Bermudan put with exercise opportunities at
    :math:`t_1 < t_2 < \dots < t_m = T` (maturity) is priced by the dynamic
    program

    .. math::

        V(t_m, x) &= (K - e^x)^+, \\
        C(t_k, x) &= \bigl(e^{(t_{k+1}-t_k)\,\mathcal{A}}\,V(t_{k+1},\cdot)\bigr)(x),
        \qquad \mathcal{A} = \tfrac{\sigma^2}{2}\partial_{xx}, \\
        V(t_k, x) &= \max\!\bigl((K-e^x)^+,\ C(t_k, x)\bigr),

    where between exercise dates the value solves the pure-heat equation, so the
    continuation operator :math:`e^{\Delta\tau\,\mathcal{A}}` is **exact Gaussian
    convolution** (one step per inter-exercise interval, no time-discretisation
    error) implemented by :func:`heat_propagate`.  The max-gluing uses the
    **exact** :func:`torch.maximum` (the reference; the trained stages use the
    Chen--Mangasarian smooth max).  This is the machine-precision target against
    which the learned backward induction is validated and its error propagation
    measured.

    For ``t`` strictly inside an inter-exercise interval the value is the pure
    continuation toward the next exercise date (the option cannot be exercised
    there); at an exercise date it includes the ``max``.  Arbitrary ``t`` tensors
    are handled by grouping equal time values (slices share one time).

    Args:
        x:              Log-price coordinate, shape ``(N,)``.
        t:              Time coordinate (scalar-valued or shape ``(N,)``).
        exercise_times: Ascending exercise dates; the last is maturity ``T``.
        K:              Strike.
        sigma:          Diffusion scale (coefficient ``sigma^2 / 2``).
        y_lo, y_hi:     Quadrature support for every convolution.
        n_quad:         Quadrature nodes per convolution.

    Returns:
        The exact value ``V(t, x)`` of shape ``(N,)``.
    """
    et = sorted(float(s) for s in exercise_times)
    m = len(et)
    if m < 2:
        raise ValueError("Need at least two exercise dates (one intermediate + maturity).")

    def payoff(y):
        return heat_put_payoff(y, K)

    # Build the value-at-exercise callables V(t_k, .) bottom-up (top = maturity).
    value_at = {m - 1: payoff}
    for k in range(m - 2, -1, -1):
        t_kp1 = et[k + 1]
        v_kp1 = value_at[k + 1]

        def make_value(v_kp1, t_kp1):
            def value(y):
                cont = heat_propagate(
                    v_kp1, y, torch.full_like(y, et[k]), t_terminal=t_kp1,
                    sigma=sigma, y_lo=y_lo, y_hi=y_hi, n_quad=n_quad,
                )
                return torch.maximum(payoff(y), cont)
            return value

        value_at[k] = make_value(v_kp1, t_kp1)

    def value_at_time(xv: torch.Tensor, t0: float) -> torch.Tensor:
        # On an exercise date: the with-exercise value.  Strictly inside an
        # interval: pure continuation toward the next exercise date.
        for k in range(m):
            if abs(t0 - et[k]) < 1e-9:
                return value_at[k](xv)
        j = next((k for k in range(m) if et[k] > t0), None)
        if j is None:  # t beyond maturity — undefined; return payoff
            return payoff(xv)
        return heat_propagate(
            value_at[j], xv, torch.full_like(xv, t0), t_terminal=et[j],
            sigma=sigma, y_lo=y_lo, y_hi=y_hi, n_quad=n_quad,
        )

    tvals = t.expand_as(x) if t.shape != x.shape else t
    out = torch.empty_like(x)
    for tv in torch.unique(tvals):
        mask = tvals == tv
        out[mask] = value_at_time(x[mask], float(tv))
    return out


def smooth_call_payoff_cm_time(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    K: float,
    T: float,
    eps0: float,
) -> torch.Tensor:
    r"""Time-dependent one-sided Chen--Mangasarian smoothing of the call payoff.

    .. math::

        \Psi(x, t) = \tfrac12\Bigl[(e^x - K)
            + \sqrt{(e^x - K)^2 + \varepsilon(t)^2}\Bigr],
        \qquad \varepsilon(t) = \varepsilon_0\,\frac{T - t}{T}.

    The bandwidth :math:`\varepsilon(t)` vanishes at the terminal time, so at
    ``t = T`` this reduces **exactly** to the payoff :math:`(e^x - K)^+`
    (no terminal bias), while for ``t < T`` it is a smooth ``C^\infty``
    extension whose Black--Scholes/heat residual is bounded — the smoothing is
    spread along the time axis rather than concentrated at the strike.  This
    contrasts with the static :func:`smooth_call_payoff` (constant bandwidth,
    never exact at ``t = T``).

    Args:
        x:    Log-price coordinate.
        t:    Time coordinate.
        K:    Strike.
        T:    Terminal time.
        eps0: Bandwidth at ``t = 0`` (price units); ``eps0 = 0`` recovers the
              exact non-differentiable payoff at all times.
    """
    diff = torch.exp(x) - K
    eps = eps0 * (T - t) / T
    return 0.5 * (diff + torch.sqrt(diff**2 + eps**2))


def heat_call_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    K: float,
    T: float,
    sigma: float,
) -> torch.Tensor:
    r"""Exact pure-heat solution with the call payoff ``(e^x - K)^+`` at ``t = T``.

    Under ``d_t u + (sigma^2 / 2) d_xx u = 0`` the solution is the Gaussian
    convolution of the terminal payoff, i.e. a zero-rate Black--Scholes call
    with the no-drift convexity correction ``e^{sigma^2 tau / 2}``:

    .. math::

        u(t, x) = e^{x + \sigma^2 \tau / 2}\, N(d_1) - K\, N(d_2),
        \qquad \tau = T - t,

    with :math:`d_1 = (x - \ln K + \sigma^2 \tau) / (\sigma \sqrt{\tau})` and
    :math:`d_2 = d_1 - \sigma \sqrt{\tau}`.  At ``tau = 0`` it reduces to the
    payoff.

    Args:
        x:     Log-price coordinate.
        t:     Time coordinate.
        K:     Strike.
        T:     Terminal time.
        sigma: Diffusion scale.
    """
    tau = torch.clamp(T - t, min=1e-12)
    sqrt_tau = sigma * torch.sqrt(tau)
    d1 = (x - math.log(K) + sigma**2 * tau) / sqrt_tau
    d2 = d1 - sqrt_tau
    return torch.exp(x + 0.5 * sigma**2 * tau) * _normal_cdf(d1) - K * _normal_cdf(d2)
