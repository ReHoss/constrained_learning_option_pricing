r"""Penalty baseline P of the barrier comparison: soft terminal and barrier conditions.

Mode P of the controlled comparison (Section 6 of the working note, Definition 10) enforces
neither the terminal condition on :math:`\Sigma_T=\{t=T\}` nor the barrier condition on
:math:`\Sigma_B=\{s=B\}` by construction: the trial solution has no factor vanishing on those faces
and no extension of the data, and both conditions enter the objective as penalty terms,

.. math::

    \mathcal J(\theta) = \mu_\Omega\!\int_Q(\mathcal L^{BS}\Phi_\theta)^2
        + \lambda_T\,\mu_{\Sigma_T}\!\int_{\Sigma_T}(\Phi_\theta-g)^2
        + \lambda_B\,\mu_{\Sigma_B}\!\int_{\Sigma_B}\Phi_\theta^2,

with :math:`\mu_\Omega, \mu_{\Sigma_T}, \mu_{\Sigma_B}` the normalised (uniform probability)
measures of the three sets, :math:`g(s)=(K-s)^+` the terminal datum and the barrier datum zero.

Far segment. The truncated training domain :math:`(B,s_\infty)\times(0,T)` has a third parabolic
boundary face, :math:`\{s=s_\infty\}`. In the hard modes of the comparison run with the far-field
Dirichlet condition, the trial solution equals :math:`\pi(s_\infty,t)` there (:math:`\pi` the
terminal profile of the extension). To keep that condition identical across the modes, so that
the terminal and barrier enforcement is the only factor that differs, P keeps it hard:

.. math::

    \Phi_\theta(s,t) = \frac{s_\infty-s}{s_\infty-B}\,\Psi_\theta(s,t) + \pi(s_\infty,t),

the factor :math:`(s_\infty-s)/(s_\infty-B)` vanishing on the far segment only (it equals 1 on
:math:`\Sigma_B` and does not vanish on :math:`\Sigma_T`), and the lift :math:`\pi(s_\infty,t)`
being constant in :math:`s`. Without the far-field condition, :math:`\Phi_\theta=\Psi_\theta`.

Boundary sampling. Each face is parametrised by its distance to the corner :math:`c=(B,T)`:
:math:`x=s-B\in(0,s_\infty-B)` on :math:`\Sigma_T`, :math:`x=\tau=T-t\in(0,T)` on
:math:`\Sigma_B`. A batch is a mixture of a uniform group and a group log-uniform in :math:`x` on
:math:`[x_{\min},\ell]` (:class:`LogUniformMixtureOnInterval`); each squared boundary residual is
weighted by :math:`w=u/m` (uniform density over mixture density), so that the weighted mean is an
unbiased estimator of the normalised boundary integral above, exactly as the interior corner
sampling (:mod:`learning_option_pricing.pricing.corner_similarity_sampling`) does for the interior
term.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from learning_option_pricing.pricing.terminal import black_scholes_put, payoff_put

#: The terminal profiles whose value at the far segment can serve as the far-field datum.
FAR_FIELD_DATUM_PROFILES = ("raw", "black_scholes")


@dataclass(frozen=True)
class LogUniformMixtureOnInterval:
    r"""Mixture of the uniform law on :math:`(0,\ell)` and the log-uniform law on :math:`[x_{\min},\ell]`.

    With :math:`n` points per batch, :math:`n_c=\operatorname{round}(p\,n)` are drawn log-uniformly,
    :math:`x = x_{\min}(\ell/x_{\min})^U`, and :math:`n-n_c` uniformly, :math:`x=\ell V`
    (:math:`U,V` uniform on :math:`[0,1)`). With :math:`\bar p=n_c/n`, :math:`u=1/\ell`,
    :math:`q(x)=1/(x\,L)` on :math:`[x_{\min},\ell]` (:math:`L=\ln(\ell/x_{\min})`) and
    :math:`m=(1-\bar p)u+\bar p q`, the weights :math:`w=u/m=1/((1-\bar p)+\bar p\,\ell/(xL))` on the
    support of :math:`q` and :math:`1/(1-\bar p)` below :math:`x_{\min}`, so that
    :math:`0<w\le1/(1-\bar p)` and :math:`\mathbb E\,\frac1n\sum_iw(x_i)f(x_i)=\frac1\ell\int_0^\ell f`
    (the two groups have fixed sizes; same argument as for the interior corner sampling).

    Attributes:
        length: Length :math:`\ell>0` of the interval.
        smallest_value: Lower end :math:`x_{\min}\in(0,\ell)` of the log-uniform law.
        log_uniform_fraction: Requested fraction :math:`p\in(0,1)` of log-uniform points.
    """

    length: float
    smallest_value: float
    log_uniform_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.smallest_value < self.length:
            raise ValueError(f"need 0 < smallest_value < length; got {self.smallest_value}, {self.length}.")
        if not 0.0 < self.log_uniform_fraction < 1.0:
            raise ValueError(f"log_uniform_fraction must lie in (0, 1) (the uniform group keeps the weights bounded "
                             f"by 1/(1-p)); got {self.log_uniform_fraction}.")

    def number_of_log_uniform_points(self, number_of_points: int) -> int:
        """Size :math:`n_c=\\operatorname{round}(p\\,n)` of the log-uniform group."""
        return int(round(self.log_uniform_fraction * number_of_points))

    def effective_log_uniform_fraction(self, number_of_points: int) -> float:
        r""":math:`\bar p=n_c/n`, the fraction entering the mixture density."""
        return self.number_of_log_uniform_points(number_of_points) / number_of_points

    def log_range(self) -> float:
        r""":math:`L=\ln(\ell/x_{\min})`."""
        return math.log(self.length / self.smallest_value)

    def importance_weights(self, x: torch.Tensor, effective_fraction: float) -> torch.Tensor:
        r""":math:`w=u/m` at the float64 values ``x``."""
        x = x.to(torch.float64)
        inside_support = (x >= self.smallest_value) & (x <= self.length)
        x_positive = torch.where(inside_support, x, torch.ones_like(x))  # value unused outside the support
        density_ratio = torch.where(inside_support, self.length / (x_positive * self.log_range()), torch.zeros_like(x))
        return 1.0 / ((1.0 - effective_fraction) + effective_fraction * density_ratio)

    def sample(self, number_of_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        """One batch, uniform group first, then log-uniform group, both from ``generator``.

        Returns:
            ``(x, weights)``, float64 tensors of shape ``(number_of_points,)``.
        """
        number_of_log_uniform_points = self.number_of_log_uniform_points(number_of_points)
        number_of_uniform_points = number_of_points - number_of_log_uniform_points
        x_uniform = torch.rand(number_of_uniform_points, generator=generator, dtype=torch.float64) * self.length
        uniform_for_log = torch.rand(number_of_log_uniform_points, generator=generator, dtype=torch.float64)
        x_log_uniform = self.smallest_value * (self.length / self.smallest_value) ** uniform_for_log
        x = torch.cat([x_uniform, x_log_uniform])
        return x, self.importance_weights(x, self.effective_log_uniform_fraction(number_of_points))


@dataclass(frozen=True)
class BarrierBoundaryPenaltySampling:
    r"""Weighted samplers of the terminal face :math:`\Sigma_T` and the barrier face :math:`\Sigma_B`.

    :math:`\Sigma_T`: :math:`s=B+x`, :math:`x` from a :class:`LogUniformMixtureOnInterval` of length
    :math:`s_\infty-B` and lower end ``smallest_distance_to_barrier``, :math:`t=T`.
    :math:`\Sigma_B`: :math:`s=B`, :math:`t=T-\tau`, :math:`\tau` from a mixture of length :math:`T`
    and lower end ``smallest_time_to_maturity``.

    Attributes:
        barrier: Knock-out barrier :math:`B>0`.
        truncation_price: :math:`s_\infty>B`.
        maturity: :math:`T>0`.
        smallest_distance_to_barrier: Lower end of the log-uniform law of :math:`s-B` on :math:`\Sigma_T`.
        smallest_time_to_maturity: Lower end of the log-uniform law of :math:`\tau` on :math:`\Sigma_B`.
        log_uniform_fraction: Fraction of log-uniform points on each face.
    """

    barrier: float
    truncation_price: float
    maturity: float
    smallest_distance_to_barrier: float = 1e-6
    smallest_time_to_maturity: float = 1e-6
    log_uniform_fraction: float = 0.5

    def terminal_face_law(self) -> LogUniformMixtureOnInterval:
        return LogUniformMixtureOnInterval(self.truncation_price - self.barrier, self.smallest_distance_to_barrier,
                                           self.log_uniform_fraction)

    def barrier_face_law(self) -> LogUniformMixtureOnInterval:
        return LogUniformMixtureOnInterval(self.maturity, self.smallest_time_to_maturity, self.log_uniform_fraction)

    def __post_init__(self) -> None:
        if not 0.0 < self.barrier < self.truncation_price:
            raise ValueError(f"need 0 < barrier < truncation_price; got {self.barrier}, {self.truncation_price}.")
        self.terminal_face_law()  # validates the remaining fields
        self.barrier_face_law()

    def sample(self, number_of_points_per_face: int, generator: torch.Generator, dtype: torch.dtype) -> dict:
        """One batch on each face, terminal face first, from ``generator``.

        Returns:
            A dict with ``terminal_s``, ``terminal_t``, ``terminal_weights``, ``barrier_s``,
            ``barrier_t``, ``barrier_weights``, all in ``dtype`` (weights computed in float64 at the
            pre-rounding coordinates).
        """
        distance_to_barrier, terminal_weights = self.terminal_face_law().sample(number_of_points_per_face, generator)
        time_to_maturity, barrier_weights = self.barrier_face_law().sample(number_of_points_per_face, generator)
        return {
            "terminal_s": (self.barrier + distance_to_barrier).to(dtype),
            "terminal_t": torch.full((number_of_points_per_face,), self.maturity, dtype=dtype),
            "terminal_weights": terminal_weights.to(dtype),
            "barrier_s": torch.full((number_of_points_per_face,), self.barrier, dtype=dtype),
            "barrier_t": (self.maturity - time_to_maturity).to(dtype),
            "barrier_weights": barrier_weights.to(dtype),
        }


def far_field_distance_factor(s: torch.Tensor, B: float, s_inf: float) -> torch.Tensor:
    r""":math:`(s_\infty-s)/(s_\infty-B)`: vanishes on the far segment only, equals 1 at :math:`s=B`."""
    if not s_inf > B:
        raise ValueError(f"far_field_distance_factor needs s_inf > B; got {s_inf=}, {B=}.")
    return (s_inf - s) / (s_inf - B)


class FarFieldDatumLift:
    r"""Lift :math:`(s,t)\mapsto\pi(s_\infty,t)`, constant in :math:`s`, of the far-field Dirichlet datum.

    ``terminal_profile="black_scholes"``: :math:`\pi(s_\infty,t)=V^e(s_\infty,t)`, the European put
    price, which is the far datum of the hard modes run with ``--black-scholes-payoff`` (their
    cutoff vanishes at :math:`s_\infty`, so their extension equals the profile there).
    ``terminal_profile="raw"``: :math:`(K-s_\infty)^+`. Contains no trained parameter, carries no
    information on the terminal or barrier datum away from :math:`s=s_\infty`.
    """

    def __init__(self, K: float, r: float, sigma: float, T: float, s_inf: float, terminal_profile: str) -> None:
        if terminal_profile not in FAR_FIELD_DATUM_PROFILES:
            raise ValueError(f"terminal_profile must be one of {FAR_FIELD_DATUM_PROFILES}; got {terminal_profile!r}.")
        self.K, self.r, self.sigma, self.T, self.s_inf = K, r, sigma, T, s_inf
        self.terminal_profile = terminal_profile

    def __call__(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        far_price = torch.full_like(s, self.s_inf) + 0.0 * t
        if self.terminal_profile == "black_scholes":
            return black_scholes_put(far_price, self.K, self.r, self.sigma, self.T - t)
        return payoff_put(far_price, self.K)


def weighted_boundary_penalties(model: torch.nn.Module, boundary_batch: dict, K: float,
                                device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Weighted mean squared boundary residuals of ``model`` on one boundary batch.

    Returns:
        ``(terminal_penalty, barrier_penalty)``:
        :math:`\frac1n\sum_iw_i(\Phi_\theta(s_i,T)-(K-s_i)^+)^2` and
        :math:`\frac1n\sum_jw_j\Phi_\theta(B,t_j)^2`, in the graph of the model parameters.
    """
    def on_device(key: str) -> torch.Tensor:
        return boundary_batch[key] if device is None else boundary_batch[key].to(device)

    terminal_s = on_device("terminal_s")
    terminal_values = model(torch.stack([terminal_s, on_device("terminal_t")], dim=1)).squeeze(-1)
    terminal_penalty = torch.mean(on_device("terminal_weights") * (terminal_values - payoff_put(terminal_s, K)) ** 2)
    barrier_values = model(torch.stack([on_device("barrier_s"), on_device("barrier_t")], dim=1)).squeeze(-1)
    barrier_penalty = torch.mean(on_device("barrier_weights") * barrier_values**2)
    return terminal_penalty, barrier_penalty


def boundary_constraint_violations(model: torch.nn.Module, K: float, B: float, T: float, s_inf: float,
                                   smallest_distance: float = 1e-6, points_per_scale: int = 2001,
                                   corner_window: float = 0.1) -> dict:
    r"""Sup-norm violations of the terminal and barrier conditions on deterministic face grids.

    Each face grid is the union of a geometric grid of the distance :math:`x` to the corner on
    :math:`[x_{\min},\ell]` and a uniform grid of :math:`(0,\ell]`, :math:`x_{\min}` =
    ``smallest_distance``; the corner point itself (:math:`x=0`), where the two data conflict, is
    excluded. Reported: :math:`\max|\Phi(s,T)-(K-s)^+|` and :math:`\max|\Phi(B,t)|` on the whole
    grid and on the part outside the corner window :math:`x>` ``corner_window``.
    """
    def face_distances(length: float) -> torch.Tensor:
        geometric = torch.logspace(math.log10(smallest_distance), math.log10(length), points_per_scale,
                                   dtype=torch.float64)
        uniform = torch.linspace(0.0, length, points_per_scale, dtype=torch.float64)[1:]
        return torch.sort(torch.cat([geometric, uniform])).values

    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    distance_to_barrier = face_distances(s_inf - B)
    time_to_maturity = face_distances(T)
    terminal_s = (B + distance_to_barrier).to(dtype)
    barrier_t = (T - time_to_maturity).to(dtype)
    with torch.no_grad():
        terminal_values = model(torch.stack([terminal_s, torch.full_like(terminal_s, T)], dim=1).to(device))
        barrier_values = model(torch.stack([torch.full_like(barrier_t, B), barrier_t], dim=1).to(device))
    terminal_error = (terminal_values.squeeze(-1).double().cpu() - payoff_put(terminal_s.double(), K)).abs()
    barrier_error = barrier_values.squeeze(-1).double().cpu().abs()
    return {
        "terminal_condition_max_abs_violation": float(terminal_error.max()),
        "terminal_condition_max_abs_violation_outside_corner": float(terminal_error[distance_to_barrier > corner_window].max()),
        "barrier_condition_max_abs_violation": float(barrier_error.max()),
        "barrier_condition_max_abs_violation_outside_corner": float(barrier_error[time_to_maturity > corner_window].max()),
        "boundary_violation_grid": {"smallest_distance": smallest_distance, "points_per_scale": points_per_scale,
                                    "corner_window": corner_window},
    }
