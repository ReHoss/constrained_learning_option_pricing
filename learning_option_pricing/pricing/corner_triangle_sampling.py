r"""Control law of the corner importance sampling: uniform on the :math:`\ell^1` corner triangle.

Control experiment of :mod:`learning_option_pricing.pricing.corner_similarity_sampling`. That
module draws its corner component in the similarity coordinates :math:`(\xi, \ln\tau)`, i.e. with a
density that follows the shape of the corner layer. The control replaces it by the plainest law
concentrated at the same corner :math:`c = (B, T)`, the uniform law on the :math:`\ell^1` triangle

.. math::

    \mathcal{T}_\delta = \{(s,t)\in\Omega:\ (s-B) + (T-t) \le \delta\},
    \qquad |\mathcal{T}_\delta| = \frac{\delta^2}{2},
    \qquad q_\delta = \frac{2}{\delta^2}\,\mathbf{1}_{\mathcal{T}_\delta},

which carries no information on the similarity variable. The comparison of the two laws at the
same corner fraction separates the effect of concentrating points at the corner from the effect of
distributing them along the similarity variable.

Domain and target measure are those of the similarity module: :math:`\Omega = (B, s_\infty)\times(0,T)`,
uniform density :math:`u = 1/((s_\infty - B)\,T)`, and the loss estimates
:math:`\int_\Omega u\,(\mathcal{L}^{BS}\Phi_\theta)^2`. The triangle lies in :math:`\Omega` when
:math:`\delta \le \min(s_\infty - B, T)`.

Corner component. With :math:`x = s - B` and :math:`\tau = T - t`, draw :math:`U, V` independent and
uniform on :math:`[0, 1)`, replace :math:`(U, V)` by :math:`(1-U, 1-V)` when :math:`U + V > 1`, and
set :math:`x = \delta U`, :math:`\tau = \delta V`. The reflection maps the upper triangle of the unit
square onto the lower one with unit Jacobian, so :math:`(x,\tau)` is uniform on the triangle
:math:`\{x\ge0,\ \tau\ge0,\ x+\tau\le\delta\}` (density :math:`2/\delta^2`). No rejection on the
domain is needed.

Exclusion of the maturity slice in the training dtype. A draw whose time :math:`t = T-\tau`, rounded
to the training dtype, equals :math:`T` is evaluated by the network at :math:`\tau = 0`, where the
similarity variable :math:`\xi = \ln(s/B)/(\sigma\sqrt{2\tau})` is not defined. Such a draw is
redrawn. In float32 with :math:`T = 1` this happens when :math:`\tau < 2^{-25}`, with probability at
most :math:`(2/\delta)\,2^{-25} = 6.0\cdot10^{-7}` per draw at :math:`\delta = 0.1`, so the density
of the accepted draws differs from :math:`q_\delta` by a relative amount of that order. The number of
redraws is returned in the statistics (``number_of_rejected_corner_draws``) and logged by the pilot
at every logged iteration, so the exclusion never binds silently.

Mixture and weights. A batch of :math:`n_f` points holds :math:`n_c = \operatorname{round}(p\,n_f)`
corner points and :math:`n_f - n_c` uniform points. With :math:`\bar p = n_c/n_f`, the mixture
density is :math:`m = (1-\bar p)\,u + \bar p\,q_\delta` and the weights are

.. math::

    w = \frac{u}{m} = \frac{1}{(1-\bar p) + \bar p\,\frac{2}{\delta^2 u}\,\mathbf{1}_{\mathcal{T}_\delta}},

two values only: :math:`1/(1-\bar p)` outside the triangle and
:math:`1/((1-\bar p) + 2\bar p/(\delta^2 u))` inside (:math:`4.16\cdot10^{-3}` for the pilot contract,
:math:`\bar p = 0.5`, :math:`\delta = 0.1`). The weighted mean
:math:`\frac{1}{n_f}\sum_i w(x_i)\,f(x_i)` is an unbiased estimator of :math:`\int_\Omega u\,f` for
every integrable :math:`f`, by the argument of the similarity module (the two groups have fixed
sizes :math:`n_f(1-\bar p)` and :math:`n_f\bar p`, so the expectation of the sum is
:math:`n_f\int m\,w\,f = n_f\int u\,f`).

Weights are computed in float64 at the float64 coordinates of each draw, before rounding to the
training dtype, as in the similarity module; a uniform draw is exactly representable, so the two
coincide, and a corner draw may cross the edge :math:`x + \tau = \delta` by rounding only within one
unit in the last place of that edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: Guard on the redraw loop of the maturity-slice exclusion (module docstring). The probability of
#: one redraw is at most 6.0e-7 per draw for the pilot contract, so one pass suffices in practice.
CORNER_TRIANGLE_REDRAW_MAXIMUM_NUMBER_OF_PASSES = 100


@dataclass(frozen=True)
class CornerTriangleSampling:
    r"""Mixture of the uniform density on :math:`\Omega` and the uniform law on the :math:`\ell^1`
    corner triangle :math:`\mathcal{T}_\delta` (module docstring).

    The interface (``sample_mixture``, ``number_of_corner_points``, ``effective_corner_fraction``,
    ``uniform_density``, ``importance_weights``) is that of
    :class:`~learning_option_pricing.pricing.corner_similarity_sampling.CornerSimilaritySampling`, so
    the training loop draws a batch from either law without distinction.

    Attributes:
        barrier: Knock-out barrier :math:`B > 0`.
        truncation_price: Upper end of the price range :math:`s_\infty > B`.
        maturity: Maturity :math:`T > 0`.
        triangle_size: Size :math:`\delta` of the triangle, :math:`0 < \delta \le \min(s_\infty - B, T)`.
        corner_fraction: Requested fraction :math:`p\in(0,1)` of corner points per batch.
    """

    barrier: float
    truncation_price: float
    maturity: float
    triangle_size: float = 0.1
    corner_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.barrier < self.truncation_price:
            raise ValueError(f"need 0 < barrier < truncation_price; got {self.barrier}, {self.truncation_price}.")
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be > 0; got {self.maturity}.")
        if not 0.0 < self.corner_fraction < 1.0:
            raise ValueError(f"corner_fraction must lie in (0, 1) (the uniform component keeps the weights "
                             f"bounded by 1/(1-p)); got {self.corner_fraction}.")
        largest_admissible_size = min(self.truncation_price - self.barrier, self.maturity)
        if not 0.0 < self.triangle_size <= largest_admissible_size:
            raise ValueError(f"triangle_size must lie in (0, min(s_inf - B, T)] = (0, {largest_admissible_size:g}] so "
                             f"that the triangle lies in the domain; got {self.triangle_size}.")

    # ---- deterministic quantities ---------------------------------------------------------

    def number_of_corner_points(self, number_of_points: int) -> int:
        """Size :math:`n_c = \\operatorname{round}(p\\,n_f)` of the corner group of a batch."""
        return int(round(self.corner_fraction * number_of_points))

    def effective_corner_fraction(self, number_of_points: int) -> float:
        r""":math:`\bar p = n_c/n_f`, the fraction that enters the mixture density."""
        return self.number_of_corner_points(number_of_points) / number_of_points

    def uniform_density(self) -> float:
        r""":math:`u = 1/((s_\infty - B)\,T)`."""
        return 1.0 / ((self.truncation_price - self.barrier) * self.maturity)

    def triangle_area(self) -> float:
        r""":math:`|\mathcal{T}_\delta| = \delta^2/2`."""
        return 0.5 * self.triangle_size**2

    def corner_density(self, s: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        r"""Density :math:`q_\delta = (2/\delta^2)\,\mathbf{1}_{\mathcal{T}_\delta}` at :math:`(s,\tau)`, float64."""
        distance_to_barrier = s.to(torch.float64) - self.barrier
        tau = tau.to(torch.float64)
        inside_triangle = ((distance_to_barrier >= 0.0) & (tau >= 0.0)
                           & (distance_to_barrier + tau <= self.triangle_size))
        return torch.where(inside_triangle, torch.full_like(tau, 1.0 / self.triangle_area()), torch.zeros_like(tau))

    def importance_weights(self, s: torch.Tensor, tau: torch.Tensor, effective_corner_fraction: float) -> torch.Tensor:
        r""":math:`w = u/m = 1/((1-\bar p) + \bar p\,q_\delta/u)` at the given float64 coordinates."""
        density_ratio = self.corner_density(s, tau) / self.uniform_density()
        return 1.0 / ((1.0 - effective_corner_fraction) + effective_corner_fraction * density_ratio)

    # ---- sampling ----------------------------------------------------------------------------

    def sample_corner_component(self, number_of_points: int, generator: torch.Generator, dtype: torch.dtype
                                ) -> tuple[torch.Tensor, torch.Tensor, int]:
        r"""Draw ``number_of_points`` points uniform on the triangle, none on the maturity slice in ``dtype``.

        A draw whose time :math:`T - \tau` rounds to :math:`T` in ``dtype`` is redrawn (module
        docstring), so the result always has exactly ``number_of_points`` entries and is a
        deterministic function of ``generator``.

        Returns:
            ``(s, tau, number_of_rejected_draws)``: float64 tensors and the total redraw count.
        """
        def draw(count: int) -> tuple[torch.Tensor, torch.Tensor]:
            uniform_for_price = torch.rand(count, generator=generator, dtype=torch.float64)
            uniform_for_time = torch.rand(count, generator=generator, dtype=torch.float64)
            reflected = uniform_for_price + uniform_for_time > 1.0
            uniform_for_price = torch.where(reflected, 1.0 - uniform_for_price, uniform_for_price)
            uniform_for_time = torch.where(reflected, 1.0 - uniform_for_time, uniform_for_time)
            return self.barrier + self.triangle_size * uniform_for_price, self.triangle_size * uniform_for_time

        def on_maturity_slice_in_dtype(tau: torch.Tensor) -> torch.Tensor:
            return (self.maturity - tau).to(dtype) >= torch.tensor(self.maturity, dtype=dtype)

        s, tau = draw(number_of_points)
        number_of_rejected_draws = 0
        for _ in range(CORNER_TRIANGLE_REDRAW_MAXIMUM_NUMBER_OF_PASSES):
            rejected = on_maturity_slice_in_dtype(tau)
            count = int(rejected.sum())
            if count == 0:
                return s, tau, number_of_rejected_draws
            number_of_rejected_draws += count
            s[rejected], tau[rejected] = draw(count)
        raise RuntimeError(f"corner triangle redraw did not converge in {CORNER_TRIANGLE_REDRAW_MAXIMUM_NUMBER_OF_PASSES} "
                           f"passes (triangle size {self.triangle_size}, dtype {dtype}).")

    def sample_mixture(self, number_of_points: int, generator: torch.Generator, dtype: torch.dtype
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r"""One batch of the mixture: uniform points first, then corner points.

        The uniform group is drawn exactly as the pilot's plain sampler (``torch.rand`` in the
        default dtype, ``s`` first, then ``t``), followed by the corner group, as in
        :meth:`CornerSimilaritySampling.sample_mixture`.

        Returns:
            ``(s, t, weights, statistics)``: ``s`` and ``t`` in ``dtype``, ``weights`` in ``dtype``
            (computed in float64 at the pre-rounding coordinates), and a dict with the group
            sizes, the redraw count and the extreme weights.
        """
        number_of_corner_points = self.number_of_corner_points(number_of_points)
        number_of_uniform_points = number_of_points - number_of_corner_points
        s_uniform = (torch.rand(number_of_uniform_points, generator=generator) * (self.truncation_price - self.barrier)
                     + self.barrier)
        t_uniform = torch.rand(number_of_uniform_points, generator=generator) * self.maturity
        s_corner, tau_corner, number_of_rejected_draws = self.sample_corner_component(
            number_of_corner_points, generator, dtype)

        s = torch.cat([s_uniform.to(torch.float64), s_corner])
        tau = torch.cat([self.maturity - t_uniform.to(torch.float64), tau_corner])
        weights = self.importance_weights(s, tau, self.effective_corner_fraction(number_of_points))
        statistics = {
            "number_of_uniform_points": number_of_uniform_points,
            "number_of_corner_points": number_of_corner_points,
            "number_of_rejected_corner_draws": number_of_rejected_draws,
            "minimum_weight": float(weights.min()),
            "maximum_weight": float(weights.max()),
            "mean_weight": float(weights.mean()),
        }
        t = torch.cat([t_uniform.to(torch.float64), self.maturity - tau_corner])
        return s.to(dtype), t.to(dtype), weights.to(dtype), statistics
