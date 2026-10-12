r"""Importance sampling of the interior collocation points towards the barrier-maturity corner.

Domain and target measure. The interior residual loss of the barrier pilot is a Monte Carlo
estimator of

.. math::

    \mathcal{J}(\theta) = \int_{\Omega} u(s,t)\,\big(\mathcal{L}^{BS}\Phi_\theta\big)^2(s,t)\,ds\,dt,
    \qquad \Omega = (B, s_\infty)\times(0, T),\qquad u = \frac{1}{(s_\infty - B)\,T},

with collocation points drawn from the uniform density :math:`u`. Under the similarity
variable :math:`\xi(s,\tau) = \ln(s/B)/(\sigma\sqrt{2\tau})`, :math:`\tau = T - t`, the corner
layer occupies a region of area of order :math:`\sqrt{\tau}` at each time to maturity, so the
uniform density places a fraction of order :math:`\varepsilon` of the points at
:math:`\tau < \varepsilon`.

Corner component. A point of the corner component is drawn with

- :math:`\tau` log-uniform on :math:`[\tau_{\min}, T]`: :math:`\tau = \tau_{\min}(T/\tau_{\min})^{U}`,
- :math:`\xi` uniform on :math:`[0, \xi_{\max}]`: :math:`\xi = \xi_{\max} V`,
- :math:`s = B\,e^{\xi\sigma\sqrt{2\tau}}`, redrawn while :math:`s > s_\infty`,

with :math:`U, V` independent and uniform on :math:`[0,1)`. With :math:`L = \ln(T/\tau_{\min})`
and :math:`A` the acceptance probability of the rejection step, the change of variables
:math:`(\xi,\tau)\mapsto(s,\tau)` (Jacobian :math:`\partial_s\xi = 1/(s\,\sigma\sqrt{2\tau})`) gives
the density of the accepted points,

.. math::

    q(s,\tau) = \frac{1}{A\,L\,\xi_{\max}}\;\frac{1}{\tau\,s\,\sigma\sqrt{2\tau}}\;
    \mathbf{1}\{\tau_{\min}\le\tau\le T,\ 0\le\xi(s,\tau)\le\xi_{\max},\ s\le s_\infty\}.

Acceptance probability (closed form). Set :math:`a = \ln(s_\infty/B)/(\sigma\sqrt{2}\,\xi_{\max})`,
so that the point is accepted with conditional probability :math:`\min(1, a\,\tau^{-1/2})`
given :math:`\tau`, and :math:`\tau_\star = a^2` clipped to :math:`[\tau_{\min}, T]`. Then

.. math::

    A = \frac{1}{L}\Big(\ln\frac{\tau_\star}{\tau_{\min}} + 2a\,\big(\tau_\star^{-1/2} - T^{-1/2}\big)\Big).

Mixture and weights. A batch of :math:`n_f` points holds :math:`n_c = \operatorname{round}(p\,n_f)`
points of the corner component and :math:`n_f - n_c` uniform points. With the effective
fraction :math:`\bar p = n_c/n_f` and the mixture density :math:`m = (1-\bar p)\,u + \bar p\,q`,
the weighted loss

.. math::

    \widehat{\mathcal{J}} = \frac{1}{n_f}\sum_{i=1}^{n_f} w(x_i)\,\big(\mathcal{L}^{BS}\Phi_\theta\big)^2(x_i),
    \qquad w = \frac{u}{m} = \frac{1}{(1-\bar p) + \bar p\,q/u},

satisfies :math:`\mathbb{E}\,\widehat{\mathcal{J}} = \mathcal{J}` (proof: the expectation of the
sum is :math:`n_f\int m\,w\,(\mathcal{L}^{BS}\Phi_\theta)^2 = n_f\int u\,(\mathcal{L}^{BS}\Phi_\theta)^2`,
since the two groups have fixed sizes :math:`n_f(1-\bar p)` and :math:`n_f\bar p`). The weights
satisfy :math:`0 < w \le 1/(1-\bar p)`, with equality outside the support of :math:`q`.

Weights are computed in float64 at the float64 coordinates of each draw, before the points are
rounded to the training dtype. A uniform draw is exactly representable, so the two coincide; for
a corner draw, the rounding of :math:`s` to float32 shifts :math:`\xi` by up to
:math:`1.4\cdot10^{-4}` at :math:`\tau = 10^{-6}`, which would move a draw from just inside
:math:`\xi\le\xi_{\max}` (weight of order :math:`10^{-8}`) to outside (weight :math:`1/(1-\bar p)`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

#: Guard on the rejection loop of the corner component. The acceptance probability is
#: 0.995 for the pilot contract (B = 0.6, s_inf = 3, sigma = 0.3, T = 1, tau_min = 1e-6,
#: xi_max = 5), so one or two passes suffice in practice.
CORNER_REJECTION_MAXIMUM_NUMBER_OF_PASSES = 100


@dataclass(frozen=True)
class CornerSimilaritySampling:
    r"""Mixture of the uniform density on :math:`\Omega` and the corner component (module docstring).

    Attributes:
        barrier: Knock-out barrier :math:`B > 0`.
        truncation_price: Upper end of the price range :math:`s_\infty > B`.
        maturity: Maturity :math:`T > 0`.
        volatility: Volatility :math:`\sigma > 0` of the similarity variable.
        corner_fraction: Requested fraction :math:`p\in(0,1)` of corner points per batch.
        smallest_time_to_maturity: Lower end :math:`\tau_{\min}\in(0,T)` of the log-uniform law.
        similarity_variable_max: Upper end :math:`\xi_{\max} > 0` of the uniform law of :math:`\xi`.
    """

    barrier: float
    truncation_price: float
    maturity: float
    volatility: float
    corner_fraction: float = 0.5
    smallest_time_to_maturity: float = 1e-6
    similarity_variable_max: float = 5.0

    def __post_init__(self) -> None:
        if not 0.0 < self.barrier < self.truncation_price:
            raise ValueError(f"need 0 < barrier < truncation_price; got {self.barrier}, {self.truncation_price}.")
        if not 0.0 < self.corner_fraction < 1.0:
            raise ValueError(f"corner_fraction must lie in (0, 1) (the uniform component keeps the weights "
                             f"bounded by 1/(1-p)); got {self.corner_fraction}.")
        if not 0.0 < self.smallest_time_to_maturity < self.maturity:
            raise ValueError(f"need 0 < smallest_time_to_maturity < maturity; got {self.smallest_time_to_maturity}, "
                             f"{self.maturity}.")
        if self.volatility <= 0.0 or self.similarity_variable_max <= 0.0:
            raise ValueError(f"volatility and similarity_variable_max must be > 0; got {self.volatility}, "
                             f"{self.similarity_variable_max}.")

    # ---- deterministic quantities ---------------------------------------------------------

    def number_of_corner_points(self, number_of_points: int) -> int:
        """Size :math:`n_c = \\operatorname{round}(p\\,n_f)` of the corner group of a batch."""
        return int(round(self.corner_fraction * number_of_points))

    def effective_corner_fraction(self, number_of_points: int) -> float:
        r""":math:`\bar p = n_c/n_f`, the fraction that enters the mixture density."""
        return self.number_of_corner_points(number_of_points) / number_of_points

    def log_time_range(self) -> float:
        r""":math:`L = \ln(T/\tau_{\min})`."""
        return math.log(self.maturity / self.smallest_time_to_maturity)

    def acceptance_probability(self) -> float:
        r"""Closed-form probability :math:`A` that a corner draw satisfies :math:`s\le s_\infty`."""
        a = math.log(self.truncation_price / self.barrier) / (self.volatility * math.sqrt(2.0) * self.similarity_variable_max)
        tau_star = min(max(a * a, self.smallest_time_to_maturity), self.maturity)
        return (math.log(tau_star / self.smallest_time_to_maturity)
                + 2.0 * a * (tau_star**-0.5 - self.maturity**-0.5)) / self.log_time_range()

    def uniform_density(self) -> float:
        r""":math:`u = 1/((s_\infty - B)\,T)`."""
        return 1.0 / ((self.truncation_price - self.barrier) * self.maturity)

    def corner_density(self, s: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        r"""Density :math:`q(s,\tau)` of the accepted corner draws (zero outside its support), float64."""
        s = s.to(torch.float64)
        tau = tau.to(torch.float64)
        tau_positive = torch.where(tau > 0, tau, torch.ones_like(tau))  # value unused where tau <= 0 (masked below)
        similarity_width = self.volatility * torch.sqrt(2.0 * tau_positive)
        xi = torch.log(s / self.barrier) / similarity_width
        inside_support = ((tau >= self.smallest_time_to_maturity) & (tau <= self.maturity)
                          & (xi >= 0.0) & (xi <= self.similarity_variable_max) & (s <= self.truncation_price))
        normalisation = self.acceptance_probability() * self.log_time_range() * self.similarity_variable_max
        density = 1.0 / (normalisation * tau_positive * s * similarity_width)
        return torch.where(inside_support, density, torch.zeros_like(density))

    def importance_weights(self, s: torch.Tensor, tau: torch.Tensor, effective_corner_fraction: float) -> torch.Tensor:
        r""":math:`w = u/m = 1/((1-\bar p) + \bar p\,q/u)` at the given float64 coordinates."""
        density_ratio = self.corner_density(s, tau) / self.uniform_density()
        return 1.0 / ((1.0 - effective_corner_fraction) + effective_corner_fraction * density_ratio)

    # ---- sampling ----------------------------------------------------------------------------

    def sample_corner_component(self, number_of_points: int, generator: torch.Generator
                                ) -> tuple[torch.Tensor, torch.Tensor, int]:
        r"""Draw ``number_of_points`` accepted corner points.

        Rejected points (:math:`s > s_\infty`) are redrawn, so the result always has exactly
        ``number_of_points`` entries and is a deterministic function of ``generator``.

        Returns:
            ``(s, tau, number_of_rejected_draws)``: float64 tensors and the total rejection count.
        """
        def draw(count: int) -> tuple[torch.Tensor, torch.Tensor]:
            uniform_for_time = torch.rand(count, generator=generator, dtype=torch.float64)
            uniform_for_similarity = torch.rand(count, generator=generator, dtype=torch.float64)
            tau = self.smallest_time_to_maturity * (self.maturity / self.smallest_time_to_maturity) ** uniform_for_time
            xi = self.similarity_variable_max * uniform_for_similarity
            return self.barrier * torch.exp(xi * self.volatility * torch.sqrt(2.0 * tau)), tau

        s, tau = draw(number_of_points)
        number_of_rejected_draws = 0
        for _ in range(CORNER_REJECTION_MAXIMUM_NUMBER_OF_PASSES):
            rejected = s > self.truncation_price
            count = int(rejected.sum())
            if count == 0:
                return s, tau, number_of_rejected_draws
            number_of_rejected_draws += count
            s[rejected], tau[rejected] = draw(count)
        raise RuntimeError(f"corner rejection sampling did not converge in {CORNER_REJECTION_MAXIMUM_NUMBER_OF_PASSES} "
                           f"passes (acceptance probability {self.acceptance_probability():.3e}).")

    def sample_mixture(self, number_of_points: int, generator: torch.Generator, dtype: torch.dtype
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        r"""One batch of the mixture: uniform points first, then corner points.

        The uniform group is drawn exactly as the pilot's plain sampler (``torch.rand`` in the
        default dtype, ``s`` first, then ``t``), followed by the corner group.

        Returns:
            ``(s, t, weights, statistics)``: ``s`` and ``t`` in ``dtype``, ``weights`` in ``dtype``
            (computed in float64 at the pre-rounding coordinates), and a dict with the group
            sizes, the rejection count and the extreme weights.
        """
        number_of_corner_points = self.number_of_corner_points(number_of_points)
        number_of_uniform_points = number_of_points - number_of_corner_points
        s_uniform = (torch.rand(number_of_uniform_points, generator=generator) * (self.truncation_price - self.barrier)
                     + self.barrier)
        t_uniform = torch.rand(number_of_uniform_points, generator=generator) * self.maturity
        s_corner, tau_corner, number_of_rejected_draws = self.sample_corner_component(number_of_corner_points, generator)

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
