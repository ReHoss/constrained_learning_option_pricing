r"""Closed-form extension fields of the butterfly datum on the real line.

Setting.  The backward evolution problem

.. math::

    P u = \partial_t u + A u = 0 \ \text{on}\ \mathbb{R} \times (0, T),
    \qquad u(\cdot, T) = g,

has the constant-coefficient generator
:math:`A = \nu\,\partial_{xx} + \mu\,\partial_x + r_0`, with
:math:`\nu \in (0, +\infty)` and :math:`\mu, r_0 \in \mathbb{R}`, and the butterfly
datum

.. math::

    g(x) = (\ell - |x - x^\star|)^+ = \sum_{k=1}^{3} w_k\,(x - a_k)^+ ,

with half-width :math:`\ell \in (0, +\infty)`, kink points
:math:`(a_k)_{1\le k\le3} = (x^\star - \ell, x^\star, x^\star + \ell)` and
first-derivative jumps :math:`(w_k)_{1\le k\le3} = (1, -2, 1)`.  Throughout,
:math:`s = T - t` is the time to the terminal slice.

Every field of this module has the form

.. math::

    h(x, t) = e^{\rho s}\, K(x + \beta s, s),
    \qquad
    K(y, s) = \sum_{k=1}^{3} w_k\, q_s(y - a_k),

where :math:`q_s` is a smoothing of the plus function :math:`y \mapsto y^+`
with :math:`q_0(y) = y^+`, and where :math:`\rho`, :math:`\beta` and
:math:`q_s` depend on the extension kind:

===========================  ==============================  ==============  ==============  ===============================================
Extension kind               Retained operator               :math:`\rho`    :math:`\beta`   :math:`q_s(y)`
===========================  ==============================  ==============  ==============  ===============================================
``transported_datum``        :math:`\mu\partial_x + r_0`     :math:`r_0`     :math:`\mu`     :math:`y^+`
``split_diffusion``          :math:`\nu\partial_{xx}`        :math:`0`       :math:`0`       Gaussian, :math:`\sigma = \sqrt{2\nu s}`
``split_diffusion_advection``:math:`\nu\partial_{xx}+\mu\partial_x` :math:`0` :math:`\mu`  Gaussian, :math:`\sigma = \sqrt{2\nu s}`
``exact_solution``           :math:`A`                       :math:`r_0`     :math:`\mu`     Gaussian, :math:`\sigma = \sqrt{2\nu s}`
``graded_gaussian``          :math:`\nu_c\partial_{xx}`      :math:`0`       :math:`0`       Gaussian, :math:`\sigma = \sqrt{2\nu_c s}`
``graded_chen_mangasarian``  none                            :math:`0`       :math:`0`       algebraic, :math:`\varepsilon = \varepsilon_0 s / T`
===========================  ==============================  ==============  ==============  ===============================================

The Gaussian smoothing is the convolution of :math:`y^+` with the centred normal
density of standard deviation :math:`\sigma`,

.. math::

    q(y) = y\,\Phi(y/\sigma) + \sigma\,\varphi(y/\sigma),
    \qquad q' = \Phi(y/\sigma), \qquad q'' = \varphi(y/\sigma)/\sigma ,

with :math:`\Phi` and :math:`\varphi` the standard normal distribution function
and density; since :math:`\partial_\sigma q = \varphi(y/\sigma)`, it satisfies
:math:`\partial_s q = \nu_k\,q''` for :math:`\sigma = \sqrt{2\nu_k s}`.  The
algebraic (Chen--Mangasarian) smoothing is the convolution of :math:`y^+` with
the kernel :math:`\tfrac{1}{2\varepsilon}(1 + (y/\varepsilon)^2)^{-3/2}`, whose
Fourier multiplier is :math:`z K_1(z)` at :math:`z = |\xi|\varepsilon`, the
multiplier of the periodic ``graded_chen_mangasarian`` field:

.. math::

    q(y) = \tfrac12\bigl(y + \sqrt{y^2 + \varepsilon^2}\bigr),
    \quad q' = \tfrac12\bigl(1 + y/\sqrt{y^2 + \varepsilon^2}\bigr),
    \quad q'' = \frac{\varepsilon^2}{2\,(y^2 + \varepsilon^2)^{3/2}},
    \quad \partial_\varepsilon q = \frac{\varepsilon}{2\sqrt{y^2 + \varepsilon^2}} .

Analytic derivatives.  With :math:`y = x + \beta s`,

.. math::

    \partial_x h = e^{\rho s} K_y, \qquad
    \partial_{xx} h = e^{\rho s} K_{yy}, \qquad
    \partial_t h = -e^{\rho s}\bigl(\rho K + \beta K_y + K_s\bigr),

where :math:`K_s = \nu_k K_{yy}` for the Gaussian kinds,
:math:`K_s = (\varepsilon_0 / T) \sum_k w_k\,\partial_\varepsilon q` for the
algebraic kind and :math:`K_s = 0` for ``transported_datum``.  Hence the forcing
:math:`P h` vanishes for ``exact_solution``, equals :math:`r_0 h` for
``split_diffusion_advection`` and :math:`\mu\,\partial_x h + r_0 h` for
``split_diffusion``, and vanishes away from the lines
:math:`\{x + \mu s = a_k\}` for ``transported_datum``, whose second derivative
is a sum of Dirac masses on these lines that pointwise evaluation does not see.

Terminal slice.  At :math:`s = 0` the field returns the datum
:meth:`ButterflyDatum.values` evaluated in the caller's dtype, so that
``field(x, T) == datum.values(x)`` holds exactly in floating point (the
hard-constrained trial solution requires this identity).  The derivatives at
:math:`s = 0` are the almost-everywhere values :math:`g'` (with the midpoint
value at a kink), :math:`0` for the second derivative, and
:math:`-(\rho g + \beta g')` for the time derivative: the Dirac masses of
:math:`g''` are absent from these pointwise values.

Numerical policy.  The closed forms are evaluated in ``float64``.  ``torch``
inputs are computed in ``torch.float64`` and cast back to the input dtype, and
the computation is differentiable, so autograd through the field remains
available for the analytic-versus-autograd cross-check.  ``numpy`` inputs return
``float64`` arrays.  The scales :math:`\sigma` and :math:`\varepsilon` are
replaced by :math:`1` where :math:`s = 0` before any division, inside a
``where`` whose other branch is taken there; this replacement never reaches a
returned value, so it is not a clamp.  Every constructor argument is validated
and a violation raises :class:`ValueError`.
"""
from __future__ import annotations

import math

import numpy as np
import torch

SQUARE_ROOT_OF_TWO_PI = math.sqrt(2.0 * math.pi)

# numpy >= 2.0 names the trapezoidal rule trapezoid; earlier versions, trapz.
_trapezoidal_rule = getattr(np, "trapezoid", None) or np.trapz

REAL_LINE_BUTTERFLY_FIELD_KINDS = (
    "transported_datum",
    "split_diffusion",
    "split_diffusion_advection",
    "exact_solution",
    "graded_gaussian",
    "graded_chen_mangasarian",
)

# Per kind: (exponential rate source, transport velocity source, smoothing).
# The sources name the generator coefficient the kind retains ("reaction",
# "advection") or None for zero.
_KIND_STRUCTURE = {
    "transported_datum": ("reaction", "advection", "none"),
    "split_diffusion": (None, None, "gaussian"),
    "split_diffusion_advection": (None, "advection", "gaussian"),
    "exact_solution": ("reaction", "advection", "gaussian"),
    "graded_gaussian": (None, None, "gaussian"),
    "graded_chen_mangasarian": (None, None, "chen_mangasarian"),
}


def _midpoint_heaviside(values):
    r"""Heaviside function with the midpoint value :math:`1/2` at zero."""
    if isinstance(values, torch.Tensor):
        return 0.5 * (1.0 + torch.sign(values))
    return 0.5 * (1.0 + np.sign(values))


class ButterflyDatum:
    r"""The butterfly datum :math:`g(x) = (\ell - |x - x^\star|)^+` on :math:`\mathbb{R}`.

    Args:
        half_width: The half-width :math:`\ell \in (0, +\infty)`.
        singular_point: The centre :math:`x^\star \in \mathbb{R}`.

    Raises:
        ValueError: If ``half_width`` is not a finite positive number or
            ``singular_point`` is not finite.
    """

    def __init__(self, half_width: float, singular_point: float) -> None:
        half_width = float(half_width)
        singular_point = float(singular_point)
        if not (math.isfinite(half_width) and half_width > 0.0):
            raise ValueError(f"half_width must be finite and positive, received {half_width!r}")
        if not math.isfinite(singular_point):
            raise ValueError(f"singular_point must be finite, received {singular_point!r}")
        self.half_width = half_width
        self.singular_point = singular_point
        self.kink_points = (
            singular_point - half_width,
            singular_point,
            singular_point + half_width,
        )
        self.first_derivative_jumps = (1.0, -2.0, 1.0)

    def values(self, x):
        r""":math:`g(x)`, for a ``numpy`` array or a ``torch`` tensor (dtype kept)."""
        if isinstance(x, torch.Tensor):
            return torch.relu(self.half_width - torch.abs(x - self.singular_point))
        return np.maximum(self.half_width - np.abs(np.asarray(x) - self.singular_point), 0.0)

    def first_derivative_values(self, x):
        r""":math:`g'(x) = \sum_k w_k H(x - a_k)`, with :math:`H(0) = 1/2` at a kink."""
        return sum(
            jump * _midpoint_heaviside(x - kink_point)
            for kink_point, jump in zip(self.kink_points, self.first_derivative_jumps)
        )


class RealLineButterflyField:
    r"""One closed-form field :math:`h(x, t) = e^{\rho s} K(x + \beta s, s)` of the
    module table, with its analytic derivatives.

    Args:
        generator_coefficients: Mapping ``{2: nu, 1: mu, 0: r0}``; order 2 is
            mandatory with ``nu > 0``, orders 1 and 0 default to zero, and no
            other order is accepted.
        datum: The :class:`ButterflyDatum`.
        extension_kind: One of :data:`REAL_LINE_BUTTERFLY_FIELD_KINDS`.
        terminal_time: The horizon :math:`T \in (0, +\infty)`.
        comparison_diffusivity: :math:`\nu_c \in (0, +\infty)`, mandatory for
            ``graded_gaussian`` and forbidden otherwise.
        initial_smoothing_scale: :math:`\varepsilon_0 \in (0, +\infty)`,
            mandatory for ``graded_chen_mangasarian`` and forbidden otherwise.

    Raises:
        ValueError: On any invalid argument.
    """

    def __init__(
        self,
        generator_coefficients: dict,
        datum: ButterflyDatum,
        *,
        extension_kind: str,
        terminal_time: float,
        comparison_diffusivity: float | None = None,
        initial_smoothing_scale: float | None = None,
    ) -> None:
        if extension_kind not in REAL_LINE_BUTTERFLY_FIELD_KINDS:
            raise ValueError(
                f"unknown extension_kind {extension_kind!r}; expected one of "
                f"{REAL_LINE_BUTTERFLY_FIELD_KINDS}"
            )
        coefficients = {int(order): float(value) for order, value in generator_coefficients.items()}
        unsupported_orders = sorted(set(coefficients) - {0, 1, 2})
        if unsupported_orders:
            raise ValueError(f"unsupported generator orders {unsupported_orders}; only 0, 1, 2")
        if not all(math.isfinite(value) for value in coefficients.values()):
            raise ValueError(f"generator coefficients must be finite, received {coefficients!r}")
        if coefficients.get(2, 0.0) <= 0.0:
            raise ValueError(f"the diffusivity (order 2) must be positive, received {coefficients!r}")
        terminal_time = float(terminal_time)
        if not (math.isfinite(terminal_time) and terminal_time > 0.0):
            raise ValueError(f"terminal_time must be finite and positive, received {terminal_time!r}")
        if (comparison_diffusivity is not None) != (extension_kind == "graded_gaussian"):
            raise ValueError(
                "comparison_diffusivity is mandatory for graded_gaussian and forbidden "
                f"otherwise; kind {extension_kind!r}, received {comparison_diffusivity!r}"
            )
        if (initial_smoothing_scale is not None) != (extension_kind == "graded_chen_mangasarian"):
            raise ValueError(
                "initial_smoothing_scale is mandatory for graded_chen_mangasarian and "
                f"forbidden otherwise; kind {extension_kind!r}, received {initial_smoothing_scale!r}"
            )
        self.generator_coefficients = coefficients
        self.diffusivity = coefficients[2]
        self.advection_coefficient = coefficients.get(1, 0.0)
        self.reaction_coefficient = coefficients.get(0, 0.0)
        self.datum = datum
        self.extension_kind = extension_kind
        self.terminal_time = terminal_time

        rate_source, velocity_source, smoothing = _KIND_STRUCTURE[extension_kind]
        self.exponential_rate = self.reaction_coefficient if rate_source == "reaction" else 0.0
        self.transport_velocity = self.advection_coefficient if velocity_source == "advection" else 0.0
        self.smoothing = smoothing
        self.smoothing_diffusivity = None
        self.initial_smoothing_scale = None
        if smoothing == "gaussian":
            if extension_kind == "graded_gaussian":
                comparison_diffusivity = float(comparison_diffusivity)
                if not (math.isfinite(comparison_diffusivity) and comparison_diffusivity > 0.0):
                    raise ValueError(
                        f"comparison_diffusivity must be finite and positive, received "
                        f"{comparison_diffusivity!r}"
                    )
                self.smoothing_diffusivity = comparison_diffusivity
            else:
                self.smoothing_diffusivity = self.diffusivity
        elif smoothing == "chen_mangasarian":
            initial_smoothing_scale = float(initial_smoothing_scale)
            if not (math.isfinite(initial_smoothing_scale) and initial_smoothing_scale > 0.0):
                raise ValueError(
                    f"initial_smoothing_scale must be finite and positive, received "
                    f"{initial_smoothing_scale!r}"
                )
            self.initial_smoothing_scale = initial_smoothing_scale

    # -- evaluation core -------------------------------------------------------

    def _broadcast(self, x, t):
        """Return ``(x64, s64, x_original, is_torch, original_dtype)`` with common shape.

        ``x64`` and ``s64 = T - t`` are ``torch.float64`` tensors; ``x_original``
        is ``x`` broadcast to the common shape in its own dtype (the input of the
        exact terminal branch).
        """
        is_torch = isinstance(x, torch.Tensor) or isinstance(t, torch.Tensor)
        if is_torch:
            reference = x if isinstance(x, torch.Tensor) else t
            x_tensor = torch.as_tensor(x, device=reference.device)
            if not torch.is_floating_point(x_tensor):
                x_tensor = x_tensor.to(torch.float64)
            t_tensor = torch.as_tensor(t, device=reference.device)
            original_dtype = x_tensor.dtype
            x64, t64 = torch.broadcast_tensors(x_tensor.to(torch.float64), t_tensor.to(torch.float64))
            x_original = torch.broadcast_tensors(x_tensor, t_tensor.to(x_tensor.dtype))[0]
            return x64, self.terminal_time - t64, x_original, True, original_dtype
        x_array, t_array = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64)
        )
        return (
            torch.as_tensor(np.ascontiguousarray(x_array)),
            self.terminal_time - torch.as_tensor(np.ascontiguousarray(t_array)),
            np.ascontiguousarray(x_array),
            False,
            None,
        )

    def _profile(self, x64, s64):
        r"""Return ``(factor, K, K_y, K_yy, K_s, terminal_mask)`` in ``float64``.

        The values at :math:`s = 0` are the almost-everywhere limits of the module
        docstring; elsewhere they are the closed forms.
        """
        terminal_mask = s64 == 0.0
        positive_mask = ~terminal_mask
        y = x64 + self.transport_velocity * s64
        factor = torch.exp(self.exponential_rate * s64)
        if self.smoothing == "none":
            profile = self.datum.values(y)
            first = self.datum.first_derivative_values(y)
            second = torch.zeros_like(y)
            time_part = torch.zeros_like(y)
        else:
            profile = torch.zeros_like(y)
            first = torch.zeros_like(y)
            second = torch.zeros_like(y)
            time_part = torch.zeros_like(y)
            if self.smoothing == "gaussian":
                variance = 2.0 * self.smoothing_diffusivity * s64
                standard_deviation = torch.sqrt(
                    torch.where(positive_mask, variance, torch.ones_like(variance))
                )
                for kink_point, jump in zip(self.datum.kink_points, self.datum.first_derivative_jumps):
                    offset = y - kink_point
                    standardised = offset / standard_deviation
                    density = torch.exp(-0.5 * standardised * standardised) / SQUARE_ROOT_OF_TWO_PI
                    distribution = torch.special.ndtr(standardised)
                    profile = profile + jump * (offset * distribution + standard_deviation * density)
                    first = first + jump * distribution
                    second = second + jump * density / standard_deviation
                time_part = self.smoothing_diffusivity * second
            else:  # chen_mangasarian
                scale = self.initial_smoothing_scale * s64 / self.terminal_time
                scale = torch.where(positive_mask, scale, torch.ones_like(scale))
                for kink_point, jump in zip(self.datum.kink_points, self.datum.first_derivative_jumps):
                    offset = y - kink_point
                    root = torch.sqrt(offset * offset + scale * scale)
                    profile = profile + jump * 0.5 * (offset + root)
                    first = first + jump * 0.5 * (1.0 + offset / root)
                    second = second + jump * 0.5 * scale * scale / (root * root * root)
                    time_part = time_part + jump * 0.5 * scale / root
                time_part = (self.initial_smoothing_scale / self.terminal_time) * time_part
            profile = torch.where(terminal_mask, self.datum.values(x64), profile)
            first = torch.where(terminal_mask, self.datum.first_derivative_values(x64), first)
            second = torch.where(terminal_mask, torch.zeros_like(second), second)
            time_part = torch.where(terminal_mask, torch.zeros_like(time_part), time_part)
        return factor, profile, first, second, time_part, terminal_mask

    def _finish(self, values64, is_torch, original_dtype):
        if is_torch:
            return values64.to(dtype=original_dtype)
        return values64.detach().numpy()

    # -- public callables --------------------------------------------------------

    def field(self, x, t):
        r""":math:`h(x, t)`; equals ``datum.values(x)`` exactly where :math:`t = T`."""
        x64, s64, x_original, is_torch, original_dtype = self._broadcast(x, t)
        factor, profile, _, _, _, terminal_mask = self._profile(x64, s64)
        values = self._finish(factor * profile, is_torch, original_dtype)
        datum_values = self.datum.values(x_original)
        if is_torch:
            return torch.where(terminal_mask, datum_values, values)
        return np.where(terminal_mask.numpy(), datum_values, values)

    def space_derivative(self, x, t):
        r""":math:`\partial_x h(x, t) = e^{\rho s} K_y`."""
        x64, s64, _, is_torch, original_dtype = self._broadcast(x, t)
        factor, _, first, _, _, _ = self._profile(x64, s64)
        return self._finish(factor * first, is_torch, original_dtype)

    def second_space_derivative(self, x, t):
        r""":math:`\partial_{xx} h(x, t) = e^{\rho s} K_{yy}` (almost-everywhere value at :math:`s = 0`)."""
        x64, s64, _, is_torch, original_dtype = self._broadcast(x, t)
        factor, _, _, second, _, _ = self._profile(x64, s64)
        return self._finish(factor * second, is_torch, original_dtype)

    def time_derivative(self, x, t):
        r""":math:`\partial_t h(x, t) = -e^{\rho s}(\rho K + \beta K_y + K_s)`."""
        x64, s64, _, is_torch, original_dtype = self._broadcast(x, t)
        factor, profile, first, _, time_part, _ = self._profile(x64, s64)
        values = -factor * (
            self.exponential_rate * profile + self.transport_velocity * first + time_part
        )
        return self._finish(values, is_torch, original_dtype)

    def forcing_values(self, x, t):
        r"""Pointwise forcing :math:`(P h)(x, t) = \partial_t h + \nu\,\partial_{xx} h
        + \mu\,\partial_x h + r_0 h`, assembled from the analytic derivatives."""
        return (
            self.time_derivative(x, t)
            + self.diffusivity * self.second_space_derivative(x, t)
            + self.advection_coefficient * self.space_derivative(x, t)
            + self.reaction_coefficient * self.field(x, t)
        )

    def terminal_datum_values(self, x):
        r""":math:`h(x, T) = g(x)`."""
        if isinstance(x, torch.Tensor):
            return self.field(x, torch.full_like(x, self.terminal_time))
        return self.field(x, np.full(np.shape(np.asarray(x)), self.terminal_time))

    def terminal_forcing_profile(self, x):
        r""":math:`(P h)(x, T)`, the almost-everywhere value (Dirac masses absent)."""
        if isinstance(x, torch.Tensor):
            return self.forcing_values(x, torch.full_like(x, self.terminal_time))
        return self.forcing_values(x, np.full(np.shape(np.asarray(x)), self.terminal_time))

    def derivative_callables(self) -> dict:
        r"""``{"dt", "dx", "dxx"}``, the key set of ``TerminalAnsatz(extension_derivative_fns=...)``."""
        return {
            "dt": self.time_derivative,
            "dx": self.space_derivative,
            "dxx": self.second_space_derivative,
        }


# ---------------------------------------------------------------------------
# Correction imposed by the singular forcing of a datum-path extension
# ---------------------------------------------------------------------------

def datum_line_source_correction_values(
    x,
    t,
    *,
    generator_coefficients: dict,
    datum: ButterflyDatum,
    terminal_time: float,
    temporal_factor,
    nodes_per_panel: int = 64,
):
    r"""Correction :math:`w` that the singular forcing of :math:`\Psi = c(t)\,g`
    imposes on a field whose pointwise residual vanishes almost everywhere.

    For :math:`\Psi = c(t)\,g` (``constant_in_time``: :math:`c = 1`;
    ``convex_raw`` with the linear factor: :math:`c(t) = t/T`), the forcing
    :math:`P\Psi` has the singular part
    :math:`f = c(t)\,\nu \sum_k w_k\,\delta_{a_k}`.  A field
    :math:`u = \Psi + (\text{correction of class } C^2)` whose pointwise residual
    vanishes almost everywhere satisfies :math:`P u = f` in the sense of
    distributions with :math:`u(\cdot, T) = g`, so :math:`u = u^\star + w` with
    :math:`P w = f` and :math:`w(\cdot, T) = 0`.  By Duhamel's formula, with the
    forward kernel :math:`e^{\sigma A}\delta_a = e^{r_0\sigma}\,G_{2\nu\sigma}(\cdot + \mu\sigma - a)`
    (:math:`G_v` the centred normal density of variance :math:`v`),

    .. math::

        w(x, t) = -\nu \sum_k w_k \int_0^{T - t} c(t + \sigma)\, e^{r_0 \sigma}
        G_{2\nu\sigma}(x + \mu\sigma - a_k)\,\mathrm{d}\sigma .

    The substitution :math:`\sigma = \tau^2` removes the integrable singularity at
    :math:`\sigma = 0`:

    .. math::

        w(x, t) = -\sqrt{\nu/\pi} \sum_k w_k \int_0^{\sqrt{T - t}}
        c(t + \tau^2)\, e^{r_0\tau^2}
        \exp\!\Bigl(-\frac{(x - a_k + \mu\tau^2)^2}{4\nu\tau^2}\Bigr)\mathrm{d}\tau .

    The integral is evaluated by Gauss--Legendre quadrature on three panels per
    point, split at :math:`\tau = |x - a_k|/(2\sqrt\nu)` (where the exponent
    reaches :math:`-1` in the absence of drift) and at
    :math:`\tau = \sqrt{(a_k - x)/\mu}` when the characteristic through
    :math:`(x, t)` meets :math:`a_k` (where the exponent vanishes).

    Args:
        x, t: Points, broadcast to a common shape (``numpy``, ``float64``).
        generator_coefficients: ``{2: nu, 1: mu, 0: r0}``.
        datum: The :class:`ButterflyDatum`.
        terminal_time: :math:`T`.
        temporal_factor: Callable :math:`c` acting on ``numpy`` arrays of times.
        nodes_per_panel: Gauss--Legendre nodes per panel.

    Returns:
        ``float64`` array of :math:`w(x, t)`.
    """
    diffusivity = float(generator_coefficients[2])
    drift = float(generator_coefficients.get(1, 0.0))
    reaction = float(generator_coefficients.get(0, 0.0))
    x_array, t_array = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64)
    )
    shape = x_array.shape
    x_flat = x_array.reshape(-1)
    t_flat = t_array.reshape(-1)
    upper_limit = np.sqrt(np.maximum(terminal_time - t_flat, 0.0))
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(int(nodes_per_panel))
    correction = np.zeros_like(x_flat)
    for kink_point, jump in zip(datum.kink_points, datum.first_derivative_jumps):
        offset = x_flat - kink_point
        onset_breakpoint = np.abs(offset) / (2.0 * math.sqrt(diffusivity))
        if drift != 0.0:
            crossing_time = -offset / drift
            crossing_breakpoint = np.sqrt(np.where(crossing_time > 0.0, crossing_time, 0.0))
        else:
            crossing_breakpoint = np.zeros_like(offset)
        breakpoints = np.sort(
            np.stack(
                [
                    np.zeros_like(offset),
                    np.minimum(onset_breakpoint, upper_limit),
                    np.minimum(crossing_breakpoint, upper_limit),
                    upper_limit,
                ],
                axis=-1,
            ),
            axis=-1,
        )
        integral = np.zeros_like(offset)
        for panel_index in range(3):
            lower = breakpoints[:, panel_index][:, None]
            upper = breakpoints[:, panel_index + 1][:, None]
            half_length = 0.5 * (upper - lower)
            tau = lower + half_length * (reference_nodes[None, :] + 1.0)
            tau_squared = tau * tau
            safe_tau_squared = np.where(tau_squared > 0.0, tau_squared, 1.0)
            exponent = -((offset[:, None] + drift * tau_squared) ** 2) / (4.0 * diffusivity * safe_tau_squared)
            integrand = (
                temporal_factor(t_flat[:, None] + tau_squared)
                * np.exp(reaction * tau_squared)
                * np.where(tau_squared > 0.0, np.exp(exponent), 0.0)
            )
            integral = integral + np.sum(half_length * reference_weights[None, :] * integrand, axis=-1)
        correction = correction - math.sqrt(diffusivity / math.pi) * jump * integral
    return correction.reshape(shape)


# ---------------------------------------------------------------------------
# Mean square of a pointwise forcing over a spatial window
# ---------------------------------------------------------------------------

def window_mean_square_of_pointwise_values(
    pointwise_values,
    *,
    spatial_window: tuple,
    terminal_time: float,
    refinement_centres,
    uniform_point_count: int = 20001,
    refinement_point_count: int = 400,
    time_node_count: int = 64,
) -> float:
    r"""Mean square :math:`\frac{1}{|W|\,T}\int_0^T\!\int_W f(x, t)^2\,\mathrm{d}x\,\mathrm{d}t`
    of a pointwise function over the window :math:`W = [x_-, x_+]`.

    It is the expectation of :math:`f^2` under the uniform law on
    :math:`W \times (0, T)`, which the training forcing channel estimates.  The time
    integral uses :math:`s = T - t = \tau^2`, so that a term of order
    :math:`s^{-1/2}` (the squared second derivative of a Gaussian field) becomes
    bounded, and Gauss--Legendre nodes in :math:`\tau`.  At each node, the space
    integral is the trapezoidal rule on the union of a uniform grid and of
    geometric grids around the centres ``refinement_centres(s)``, with offsets from
    :math:`10^{-8}` to :math:`2`, which resolve the layers of width
    :math:`\sqrt{s}` around the kinks.

    It applies only to a function whose mean square is finite: a forcing whose
    square has a non-integrable singularity at the terminal slice returns a value
    that depends on the resolution, and is not to be evaluated with it.
    """
    lower, upper = float(spatial_window[0]), float(spatial_window[1])
    reference_nodes, reference_weights = np.polynomial.legendre.leggauss(int(time_node_count))
    upper_tau = math.sqrt(terminal_time)
    tau_nodes = 0.5 * upper_tau * (reference_nodes + 1.0)
    tau_weights = 0.5 * upper_tau * reference_weights
    uniform_grid = np.linspace(lower, upper, int(uniform_point_count))
    geometric_offsets = np.geomspace(1e-8, 2.0, int(refinement_point_count))
    signed_offsets = np.concatenate([-geometric_offsets[::-1], [0.0], geometric_offsets])
    integral = 0.0
    for tau, weight in zip(tau_nodes, tau_weights):
        time_to_terminal = tau * tau
        centres = np.asarray(refinement_centres(time_to_terminal), dtype=np.float64)
        refined = (centres[:, None] + signed_offsets[None, :]).reshape(-1)
        grid = np.unique(np.concatenate([uniform_grid, refined[(refined > lower) & (refined < upper)]]))
        values = np.asarray(
            pointwise_values(grid, np.full_like(grid, terminal_time - time_to_terminal)),
            dtype=np.float64,
        )
        spatial_integral = float(_trapezoidal_rule(values * values, grid))
        integral += weight * 2.0 * tau * spatial_integral
    return integral / ((upper - lower) * terminal_time)
