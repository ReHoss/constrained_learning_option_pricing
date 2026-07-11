r"""Real-space closed-form extension fields for band-limited periodic data.

Setting.  On the circle :math:`[0, 2\pi)` over the strip
:math:`(0, T) \times [0, 2\pi)`, a band-limited real terminal datum

.. math::

    g(x) = \sum_{k=1}^{K} \bigl[ a_k \cos(k x) + b_k \sin(k x) \bigr]

is extended from the terminal time :math:`t = T` into the strip by one of the
closed-form extensions of the stage-2 specification
(``documents/methodology/stage2_trained_ablation_specification.md``,
Section 1.2).  The spatial generator is the constant-coefficient operator
:math:`A = \nu\,\partial_{xx} + \mu\,\partial_x + r_0` with diffusivity
:math:`\nu \in (0, +\infty)`, advection coefficient :math:`\mu \in \mathbb{R}`
and reaction coefficient :math:`r_0 \in \mathbb{R}`; the residual operator is
:math:`P u = \partial_t u + A u`.

Every extension field is the finite sum of spectral components [#]_

.. math::

    h(x, t) = \sum_{k=1}^{K} e^{-(T - t)\, r_k}
    \bigl[ a_k \cos\theta_k(x, t) + b_k \sin\theta_k(x, t) \bigr],
    \qquad
    \theta_k(x, t) = k x + (T - t)\, \mu_{\mathrm{adv}}\, k,

where the per-component decay rate :math:`r_k` and the phase-advection
velocity :math:`\mu_{\mathrm{adv}}` depend on the extension kind:

===============================  =============================  ======================
Extension kind                   Decay rate :math:`r_k`         :math:`\mu_{\mathrm{adv}}`
===============================  =============================  ======================
``split_diffusion``              :math:`\nu k^2`                :math:`0`
``split_diffusion_advection``    :math:`\nu k^2`                :math:`\mu`
``graded_gaussian``              :math:`\nu_c k^2`              :math:`0`
``exact_solution``               :math:`\nu k^2 - r_0`          :math:`\mu`
===============================  =============================  ======================

with :math:`\nu_c \ge 0` the comparison diffusivity of the graded Gaussian
extension.  The analytic derivatives follow by differentiating each spectral
component:

.. math::

    \partial_t h = \sum_{k=1}^{K} e^{-(T-t) r_k}
    \bigl[ r_k \bigl( a_k \cos\theta_k + b_k \sin\theta_k \bigr)
    + \mu_{\mathrm{adv}} k \bigl( a_k \sin\theta_k - b_k \cos\theta_k \bigr)
    \bigr],

.. math::

    \partial_x h = \sum_{k=1}^{K} e^{-(T-t) r_k}\, k
    \bigl[ -a_k \sin\theta_k + b_k \cos\theta_k \bigr],
    \qquad
    \partial_{xx} h = -\sum_{k=1}^{K} e^{-(T-t) r_k}\, k^2
    \bigl[ a_k \cos\theta_k + b_k \sin\theta_k \bigr].

Numerical policy (specification Section 1.4 item 2).  All coefficient arrays
are synthesised in ``float64`` at construction.  Evaluation accepts either
``numpy`` arrays (``float64`` path) or ``torch`` tensors (internal ``float64``
computation, result cast to the input dtype at the end; the computation is
differentiable, so autograd through the field remains available for
cross-checks).  The terminal identity ``field(x, T) == terminal_datum_values(x)``
holds exactly in floating point: at :math:`t = T` the decay factor is
:math:`e^{0} = 1` exactly, the phase advection vanishes exactly, and the
summation order of the component sum is fixed (the same reduction as
:meth:`PeriodicExtensionField.terminal_datum_values`).

Validation policy.  Every constructor argument is validated and a violation
raises :class:`ValueError`; nothing is silently clamped.  In particular the
``exact_solution`` kind validates the dissipativity bound
:math:`\min_k (\nu k^2 - r_0) \ge -10^{-12}` over the retained band before any
decay factor is synthesised, mirroring
:class:`learning_option_pricing.pde.terminal_data_extensions.ExactSolutionExtension`.

.. [#] A *spectral component* is one summand of the finite trigonometric sum
   above (the terminology preferred here over the synonymous physics term
   *mode*).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

TWO_PI = 2.0 * math.pi

# Dissipativity bound for the exact-solution decay rates, mirroring
# learning_option_pricing.pde.periodic_spectral_toolbox.DISSIPATIVITY_TOLERANCE:
# the validation raises ValueError, it never silently clamps.
DISSIPATIVITY_TOLERANCE = 1.0e-12

EXTENSION_FIELD_KINDS = (
    "split_diffusion",
    "split_diffusion_advection",
    "graded_gaussian",
    "exact_solution",
)

SUPPORTED_GENERATOR_ORDERS = (0, 1, 2)


def bandlimited_bernoulli_cosine_coefficients(
    truncation_wavenumber: int,
) -> np.ndarray:
    r"""Real cosine coefficients of the band-limited Bernoulli datum.

    The stage-2 terminal datum is the truncation of the regularity-index-1
    periodised Bernoulli datum,

    .. math::

        g(x) = \sum_{k=1}^{K_g} \frac{\cos(k x)}{\pi^2 k^2},

    so the real cosine coefficient at wavenumber :math:`k` is
    :math:`a_k = 1 / (\pi^2 k^2)` (twice the complex coefficient
    :math:`c_k = 1 / (2 \pi^2 k^2)` of
    :class:`learning_option_pricing.pde.periodic_spectral_toolbox.PeriodisedBernoulliDatum`
    with regularity index 1, restricted to the retained band).

    Args:
        truncation_wavenumber: Band edge :math:`K_g \ge 1`.

    Returns:
        ``float64`` array of shape ``(truncation_wavenumber,)`` with entry
        ``j`` equal to :math:`1 / (\pi^2 (j+1)^2)`.

    Raises:
        ValueError: If ``truncation_wavenumber`` is not a positive integer.
    """
    if (
        int(truncation_wavenumber) != truncation_wavenumber
        or truncation_wavenumber < 1
    ):
        raise ValueError(
            "truncation_wavenumber must be a positive integer, received "
            f"{truncation_wavenumber!r}"
        )
    retained_wavenumbers = np.arange(
        1, int(truncation_wavenumber) + 1, dtype=np.float64
    )
    return 1.0 / (math.pi**2 * retained_wavenumbers**2)


def _validated_generator_coefficients(
    generator_coefficients: dict[int, float],
) -> dict[int, float]:
    """Validate and normalise the generator coefficient mapping.

    The mapping must use the differential orders 0, 1, 2 only, must contain
    the order 2, and the diffusivity (the order-2 coefficient) must be
    strictly positive — the standing assumption of the stage-2 specification
    (Section 0).

    Raises:
        ValueError: On an empty mapping, an unsupported order, a missing
            order 2, or a non-positive diffusivity.
    """
    if not generator_coefficients:
        raise ValueError("generator_coefficients must not be empty")
    normalised: dict[int, float] = {}
    for order, coefficient in generator_coefficients.items():
        if int(order) != order or int(order) not in SUPPORTED_GENERATOR_ORDERS:
            raise ValueError(
                "generator_coefficients orders must belong to "
                f"{SUPPORTED_GENERATOR_ORDERS}, received order {order!r}"
            )
        normalised[int(order)] = float(coefficient)
    if 2 not in normalised:
        raise ValueError(
            "generator_coefficients must contain the differential order 2 "
            "(the diffusivity), received orders "
            f"{sorted(generator_coefficients)}"
        )
    if normalised[2] <= 0.0:
        raise ValueError(
            "the diffusivity (order-2 coefficient) must be strictly "
            f"positive, received {normalised[2]!r}"
        )
    return normalised


class PeriodicExtensionField:
    r"""Closed-form extension field on the circle as a finite component sum.

    The field, its analytic derivatives, and the forcing are the closed forms
    of the module docstring, evaluated as vectorised finite sums over the
    retained band :math:`k = 1, \ldots, K`.  Coefficient arrays are
    synthesised in ``float64`` at construction; ``torch`` coefficient tensors
    are created lazily per device and cached (specification: "coefficient
    tensors precomputed at construction, moved to the training device").

    Args:
        generator_coefficients: Mapping from differential order to real
            coefficient, ``{2: nu, 1: mu, 0: r0}`` (orders 1 and 0 optional,
            defaulting to zero; order 2 mandatory with ``nu > 0``).
        cosine_coefficients: Real cosine amplitudes :math:`(a_k)_{k=1}^{K}`
            of the terminal datum, one-dimensional, index ``j`` holding the
            amplitude at wavenumber ``j + 1``.
        extension_kind: One of :data:`EXTENSION_FIELD_KINDS`.
        sine_coefficients: Optional real sine amplitudes
            :math:`(b_k)_{k=1}^{K}`; defaults to zeros (the generator cells
            of the stage-2 specification use a pure cosine datum; the control
            cell uses a single sine component).
        comparison_diffusivity: The diffusivity :math:`\nu_c \ge 0` of the
            comparison heat semigroup; mandatory for the ``graded_gaussian``
            kind and forbidden for every other kind.
        terminal_time: The horizon :math:`T > 0`.

    Raises:
        ValueError: On an unknown kind, invalid generator coefficients,
            non-finite or mis-shaped datum coefficients, an inconsistent
            ``comparison_diffusivity``, a non-positive ``terminal_time``, or
            (``exact_solution`` kind) a violation of the dissipativity bound
            over the retained band.
    """

    def __init__(
        self,
        generator_coefficients: dict[int, float],
        cosine_coefficients,
        *,
        extension_kind: str,
        sine_coefficients=None,
        comparison_diffusivity: float | None = None,
        terminal_time: float = 1.0,
    ) -> None:
        if extension_kind not in EXTENSION_FIELD_KINDS:
            raise ValueError(
                f"unknown extension_kind {extension_kind!r}; choose from "
                f"{EXTENSION_FIELD_KINDS}"
            )
        if terminal_time <= 0.0:
            raise ValueError(
                "terminal_time must be strictly positive, received "
                f"{terminal_time!r}"
            )
        normalised_coefficients = _validated_generator_coefficients(
            generator_coefficients
        )
        cosine_array = np.asarray(cosine_coefficients, dtype=np.float64)
        if cosine_array.ndim != 1 or cosine_array.size < 1:
            raise ValueError(
                "cosine_coefficients must be a one-dimensional array with at "
                f"least one entry, received shape {cosine_array.shape}"
            )
        if sine_coefficients is None:
            sine_array = np.zeros_like(cosine_array)
        else:
            sine_array = np.asarray(sine_coefficients, dtype=np.float64)
            if sine_array.shape != cosine_array.shape:
                raise ValueError(
                    "sine_coefficients must have the same shape as "
                    f"cosine_coefficients, received {sine_array.shape} "
                    f"against {cosine_array.shape}"
                )
        if not (np.all(np.isfinite(cosine_array)) and np.all(np.isfinite(sine_array))):
            raise ValueError("datum coefficients must all be finite")

        if extension_kind == "graded_gaussian":
            if comparison_diffusivity is None:
                raise ValueError(
                    "extension_kind 'graded_gaussian' requires "
                    "comparison_diffusivity"
                )
            if comparison_diffusivity < 0.0:
                raise ValueError(
                    "comparison_diffusivity must be non-negative (a negative "
                    "value makes the comparison semigroup antidissipative), "
                    f"received {comparison_diffusivity!r}"
                )
        elif comparison_diffusivity is not None:
            raise ValueError(
                "comparison_diffusivity applies to the 'graded_gaussian' "
                f"kind only, received {comparison_diffusivity!r} with "
                f"extension_kind {extension_kind!r}"
            )

        self.extension_kind = extension_kind
        self.terminal_time = float(terminal_time)
        self.generator_coefficients = normalised_coefficients
        self.diffusivity = normalised_coefficients[2]
        self.advection_coefficient = normalised_coefficients.get(1, 0.0)
        self.reaction_coefficient = normalised_coefficients.get(0, 0.0)
        self.comparison_diffusivity = (
            float(comparison_diffusivity)
            if comparison_diffusivity is not None
            else None
        )
        self.truncation_wavenumber = int(cosine_array.size)

        # float64 coefficient synthesis (numerical policy of the spec).
        wavenumbers = np.arange(
            1, self.truncation_wavenumber + 1, dtype=np.float64
        )
        if extension_kind in ("split_diffusion", "split_diffusion_advection"):
            decay_rates = self.diffusivity * wavenumbers**2
        elif extension_kind == "graded_gaussian":
            decay_rates = self.comparison_diffusivity * wavenumbers**2
        else:  # exact_solution
            decay_rates = (
                self.diffusivity * wavenumbers**2 - self.reaction_coefficient
            )
            worst_index = int(np.argmin(decay_rates))
            if decay_rates[worst_index] < -DISSIPATIVITY_TOLERANCE:
                raise ValueError(
                    "dissipativity violated for the exact-solution field: "
                    f"nu k^2 - r0 = {decay_rates[worst_index]:.6e} < "
                    f"-{DISSIPATIVITY_TOLERANCE:.0e} at wavenumber "
                    f"k = {worst_index + 1}"
                )
        if extension_kind in ("split_diffusion_advection", "exact_solution"):
            phase_advection_velocity = self.advection_coefficient
        else:
            phase_advection_velocity = 0.0

        self._numpy_coefficient_arrays = {
            "wavenumbers": wavenumbers,
            "decay_rates": decay_rates,
            "cosine_amplitudes": cosine_array.copy(),
            "sine_amplitudes": sine_array.copy(),
            "phase_advection_rates": phase_advection_velocity * wavenumbers,
        }
        # torch float64 coefficient tensors, cached per device.
        self._torch_coefficient_cache: dict[torch.device, dict[str, torch.Tensor]] = {}

    # -- coefficient access -------------------------------------------------

    def _coefficient_arrays(self, reference):
        """Return the coefficient arrays in the backend of ``reference``.

        ``reference`` is a sample input (``torch.Tensor`` or array-like);
        torch coefficient tensors are created on its device on first use and
        cached.
        """
        if isinstance(reference, torch.Tensor):
            device = reference.device
            if device not in self._torch_coefficient_cache:
                self._torch_coefficient_cache[device] = {
                    name: torch.as_tensor(
                        array, dtype=torch.float64, device=device
                    )
                    for name, array in self._numpy_coefficient_arrays.items()
                }
            return self._torch_coefficient_cache[device]
        return self._numpy_coefficient_arrays

    # -- synthesis core -------------------------------------------------------

    def _broadcast_float64(self, x, t):
        """Broadcast ``x`` and ``t`` to a common shape in ``float64``.

        Returns ``(x64, time_to_terminal, backend_is_torch, original_dtype)``
        where ``time_to_terminal = T - t``.
        """
        if isinstance(x, torch.Tensor) or isinstance(t, torch.Tensor):
            x_tensor = torch.as_tensor(x)
            t_tensor = torch.as_tensor(
                t, dtype=x_tensor.dtype, device=x_tensor.device
            ) if not isinstance(t, torch.Tensor) else t
            original_dtype = x_tensor.dtype
            x64 = x_tensor.to(dtype=torch.float64)
            t64 = t_tensor.to(dtype=torch.float64)
            x64, t64 = torch.broadcast_tensors(x64, t64)
            return x64, self.terminal_time - t64, True, original_dtype
        x64 = np.asarray(x, dtype=np.float64)
        t64 = np.asarray(t, dtype=np.float64)
        x64, t64 = np.broadcast_arrays(x64, t64)
        return x64, self.terminal_time - t64, False, None

    def _component_terms(self, x, t):
        """Per-component decay factors and trigonometric arguments.

        Returns ``(decay_factors, cos_theta, sin_theta, arrays, is_torch,
        original_dtype, batch_shape)`` with the component axis appended last
        (shape ``batch_shape + (K,)``).  The reduction over that fixed last
        axis is what makes the terminal identity exact (see the module
        docstring).
        """
        x64, time_to_terminal, is_torch, original_dtype = self._broadcast_float64(x, t)
        arrays = self._coefficient_arrays(x64 if is_torch else None)
        wavenumbers = arrays["wavenumbers"]
        x_expanded = x64[..., None]
        s_expanded = time_to_terminal[..., None]
        theta = x_expanded * wavenumbers + s_expanded * arrays["phase_advection_rates"]
        if is_torch:
            decay_factors = torch.exp(-s_expanded * arrays["decay_rates"])
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
        else:
            decay_factors = np.exp(-s_expanded * arrays["decay_rates"])
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
        return (
            decay_factors,
            cos_theta,
            sin_theta,
            arrays,
            is_torch,
            original_dtype,
            x64.shape,
        )

    @staticmethod
    def _cast_back(values, is_torch, original_dtype):
        """Cast a ``float64`` result to the caller's dtype (torch path only)."""
        if is_torch:
            return values.to(dtype=original_dtype)
        return values

    # -- public callables (closed forms of the spec, Sections 1.2–1.3) -------

    def field(self, x, t):
        r"""Extension field :math:`h(x, t)` as a finite component sum."""
        decay, cos_theta, sin_theta, arrays, is_torch, dtype, _ = (
            self._component_terms(x, t)
        )
        component_values = decay * (
            arrays["cosine_amplitudes"] * cos_theta
            + arrays["sine_amplitudes"] * sin_theta
        )
        return self._cast_back(component_values.sum(-1), is_torch, dtype)

    def time_derivative(self, x, t):
        r"""Analytic time derivative :math:`\partial_t h(x, t)`."""
        decay, cos_theta, sin_theta, arrays, is_torch, dtype, _ = (
            self._component_terms(x, t)
        )
        cosine_amplitudes = arrays["cosine_amplitudes"]
        sine_amplitudes = arrays["sine_amplitudes"]
        component_values = decay * (
            arrays["decay_rates"]
            * (cosine_amplitudes * cos_theta + sine_amplitudes * sin_theta)
            + arrays["phase_advection_rates"]
            * (cosine_amplitudes * sin_theta - sine_amplitudes * cos_theta)
        )
        return self._cast_back(component_values.sum(-1), is_torch, dtype)

    def space_derivative(self, x, t):
        r"""Analytic space derivative :math:`\partial_x h(x, t)`."""
        decay, cos_theta, sin_theta, arrays, is_torch, dtype, _ = (
            self._component_terms(x, t)
        )
        component_values = decay * (
            arrays["wavenumbers"]
            * (
                -arrays["cosine_amplitudes"] * sin_theta
                + arrays["sine_amplitudes"] * cos_theta
            )
        )
        return self._cast_back(component_values.sum(-1), is_torch, dtype)

    def second_space_derivative(self, x, t):
        r"""Analytic second space derivative :math:`\partial_{xx} h(x, t)`."""
        decay, cos_theta, sin_theta, arrays, is_torch, dtype, _ = (
            self._component_terms(x, t)
        )
        component_values = -decay * (
            arrays["wavenumbers"] ** 2
            * (
                arrays["cosine_amplitudes"] * cos_theta
                + arrays["sine_amplitudes"] * sin_theta
            )
        )
        return self._cast_back(component_values.sum(-1), is_torch, dtype)

    def forcing_values(self, x, t):
        r"""Forcing :math:`(P h)(x, t) = \partial_t h + \nu\,\partial_{xx} h
        + \mu\,\partial_x h + r_0 h`, assembled from the analytic derivatives.
        """
        return (
            self.time_derivative(x, t)
            + self.diffusivity * self.second_space_derivative(x, t)
            + self.advection_coefficient * self.space_derivative(x, t)
            + self.reaction_coefficient * self.field(x, t)
        )

    def terminal_datum_values(self, x):
        r"""Terminal datum :math:`g(x)`, with the same fixed summation order
        as :meth:`field`, so that ``field(x, T) == terminal_datum_values(x)``
        holds exactly in floating point.
        """
        if isinstance(x, torch.Tensor):
            terminal_time_like = torch.full_like(x, self.terminal_time)
        else:
            terminal_time_like = np.full(
                np.shape(np.asarray(x)), self.terminal_time
            )
        return self.field(x, terminal_time_like)

    def terminal_forcing_profile(self, x):
        r"""Terminal forcing profile :math:`(P h)(x, T)` (specification
        Section 3.5), assembled analytically at :math:`t = T`.
        """
        if isinstance(x, torch.Tensor):
            terminal_time_like = torch.full_like(x, self.terminal_time)
        else:
            terminal_time_like = np.full(
                np.shape(np.asarray(x)), self.terminal_time
            )
        return self.forcing_values(x, terminal_time_like)

    def derivative_callables(self):
        r"""The analytic-derivative mapping for the training bypass.

        Returns:
            ``{"dt": time_derivative, "dx": space_derivative,
            "dxx": second_space_derivative}`` — the exact key set expected by
            ``TerminalAnsatz(extension_derivative_fns=...)``.
        """
        return {
            "dt": self.time_derivative,
            "dx": self.space_derivative,
            "dxx": self.second_space_derivative,
        }


# ---------------------------------------------------------------------------
# Registry builders (uniform call signature; the schema keys of the spec)
# ---------------------------------------------------------------------------


def build_split_diffusion_extension_field(
    generator_coefficients: dict[int, float],
    cosine_coefficients,
    *,
    sine_coefficients=None,
    comparison_diffusivity: float | None = None,
    terminal_time: float = 1.0,
) -> PeriodicExtensionField:
    r"""Split semigroup extension for the subset :math:`\{\partial_{xx}\}`
    (specification V3); the forcing satisfies :math:`P h = \mu\,\partial_x h
    + r_0 h`.
    """
    if comparison_diffusivity is not None:
        raise ValueError(
            "comparison_diffusivity applies to the graded Gaussian extension "
            f"only, received {comparison_diffusivity!r}"
        )
    return PeriodicExtensionField(
        generator_coefficients,
        cosine_coefficients,
        extension_kind="split_diffusion",
        sine_coefficients=sine_coefficients,
        terminal_time=terminal_time,
    )


def build_split_diffusion_advection_extension_field(
    generator_coefficients: dict[int, float],
    cosine_coefficients,
    *,
    sine_coefficients=None,
    comparison_diffusivity: float | None = None,
    terminal_time: float = 1.0,
) -> PeriodicExtensionField:
    r"""Split semigroup extension for the subset
    :math:`\{\partial_{xx}, \partial_x\}` (specification V4); the forcing
    satisfies :math:`P h = r_0 h`.
    """
    if comparison_diffusivity is not None:
        raise ValueError(
            "comparison_diffusivity applies to the graded Gaussian extension "
            f"only, received {comparison_diffusivity!r}"
        )
    return PeriodicExtensionField(
        generator_coefficients,
        cosine_coefficients,
        extension_kind="split_diffusion_advection",
        sine_coefficients=sine_coefficients,
        terminal_time=terminal_time,
    )


def build_graded_gaussian_extension_field(
    generator_coefficients: dict[int, float],
    cosine_coefficients,
    *,
    sine_coefficients=None,
    comparison_diffusivity: float | None = None,
    terminal_time: float = 1.0,
) -> PeriodicExtensionField:
    r"""Graded Gaussian extension with comparison diffusivity :math:`\nu_c`
    (specification V5/V6); the per-wavenumber forcing coefficient is
    :math:`(a(k) + \nu_c k^2)\,\hat h(k, t)`.
    """
    return PeriodicExtensionField(
        generator_coefficients,
        cosine_coefficients,
        extension_kind="graded_gaussian",
        sine_coefficients=sine_coefficients,
        comparison_diffusivity=comparison_diffusivity,
        terminal_time=terminal_time,
    )


def exact_solution_field(
    generator_coefficients: dict[int, float],
    cosine_coefficients,
    *,
    sine_coefficients=None,
    comparison_diffusivity: float | None = None,
    terminal_time: float = 1.0,
) -> PeriodicExtensionField:
    r"""Exact solution :math:`u^\star` of the backward evolution problem as an
    extension field (specification V7 and Section 3.2); the forcing vanishes
    identically.
    """
    if comparison_diffusivity is not None:
        raise ValueError(
            "comparison_diffusivity applies to the graded Gaussian extension "
            f"only, received {comparison_diffusivity!r}"
        )
    return PeriodicExtensionField(
        generator_coefficients,
        cosine_coefficients,
        extension_kind="exact_solution",
        sine_coefficients=sine_coefficients,
        terminal_time=terminal_time,
    )


# Registry mapping the string-valued schema keys of the variant catalogue to
# constructors (specification Section 1.4 item 2).
EXTENSION_FIELD_REGISTRY = {
    "split_diffusion": build_split_diffusion_extension_field,
    "split_diffusion_advection": build_split_diffusion_advection_extension_field,
    "graded_gaussian": build_graded_gaussian_extension_field,
    "exact_solution": exact_solution_field,
}


# ---------------------------------------------------------------------------
# Single-spectral-component sine control cell (specification Section 1.3)
# ---------------------------------------------------------------------------


def sine_cell_matched_exponential_rate(
    diffusivity: float, wavenumber: int
) -> float:
    r"""Matched exponential interpolation rate :math:`\gamma = \nu k_0^2` of
    the single-spectral-component sine cell on the circle.

    Caution (specification D10): the default rate of
    ``make_interpolation_coefficient`` is the eigenvalue-matched value of the
    **unit-interval** family, :math:`\sigma^2 \pi^2 / 2`; on the circle the
    matched rate for the wavenumber :math:`k_0` is :math:`\nu k_0^2`, so this
    value must always be passed explicitly — relying on the library default
    silently mismatches the factor by :math:`\pi^2`.

    Args:
        diffusivity: The diffusivity :math:`\nu > 0` of the pure-heat
            generator.
        wavenumber: The single retained wavenumber :math:`k_0 \ge 1`.

    Returns:
        The rate :math:`\gamma = \nu k_0^2` as a float.

    Raises:
        ValueError: If ``diffusivity`` is not strictly positive or
            ``wavenumber`` is not a positive integer.
    """
    if diffusivity <= 0.0:
        raise ValueError(
            f"diffusivity must be strictly positive, received {diffusivity!r}"
        )
    if int(wavenumber) != wavenumber or wavenumber < 1:
        raise ValueError(
            f"wavenumber must be a positive integer, received {wavenumber!r}"
        )
    return float(diffusivity) * float(int(wavenumber)) ** 2


@dataclass(frozen=True)
class SingleComponentSineCell:
    r"""The single-spectral-component sine control cell of the specification.

    Datum :math:`g(x) = A \sin(k_0 x)`; pure-heat generator
    :math:`\nu\,\partial_{xx}`; exact solution
    :math:`u^\star(x, t) = A\, e^{-\nu k_0^2 (T - t)} \sin(k_0 x)`; matched
    exponential interpolation rate :math:`\gamma = \nu k_0^2` held as an
    **explicit** field (never inferred from a library default; specification
    D10).

    Attributes:
        diffusivity: The diffusivity :math:`\nu > 0`.
        wavenumber: The single retained wavenumber :math:`k_0`.
        amplitude: The datum amplitude :math:`A`.
        terminal_time: The horizon :math:`T > 0`.
        matched_exponential_rate: The explicit rate :math:`\gamma = \nu k_0^2`.
        exact_solution: The exact solution :math:`u^\star` as a
            :class:`PeriodicExtensionField` of kind ``exact_solution``.
    """

    diffusivity: float
    wavenumber: int
    amplitude: float
    terminal_time: float
    matched_exponential_rate: float
    exact_solution: PeriodicExtensionField

    def terminal_datum_values(self, x):
        r"""Datum :math:`g(x) = A \sin(k_0 x)` (fixed summation order of the
        exact-solution field, so the terminal identity is exact)."""
        return self.exact_solution.terminal_datum_values(x)


def make_single_component_sine_cell(
    diffusivity: float = 0.125,
    wavenumber: int = 1,
    amplitude: float = 1.0,
    terminal_time: float = 1.0,
) -> SingleComponentSineCell:
    r"""Build the single-spectral-component sine control cell.

    The defaults reproduce the specification's control cell
    (``heat_sine_single_component``): :math:`\nu = 0.125`, :math:`k_0 = 1`,
    :math:`A = 1`, :math:`T = 1`, hence the matched rate
    :math:`\gamma = \nu k_0^2 = 0.125`.

    Raises:
        ValueError: Propagated from
            :func:`sine_cell_matched_exponential_rate` or
            :class:`PeriodicExtensionField` on invalid arguments.
    """
    matched_rate = sine_cell_matched_exponential_rate(diffusivity, wavenumber)
    retained_band_size = int(wavenumber)
    cosine_coefficients = np.zeros(retained_band_size, dtype=np.float64)
    sine_coefficients = np.zeros(retained_band_size, dtype=np.float64)
    sine_coefficients[retained_band_size - 1] = float(amplitude)
    exact_solution = exact_solution_field(
        {2: float(diffusivity)},
        cosine_coefficients,
        sine_coefficients=sine_coefficients,
        terminal_time=terminal_time,
    )
    return SingleComponentSineCell(
        diffusivity=float(diffusivity),
        wavenumber=int(wavenumber),
        amplitude=float(amplitude),
        terminal_time=float(terminal_time),
        matched_exponential_rate=matched_rate,
        exact_solution=exact_solution,
    )
