r"""Analytic terminal-data extensions on the circle and their strip forcing.

Setting.  A terminal datum :math:`g` with exact Fourier coefficients
:math:`c_k` (see :mod:`learning_option_pricing.pde.periodic_spectral_toolbox`)
is extended from the terminal time :math:`t = T` into the strip
:math:`(0, T) \times [0, 2\pi)` by an analytic extension :math:`h` with
per-wavenumber coefficient :math:`\hat h(k, t)`.  For a generator with
symbol :math:`a(k)`, the forcing of the extension is

.. math::

    \widehat{Lh}(k, t) = \partial_t \hat h(k, t) + a(k)\, \hat h(k, t),

and the squared strip norm of the forcing is (Parseval)

.. math::

    \|Lh\|^2_{L^2((0,T);\,L^2(0,2\pi))}
    = 2\pi \sum_{0 < |k| \le K_{\max}} \int_0^T |\widehat{Lh}(k, t)|^2\, dt .

Every extension class exposes :math:`\hat h(k, t)`, its analytic time
derivative :math:`\partial_t \hat h(k, t)`, the forcing coefficient, and a
closed-form value of the per-wavenumber time integral
:math:`\int_0^T |\widehat{Lh}(k, t)|^2 dt` — no time quadrature is used.

Terminal-distance factor.  The convex trial solution uses the linear factor
:math:`d_T(t) = 1 - t/T` with :math:`d_T'(t) = -1/T`; the extension is
:math:`h = (1 - d_T(t))\, g`, so :math:`\hat h(k, t) = (t/T)\, c_k` and the
extension coefficient equals the datum coefficient at :math:`t = T`.

Closed-form time integrals per wavenumber (implemented exactly as derived):

* split / graded: :math:`|b(k)|^2 |c_k|^2\, \varphi(2 \operatorname{Re}
  a_A(k))` with :math:`\varphi(z) = (e^{zT} - 1)/z` for :math:`z \ne 0` and
  :math:`\varphi(0) = T` (the :math:`z \to 0` limit is implemented as an
  explicit branch on the exact zero, not through an epsilon);
* constant-in-time: :math:`T\, |a(k)|^2 |c_k|^2`;
* convex raw: with :math:`\alpha = 1/T` and :math:`\beta = a(k)/T`,
  :math:`\int_0^T |\alpha + \beta t|^2 dt = |\alpha|^2 T +
  \operatorname{Re}(\overline{\alpha} \beta)\, T^2 + |\beta|^2 T^3 / 3`;
* exact solution: :math:`0` identically.

Dissipativity is validated (never silently enforced) before any semigroup
factor or exact solution is evaluated; a violation raises
:class:`ValueError` with the offending wavenumber and real part.
"""
from __future__ import annotations

import abc
import math

import numpy as np

from learning_option_pricing.pde.periodic_spectral_toolbox import (
    ConstantCoefficientGenerator,
)


TWO_PI = 2.0 * math.pi


def exponential_time_integral_factor(decay_rate, terminal_time: float) -> np.ndarray:
    r"""Closed-form factor :math:`\varphi(z) = \int_0^T e^{z(T-t)}\,dt = (e^{zT} - 1)/z`.

    The :math:`z \to 0` limit :math:`\varphi(0) = T` is implemented as an
    explicit branch on the exact zero (never through a denominator epsilon):
    entries with ``decay_rate == 0.0`` receive ``terminal_time`` directly,
    and every other entry is evaluated as ``expm1(z * T) / z``, which is
    numerically stable for small nonzero ``z``.

    Args:
        decay_rate: Real array (or scalar) of decay rates ``z``.
        terminal_time: The horizon ``T > 0``.

    Returns:
        ``float64`` array of the same shape as ``decay_rate``.
    """
    decay_rate_array = np.atleast_1d(np.asarray(decay_rate, dtype=np.float64))
    factor_values = np.full(decay_rate_array.shape, float(terminal_time))
    nonzero_mask = decay_rate_array != 0.0
    factor_values[nonzero_mask] = (
        np.expm1(decay_rate_array[nonzero_mask] * terminal_time)
        / decay_rate_array[nonzero_mask]
    )
    return factor_values.reshape(np.shape(decay_rate))


class TerminalDataExtension(abc.ABC):
    r"""Base class for analytic extensions of a terminal datum into the strip.

    Subclasses implement the extension coefficient :math:`\hat h(k, t)`, its
    analytic time derivative, and the closed-form per-wavenumber time
    integral of the squared forcing.  The forcing coefficient is assembled
    here as :math:`\partial_t \hat h + a(k) \hat h`.

    All coefficient methods are vectorised: ``wavenumbers`` and ``time`` may
    be any broadcast-compatible array shapes (e.g. ``time`` of shape
    ``(L, 1)`` against ``wavenumbers`` of shape ``(M,)`` yields ``(L, M)``).

    Args:
        datum: Terminal datum exposing ``fourier_coefficients(wavenumbers)``.
        generator: The constant-coefficient generator with symbol ``a``.
        terminal_time: The horizon ``T > 0`` (the study uses ``T = 1.0``).

    Raises:
        ValueError: If ``terminal_time`` is not strictly positive.
    """

    def __init__(
        self,
        datum,
        generator: ConstantCoefficientGenerator,
        terminal_time: float = 1.0,
    ) -> None:
        if terminal_time <= 0.0:
            raise ValueError(
                f"terminal_time must be strictly positive, received "
                f"{terminal_time!r}"
            )
        self.datum = datum
        self.generator = generator
        self.terminal_time = float(terminal_time)

    @abc.abstractmethod
    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        r"""Extension coefficient :math:`\hat h(k, t)` (``complex128``)."""

    @abc.abstractmethod
    def extension_coefficient_time_derivative(self, wavenumbers, time) -> np.ndarray:
        r"""Analytic time derivative :math:`\partial_t \hat h(k, t)` (``complex128``)."""

    @abc.abstractmethod
    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        r"""Closed-form :math:`\int_0^T |\widehat{Lh}(k, t)|^2\, dt` per wavenumber."""

    def forcing_coefficient(self, wavenumbers, time) -> np.ndarray:
        r"""Forcing coefficient :math:`\widehat{Lh}(k, t) = \partial_t \hat h + a(k) \hat h`."""
        return self.extension_coefficient_time_derivative(
            wavenumbers, time
        ) + self.generator.symbol(wavenumbers) * self.extension_coefficient(
            wavenumbers, time
        )


class ConvexRawExtension(TerminalDataExtension):
    r"""Convex trial extension :math:`h = (1 - d_T(t))\, g` with the linear factor.

    With the terminal-distance factor :math:`d_T(t) = 1 - t/T` (so
    :math:`1 - d_T(t) = t/T` and :math:`d_T'(t) = -1/T`):

    .. math::

        \hat h(k, t) = \frac{t}{T} c_k,
        \qquad
        \partial_t \hat h(k, t) = \frac{c_k}{T},
        \qquad
        \widehat{Lh}(k, t) = c_k \Bigl( \frac{1}{T} + \frac{t}{T} a(k) \Bigr).

    At :math:`t = T` the extension coefficient equals the datum coefficient
    :math:`c_k`, as required of a terminal-data extension.

    Closed-form time integral: with :math:`\alpha = 1/T` and
    :math:`\beta = a(k)/T`,

    .. math::

        \int_0^T |\widehat{Lh}(k, t)|^2\, dt
        = |c_k|^2 \bigl( |\alpha|^2 T
        + \operatorname{Re}(\overline{\alpha} \beta)\, T^2
        + |\beta|^2 T^3 / 3 \bigr).
    """

    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        time_array = np.asarray(time, dtype=np.float64)
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        return (time_array / self.terminal_time) * datum_coefficients

    def extension_coefficient_time_derivative(self, wavenumbers, time) -> np.ndarray:
        time_array = np.asarray(time, dtype=np.float64)
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        return np.broadcast_to(
            datum_coefficients / self.terminal_time,
            np.broadcast_shapes(time_array.shape, datum_coefficients.shape),
        ).copy()

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        symbol_values = self.generator.symbol(wavenumbers)
        alpha = 1.0 / self.terminal_time
        beta = symbol_values / self.terminal_time
        polynomial_time_integral = (
            abs(alpha) ** 2 * self.terminal_time
            + np.real(np.conjugate(alpha) * beta) * self.terminal_time**2
            + np.abs(beta) ** 2 * self.terminal_time**3 / 3.0
        )
        return np.abs(datum_coefficients) ** 2 * polynomial_time_integral


class ConstantInTimeExtension(TerminalDataExtension):
    r"""Constant-in-time extension :math:`\hat h(k, t) = c_k`.

    The time derivative vanishes and the forcing is
    :math:`\widehat{Lh}(k, t) = a(k)\, c_k`, independent of time, so the
    closed-form time integral is :math:`T\, |a(k)|^2 |c_k|^2`.
    """

    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        time_array = np.asarray(time, dtype=np.float64)
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        return np.broadcast_to(
            datum_coefficients,
            np.broadcast_shapes(time_array.shape, datum_coefficients.shape),
        ).copy()

    def extension_coefficient_time_derivative(self, wavenumbers, time) -> np.ndarray:
        time_array = np.asarray(time, dtype=np.float64)
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        return np.zeros(
            np.broadcast_shapes(time_array.shape, datum_coefficients.shape),
            dtype=np.complex128,
        )

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        symbol_values = self.generator.symbol(wavenumbers)
        return (
            self.terminal_time
            * np.abs(symbol_values) ** 2
            * np.abs(datum_coefficients) ** 2
        )


class SplitSemigroupExtension(TerminalDataExtension):
    r"""Semigroup extension driven by a dissipative subset of the generator.

    For a subset :math:`A` of the generator orders with subset symbol
    :math:`a_A(k)` and defect symbol :math:`b(k) = a(k) - a_A(k)`:

    .. math::

        \hat h(k, t) = e^{(T - t)\, a_A(k)}\, c_k,
        \qquad
        \partial_t \hat h(k, t) = -a_A(k)\, \hat h(k, t),

    so the forcing satisfies the identity
    :math:`\widehat{Lh}(k, t) = b(k)\, \hat h(k, t)` (verified to machine
    precision in the unit tests).  Dissipativity of the subset symbol over
    the supplied band is validated before every semigroup evaluation.

    Closed-form time integral:
    :math:`|b(k)|^2 |c_k|^2\, \varphi(2 \operatorname{Re} a_A(k))` with
    :math:`\varphi` from :func:`exponential_time_integral_factor`.

    Args:
        datum: Terminal datum exposing ``fourier_coefficients``.
        generator: The constant-coefficient generator.
        subset_orders: Differential orders defining the subset symbol.
        terminal_time: The horizon ``T > 0``.
    """

    def __init__(
        self,
        datum,
        generator: ConstantCoefficientGenerator,
        subset_orders,
        terminal_time: float = 1.0,
    ) -> None:
        super().__init__(datum, generator, terminal_time)
        self.generator_split = generator.split(subset_orders)
        self.subset_orders = self.generator_split.subset_orders
        self.defect_order = self.generator_split.defect_order

    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        time_array = np.asarray(time, dtype=np.float64)
        semigroup_values = self.generator.semigroup_multiplier(
            self.terminal_time - time_array, wavenumbers, self.subset_orders
        )
        return semigroup_values * self.datum.fourier_coefficients(wavenumbers)

    def extension_coefficient_time_derivative(self, wavenumbers, time) -> np.ndarray:
        subset_symbol_values = self.generator_split.subset_symbol(wavenumbers)
        return -subset_symbol_values * self.extension_coefficient(wavenumbers, time)

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        self.generator.validate_dissipativity(wavenumbers, self.subset_orders)
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        defect_symbol_values = self.generator_split.defect_symbol(wavenumbers)
        subset_symbol_values = self.generator_split.subset_symbol(wavenumbers)
        decay_factor = exponential_time_integral_factor(
            2.0 * np.real(subset_symbol_values), self.terminal_time
        )
        return (
            np.abs(defect_symbol_values) ** 2
            * np.abs(datum_coefficients) ** 2
            * decay_factor
        )


class GradedGaussianExtension(TerminalDataExtension):
    r"""Gaussian-graded extension :math:`\hat h(k, t) = e^{-(T-t)\, \nu_c k^2}\, c_k`.

    The comparison diffusivity :math:`\nu_c \ge 0` need not equal the
    generator's own diffusivity: this extension coincides with
    :class:`SplitSemigroupExtension` for the operator
    :math:`A = \nu_c \partial_{xx}` even when :math:`\nu_c` differs from the
    diffusivity present in the generator.  The forcing is

    .. math::

        \widehat{Lh}(k, t) = \bigl( a(k) + \nu_c k^2 \bigr)\, \hat h(k, t),

    and the closed-form time integral is
    :math:`|a(k) + \nu_c k^2|^2 |c_k|^2\, \varphi(-2 \nu_c k^2)` with
    :math:`\varphi` from :func:`exponential_time_integral_factor`.

    Args:
        datum: Terminal datum exposing ``fourier_coefficients``.
        generator: The constant-coefficient generator.
        comparison_diffusivity: The diffusivity :math:`\nu_c \ge 0` of the
            comparison heat semigroup.
        terminal_time: The horizon ``T > 0``.

    Raises:
        ValueError: If ``comparison_diffusivity`` is negative (the comparison
            semigroup would be antidissipative).
    """

    def __init__(
        self,
        datum,
        generator: ConstantCoefficientGenerator,
        comparison_diffusivity: float,
        terminal_time: float = 1.0,
    ) -> None:
        super().__init__(datum, generator, terminal_time)
        if comparison_diffusivity < 0.0:
            raise ValueError(
                "comparison_diffusivity must be non-negative (a negative "
                "value makes the comparison semigroup antidissipative), "
                f"received {comparison_diffusivity!r}"
            )
        self.comparison_diffusivity = float(comparison_diffusivity)

    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
        time_array = np.asarray(time, dtype=np.float64)
        gaussian_factor = np.exp(
            -(self.terminal_time - time_array)
            * self.comparison_diffusivity
            * wavenumber_array**2
        )
        return gaussian_factor * self.datum.fourier_coefficients(wavenumbers)

    def extension_coefficient_time_derivative(self, wavenumbers, time) -> np.ndarray:
        wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
        return (
            self.comparison_diffusivity
            * wavenumber_array**2
            * self.extension_coefficient(wavenumbers, time)
        )

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
        datum_coefficients = self.datum.fourier_coefficients(wavenumbers)
        defect_symbol_values = (
            self.generator.symbol(wavenumbers)
            + self.comparison_diffusivity * wavenumber_array**2
        )
        decay_factor = exponential_time_integral_factor(
            -2.0 * self.comparison_diffusivity * wavenumber_array**2,
            self.terminal_time,
        )
        return (
            np.abs(defect_symbol_values) ** 2
            * np.abs(datum_coefficients) ** 2
            * decay_factor
        )


class ExactSolutionExtension(TerminalDataExtension):
    r"""Exact solution :math:`\hat h(k, t) = e^{(T-t)\, a(k)}\, c_k` of the evolution.

    The time derivative is :math:`\partial_t \hat h = -a(k)\, \hat h`, so the
    forcing :math:`\partial_t \hat h + a(k) \hat h` vanishes identically; the
    base-class assembly reproduces this cancellation exactly in
    floating-point arithmetic because both terms are the same computed
    product :math:`a(k)\, \hat h(k, t)` with opposite signs.  Dissipativity
    of the full symbol over the supplied band is validated before every
    evaluation.

    The closed-form time integral of the squared forcing is zero.
    """

    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        self.generator.validate_dissipativity(wavenumbers)
        time_array = np.asarray(time, dtype=np.float64)
        symbol_values = self.generator.symbol(wavenumbers)
        return np.exp(
            (self.terminal_time - time_array) * symbol_values
        ) * self.datum.fourier_coefficients(wavenumbers)

    def extension_coefficient_time_derivative(self, wavenumbers, time) -> np.ndarray:
        symbol_values = self.generator.symbol(wavenumbers)
        return -symbol_values * self.extension_coefficient(wavenumbers, time)

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        self.generator.validate_dissipativity(wavenumbers)
        wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
        return np.zeros(wavenumber_array.shape, dtype=np.float64)


def total_strip_forcing_squared(
    extension: TerminalDataExtension, wavenumbers: np.ndarray
) -> float:
    r"""Squared strip norm of the forcing over the supplied wavenumber band.

    Evaluates (Parseval, closed-form time integrals; no time quadrature)

    .. math::

        \|Lh\|^2_{L^2((0,T);\,L^2(0,2\pi))}
        = 2\pi \sum_{k \in \text{band}} \int_0^T |\widehat{Lh}(k, t)|^2\, dt .

    Args:
        extension: A :class:`TerminalDataExtension` instance.
        wavenumbers: The band of integer wavenumbers to sum over, typically
            :func:`learning_option_pricing.pde.periodic_spectral_toolbox.symmetric_wavenumber_band`
            (which excludes :math:`k = 0`).

    Returns:
        The squared strip norm as a float.
    """
    return float(
        TWO_PI * np.sum(extension.squared_forcing_time_integral(wavenumbers))
    )
