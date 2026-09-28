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


def chen_mangasarian_multiplier(scaled_wavenumber) -> np.ndarray:
    r"""Fourier multiplier :math:`m(z) = z K_1(z)` of the Chen--Mangasarian kernel.

    The Chen--Mangasarian smoothing of the ramp,
    :math:`\tfrac12\bigl(y + \sqrt{y^2 + \varepsilon^2}\bigr)`, is the convolution of
    :math:`y^+` with the kernel
    :math:`\varphi_\varepsilon(y) = \varepsilon^2 / \bigl(2 (y^2 + \varepsilon^2)^{3/2}\bigr)`,
    of unit mass, whose Fourier transform is
    :math:`\hat\varphi_\varepsilon(\xi) = \varepsilon|\xi|\, K_1(\varepsilon|\xi|)`,
    with :math:`K_1` the modified Bessel function of the second kind.  On the
    circle the periodised kernel has the same multiplier at each integer
    wavenumber (Poisson summation).  The value at :math:`z = 0` is the limit
    :math:`\lim_{z \to 0} z K_1(z) = 1`, set explicitly because
    :math:`K_1(0) = +\infty`.

    Args:
        scaled_wavenumber: The argument :math:`z = \varepsilon |k| \ge 0`.

    Returns:
        ``float64`` array of :math:`z K_1(z)`, equal to ``1`` where ``z == 0``.
    """
    from scipy.special import k1

    z = np.abs(np.asarray(scaled_wavenumber, dtype=np.float64))
    positive = z > 0.0
    return np.where(positive, z * k1(np.where(positive, z, 1.0)), 1.0)


def chen_mangasarian_multiplier_derivative(scaled_wavenumber) -> np.ndarray:
    r"""Derivative :math:`m'(z) = -z K_0(z)` of the Chen--Mangasarian multiplier.

    Follows from the recurrence :math:`(z K_1(z))' = -z K_0(z)`.  The value at
    :math:`z = 0` is the limit :math:`\lim_{z \to 0} z K_0(z) = 0`
    (since :math:`K_0(z) = -\ln(z/2) - \gamma + o(1)` as :math:`z \to 0`).

    Args:
        scaled_wavenumber: The argument :math:`z = \varepsilon |k| \ge 0`.

    Returns:
        ``float64`` array of :math:`-z K_0(z)`, equal to ``0`` where ``z == 0``.
    """
    from scipy.special import k0

    z = np.abs(np.asarray(scaled_wavenumber, dtype=np.float64))
    positive = z > 0.0
    return np.where(positive, -z * k0(np.where(positive, z, 1.0)), 0.0)


class GradedChenMangasarianExtension(TerminalDataExtension):
    r"""Linearly graded Chen--Mangasarian extension.

    The datum is convolved with the Chen--Mangasarian kernel at the graded
    scale :math:`\varepsilon(t) = \varepsilon_0 (T - t)/T`, which vanishes at the
    terminal slice:

    .. math::

        \hat h(k, t) = m\bigl(|k|\, \varepsilon(t)\bigr)\, c_k,
        \qquad m(z) = z K_1(z),

    so :math:`\hat h(k, T) = c_k` and the datum is met exactly.  With
    :math:`\partial_t \varepsilon = -\varepsilon_0/T`,

    .. math::

        \partial_t \hat h(k, t) = -\frac{\varepsilon_0 |k|}{T}\,
        m'\bigl(|k|\, \varepsilon(t)\bigr)\, c_k
        = \frac{\varepsilon_0 |k|}{T}\, z K_0(z)\, c_k ,
        \qquad z = |k|\, \varepsilon(t),

    which is bounded on :math:`[0, T]` and vanishes at :math:`t = T`.  The
    family :math:`\{\varphi_\varepsilon\}` is not a semigroup, so no schedule makes
    this extension cancel the principal part of the generator; at the slice
    its forcing coefficient reduces to :math:`a(k)\, c_k`, that of the
    constant-in-time extension.  The linear grading is used because the
    parabolic grading :math:`\varepsilon \propto \sqrt{T - t}` makes
    :math:`\partial_t \hat h` diverge logarithmically at the slice.

    The squared-forcing time integral has no closed form; it is evaluated by
    Gauss--Legendre quadrature in time (the only quadrature among the
    extensions of this module), with the node count stated in the docstring
    of :meth:`squared_forcing_time_integral`.

    Args:
        datum: Terminal datum exposing ``fourier_coefficients``.
        generator: The constant-coefficient generator.
        initial_smoothing_scale: The scale :math:`\varepsilon_0 > 0` at
            :math:`t = 0`.
        terminal_time: The horizon ``T > 0``.

    Raises:
        ValueError: If ``initial_smoothing_scale`` is not strictly positive.
    """

    GAUSS_LEGENDRE_TIME_NODES = 256

    def __init__(
        self,
        datum,
        generator: ConstantCoefficientGenerator,
        initial_smoothing_scale: float,
        terminal_time: float = 1.0,
    ) -> None:
        super().__init__(datum, generator, terminal_time)
        if not initial_smoothing_scale > 0.0:
            raise ValueError(
                "initial_smoothing_scale must be strictly positive (a zero scale "
                "is the constant-in-time extension), received "
                f"{initial_smoothing_scale!r}"
            )
        self.initial_smoothing_scale = float(initial_smoothing_scale)

    def smoothing_scale_at(self, time) -> np.ndarray:
        r"""Graded scale :math:`\varepsilon(t) = \varepsilon_0 (T - t) / T`."""
        time_array = np.asarray(time, dtype=np.float64)
        return (
            self.initial_smoothing_scale
            * (self.terminal_time - time_array)
            / self.terminal_time
        )

    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        wavenumber_array = np.abs(np.asarray(wavenumbers, dtype=np.float64))
        scaled = wavenumber_array * self.smoothing_scale_at(time)
        return chen_mangasarian_multiplier(scaled) * self.datum.fourier_coefficients(
            wavenumbers
        )

    def extension_coefficient_time_derivative(self, wavenumbers, time) -> np.ndarray:
        wavenumber_array = np.abs(np.asarray(wavenumbers, dtype=np.float64))
        scaled = wavenumber_array * self.smoothing_scale_at(time)
        rate = self.initial_smoothing_scale * wavenumber_array / self.terminal_time
        return (
            -rate
            * chen_mangasarian_multiplier_derivative(scaled)
            * self.datum.fourier_coefficients(wavenumbers)
        )

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        r"""Per-wavenumber :math:`\int_0^T |\widehat{Lh}(k, t)|^2\, dt` by
        Gauss--Legendre quadrature with ``GAUSS_LEGENDRE_TIME_NODES`` nodes on
        :math:`[0, T]`.  The integrand is continuous on :math:`[0, T]`; its
        only non-analytic point is :math:`t = T`, where
        :math:`z K_0(z) = -z \ln z + O(z)`.
        """
        nodes, weights = np.polynomial.legendre.leggauss(self.GAUSS_LEGENDRE_TIME_NODES)
        times = 0.5 * self.terminal_time * (nodes + 1.0)
        half_length_weights = 0.5 * self.terminal_time * weights
        wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
        forcing = self.forcing_coefficient(wavenumber_array[None, :], times[:, None])
        return np.sum(half_length_weights[:, None] * np.abs(forcing) ** 2, axis=0)


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
