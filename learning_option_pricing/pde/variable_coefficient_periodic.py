r"""Variable-coefficient generators on the circle with one trigonometric harmonic.

Setting.  On the circle :math:`\mathbb T = \mathbb R / (2\pi\mathbb Z)` the spatial
generator is

.. math::

    L^X u = \sum_{j} c_j(x)\, \partial_x^j u,
    \qquad
    c_j(x) = c_{j,0} + c_{j,1} \cos(x - \varphi_j),

with at most one harmonic per coefficient.  This is the class of the
variable-coefficient cells of the pre-registration
``documents/methodology/2026-09-29_preregistration_variable_coefficient_split.md``
(local volatility in log-price at order 2, a variable biharmonic coefficient at
order 4).

Fourier action.  On :math:`e^{imx}`, the term :math:`c_j\,\partial_x^j` gives

.. math::

    c_{j,0} (im)^j e^{imx}
    + \tfrac{c_{j,1}}{2} (im)^j \bigl( e^{-i\varphi_j} e^{i(m+1)x}
    + e^{i\varphi_j} e^{i(m-1)x} \bigr),

so the Fourier coefficient of :math:`L^X f` at wavenumber :math:`k` is

.. math::

    \widehat{L^X f}(k) = a_0(k)\,\hat f(k) + \ell(k-1)\,\hat f(k-1)
    + \upsilon(k+1)\,\hat f(k+1),

with the mean symbol :math:`a_0(m) = \sum_j c_{j,0} (im)^j`, the lower coupling
:math:`\ell(m) = \sum_j \tfrac{c_{j,1}}{2} e^{-i\varphi_j} (im)^j` and the upper
coupling :math:`\upsilon(m) = \sum_j \tfrac{c_{j,1}}{2} e^{i\varphi_j} (im)^j`.  A
band-limited :math:`f` of band :math:`M` is mapped to band :math:`M+1`, exactly.

Contents.

* :class:`TrigonometricCoefficient` — one coefficient, evaluable on ``numpy``
  arrays and ``torch`` tensors (the training operators use the latter).
* :class:`VariableCoefficientGenerator` — the generator, its Fourier action, its
  Galerkin matrix, and the constant-coefficient operators obtained by freezing
  the principal coefficient at a point or at its mean.
* :class:`BandLimitedDatum` — a datum restricted to :math:`0 < |k| \le K`.
* :class:`GalerkinReferenceSolution` — the reference solution
  :math:`u^\star(\cdot, t) = \exp\bigl((T - t) L^X_N\bigr) g` of the terminal-value
  problem, the generator having no closed-form semigroup.
* Exact spectral counterparts of the four extensions of the study
  (:class:`VariableCoefficientConstantInTimeExtension`,
  :class:`VariableCoefficientConvexRawExtension`,
  :class:`VariableCoefficientFrozenSplitExtension`), with the per-wavenumber time
  integral of the squared forcing in closed form.  Their interface matches the
  one the stage-2 runner and aggregator use for the constant-coefficient
  extensions (``forcing_coefficient``, ``squared_forcing_time_integral``).

Validation policy: invalid arguments raise :class:`ValueError`; nothing is
silently clamped.
"""
from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Coefficients and generator
# ---------------------------------------------------------------------------


class TrigonometricCoefficient:
    r"""Coefficient :math:`c(x) = c_0 + c_1 \cos(x - \varphi)`.

    Args:
        constant: The mean :math:`c_0 \in \mathbb R`.
        amplitude: The amplitude :math:`c_1 \in \mathbb R`.
        phase: The phase :math:`\varphi \in \mathbb R`.
    """

    def __init__(self, constant: float, amplitude: float = 0.0, phase: float = 0.0) -> None:
        for name, value in (("constant", constant), ("amplitude", amplitude), ("phase", phase)):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, received {value!r}")
        self.constant = float(constant)
        self.amplitude = float(amplitude)
        self.phase = float(phase)

    @classmethod
    def from_specification(cls, specification) -> "TrigonometricCoefficient":
        """Build from a float (constant coefficient) or a mapping with keys
        ``constant``, ``amplitude``, ``phase`` (the catalogue form)."""
        if isinstance(specification, dict):
            return cls(
                specification["constant"],
                specification.get("amplitude", 0.0),
                specification.get("phase", 0.0),
            )
        return cls(float(specification))

    def is_constant(self) -> bool:
        return self.amplitude == 0.0

    def __call__(self, x):
        r"""Values :math:`c(x)` on a ``torch`` tensor or an array-like."""
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is a project dependency
            torch = None
        if torch is not None and isinstance(x, torch.Tensor):
            return self.constant + self.amplitude * torch.cos(x - self.phase)
        x_array = np.asarray(x, dtype=np.float64)
        return self.constant + self.amplitude * np.cos(x_array - self.phase)

    def value_at(self, point: float) -> float:
        return self.constant + self.amplitude * math.cos(point - self.phase)

    @property
    def minimum(self) -> float:
        return self.constant - abs(self.amplitude)

    @property
    def maximum(self) -> float:
        return self.constant + abs(self.amplitude)

    def description(self) -> dict:
        """Serialisable description (for metadata and logs)."""
        return {"constant": self.constant, "amplitude": self.amplitude, "phase": self.phase}


def _even_order_dissipative_sign(order: int) -> float:
    r"""Sign :math:`\sigma_p` such that :math:`c\,\partial_x^{2p}` is dissipative iff
    :math:`\sigma_p c > 0`: the symbol of :math:`\partial_x^{2p}` is
    :math:`(ik)^{2p} = (-1)^p k^{2p}`, so :math:`\sigma_p = (-1)^{p+1}`."""
    return (-1.0) ** (order // 2 + 1)


class VariableCoefficientGenerator:
    r"""Generator :math:`L^X = \sum_j c_j(x)\,\partial_x^j` with trigonometric coefficients.

    Args:
        coefficients: Mapping from differential order (0 to 4) to a float, a
            :class:`TrigonometricCoefficient`, or its catalogue specification.
        name: Descriptive name used in error messages and logs.

    Raises:
        ValueError: On an unsupported order, a missing even principal order
            (2 or 4), or a principal coefficient that is not of dissipative
            sign at every point of the circle.
    """

    SUPPORTED_ORDERS = (0, 1, 2, 3, 4)

    def __init__(self, coefficients: dict, name: str) -> None:
        if not coefficients:
            raise ValueError(f"generator {name!r}: empty coefficient mapping")
        self.name = str(name)
        self.coefficients: dict[int, TrigonometricCoefficient] = {}
        for order, specification in coefficients.items():
            if int(order) != order or int(order) not in self.SUPPORTED_ORDERS:
                raise ValueError(
                    f"generator {name!r}: orders must belong to {self.SUPPORTED_ORDERS}, "
                    f"received {order!r}"
                )
            coefficient = (
                specification
                if isinstance(specification, TrigonometricCoefficient)
                else TrigonometricCoefficient.from_specification(specification)
            )
            self.coefficients[int(order)] = coefficient
        even_orders = [order for order in self.coefficients if order % 2 == 0 and order > 0]
        if not even_orders:
            raise ValueError(f"generator {name!r}: no even principal order (2 or 4)")
        self.principal_order = max(even_orders)
        if max(self.coefficients) != self.principal_order:
            raise ValueError(
                f"generator {name!r}: the highest order {max(self.coefficients)} is odd"
            )
        principal = self.coefficients[self.principal_order]
        sign = _even_order_dissipative_sign(self.principal_order)
        worst_value = min(sign * principal.minimum, sign * principal.maximum)
        if worst_value <= 0.0:
            raise ValueError(
                f"generator {name!r}: the principal order-{self.principal_order} "
                f"coefficient ranges over [{principal.minimum}, {principal.maximum}] and "
                "is not of dissipative sign at every point"
            )

    # -- descriptions and runtime forms ------------------------------------

    def description(self) -> dict:
        return {order: c.description() for order, c in sorted(self.coefficients.items())}

    def runtime_coefficients(self) -> dict:
        r"""Mapping order :math:`\to` coefficient for the training operators:
        a float for a constant coefficient (so the constant-coefficient channels
        are computed exactly as before), the :class:`TrigonometricCoefficient`
        itself (a callable of the spatial coordinate) otherwise."""
        return {
            order: (c.constant if c.is_constant() else c)
            for order, c in sorted(self.coefficients.items())
        }

    def frozen_principal_coefficients(self, at) -> dict[int, float]:
        r"""Constant-coefficient principal operator :math:`c\,\partial_x^{2p}`,
        frozen at the point ``at`` (a float) or at the mean (``at="mean"``)."""
        principal = self.coefficients[self.principal_order]
        if isinstance(at, str):
            if at != "mean":
                raise ValueError(f"'at' must be a point or 'mean', received {at!r}")
            value = principal.constant
        else:
            value = principal.value_at(float(at))
        return {self.principal_order: value}

    # -- Fourier action ------------------------------------------------------

    def _symbol_parts(self, wavenumbers):
        """Mean symbol a_0, lower coupling l and upper coupling u at ``wavenumbers``."""
        m = np.asarray(wavenumbers, dtype=np.float64)
        mean_symbol = np.zeros(m.shape, dtype=np.complex128)
        lower = np.zeros(m.shape, dtype=np.complex128)
        upper = np.zeros(m.shape, dtype=np.complex128)
        for order, c in self.coefficients.items():
            derivative_symbol = (1j * m) ** order
            mean_symbol = mean_symbol + c.constant * derivative_symbol
            if not c.is_constant():
                lower = lower + 0.5 * c.amplitude * np.exp(-1j * c.phase) * derivative_symbol
                upper = upper + 0.5 * c.amplitude * np.exp(1j * c.phase) * derivative_symbol
        return mean_symbol, lower, upper

    def mean_symbol(self, wavenumbers) -> np.ndarray:
        return self._symbol_parts(wavenumbers)[0]

    def couplings(self, wavenumbers):
        """Return (lower, upper) couplings at ``wavenumbers``."""
        _, lower, upper = self._symbol_parts(wavenumbers)
        return lower, upper

    def apply_fourier(self, coefficients_dense: np.ndarray) -> np.ndarray:
        r"""Exact :math:`\widehat{L^X f}` on band :math:`M + 1` from :math:`\hat f` on
        band :math:`M` (dense arrays ordered :math:`k = -M, \dots, M`)."""
        f = np.asarray(coefficients_dense, dtype=np.complex128)
        band = (f.shape[-1] - 1) // 2
        if f.shape[-1] != 2 * band + 1:
            raise ValueError("dense coefficient arrays must have odd length 2M + 1")
        m = np.arange(-band, band + 1)
        mean_symbol, lower, upper = self._symbol_parts(m)
        out = np.zeros(f.shape[:-1] + (2 * band + 3,), dtype=np.complex128)
        # Output index of wavenumber k is k + band + 1.
        out[..., 1:-1] += mean_symbol * f
        out[..., 2:] += lower * f      # f_m contributes to k = m + 1
        out[..., :-2] += upper * f     # f_m contributes to k = m - 1
        return out

    def galerkin_matrix(self, band: int) -> np.ndarray:
        r"""Galerkin matrix on :math:`|k| \le N` (dense, ordered :math:`-N, \dots, N`):
        the couplings leaving the band are truncated."""
        if int(band) != band or band < 1:
            raise ValueError(f"band must be a positive integer, received {band!r}")
        m = np.arange(-band, band + 1)
        mean_symbol, lower, upper = self._symbol_parts(m)
        matrix = np.diag(mean_symbol)
        size = 2 * band + 1
        rows = np.arange(1, size)
        matrix[rows, rows - 1] = lower[:-1]   # entry (k, k-1) = l(k-1)
        matrix[rows - 1, rows] = upper[1:]    # entry (k, k+1) = u(k+1)
        return matrix


def generator_from_specification(specification: dict, name: str) -> VariableCoefficientGenerator:
    """Build a :class:`VariableCoefficientGenerator` from a catalogue mapping."""
    return VariableCoefficientGenerator(
        {int(order): value for order, value in specification.items()}, name
    )


# ---------------------------------------------------------------------------
# Datum and reference solution
# ---------------------------------------------------------------------------


class BandLimitedDatum:
    r"""Datum restricted to the band :math:`0 < |k| \le K` of a base datum.

    The base datum exposes ``fourier_coefficients(wavenumbers)`` (for instance
    :class:`learning_option_pricing.pde.periodic_spectral_toolbox.PeriodisedBernoulliDatum`).
    Outside the band the coefficients are zero, which the variable-coefficient
    coupling requires: it reads :math:`\hat g` one wavenumber beyond the band.
    """

    def __init__(self, base_datum, band_edge: int) -> None:
        if int(band_edge) != band_edge or band_edge < 1:
            raise ValueError(f"band_edge must be a positive integer, received {band_edge!r}")
        self.base_datum = base_datum
        self.band_edge = int(band_edge)

    def fourier_coefficients(self, wavenumbers) -> np.ndarray:
        k = np.asarray(wavenumbers)
        values = np.zeros(k.shape, dtype=np.complex128)
        inside = (k != 0) & (np.abs(k) <= self.band_edge)
        if np.any(inside):
            values[inside] = self.base_datum.fourier_coefficients(k[inside])
        return values

    def dense_coefficients(self, band: int) -> np.ndarray:
        """Coefficients on :math:`k = -N, \\dots, N` (zero-padded beyond the datum band)."""
        return self.fourier_coefficients(np.arange(-band, band + 1))


def synthesise_dense(coefficients_dense: np.ndarray, x) -> np.ndarray:
    r"""Real part of :math:`\sum_k \hat f(k) e^{ikx}` at the points ``x``."""
    f = np.asarray(coefficients_dense, dtype=np.complex128)
    band = (f.shape[-1] - 1) // 2
    k = np.arange(-band, band + 1)
    x_array = np.asarray(x, dtype=np.float64)
    phases = np.exp(1j * x_array[..., None] * k)
    return np.real(phases @ f)


class GalerkinReferenceSolution:
    r"""Reference solution :math:`u^\star(\cdot, t) = \exp\bigl((T - t) L^X_N\bigr) g`.

    The Fourier–Galerkin matrix :math:`L^X_N` on :math:`|k| \le N` is exponentiated
    with :func:`scipy.linalg.expm` at each distinct evaluation time (results are
    cached per time).  At :math:`t = T` the datum is returned exactly.

    Args:
        generator: The :class:`VariableCoefficientGenerator`.
        datum: A :class:`BandLimitedDatum`.
        galerkin_band: The truncation :math:`N`, larger than the datum band.
        terminal_time: The horizon :math:`T > 0`.
    """

    def __init__(self, generator, datum, galerkin_band: int, terminal_time: float) -> None:
        if galerkin_band <= datum.band_edge:
            raise ValueError(
                f"galerkin_band {galerkin_band} must exceed the datum band {datum.band_edge}"
            )
        if not terminal_time > 0.0:
            raise ValueError(f"terminal_time must be positive, received {terminal_time!r}")
        self.generator = generator
        self.datum = datum
        self.galerkin_band = int(galerkin_band)
        self.terminal_time = float(terminal_time)
        self._matrix = generator.galerkin_matrix(self.galerkin_band)
        self._datum_dense = datum.dense_coefficients(self.galerkin_band)
        self._cache: dict[float, np.ndarray] = {}

    def fourier_coefficients_at(self, time: float) -> np.ndarray:
        """Dense coefficients of :math:`u^\\star(\\cdot, t)` on :math:`|k| \\le N`."""
        key = float(time)
        if key not in self._cache:
            time_to_terminal = self.terminal_time - key
            if time_to_terminal < 0.0:
                raise ValueError(f"time {time} lies beyond the horizon {self.terminal_time}")
            if time_to_terminal == 0.0:
                self._cache[key] = self._datum_dense.copy()
            else:
                from scipy.linalg import expm

                self._cache[key] = expm(time_to_terminal * self._matrix) @ self._datum_dense
        return self._cache[key]

    def field(self, x, t) -> np.ndarray:
        r"""Values :math:`u^\star(x, t)` (``float64``); ``x`` and ``t`` broadcast.

        The evaluation is grouped by distinct values of ``t``, so it is intended
        for a small set of evaluation times (the runner's time slices)."""
        x_array = np.asarray(x, dtype=np.float64)
        t_array = np.asarray(t, dtype=np.float64)
        x_array, t_array = np.broadcast_arrays(x_array, t_array)
        values = np.empty(x_array.shape, dtype=np.float64)
        for time in np.unique(t_array):
            mask = t_array == time
            values[mask] = synthesise_dense(self.fourier_coefficients_at(float(time)), x_array[mask])
        return values

    def terminal_datum_values(self, x) -> np.ndarray:
        return synthesise_dense(self._datum_dense, x)


def galerkin_convergence_deviation(
    generator, datum, galerkin_band: int, refined_band: int, terminal_time: float, times, x
) -> float:
    r"""Largest relative :math:`\ell^2` deviation, over ``times``, between the
    references at truncations ``galerkin_band`` and ``refined_band`` on the
    points ``x``."""
    coarse = GalerkinReferenceSolution(generator, datum, galerkin_band, terminal_time)
    fine = GalerkinReferenceSolution(generator, datum, refined_band, terminal_time)
    worst = 0.0
    for time in times:
        u_coarse = coarse.field(x, np.full_like(np.asarray(x, dtype=np.float64), float(time)))
        u_fine = fine.field(x, np.full_like(np.asarray(x, dtype=np.float64), float(time)))
        scale = float(np.linalg.norm(u_fine))
        deviation = float(np.linalg.norm(u_coarse - u_fine))
        worst = max(worst, deviation / scale if scale > 0.0 else deviation)
    return worst


def relative_deviations_from_fields(first_values, second_values, times) -> dict:
    r"""Relative :math:`\ell^2` deviations between two fields sampled on times :math:`\times` points.

    Both arrays have shape ``(len(times), number of points)``; the second field is the
    normalising one.  Returns

    * ``per_time``: :math:`\lVert u_1(\cdot,t) - u_2(\cdot,t)\rVert / \lVert u_2(\cdot,t)\rVert`
      at each time;
    * ``maximum_over_times`` and ``time_of_maximum``;
    * ``space_time``: :math:`\lVert u_1 - u_2\rVert / \lVert u_2\rVert` over all times and
      points together, which the maximum over times bounds from above.

    Raises:
        ValueError: On mismatched shapes, or if the normalising field vanishes at some
            time (no epsilon is added to a denominator).
    """
    first = np.asarray(first_values, dtype=np.float64)
    second = np.asarray(second_values, dtype=np.float64)
    time_values = [float(t) for t in times]
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != len(time_values):
        raise ValueError(
            f"fields must share the shape (number of times, number of points); received "
            f"{first.shape} and {second.shape} for {len(time_values)} times"
        )
    normalising_norms = np.linalg.norm(second, axis=1)
    if np.any(normalising_norms == 0.0):
        raise ValueError("the normalising field vanishes at some time; the relative deviation is undefined")
    per_time = np.linalg.norm(first - second, axis=1) / normalising_norms
    index_of_maximum = int(np.argmax(per_time))
    return {
        "per_time": [float(value) for value in per_time],
        "maximum_over_times": float(per_time[index_of_maximum]),
        "time_of_maximum": time_values[index_of_maximum],
        "space_time": float(np.linalg.norm(first - second) / np.linalg.norm(second)),
    }


def galerkin_reference_deviations(
    generator, datum, first_band: int, second_band: int, terminal_time: float, times, x
) -> dict:
    r"""Deviations between the references at truncations ``first_band`` and ``second_band``.

    The references are sampled at the points ``x`` and the times ``times``; the second
    truncation normalises.  Returns the dictionary of
    :func:`relative_deviations_from_fields`, whose ``maximum_over_times`` entry equals
    :func:`galerkin_convergence_deviation` and whose ``space_time`` entry is the ratio of
    the norms over the whole space-time sample.
    """
    x_array = np.asarray(x, dtype=np.float64)
    fields = []
    for band in (first_band, second_band):
        reference = GalerkinReferenceSolution(generator, datum, band, terminal_time)
        fields.append(np.stack([
            reference.field(x_array, np.full_like(x_array, float(time))) for time in times
        ]))
    return relative_deviations_from_fields(fields[0], fields[1], times)


# ---------------------------------------------------------------------------
# Exact spectral counterparts of the extensions
# ---------------------------------------------------------------------------


def complex_time_integral_factor(rate, terminal_time: float) -> np.ndarray:
    r""":math:`\int_0^T e^{z s}\, ds = (e^{zT} - 1)/z`, for complex ``z``; the exact
    value :math:`T` is used where :math:`z = 0` (no denominator epsilon)."""
    z = np.asarray(rate, dtype=np.complex128)
    out = np.full(z.shape, complex(terminal_time), dtype=np.complex128)
    nonzero = z != 0
    out[nonzero] = np.expm1(z[nonzero] * terminal_time) / z[nonzero]
    return out


class _VariableCoefficientExtensionBase:
    """Shared construction: generator, band-limited datum, horizon."""

    def __init__(self, generator, datum, terminal_time: float = 1.0) -> None:
        if not terminal_time > 0.0:
            raise ValueError(f"terminal_time must be positive, received {terminal_time!r}")
        self.generator = generator
        self.datum = datum
        self.terminal_time = float(terminal_time)

    @property
    def forcing_band_edge(self) -> int:
        """The forcing lives on :math:`|k| \\le K + 1` (one harmonic in the coefficients)."""
        return self.datum.band_edge + 1

    def _generator_applied_to_datum(self, wavenumbers) -> np.ndarray:
        """:math:`\\widehat{L^X g}(k)` at ``wavenumbers`` (exact)."""
        k = np.asarray(wavenumbers)
        g = self.datum.fourier_coefficients
        mean_symbol = self.generator.mean_symbol(k)
        lower, _ = self.generator.couplings(k - 1)
        _, upper = self.generator.couplings(k + 1)
        return mean_symbol * g(k) + lower * g(k - 1) + upper * g(k + 1)


class VariableCoefficientConstantInTimeExtension(_VariableCoefficientExtensionBase):
    r"""Extension :math:`h = g`; forcing :math:`L^X g`, independent of time."""

    def forcing_coefficient(self, wavenumbers, time) -> np.ndarray:
        values = self._generator_applied_to_datum(wavenumbers)
        return np.broadcast_to(values, np.broadcast(np.asarray(wavenumbers), np.asarray(time)).shape).copy()

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        return self.terminal_time * np.abs(self._generator_applied_to_datum(wavenumbers)) ** 2


class VariableCoefficientConvexRawExtension(_VariableCoefficientExtensionBase):
    r"""Extension :math:`h = (t/T)\,g`; forcing :math:`g/T + (t/T)\,L^X g`, and

    .. math::

        \int_0^T \Bigl|\tfrac{\hat g}{T} + \tfrac{t}{T}\widehat{L^X g}\Bigr|^2 dt
        = T|a|^2 + T\,\mathrm{Re}(a\bar b) + \tfrac{T}{3}|b|^2,
        \quad a = \hat g/T,\ b = \widehat{L^X g}.
    """

    def forcing_coefficient(self, wavenumbers, time) -> np.ndarray:
        k = np.asarray(wavenumbers)
        t = np.asarray(time, dtype=np.float64)
        return (
            self.datum.fourier_coefficients(k) / self.terminal_time
            + (t / self.terminal_time) * self._generator_applied_to_datum(k)
        )

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        k = np.asarray(wavenumbers)
        a = self.datum.fourier_coefficients(k) / self.terminal_time
        b = self._generator_applied_to_datum(k)
        T = self.terminal_time
        return T * np.abs(a) ** 2 + T * np.real(a * np.conj(b)) + (T / 3.0) * np.abs(b) ** 2


class VariableCoefficientFrozenSplitExtension(_VariableCoefficientExtensionBase):
    r"""Split extension :math:`h = e^{(T-t)A} g` for a constant-coefficient ``A``.

    With :math:`\lambda(m)` the symbol of ``A``, :math:`h_m(s) = e^{s\lambda(m)}\hat g_m`
    (:math:`s = T - t`), and the forcing :math:`(L^X - A)h` has the coefficient

    .. math::

        \hat F_k(s) = \sum_{m \in \{k-1, k, k+1\}} \gamma_{k,m}\, e^{s\lambda(m)}\hat g_m,
        \quad \gamma_{k,k} = a_0(k) - \lambda(k),\ \gamma_{k,k-1} = \ell(k-1),\
        \gamma_{k,k+1} = \upsilon(k+1),

    whose squared modulus integrates in closed form:
    :math:`\sum_{m,m'} \gamma_{k,m}\bar\gamma_{k,m'}\hat g_m\bar{\hat g}_{m'}
    \int_0^T e^{s(\lambda(m) + \bar\lambda(m'))} ds`.

    Args:
        generator: The :class:`VariableCoefficientGenerator` :math:`L^X`.
        datum: A :class:`BandLimitedDatum`.
        retained_coefficients: Mapping order :math:`\to` float of ``A`` (from
            :meth:`VariableCoefficientGenerator.frozen_principal_coefficients`).
        terminal_time: The horizon :math:`T > 0`.
    """

    def __init__(self, generator, datum, retained_coefficients: dict, terminal_time: float = 1.0) -> None:
        super().__init__(generator, datum, terminal_time)
        self.retained_coefficients = {int(o): float(c) for o, c in retained_coefficients.items()}

    def retained_symbol(self, wavenumbers) -> np.ndarray:
        m = np.asarray(wavenumbers, dtype=np.float64)
        symbol = np.zeros(m.shape, dtype=np.complex128)
        for order, value in self.retained_coefficients.items():
            symbol = symbol + value * (1j * m) ** order
        return symbol

    def _terms(self, k):
        """(gamma, lambda, g) for m = k-1, k, k+1, stacked on a trailing axis."""
        neighbours = np.stack([k - 1, k, k + 1], axis=-1)
        mean_symbol = self.generator.mean_symbol(k)
        lower, _ = self.generator.couplings(k - 1)
        _, upper = self.generator.couplings(k + 1)
        gamma = np.stack(
            [lower, mean_symbol - self.retained_symbol(k), upper], axis=-1
        )
        rates = self.retained_symbol(neighbours)
        datum_values = self.datum.fourier_coefficients(neighbours)
        return gamma, rates, datum_values

    def extension_coefficient(self, wavenumbers, time) -> np.ndarray:
        k = np.asarray(wavenumbers)
        s = self.terminal_time - np.asarray(time, dtype=np.float64)
        return np.exp(s * self.retained_symbol(k)) * self.datum.fourier_coefficients(k)

    def forcing_coefficient(self, wavenumbers, time) -> np.ndarray:
        k = np.asarray(wavenumbers)
        s = np.asarray(self.terminal_time - np.asarray(time, dtype=np.float64))
        gamma, rates, datum_values = self._terms(k)
        return np.sum(gamma * np.exp(s[..., None] * rates) * datum_values, axis=-1)

    def squared_forcing_time_integral(self, wavenumbers) -> np.ndarray:
        k = np.asarray(wavenumbers)
        gamma, rates, datum_values = self._terms(k)
        amplitudes = gamma * datum_values                       # (..., 3)
        pair_rates = rates[..., :, None] + np.conj(rates[..., None, :])
        factors = complex_time_integral_factor(pair_rates, self.terminal_time)
        integral = np.einsum("...m,...n,...mn->...", amplitudes, np.conj(amplitudes), factors)
        return np.real(integral)


def build_variable_coefficient_extension(
    variant_name: str, generator, datum, singular_point: float, terminal_time: float = 1.0
):
    """Spectral counterpart of a variant of the variable-coefficient cells.

    ``variant_name`` is one of ``convex_raw``, ``constant_in_time``,
    ``split_frozen_singular`` (principal coefficient frozen at
    ``singular_point``), ``split_frozen_mean``; any other name returns ``None``.
    """
    if variant_name == "convex_raw":
        return VariableCoefficientConvexRawExtension(generator, datum, terminal_time)
    if variant_name == "constant_in_time":
        return VariableCoefficientConstantInTimeExtension(generator, datum, terminal_time)
    if variant_name == "split_frozen_singular":
        return VariableCoefficientFrozenSplitExtension(
            generator, datum, generator.frozen_principal_coefficients(singular_point), terminal_time
        )
    if variant_name == "split_frozen_mean":
        return VariableCoefficientFrozenSplitExtension(
            generator, datum, generator.frozen_principal_coefficients("mean"), terminal_time
        )
    return None


def full_wavenumber_band(band_edge: int) -> np.ndarray:
    r"""Integer wavenumbers :math:`|k| \le K`, **including** :math:`k = 0`: a
    variable coefficient moves datum content into the zero mode of the forcing,
    which the constant-coefficient band (which excludes :math:`k = 0`) omits."""
    return np.arange(-int(band_edge), int(band_edge) + 1)


def strip_forcing_energy(extension, band_edge: int | None = None) -> float:
    r""":math:`\lVert Lh \rVert^2_{L^2(Q)} = 2\pi \sum_{|k| \le K+1} \int_0^T |\hat F_k|^2\,dt`."""
    edge = extension.forcing_band_edge if band_edge is None else int(band_edge)
    return float(TWO_PI * np.sum(extension.squared_forcing_time_integral(full_wavenumber_band(edge))))
