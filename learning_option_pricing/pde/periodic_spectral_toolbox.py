r"""Exact Fourier toolbox on the circle for the spectral forcing study.

Domain and convention.  All objects are defined on the circle :math:`[0, 2\pi)`.
The Fourier coefficient of a function :math:`f` at the integer wavenumber
:math:`k` is

.. math::

    c_k = \frac{1}{2\pi} \int_0^{2\pi} f(x)\, e^{-ikx}\, dx,
    \qquad
    f(x) = \sum_{k \in \mathbb{Z}} c_k e^{ikx},

so Parseval's identity reads
:math:`\|f\|_{L^2(0,2\pi)}^2 = 2\pi \sum_k |c_k|^2`.

Measurement policy.  Every measurement in the study is made from the exact
analytic Fourier coefficients, evaluated (vectorised, ``complex128``) over
integer wavenumber arrays; the FFT of sampled values is never used as a
measurement.  FFT synthesis is provided only to plot a datum on a grid
(:func:`synthesise_datum_on_grid`), and truncates the Fourier series to the
band of wavenumbers resolvable on the grid — the band truncation is stated in
that function's docstring.

Contents.

* Terminal-datum classes with exact coefficients:
  :class:`PeriodisedBernoulliDatum` (single break point, regularity index
  :math:`\rho \in \{0, 1, 2\}`) and :class:`SquareWaveDatum` (two break
  points; the single-break-point floor prediction does not apply verbatim, so
  that case is measured only).
* :class:`ConstantCoefficientGenerator` — a constant-coefficient spatial
  generator through its Fourier symbol
  :math:`a(k) = \sum_j a_j (ik)^j`, with symbol splitting, dissipativity
  validation, and the semigroup multiplier :math:`e^{s a_A(k)}`.
* Named generator factories: :func:`advection_diffusion_reaction`,
  :func:`black_scholes_log_price`, :func:`biharmonic_advection_reaction`.
* The operator-channel floor
  :math:`\mathrm{floor}(K) = 2\pi \sum_{0 < |k| \le K} |a(k)|^2 |c_k|^2`
  and its predicted power-law constant and exponent for a single-break-point
  Bernoulli datum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


TWO_PI = 2.0 * math.pi

# Dissipativity bound: a symbol (or subset symbol) may only be exponentiated
# into a semigroup if max_k Re a(k) <= DISSIPATIVITY_TOLERANCE over the
# working band.  The validation raises ValueError with the offending values;
# it is never a silent clamp.
DISSIPATIVITY_TOLERANCE = 1.0e-12

# Squared L^2(0,1) norms of the Bernoulli polynomials B_n:
# integral_0^1 B_n(y)^2 dy = (n!)^2 / (2n)! * |B_{2n}| (Bernoulli number).
_SQUARED_UNIT_INTERVAL_NORM_OF_BERNOULLI_POLYNOMIAL = {
    1: 1.0 / 12.0,   # B_1(y) = y - 1/2
    2: 1.0 / 180.0,  # B_2(y) = y^2 - y + 1/6
    3: 1.0 / 840.0,  # B_3(y) = y^3 - (3/2) y^2 + (1/2) y
}


def symmetric_wavenumber_band(maximum_wavenumber: int) -> np.ndarray:
    """Return the integer wavenumbers ``k`` with ``0 < |k| <= maximum_wavenumber``.

    The zero wavenumber is excluded: every datum in this toolbox has a
    vanishing mean, so the ``k = 0`` term contributes nothing to any sum.

    Args:
        maximum_wavenumber: Band edge ``K``; must be a positive integer.

    Returns:
        Array of shape ``(2 * maximum_wavenumber,)`` with dtype ``int64``,
        ordered ``-K, ..., -1, 1, ..., K``.

    Raises:
        ValueError: If ``maximum_wavenumber`` is not a positive integer.
    """
    if int(maximum_wavenumber) != maximum_wavenumber or maximum_wavenumber < 1:
        raise ValueError(
            "maximum_wavenumber must be a positive integer, received "
            f"{maximum_wavenumber!r}"
        )
    positive_wavenumbers = np.arange(1, int(maximum_wavenumber) + 1, dtype=np.int64)
    return np.concatenate([-positive_wavenumbers[::-1], positive_wavenumbers])


# ---------------------------------------------------------------------------
# Terminal-datum classes with exact Fourier coefficients
# ---------------------------------------------------------------------------


class PeriodisedBernoulliDatum:
    r"""Periodised Bernoulli polynomial :math:`g(x) = B_{\rho+1}(x / 2\pi)` on the circle.

    The regularity index :math:`\rho \in \{0, 1, 2\}` counts the derivatives
    that remain continuous across the single break point :math:`x^\star = 0`:
    the datum belongs to :math:`C^{\rho-1}` (piecewise polynomial), its
    :math:`\rho`-th derivative jumps at :math:`x^\star = 0`, and its exact
    Fourier coefficients are

    .. math::

        c_0 = 0,
        \qquad
        c_k = -\frac{(\rho+1)!}{(2\pi i k)^{\rho+1}} \quad (k \ne 0),

    so that :math:`|c_k| = (\rho+1)! / (2\pi |k|)^{\rho+1}` holds exactly,
    with no remainder term.

    Jump of the :math:`\rho`-th derivative.  Differentiating
    :math:`B_n(x/2\pi)` gives :math:`(n / 2\pi) B_{n-1}(x/2\pi)`; after
    :math:`\rho` derivatives,
    :math:`g^{(\rho)}(x) = (\rho+1)!\,(2\pi)^{-\rho} B_1(\{x/2\pi\})`, and
    :math:`B_1(\{y\}) = \{y\} - 1/2` jumps by :math:`-1` at :math:`y = 0`.
    Hence the jump of the :math:`\rho`-th derivative equals
    :math:`-(\rho+1)!\,(2\pi)^{-\rho}` (attribute
    :attr:`jump_of_rho_derivative`).

    Attributes:
        regularity_index: The index :math:`\rho`.
        break_point: The single break point, ``0.0``.
        jump_of_rho_derivative: Jump of :math:`g^{(\rho)}` at the break point,
            equal to :math:`-(\rho+1)!\,(2\pi)^{-\rho}`.
        squared_l2_norm: The exact value of
            :math:`\|g\|_{L^2(0,2\pi)}^2 = 2\pi \int_0^1 B_{\rho+1}(y)^2\,dy`.
    """

    def __init__(self, regularity_index: int) -> None:
        if regularity_index not in (0, 1, 2):
            raise ValueError(
                "regularity_index must belong to {0, 1, 2}, received "
                f"{regularity_index!r}"
            )
        self.regularity_index = int(regularity_index)
        self.polynomial_degree = self.regularity_index + 1
        self.break_point = 0.0
        self.jump_of_rho_derivative = (
            -math.factorial(self.polynomial_degree) * TWO_PI ** (-self.regularity_index)
        )
        self.squared_l2_norm = (
            TWO_PI
            * _SQUARED_UNIT_INTERVAL_NORM_OF_BERNOULLI_POLYNOMIAL[self.polynomial_degree]
        )

    def fourier_coefficients(self, wavenumbers: np.ndarray) -> np.ndarray:
        r"""Exact Fourier coefficients :math:`c_k` at the given integer wavenumbers.

        Args:
            wavenumbers: Array of integer wavenumbers (any shape); internally
                cast to ``float64`` before exponentiation so that large bands
                (e.g. :math:`|k| = 2^{22}`) do not overflow integer arithmetic.

        Returns:
            ``complex128`` array of the same shape, with
            :math:`c_0 = 0` and
            :math:`c_k = -(\rho+1)! / (2\pi i k)^{\rho+1}` for :math:`k \ne 0`.
        """
        wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
        coefficient_values = np.zeros(wavenumber_array.shape, dtype=np.complex128)
        nonzero_mask = wavenumber_array != 0.0
        denominator = (
            TWO_PI * 1j * wavenumber_array[nonzero_mask]
        ) ** self.polynomial_degree
        coefficient_values[nonzero_mask] = (
            -float(math.factorial(self.polynomial_degree)) / denominator
        )
        return coefficient_values

    def pointwise_values(self, spatial_points: np.ndarray) -> np.ndarray:
        r"""Closed-form values :math:`g(x) = B_{\rho+1}(\{x / 2\pi\})`.

        At the break point :math:`x = 0` (only relevant for
        :math:`\rho = 0`, where the datum itself is discontinuous) the
        midpoint convention is used: the returned value is the average of the
        one-sided limits, which is the value to which the Fourier series
        converges there.

        Args:
            spatial_points: Array of positions on the circle (any real values;
                reduced modulo :math:`2\pi`).

        Returns:
            ``float64`` array of the same shape.
        """
        fractional_part = np.mod(
            np.asarray(spatial_points, dtype=np.float64) / TWO_PI, 1.0
        )
        if self.polynomial_degree == 1:
            values = fractional_part - 0.5
            # Midpoint convention at the discontinuity: B_1 has one-sided
            # limits -1/2 (right) and +1/2 (left) at y = 0, average 0.
            values = np.where(fractional_part == 0.0, 0.0, values)
        elif self.polynomial_degree == 2:
            values = fractional_part**2 - fractional_part + 1.0 / 6.0
        else:  # polynomial_degree == 3
            values = (
                fractional_part**3 - 1.5 * fractional_part**2 + 0.5 * fractional_part
            )
        return values


class SquareWaveDatum:
    r"""Square wave on the circle: :math:`g = +1` on :math:`(0, \pi)`, :math:`-1` on :math:`(\pi, 2\pi)`.

    Exact Fourier coefficients:

    .. math::

        c_0 = 0,
        \qquad
        c_k = \frac{2}{i \pi k} \ (k \text{ odd}),
        \qquad
        c_k = 0 \ (k \text{ even}).

    The regularity index is :math:`\rho = 0` and there are two break points:
    the datum jumps by :math:`+2` at :math:`x = 0` and by :math:`-2` at
    :math:`x = \pi`.  This is the multi-singularity extension case: the
    single-break-point floor prediction (see
    :func:`predicted_operator_channel_floor_constant`) does not apply
    verbatim, and the corresponding quantities are measured only.

    Attributes:
        regularity_index: ``0``.
        break_point_jumps: Mapping from break point to the jump of the datum
            there, ``{0.0: +2.0, pi: -2.0}``.
        squared_l2_norm: :math:`\|g\|_{L^2(0,2\pi)}^2 = 2\pi` (the datum has
            unit modulus almost everywhere).
    """

    def __init__(self) -> None:
        self.regularity_index = 0
        self.break_point_jumps = {0.0: 2.0, math.pi: -2.0}
        self.squared_l2_norm = TWO_PI

    def fourier_coefficients(self, wavenumbers: np.ndarray) -> np.ndarray:
        """Exact Fourier coefficients at the given integer wavenumbers.

        Args:
            wavenumbers: Array of integer wavenumbers (any shape).

        Returns:
            ``complex128`` array of the same shape, with
            ``c_k = 2 / (i pi k)`` for odd ``k`` and ``0`` otherwise.
        """
        wavenumber_array = np.asarray(wavenumbers)
        integer_wavenumbers = np.asarray(np.rint(wavenumber_array), dtype=np.int64)
        coefficient_values = np.zeros(wavenumber_array.shape, dtype=np.complex128)
        odd_mask = integer_wavenumbers % 2 != 0
        coefficient_values[odd_mask] = 2.0 / (
            1j * math.pi * integer_wavenumbers[odd_mask].astype(np.float64)
        )
        return coefficient_values

    def pointwise_values(self, spatial_points: np.ndarray) -> np.ndarray:
        r"""Closed-form values of the square wave, with the midpoint convention.

        At the break points :math:`x = 0` and :math:`x = \pi` the returned
        value is ``0``, the average of the one-sided limits (the value to
        which the Fourier series converges there).

        Args:
            spatial_points: Array of positions on the circle (any real values;
                reduced modulo :math:`2\pi`).

        Returns:
            ``float64`` array of the same shape.
        """
        reduced_points = np.mod(np.asarray(spatial_points, dtype=np.float64), TWO_PI)
        values = np.where(reduced_points < math.pi, 1.0, -1.0)
        on_break_point = (reduced_points == 0.0) | (reduced_points == math.pi)
        return np.where(on_break_point, 0.0, values)


def synthesise_datum_on_grid(
    datum, number_of_grid_points: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Synthesise datum values on a uniform grid by inverse FFT of the exact coefficients.

    Band truncation: on an ``N``-point grid the synthesis retains only the
    wavenumbers :math:`-N/2 \le k \le N/2 - 1` (the band resolvable on the
    grid); the discarded tail has coefficient magnitudes
    :math:`O(|k|^{-(\rho+1)})`, so the truncation error decreases with ``N``.
    This function is intended for plotting :math:`g(x)` only — measurements
    always use the exact coefficients directly.

    Args:
        datum: Object exposing ``fourier_coefficients(wavenumbers)``.
        number_of_grid_points: Grid size ``N`` (a power of two is efficient
            but not required).

    Returns:
        Tuple ``(grid_points, synthesised_values)`` where ``grid_points`` is
        ``np.linspace(0, 2*pi, N, endpoint=False)`` and
        ``synthesised_values`` is the real part of the truncated Fourier sum
        at those points.
    """
    grid_points = np.linspace(0.0, TWO_PI, number_of_grid_points, endpoint=False)
    fft_ordered_wavenumbers = np.rint(
        np.fft.fftfreq(number_of_grid_points, d=1.0 / number_of_grid_points)
    ).astype(np.int64)
    coefficient_values = datum.fourier_coefficients(fft_ordered_wavenumbers)
    # numpy's inverse FFT divides by N, so multiplying by N recovers the
    # plain truncated sum  sum_k c_k exp(i k x_n).
    synthesised_values = np.real(
        np.fft.ifft(coefficient_values) * number_of_grid_points
    )
    return grid_points, synthesised_values


# ---------------------------------------------------------------------------
# Constant-coefficient generator through its Fourier symbol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorSplit:
    r"""Splitting of a generator symbol into a subset part and its defect.

    For a generator with symbol :math:`a(k)` and a subset :math:`A` of its
    differential orders, the split is
    :math:`a(k) = a_A(k) + b(k)` with
    :math:`a_A(k) = \sum_{j \in A} a_j (ik)^j` (the subset symbol) and
    :math:`b(k) = a(k) - a_A(k)` (the defect symbol).  The defect order is
    :math:`m = \max\{j \notin A\}`, or :math:`-\infty` when the subset
    exhausts every order (then :math:`b = 0` identically).

    Attributes:
        full_generator: The generator being split.
        subset_orders: Sorted tuple of the orders in the subset.
        defect_orders: Sorted tuple of the remaining orders.
        defect_order: ``max(defect_orders)`` as a float, or ``-inf`` when the
            defect is empty.
    """

    full_generator: "ConstantCoefficientGenerator"
    subset_orders: tuple[int, ...]
    defect_orders: tuple[int, ...]
    defect_order: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "defect_order",
            float(max(self.defect_orders)) if self.defect_orders else float("-inf"),
        )

    def subset_symbol(self, wavenumbers: np.ndarray) -> np.ndarray:
        r"""Evaluate the subset symbol :math:`a_A(k) = \sum_{j \in A} a_j (ik)^j`."""
        return _evaluate_symbol(
            self.full_generator.coefficients, self.subset_orders, wavenumbers
        )

    def defect_symbol(self, wavenumbers: np.ndarray) -> np.ndarray:
        r"""Evaluate the defect symbol :math:`b(k) = \sum_{j \notin A} a_j (ik)^j`."""
        return _evaluate_symbol(
            self.full_generator.coefficients, self.defect_orders, wavenumbers
        )


def _evaluate_symbol(
    coefficients: dict[int, float],
    orders: tuple[int, ...],
    wavenumbers: np.ndarray,
) -> np.ndarray:
    """Evaluate ``sum_{j in orders} coefficients[j] * (i k)**j`` as ``complex128``.

    The wavenumbers are cast to ``float64`` before exponentiation so that
    high orders on large bands (e.g. ``k**4`` at ``|k| = 2**22``) do not
    overflow integer arithmetic.
    """
    wavenumber_array = np.asarray(wavenumbers, dtype=np.float64)
    symbol_values = np.zeros(wavenumber_array.shape, dtype=np.complex128)
    for order in sorted(orders):
        symbol_values += coefficients[order] * (1j * wavenumber_array) ** order
    return symbol_values


class ConstantCoefficientGenerator:
    r"""Constant-coefficient spatial generator represented by its Fourier symbol.

    The generator :math:`A = \sum_j a_j \partial_x^j` acts on the spectral
    component :math:`e^{ikx}` as multiplication by the symbol

    .. math::

        a(k) = \sum_j a_j (ik)^j \in \mathbb{C},

    evaluated in ``complex128``.  The maximal order :math:`2p` must be even;
    the principal constant is
    :math:`a_0 = \lim_{|k| \to \infty} |a(k)| / k^{2p} = |a_{2p}|`.

    Dissipativity is validated, never silently enforced: before a symbol (or
    subset symbol) is exponentiated into a semigroup, the bound
    :math:`\max_k \operatorname{Re} a(k) \le 10^{-12}` must hold over the
    working band, and a violation raises :class:`ValueError` reporting the
    offending wavenumber and real part.

    Args:
        coefficients: Mapping from differential order :math:`j` (non-negative
            integer) to the real coefficient :math:`a_j`.
        name: Descriptive name used in error messages and reports.

    Raises:
        ValueError: If the coefficient mapping is empty, contains a negative
            or non-integer order, has an odd maximal order, or has a vanishing
            coefficient at the maximal order.
    """

    def __init__(self, coefficients: dict[int, float], name: str) -> None:
        if not coefficients:
            raise ValueError(
                f"generator '{name}': the coefficient mapping is empty"
            )
        for order in coefficients:
            if int(order) != order or order < 0:
                raise ValueError(
                    f"generator '{name}': differential orders must be "
                    f"non-negative integers, received {order!r}"
                )
        self.coefficients = {int(order): float(value) for order, value in coefficients.items()}
        self.name = str(name)
        self.max_order = max(self.coefficients)
        if self.max_order % 2 != 0:
            raise ValueError(
                f"generator '{self.name}': the maximal order must be even, "
                f"received max order {self.max_order} "
                f"(coefficients {self.coefficients})"
            )
        if self.coefficients[self.max_order] == 0.0:
            raise ValueError(
                f"generator '{self.name}': the coefficient of the maximal "
                f"order {self.max_order} vanishes, so the principal constant "
                "is undefined"
            )
        self.half_order = self.max_order // 2
        # a_0 = lim_{|k| -> infinity} |a(k)| / k^{2p} = |a_{2p}|.
        self.principal_constant = abs(self.coefficients[self.max_order])

    def symbol(self, wavenumbers: np.ndarray) -> np.ndarray:
        r"""Evaluate the full symbol :math:`a(k) = \sum_j a_j (ik)^j`.

        Args:
            wavenumbers: Array of integer wavenumbers (any shape).

        Returns:
            ``complex128`` array of the same shape.
        """
        return _evaluate_symbol(
            self.coefficients, tuple(self.coefficients), wavenumbers
        )

    def split(self, subset_orders) -> GeneratorSplit:
        """Split the symbol into a subset part and its defect.

        Args:
            subset_orders: Iterable of differential orders to place in the
                subset; every order must be present in the generator's
                coefficient mapping.

        Returns:
            The corresponding :class:`GeneratorSplit`.

        Raises:
            ValueError: If an order in ``subset_orders`` is not an order of
                this generator.
        """
        requested_orders = tuple(sorted(set(int(order) for order in subset_orders)))
        unknown_orders = [
            order for order in requested_orders if order not in self.coefficients
        ]
        if unknown_orders:
            raise ValueError(
                f"generator '{self.name}': subset orders {unknown_orders} are "
                f"not among the generator orders {sorted(self.coefficients)}"
            )
        remaining_orders = tuple(
            sorted(set(self.coefficients) - set(requested_orders))
        )
        return GeneratorSplit(
            full_generator=self,
            subset_orders=requested_orders,
            defect_orders=remaining_orders,
        )

    def validate_dissipativity(
        self, wavenumbers: np.ndarray, subset_orders=None
    ) -> None:
        r"""Require :math:`\max_k \operatorname{Re} a(k) \le 10^{-12}` over the band.

        Args:
            wavenumbers: The working band over which the semigroup (or exact
                solution) will be evaluated.
            subset_orders: When given, the subset symbol :math:`a_A` is
                validated instead of the full symbol.

        Raises:
            ValueError: If the bound is violated; the message reports the
                worst offending wavenumber and the real part attained there.
        """
        if subset_orders is None:
            symbol_values = self.symbol(wavenumbers)
            symbol_description = f"full symbol of generator '{self.name}'"
        else:
            generator_split = self.split(subset_orders)
            symbol_values = generator_split.subset_symbol(wavenumbers)
            symbol_description = (
                f"subset symbol (orders {list(generator_split.subset_orders)}) "
                f"of generator '{self.name}'"
            )
        real_parts = np.real(np.asarray(symbol_values))
        worst_index = int(np.argmax(real_parts))
        worst_real_part = float(real_parts.flat[worst_index])
        if worst_real_part > DISSIPATIVITY_TOLERANCE:
            offending_wavenumber = np.asarray(wavenumbers).flat[worst_index]
            raise ValueError(
                f"dissipativity violated for the {symbol_description}: "
                f"Re a(k) = {worst_real_part:.6e} > "
                f"{DISSIPATIVITY_TOLERANCE:.0e} at wavenumber "
                f"k = {offending_wavenumber}"
            )

    def semigroup_multiplier(
        self, elapsed_time, wavenumbers: np.ndarray, subset_orders
    ) -> np.ndarray:
        r"""Semigroup multiplier :math:`e^{s\, a_A(k)}` of the subset symbol.

        Dissipativity of the subset symbol over the supplied band is
        validated before exponentiation (raising :class:`ValueError` on
        violation), and the elapsed time must be non-negative.

        Args:
            elapsed_time: Semigroup time :math:`s \ge 0`; scalar or array
                broadcastable against ``wavenumbers``.
            wavenumbers: Working band of integer wavenumbers.
            subset_orders: Orders defining the subset symbol :math:`a_A`.

        Returns:
            ``complex128`` array ``exp(elapsed_time * a_A(wavenumbers))``
            (broadcast shape).

        Raises:
            ValueError: If ``elapsed_time`` has a negative entry, or if the
                subset symbol violates the dissipativity bound on the band.
        """
        elapsed_time_array = np.asarray(elapsed_time, dtype=np.float64)
        if np.any(elapsed_time_array < 0.0):
            raise ValueError(
                f"generator '{self.name}': the semigroup multiplier requires "
                "a non-negative elapsed time, received minimum "
                f"{float(np.min(elapsed_time_array)):.6e}"
            )
        self.validate_dissipativity(wavenumbers, subset_orders)
        subset_symbol_values = self.split(subset_orders).subset_symbol(wavenumbers)
        return np.exp(elapsed_time_array * subset_symbol_values)


# ---------------------------------------------------------------------------
# Named generator instances
# ---------------------------------------------------------------------------


def advection_diffusion_reaction() -> ConstantCoefficientGenerator:
    r"""Generator G1: advection–diffusion–reaction, half order :math:`p = 1`.

    Coefficients ``{2: 0.7, 1: 1.3, 0: -0.4}``, hence the symbol
    :math:`a(k) = -0.7 k^2 + 1.3\,ik - 0.4` and the principal constant
    :math:`a_0 = 0.7`.
    """
    return ConstantCoefficientGenerator(
        coefficients={2: 0.7, 1: 1.3, 0: -0.4},
        name="advection_diffusion_reaction",
    )


def black_scholes_log_price(
    volatility: float = 0.5, risk_free_rate: float = 0.03
) -> ConstantCoefficientGenerator:
    r"""Generator G2: Black–Scholes generator in the log-price coordinate.

    With volatility :math:`\sigma` and risk-free rate :math:`r`, the spatial
    generator of the Black–Scholes equation in the log-price coordinate is
    :math:`\tfrac{\sigma^2}{2}\partial_{xx} + (r - \tfrac{\sigma^2}{2})
    \partial_x - r`.  The defaults :math:`\sigma = 0.5`, :math:`r = 0.03`
    give the coefficients ``{2: 0.125, 1: -0.095, 0: -0.03}`` and the
    principal constant :math:`a_0 = 0.125`.
    """
    diffusion_coefficient = 0.5 * volatility**2
    return ConstantCoefficientGenerator(
        coefficients={
            2: diffusion_coefficient,
            1: risk_free_rate - diffusion_coefficient,
            0: -risk_free_rate,
        },
        name="black_scholes_log_price",
    )


def biharmonic_advection_reaction() -> ConstantCoefficientGenerator:
    r"""Generator G3: biharmonic–advection–reaction, half order :math:`p = 2`.

    Coefficients ``{4: -0.05, 1: 1.3, 0: -0.4}``, hence the symbol
    :math:`a(k) = -0.05 k^4 + 1.3\,ik - 0.4` — note that
    :math:`(ik)^4 = k^4`, so the dissipative order-4 coefficient must be
    negative.  The principal constant is :math:`a_0 = 0.05`.
    """
    return ConstantCoefficientGenerator(
        coefficients={4: -0.05, 1: 1.3, 0: -0.4},
        name="biharmonic_advection_reaction",
    )


# ---------------------------------------------------------------------------
# Operator-channel floor and its predicted power law
# ---------------------------------------------------------------------------


def operator_channel_floor(
    generator: ConstantCoefficientGenerator,
    datum,
    maximum_wavenumber: int,
) -> float:
    r"""Band-truncated operator-channel floor
    :math:`\mathrm{floor}(K) = 2\pi \sum_{0 < |k| \le K} |a(k)|^2 |c_k|^2`.

    This is the squared :math:`L^2(0, 2\pi)` norm of :math:`A g` truncated to
    the band :math:`|k| \le K` (Parseval), evaluated from the exact analytic
    coefficients — never from an FFT of sampled values.

    Args:
        generator: The constant-coefficient generator with symbol ``a``.
        datum: Terminal datum exposing ``fourier_coefficients``.
        maximum_wavenumber: Band edge :math:`K`.

    Returns:
        The floor value as a float.
    """
    wavenumber_band = symmetric_wavenumber_band(maximum_wavenumber)
    symbol_values = generator.symbol(wavenumber_band)
    coefficient_values = datum.fourier_coefficients(wavenumber_band)
    return float(
        TWO_PI
        * np.sum(np.abs(symbol_values) ** 2 * np.abs(coefficient_values) ** 2)
    )


def predicted_floor_exponent(
    generator: ConstantCoefficientGenerator, datum
) -> int:
    r"""Predicted growth exponent :math:`e = 4p - 2\rho - 1` of the floor.

    For :math:`e > 0` the floor obeys
    :math:`\mathrm{floor}(K) = C_{\mathrm{pred}} K^e (1 + o(1))` as
    :math:`K \to \infty`; for :math:`e < 0` the floor saturates to its finite
    total sum.
    """
    return 4 * generator.half_order - 2 * datum.regularity_index - 1


def predicted_operator_channel_floor_constant(
    generator: ConstantCoefficientGenerator, datum
) -> float:
    r"""Predicted floor constant :math:`C_{\mathrm{pred}} = a_0^2\,J^2 / (\pi e)`.

    Here :math:`a_0` is the principal constant of the generator, :math:`J`
    is the jump of the :math:`\rho`-th derivative of the datum at its single
    break point, and :math:`e = 4p - 2\rho - 1 > 0` is the growth exponent.
    The prediction applies to a datum with a single break point (a
    :class:`PeriodisedBernoulliDatum`); for a multi-break datum such as
    :class:`SquareWaveDatum` the single-break-point prediction does not apply
    verbatim and the floor is measured only.

    Raises:
        ValueError: If the exponent satisfies :math:`e \le 0` (saturation
            regime: the floor tends to a finite limit and no power law
            holds), or if the datum does not expose a single-break-point
            jump.
    """
    growth_exponent = predicted_floor_exponent(generator, datum)
    if growth_exponent <= 0:
        raise ValueError(
            "the predicted power-law scaling of the floor requires a "
            "positive growth exponent, but "
            f"e = 4p - 2*rho - 1 = {growth_exponent} for generator "
            f"'{generator.name}' (p = {generator.half_order}) and regularity "
            f"index rho = {datum.regularity_index}; in this regime the floor "
            "saturates to its finite total sum"
        )
    if not hasattr(datum, "jump_of_rho_derivative"):
        raise ValueError(
            "the floor power-law constant is predicted only for a datum with "
            "a single break point exposing 'jump_of_rho_derivative'; the "
            f"received datum of type {type(datum).__name__} does not (for a "
            "multi-break datum the single-break-point prediction does not "
            "apply verbatim, and the floor is measured only)"
        )
    return (
        generator.principal_constant**2
        * datum.jump_of_rho_derivative**2
        / (math.pi * growth_exponent)
    )
