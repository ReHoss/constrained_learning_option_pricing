r"""Extension fields on the circle whose trace is the datum itself, with an adaptive Fourier band.

A field :math:`h(\cdot, t) = e^{sA} g`, where :math:`s = T - t` is the time elapsed since the
datum (the runner's terminal slice :math:`t = T` carries the datum), is known for a datum
:math:`g` given both by a closed form and by its cosine series :math:`\sum_k a_k \cos(kx)`.
A finite band of that series has the trace :math:`g_K`, the truncated datum, which is a
different initial condition.  This module evaluates the field so that its trace is
:math:`g` itself:

* at :math:`s = 0`, the closed form of :math:`g` (no series);
* at :math:`s > 0`, the cosine series over a band :math:`K(s)` chosen point by point.  The
  decay rates of the field are bounded below by :math:`c k^{r} - b` (:math:`r` the order of
  the retained principal term, :math:`c` its coefficient, :math:`b \ge 0` an offset computed
  from the rates), and :math:`K(s)` is the smallest power of two, not below the base band,
  with :math:`s\,(c K^{r} - b) \ge Z`.  The omitted components then carry a factor at most
  :math:`e^{-Z}`, and with :math:`|a_k| \le A k^{-q}`, :math:`q > 1`, the omitted part of the
  field is at most :math:`A\, e^{-Z} K^{1-q} / (q - 1)`.  For the square-root smoothings the
  components carry the multiplier :math:`m(z) = z K_1(z)` at :math:`z = k\,\varepsilon_0 (s/T)^{q_t}`
  (:math:`q_t = 1` graded, :math:`q_t = 1/2` square-root time), the criterion reads
  :math:`s^{q_t} c K \ge Z` with :math:`c = \varepsilon_0 / T^{q_t}`, and the bound uses
  :math:`m(s^{q_t} c K)` in place of :math:`e^{-Z}`.

Above the maximum band the field is evaluated on the maximum band; this truncation is reported
(WARNING on first occurrence, DEBUG afterwards), with the number of points, the smallest
:math:`s` and the bound of the omitted part, never applied silently.  Derivatives at
:math:`s = 0` are not defined for a datum with a first-derivative discontinuity; they are
evaluated on the base band and reported in the same way.

The field of each band is the :class:`~learning_option_pricing.pde.periodic_extension_fields.
PeriodicExtensionField` returned by the same builder, so that at every point the value is the
closed form of that module on the band chosen for the point.
"""

from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSION_KINDS = (
    "split_diffusion",
    "split_diffusion_advection",
    "split_principal",
    "split_principal_advection",
    "exact_solution",
    "graded_chen_mangasarian",
    "square_root_smoothing_sqrt_time",
    "square_root_smoothing_sqrt_time_transported",
)
# Square-root smoothings: the component of wavenumber k carries the multiplier m(z) = z K_1(z) at
# z = k eps(s), eps(s) = eps_0 (s/T)^q (q = 1 graded, q = 1/2 for the square-root-time kinds).
# The multiplier decreases from 1 to 0, so the omitted components are bounded with m itself.
SQUARE_ROOT_SMOOTHING_KINDS = (
    "graded_chen_mangasarian",
    "square_root_smoothing_sqrt_time",
    "square_root_smoothing_sqrt_time_transported",
)
DEFAULT_TAIL_EXPONENT = 40.0
# Points times components evaluated per call of a band field: 2^24 float64 entries per
# intermediate array (128 MB), the size of a training batch of 4096 points on 4096 components.
POINTS_TIMES_COMPONENTS_PER_CHUNK = 2**24
DERIVATIVE_METHOD_NAMES = {
    "dt": "time_derivative",
    "dx": "space_derivative",
    "dxx": "second_space_derivative",
    "dxxx": "third_space_derivative",
    "dxxxx": "fourth_space_derivative",
    "dxxxxx": "fifth_space_derivative",
    "dxxxxxx": "sixth_space_derivative",
}
TWO_PI = 2.0 * math.pi


def periodised_bernoulli_values(x):
    r"""Datum :math:`g(x) = y^2 - y + 1/6`, :math:`y = (x \bmod 2\pi)/(2\pi)`, in closed form.

    Its cosine series is :math:`\sum_{k \ge 1} \cos(kx)/(\pi^2 k^2)`; it is continuous with a
    first-derivative discontinuity at :math:`x = 0 \bmod 2\pi`.  Accepts numpy arrays or torch
    tensors and keeps the dtype; the value is computed in ``float64`` and then rounded to that
    dtype, so that it equals bit for bit the trace of :class:`ExactTraceAdaptiveBandField`.
    """
    if isinstance(x, torch.Tensor):
        unit_position = torch.remainder(x.to(torch.float64), TWO_PI) / TWO_PI
        return (unit_position * unit_position - unit_position + 1.0 / 6.0).to(x.dtype)
    array = np.asarray(x)
    unit_position = np.mod(array.astype(np.float64), TWO_PI) / TWO_PI
    values = unit_position * unit_position - unit_position + 1.0 / 6.0
    return values.astype(array.dtype) if np.issubdtype(array.dtype, np.floating) else values


class ExactTraceAdaptiveBandField:
    r"""Field :math:`e^{sA} g` on the circle with trace :math:`g` and a band adapted to :math:`s`.

    Args:
        field_builder: A builder of :data:`~learning_option_pricing.pde.periodic_extension_fields.
            EXTENSION_FIELD_REGISTRY` (same signature), called once per band used.
        generator_coefficients: Mapping from differential order to coefficient of the operator.
        datum_values: Closed form of the datum, for numpy arrays and torch tensors.
        cosine_coefficients_for_band: Callable returning the cosine amplitudes
            :math:`(a_k)_{1 \le k \le K}` for a band :math:`K`.
        datum_amplitude_constant: The constant :math:`A` of :math:`|a_k| \le A k^{-q}`.
        datum_amplitude_exponent: The exponent :math:`q > 1`.
        terminal_time: The horizon :math:`T`.
        base_band: The smallest band, used wherever it suffices.
        maximum_band: The largest band; points needing more are evaluated on it and reported.
        tail_exponent: The exponent :math:`Z` of the band criterion.
        builder_keyword_arguments: Extra keyword arguments of the builder.

    Raises:
        ValueError: On an extension kind without exponential decay of its components (the
            band criterion needs it), or on inconsistent bands or exponents.
    """

    def __init__(
        self,
        field_builder: Callable,
        generator_coefficients: dict[int, float],
        *,
        datum_values: Callable,
        cosine_coefficients_for_band: Callable[[int], np.ndarray],
        datum_amplitude_constant: float,
        datum_amplitude_exponent: float,
        terminal_time: float,
        base_band: int,
        maximum_band: int,
        tail_exponent: float = DEFAULT_TAIL_EXPONENT,
        builder_keyword_arguments: dict | None = None,
    ) -> None:
        if not 1 <= int(base_band) <= int(maximum_band):
            raise ValueError(f"bands must satisfy 1 <= base_band <= maximum_band, got {base_band}, {maximum_band}")
        if not tail_exponent > 0.0:
            raise ValueError(f"tail_exponent must be positive, got {tail_exponent!r}")
        if not datum_amplitude_exponent > 1.0:
            raise ValueError(f"datum_amplitude_exponent must exceed 1, got {datum_amplitude_exponent!r}")
        self._field_builder = field_builder
        self._builder_keyword_arguments = dict(builder_keyword_arguments or {})
        self._datum_values = datum_values
        self._cosine_coefficients_for_band = cosine_coefficients_for_band
        self.generator_coefficients = dict(generator_coefficients)
        self.datum_amplitude_constant = float(datum_amplitude_constant)
        self.datum_amplitude_exponent = float(datum_amplitude_exponent)
        self.terminal_time = float(terminal_time)
        self.base_band = int(base_band)
        self.maximum_band = int(maximum_band)
        self.tail_exponent = float(tail_exponent)
        self._fields_by_band: dict = {}
        self.base_field = self._field_for_band(self.base_band)
        self.extension_kind = self.base_field.extension_kind
        if self.extension_kind not in SUPPORTED_EXTENSION_KINDS:
            raise ValueError(
                f"extension kind {self.extension_kind!r} has no exponential decay of its components; "
                f"supported kinds: {SUPPORTED_EXTENSION_KINDS}")
        # Exponent q of the decay law s^q (c k^r - b): q = 1 for the semigroups, and the width-law
        # exponent for the square-root smoothings.
        self.decay_time_exponent = 1.0
        if self.extension_kind in ("split_diffusion", "split_diffusion_advection"):
            self.decay_order, self.decay_coefficient = 2, float(self.base_field.diffusivity)
        elif self.extension_kind in SQUARE_ROOT_SMOOTHING_KINDS:
            # Convolution of the datum with the square-root kernel of width eps(s) = eps_0 (s/T)^q:
            # the multiplier z K_1(z), z = k eps(s) = s^q (eps_0 / T^q) k, is at most
            # sqrt(pi z / 2) e^{-z} (1 + 3/(8z)) for z >= 1, so its components decay like
            # e^{-s^q c k} with c = eps_0 / T^q (order one). The transported kind has the further
            # factor e^{s r_0} <= 1 (r_0 <= 0, validated by the field).
            self.decay_order = 1
            self.decay_time_exponent = float(self.base_field.smoothing_time_exponent)
            self.decay_coefficient = (float(self.base_field.initial_smoothing_scale)
                                      / self.terminal_time ** self.decay_time_exponent)
        else:
            self.decay_order = int(self.base_field.principal_order)
            self.decay_coefficient = abs(float(self.base_field.generator_coefficients[self.decay_order]))
        if not self.decay_coefficient > 0.0:
            raise ValueError(f"the retained principal coefficient must be non-zero, got {self.decay_coefficient!r}")
        self.decay_rate_offset = self._decay_rate_offset(self.base_field)
        self._report_counts = {"maximum_band": 0, "datum_slice_derivative": 0}

    # -- bands ----------------------------------------------------------------------------

    def _decay_rate_offset(self, field) -> float:
        """Smallest b >= 0 with decay_rate_k >= c k^r - b on the band of ``field``."""
        if field.extension_kind in SQUARE_ROOT_SMOOTHING_KINDS:
            return 0.0   # the multiplier bound applies with b = 0 (the transported factor is at most 1)
        arrays = field._numpy_coefficient_arrays
        lower_bound = self.decay_coefficient * arrays["wavenumbers"] ** self.decay_order
        return float(max(0.0, np.max(lower_bound - arrays["decay_rates"])))

    def _field_for_band(self, band: int):
        if band not in self._fields_by_band:
            field = self._field_builder(
                self.generator_coefficients, self._cosine_coefficients_for_band(band),
                terminal_time=self.terminal_time, **self._builder_keyword_arguments)
            if self._fields_by_band and self._decay_rate_offset(field) > self.decay_rate_offset + 1e-12:
                raise ValueError(f"the decay-rate lower bound of the base band fails on band {band}")
            self._fields_by_band[band] = field
        return self._fields_by_band[band]

    def required_band(self, time_since_datum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Band of each positive :math:`s` and the mask of the points capped at the maximum band."""
        time_since_datum = np.asarray(time_since_datum, dtype=np.float64)
        if np.any(time_since_datum <= 0.0):
            raise ValueError("required_band applies to positive times since the datum")
        # s^q, with the semigroup arithmetic s itself unchanged for q = 1.
        decay_time = (time_since_datum if self.decay_time_exponent == 1.0
                      else time_since_datum ** self.decay_time_exponent)
        needed = ((self.tail_exponent + self.decay_rate_offset * time_since_datum)
                  / (self.decay_coefficient * decay_time)) ** (1.0 / self.decay_order)
        exponents = np.ceil(np.log2(np.maximum(needed, 1.0)))
        bands = np.where(needed <= self.base_band, float(self.base_band), 2.0 ** exponents)
        capped = bands > self.maximum_band
        return np.minimum(bands, float(self.maximum_band)).astype(np.int64), capped

    def omitted_part_bound(self, time_since_datum: float, band: int) -> float:
        r"""Bound of the omitted part of the field, :math:`A\,\phi\,K^{1-q}/(q-1)`.

        For the semigroups, :math:`\phi = e^{-s(cK^r - b)}`.  For the square-root smoothings,
        :math:`\phi = m(c\,s^{q_t} K)` with the multiplier :math:`m(z) = z K_1(z)` itself, which
        decreases from 1, so that it bounds every omitted component; the factor
        :math:`e^{-z}` alone would understate it by up to :math:`\sqrt{\pi z/2}\,(1 + 3/(8z))`
        (corrected 2026-10-21, before which the warnings of the graded kind used :math:`e^{-z}`).
        """
        if self.extension_kind in SQUARE_ROOT_SMOOTHING_KINDS:
            from learning_option_pricing.pde.terminal_data_extensions import chen_mangasarian_multiplier

            scaled = self.decay_coefficient * time_since_datum ** self.decay_time_exponent * band
            factor = float(chen_mangasarian_multiplier(scaled))
            return (self.datum_amplitude_constant * factor * band ** (1.0 - self.datum_amplitude_exponent)
                    / (self.datum_amplitude_exponent - 1.0))
        exponent = time_since_datum * (self.decay_coefficient * band**self.decay_order - self.decay_rate_offset)
        return (self.datum_amplitude_constant * math.exp(-exponent) * band ** (1.0 - self.datum_amplitude_exponent)
                / (self.datum_amplitude_exponent - 1.0))

    def _report(self, key: str, count: int, message: str) -> None:
        first = self._report_counts[key] == 0
        self._report_counts[key] += count
        (LOGGER.warning if first else LOGGER.debug)(
            "%s (%d point(s) this call, %d in total)", message, count, self._report_counts[key])

    # -- evaluation -----------------------------------------------------------------------

    def _evaluate(self, method_name: str, x, t):
        is_torch = isinstance(x, torch.Tensor) or isinstance(t, torch.Tensor)
        if is_torch:
            x_tensor = torch.as_tensor(x)
            t_tensor = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, device=x_tensor.device)
            original_dtype = x_tensor.dtype
            x_original, _ = torch.broadcast_tensors(x_tensor, t_tensor.to(x_tensor.dtype))
            x_original_flat = x_original.reshape(-1)
            x64, t64 = torch.broadcast_tensors(x_tensor.to(torch.float64), t_tensor.to(torch.float64, copy=False))
            shape, x_flat, t_flat = x64.shape, x64.reshape(-1), t64.reshape(-1)
            time_since_datum = (self.terminal_time - t_flat).detach().cpu().numpy()
            output = torch.empty(x_flat.shape, dtype=torch.float64, device=x_flat.device)
        else:
            x64, t64 = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
            shape, x_flat, t_flat = x64.shape, x64.reshape(-1), t64.reshape(-1)
            x_original_flat = x_flat
            time_since_datum = self.terminal_time - t_flat
            output = np.empty(x_flat.shape, dtype=np.float64)
        if np.any(time_since_datum < 0.0):
            raise ValueError(f"times beyond the datum slice t = {self.terminal_time} are not defined")

        def select(indices):
            return torch.as_tensor(indices, device=x_flat.device) if is_torch else indices

        on_datum_slice = np.flatnonzero(time_since_datum == 0.0)
        if on_datum_slice.size:
            index = select(on_datum_slice)
            if method_name == "field":
                # The datum on the input in its own dtype: the trace then equals datum_values(x) bit for bit.
                datum = self._datum_values(x_original_flat[index])
                output[index] = datum.to(torch.float64) if is_torch else datum
            else:
                self._report("datum_slice_derivative", on_datum_slice.size,
                             f"[exact-trace {self.extension_kind}] {method_name} at the datum slice s = 0 is not "
                             f"defined for a datum with a first-derivative discontinuity; evaluated on the base "
                             f"band K = {self.base_band}")
                output[index] = getattr(self.base_field, method_name)(x_flat[index], t_flat[index])
        positive = np.flatnonzero(time_since_datum > 0.0)
        if positive.size:
            bands, capped = self.required_band(time_since_datum[positive])
            if np.any(capped):
                smallest = float(np.min(time_since_datum[positive][capped]))
                self._report("maximum_band", int(np.count_nonzero(capped)),
                             f"[exact-trace {self.extension_kind}] times since the datum down to s = {smallest:.3e} "
                             f"need a band above the maximum {self.maximum_band}; evaluated on it, omitted part "
                             f"of the field at most {self.omitted_part_bound(smallest, self.maximum_band):.3e}")
            for band in np.unique(bands):
                band_indices = positive[bands == band]
                field = self._field_for_band(int(band))
                chunk = max(1, POINTS_TIMES_COMPONENTS_PER_CHUNK // int(band))
                for start in range(0, band_indices.size, chunk):
                    index = select(band_indices[start:start + chunk])
                    output[index] = getattr(field, method_name)(x_flat[index], t_flat[index])
        output = output.reshape(shape)
        return output.to(dtype=original_dtype) if is_torch else output

    def field(self, x, t):
        r"""Field :math:`h(x, t)`; equal to the closed form of :math:`g` where :math:`t = T`."""
        return self._evaluate("field", x, t)

    def time_derivative(self, x, t):
        return self._evaluate("time_derivative", x, t)

    def space_derivative(self, x, t):
        return self._evaluate("space_derivative", x, t)

    def second_space_derivative(self, x, t):
        return self._evaluate("second_space_derivative", x, t)

    def third_space_derivative(self, x, t):
        return self._evaluate("third_space_derivative", x, t)

    def fourth_space_derivative(self, x, t):
        return self._evaluate("fourth_space_derivative", x, t)

    def fifth_space_derivative(self, x, t):
        return self._evaluate("fifth_space_derivative", x, t)

    def sixth_space_derivative(self, x, t):
        return self._evaluate("sixth_space_derivative", x, t)

    def forcing_values(self, x, t):
        r"""Forcing :math:`(Ph)(x, t)` of the band field of each point."""
        return self._evaluate("forcing_values", x, t)

    def terminal_datum_values(self, x):
        r"""The datum :math:`g(x)` in closed form, equal to ``field(x, T)``."""
        return self._datum_values(x)

    def terminal_forcing_profile(self, x):
        r"""Forcing at :math:`t = T`, defined on the base band only (reported)."""
        if isinstance(x, torch.Tensor):
            terminal = torch.full_like(x, self.terminal_time)
        else:
            terminal = np.full(np.shape(np.asarray(x)), self.terminal_time)
        return self.forcing_values(x, terminal)

    def derivative_callables(self) -> dict:
        """The analytic-derivative mapping, with the keys of the base field."""
        return {key: getattr(self, DERIVATIVE_METHOD_NAMES[key]) for key in self.base_field.derivative_callables()}

    def __getattr__(self, name: str):
        # Coefficient attributes (diagnostics on the base band) come from the base field.
        if name.startswith("__") or name in ("base_field", "_fields_by_band"):
            raise AttributeError(name)
        return getattr(self.base_field, name)
