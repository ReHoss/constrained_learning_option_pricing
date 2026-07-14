"""The measured cancellation cutoff of a trained network, without any clamp.

The cutoff :math:`k^\\star` is the wavenumber at which a trained network stops
cancelling the extension forcing: below it the achieved residual is small
compared with the forcing, above it the residual matches the forcing. It is the
measured counterpart of the reachable-band edge of the ideal-filter proposition.

The quantity it is read from is the *cancellation ratio*

.. math::

    \\rho_k = \\frac{|\\hat r_k|^2}{|\\widehat{\\mathcal{L}h}(k)|^2} ,

the achieved-residual power over the forcing power at the wavenumber ``k``.

Why this needs care, and why no clamp appears here
--------------------------------------------------

The ratio is a quotient of two quantities that both become numerically
meaningless at large ``k``, and for *opposite* reasons:

* the forcing of a split-generator extension is suppressed like a Gaussian in
  ``k`` — measured on the production runs it falls from :math:`2\\cdot10^{-5}`
  at ``k = 1`` to :math:`4\\cdot10^{-188}` at ``k = 128``;
* the achieved residual cannot fall below the round-off level of the autograd
  evaluation and of the transform — measured on the same runs, it flattens at
  about :math:`3\\cdot10^{-19}` and stays there.

So for ``k`` beyond roughly 30 the ratio is a numerical floor divided by a
physically vanishing forcing, and it grows without bound (values above
:math:`10^{160}` are observed *inside* the datum band). An earlier version of
this estimator truncated the ratio at 1.5 to keep those values from destroying
the running mean. That truncation was a clamp over a broken guard: it was
load-bearing — it made the cutoff of the ``G1`` cell vanish on two seeds of
three, manufacturing a seed instability that does not exist — and its value had
no derivation.

The defect is not in the ratio's magnitude but in its *domain*: the ratio simply
carries no information where the forcing lies below the residual's own numerical
floor. That floor is **measured, not chosen**. The stage-two datum is
band-limited, so the forcing vanishes *identically* above its band edge; the
residual power at those wavenumbers is therefore pure round-off, and reading its
maximum there gives the floor directly from the run. Restricting the ratio to the
wavenumbers whose forcing stands above that floor makes the ratio bounded by
construction, and no value is ever truncated.
"""

from __future__ import annotations

import numpy as np

# Width of the centred running mean applied to the cancellation ratio before
# thresholding. The ratio is noisy per wavenumber (it is an average over five
# time slices), and smoothing before thresholding makes the cutoff robust. This
# is a smoother, declared as part of the estimator -- it is not a clamp: it
# rewrites no value, it averages them.
DEFAULT_RUNNING_MEAN_WINDOW = 7

# The cutoff is declared at the wavenumber where half of the forcing power is no
# longer cancelled. The value 1/2 is the definition of the cutoff, not a guard.
DEFAULT_CANCELLATION_THRESHOLD = 0.5


def residual_numerical_floor(
    forcing_power: np.ndarray, residual_power: np.ndarray
) -> float:
    """The residual's numerical floor, read off the run rather than chosen.

    The datum being band-limited, the forcing vanishes identically above its
    band edge. The residual power measured at those wavenumbers is the round-off
    level of the computation: the smallest residual the evaluation can represent.

    Args:
        forcing_power: Forcing power per wavenumber bin, ``|Lh(k)|^2``.
        residual_power: Achieved-residual power per wavenumber bin, ``|r_k|^2``.

    Returns:
        The largest residual power over the wavenumbers at which the forcing is
        exactly zero.

    Raises:
        ValueError: If the forcing vanishes at no wavenumber. The floor is then
            not measurable from this run, and it must not be replaced by a
            chosen constant -- the caller has to decide explicitly what to do.
    """
    forcing_power = np.asarray(forcing_power, dtype=float)
    residual_power = np.asarray(residual_power, dtype=float)
    zero_forcing_bins = forcing_power == 0.0
    if not bool(zero_forcing_bins.any()):
        raise ValueError(
            "the residual numerical floor cannot be measured: the forcing is "
            "non-zero at every wavenumber, so no bin isolates the round-off "
            "level of the residual. It must not be replaced by a chosen "
            "constant."
        )
    return float(residual_power[zero_forcing_bins].max())


def cancellation_ratio(
    forcing_power: np.ndarray, residual_power: np.ndarray
) -> np.ndarray:
    """The raw cancellation ratio, unbounded and untruncated.

    The value at a wavenumber of zero forcing is set to zero: the ratio is
    undefined there and those wavenumbers are excluded from the cutoff by
    :func:`informative_band` in any case.

    Args:
        forcing_power: Forcing power per wavenumber bin.
        residual_power: Achieved-residual power per wavenumber bin.

    Returns:
        The ratio ``|r_k|^2 / |Lh(k)|^2``, with zeros where the forcing vanishes.
    """
    forcing_power = np.asarray(forcing_power, dtype=float)
    residual_power = np.asarray(residual_power, dtype=float)
    nonzero_forcing = forcing_power > 0.0
    return np.where(
        nonzero_forcing,
        residual_power / np.where(nonzero_forcing, forcing_power, 1.0),
        0.0,
    )


def informative_band(
    forcing_power: np.ndarray, residual_power: np.ndarray
) -> np.ndarray:
    """The wavenumbers at which the cancellation ratio carries information.

    These are the wavenumbers whose forcing power stands strictly above the
    residual's numerical floor. The zeroth bin is always excluded: the datum has
    zero mean (``c_0 = 0``), so it carries no forcing.

    Args:
        forcing_power: Forcing power per wavenumber bin.
        residual_power: Achieved-residual power per wavenumber bin.

    Returns:
        A boolean mask over the wavenumber bins.
    """
    forcing_power = np.asarray(forcing_power, dtype=float)
    floor = residual_numerical_floor(forcing_power, residual_power)
    mask = forcing_power > floor
    mask[0] = False
    return mask


def measured_cancellation_cutoff(
    forcing_power: np.ndarray,
    residual_power: np.ndarray,
    wavenumber_bins: np.ndarray | None = None,
    running_mean_window: int = DEFAULT_RUNNING_MEAN_WINDOW,
    cancellation_threshold: float = DEFAULT_CANCELLATION_THRESHOLD,
) -> dict:
    """The measured cutoff, its running mean, and the band it was read on.

    The running mean at a wavenumber averages the cancellation ratio over the
    neighbours that are themselves informative -- never over a wavenumber outside
    the band, and never over a zero pad. The cutoff is the least informative
    wavenumber at which that mean reaches the threshold.

    Args:
        forcing_power: Forcing power per wavenumber bin.
        residual_power: Achieved-residual power per wavenumber bin.
        wavenumber_bins: The wavenumber of each bin; defaults to ``0, 1, 2, ...``.
        running_mean_window: Width of the centred running mean (odd).
        cancellation_threshold: The level defining the cutoff.

    Returns:
        A dictionary with ``cutoff`` (an ``int``, or ``None`` when the running
        mean never reaches the threshold on the informative band -- which means
        the network cancels more than half the forcing everywhere it is
        measurable), ``running_mean`` (``nan`` off the band), ``in_band_mask``,
        ``ratio`` and ``residual_floor``.

    Raises:
        ValueError: If ``running_mean_window`` is not a positive odd integer, or
            if the residual floor is not measurable.
    """
    if running_mean_window < 1 or running_mean_window % 2 == 0:
        raise ValueError(
            "running_mean_window must be a positive odd integer, received "
            f"{running_mean_window!r}"
        )
    forcing_power = np.asarray(forcing_power, dtype=float)
    residual_power = np.asarray(residual_power, dtype=float)
    if wavenumber_bins is None:
        wavenumber_bins = np.arange(forcing_power.size)
    wavenumber_bins = np.asarray(wavenumber_bins)

    floor = residual_numerical_floor(forcing_power, residual_power)
    ratio = cancellation_ratio(forcing_power, residual_power)
    in_band_mask = informative_band(forcing_power, residual_power)

    half_window = running_mean_window // 2
    running_mean = np.full(ratio.shape, np.nan)
    in_band_indices = np.flatnonzero(in_band_mask)
    for bin_index in in_band_indices:
        neighbours = [
            j
            for j in range(bin_index - half_window, bin_index + half_window + 1)
            if 0 <= j < ratio.size and in_band_mask[j]
        ]
        running_mean[bin_index] = float(ratio[neighbours].mean())

    cutoff = None
    for bin_index in in_band_indices:
        if running_mean[bin_index] >= cancellation_threshold:
            cutoff = int(wavenumber_bins[bin_index])
            break

    return {
        "cutoff": cutoff,
        "running_mean": running_mean,
        "in_band_mask": in_band_mask,
        "ratio": ratio,
        "residual_floor": floor,
    }
