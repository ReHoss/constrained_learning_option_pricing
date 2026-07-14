"""Tests for the clamp-free measured cancellation cutoff."""

import numpy as np
import pytest

from learning_option_pricing.pde.cancellation_cutoff import (
    cancellation_ratio,
    informative_band,
    measured_cancellation_cutoff,
    residual_numerical_floor,
)


def test_residual_floor_is_read_from_the_zero_forcing_bins():
    forcing_power = np.array([0.0, 1.0, 1.0e-3, 0.0, 0.0])
    residual_power = np.array([9.0, 0.5, 0.5, 2.0e-19, 7.0e-19])
    # bin 0 and bins 3, 4 have zero forcing; the floor is the largest residual
    # among them, and bin 0's residual counts too.
    assert residual_numerical_floor(forcing_power, residual_power) == 9.0


def test_residual_floor_refuses_to_be_guessed_when_the_forcing_never_vanishes():
    forcing_power = np.array([1.0, 1.0, 1.0])
    residual_power = np.array([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="cannot be measured"):
        residual_numerical_floor(forcing_power, residual_power)


def test_the_ratio_is_never_truncated():
    """The estimator carries no ceiling: an astronomic ratio survives intact."""
    forcing_power = np.array([0.0, 1.0, 1.0e-200, 0.0])
    residual_power = np.array([0.0, 0.25, 1.0e-19, 1.0e-19])
    ratio = cancellation_ratio(forcing_power, residual_power)
    assert ratio[1] == pytest.approx(0.25)
    assert ratio[2] == pytest.approx(1.0e181)  # not clipped to 1.5, or anything
    assert ratio[0] == 0.0 and ratio[3] == 0.0  # undefined where forcing vanishes


def test_the_informative_band_excludes_the_forcing_below_the_residual_floor():
    forcing_power = np.array([0.0, 1.0, 1.0e-10, 1.0e-30, 0.0])
    residual_power = np.array([0.0, 0.1, 1.0e-11, 1.0e-19, 1.0e-19])
    # floor = 1e-19 (largest residual where the forcing is exactly zero)
    mask = informative_band(forcing_power, residual_power)
    assert not mask[0]  # c_0 = 0 always excluded
    assert mask[1]  # forcing 1.0 > 1e-19
    assert mask[2]  # forcing 1e-10 > 1e-19
    assert not mask[3]  # forcing 1e-30 < 1e-19: below the residual's own floor
    assert not mask[4]  # zero forcing


def test_cutoff_is_the_first_wavenumber_whose_running_mean_reaches_one_half():
    # Ratio 0 up to k = 4, then 1 from k = 5 on: with a 3-point centred mean the
    # first bin whose mean reaches 1/2 is k = 5 (mean of 0, 1, 1 = 2/3).
    forcing_power = np.array([0.0] + [1.0] * 8 + [0.0, 0.0])
    residual_power = np.array([0.0] + [0.0] * 4 + [1.0] * 4 + [1.0e-19, 1.0e-19])
    result = measured_cancellation_cutoff(
        forcing_power, residual_power, running_mean_window=3
    )
    assert result["cutoff"] == 5
    assert result["residual_floor"] == pytest.approx(1.0e-19)


def test_cutoff_is_absent_when_the_network_cancels_throughout():
    forcing_power = np.array([0.0] + [1.0] * 6 + [0.0])
    residual_power = np.array([0.0] + [0.01] * 6 + [1.0e-19])
    result = measured_cancellation_cutoff(forcing_power, residual_power)
    assert result["cutoff"] is None


def test_the_running_mean_never_reaches_outside_the_informative_band():
    """A wavenumber below the residual floor cannot leak into its neighbour.

    This is the defect the removed clamp was covering: a zero-padded convolution
    let an uninformative wavenumber, whose ratio is astronomic, contaminate the
    mean of its in-band neighbours.
    """
    # k = 1, 2, 3 informative with ratio 0; k = 4 has a vanishing forcing and a
    # ratio of 1e181; it must not raise the mean at k = 3.
    forcing_power = np.array([0.0, 1.0, 1.0, 1.0, 1.0e-200, 0.0])
    residual_power = np.array([0.0, 0.0, 0.0, 0.0, 1.0e-19, 1.0e-19])
    result = measured_cancellation_cutoff(
        forcing_power, residual_power, running_mean_window=3
    )
    assert not result["in_band_mask"][4]
    assert result["running_mean"][3] == pytest.approx(0.0)
    assert result["cutoff"] is None


def test_running_mean_window_must_be_odd():
    forcing_power = np.array([0.0, 1.0, 0.0])
    residual_power = np.array([0.0, 1.0, 1.0e-19])
    with pytest.raises(ValueError, match="odd"):
        measured_cancellation_cutoff(
            forcing_power, residual_power, running_mean_window=4
        )
