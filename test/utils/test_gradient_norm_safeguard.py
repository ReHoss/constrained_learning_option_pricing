"""Tests for learning_option_pricing.utils.gradient_norm_safeguard."""
from __future__ import annotations

import logging
import math

import pytest
import torch

from learning_option_pricing.utils.gradient_norm_safeguard import (
    GradientNormSafeguardMonitor,
)

LOGGER_NAME = "test_gradient_norm_safeguard"


def _model_with_gradients(seed: int = 0) -> torch.nn.Module:
    """A small linear model whose gradients are populated by one backward pass."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 3)
    inputs = torch.randn(8, 4)
    loss = (model(inputs) ** 2).mean()
    loss.backward()
    return model


def _global_gradient_norm(model: torch.nn.Module) -> float:
    return float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())))


def test_large_threshold_never_binds_and_leaves_gradients_unchanged():
    model = _model_with_gradients()
    gradients_before = [p.grad.clone() for p in model.parameters()]
    expected_norm = _global_gradient_norm(model)
    monitor = GradientNormSafeguardMonitor(
        1.0e12, logger=logging.getLogger(LOGGER_NAME), label="unit"
    )
    returned_norm = monitor.clip_and_record(model.parameters(), update_index=1)
    assert returned_norm == pytest.approx(expected_norm, rel=1e-6)
    for before, parameter in zip(gradients_before, model.parameters()):
        assert torch.equal(before, parameter.grad)
    summary = monitor.summary()
    assert summary["gradient_norm_safeguard_activation_count"] == 0
    assert summary["gradient_norm_safeguard_first_activation_update"] is None
    assert summary["gradient_norm_safeguard_smallest_scaling_factor"] is None
    assert summary["gradient_norm_largest_pre_clip"] == pytest.approx(expected_norm, rel=1e-6)
    assert summary["gradient_norm_largest_pre_clip_update"] == 1
    assert summary["gradient_norm_updates_recorded"] == 1


def test_small_threshold_binds_rescales_and_warns_once(caplog):
    threshold = 1.0e-3
    monitor = GradientNormSafeguardMonitor(
        threshold, logger=logging.getLogger(LOGGER_NAME), label="unit"
    )
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    for update_index in (1, 2, 3):
        model = _model_with_gradients(seed=update_index)
        pre_clip_norm = monitor.clip_and_record(model.parameters(), update_index)
        assert pre_clip_norm > threshold
        assert _global_gradient_norm(model) <= threshold * (1.0 + 1e-5)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    debug_reports = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and "safeguard bound" in r.getMessage()
    ]
    assert len(warnings) == 1 and "update 1" in warnings[0].getMessage()
    assert len(debug_reports) == 2
    summary = monitor.summary()
    assert summary["gradient_norm_safeguard_activation_count"] == 3
    assert summary["gradient_norm_safeguard_first_activation_update"] == 1
    assert 0.0 < summary["gradient_norm_safeguard_smallest_scaling_factor"] < 1.0
    assert summary["gradient_norm_largest_pre_clip"] == max(monitor.pre_clip_norm_per_update)


def test_non_finite_norm_is_counted_and_reported(caplog):
    model = _model_with_gradients()
    next(model.parameters()).grad[0, 0] = float("inf")
    monitor = GradientNormSafeguardMonitor(
        1.0e12, logger=logging.getLogger(LOGGER_NAME), label="unit"
    )
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    returned_norm = monitor.clip_and_record(model.parameters(), update_index=7)
    assert not math.isfinite(returned_norm)
    summary = monitor.summary()
    assert summary["gradient_norm_non_finite_count"] == 1
    assert summary["gradient_norm_first_non_finite_update"] == 7
    assert summary["gradient_norm_largest_pre_clip"] is None
    assert summary["gradient_norm_safeguard_activation_count"] == 0
    assert any(
        r.levelno == logging.WARNING and "non-finite" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize("invalid_threshold", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_threshold_raises(invalid_threshold):
    with pytest.raises(ValueError):
        GradientNormSafeguardMonitor(
            invalid_threshold, logger=logging.getLogger(LOGGER_NAME), label="unit"
        )
