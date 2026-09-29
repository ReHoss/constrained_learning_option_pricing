r"""Gradient-norm safeguard that reports every activation.

A clip of the global gradient norm is a numerical safeguard that can change the
update, so it must never act silently (global rule on numerical clamps): its first
activation is logged at WARNING level, the later ones at DEBUG level, and the run
records how many updates it affected and the most extreme raw (pre-clip) value.

:class:`GradientNormSafeguardMonitor` wraps :func:`torch.nn.utils.clip_grad_norm_`
and keeps the pre-clip norm of **every** update, so that the record of a run covers
all its updates rather than the logged iterations only.  PyTorch multiplies the
gradients by :math:`\min\bigl(1, c/(\lVert g\rVert + 10^{-6})\bigr)`, with :math:`c` the
threshold; an update is counted as an activation exactly when this factor is below
one.  A non-finite norm is recorded separately: PyTorch then multiplies the
gradients by a non-finite or zero factor, which is reported at WARNING level on its
first occurrence.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable

# The additive constant of the scaling factor in torch.nn.utils.clip_grad_norm_,
# ``max_norm / (total_norm + 1e-6)`` (unchanged across the torch 2.x releases
# used by this project); reproduced so that the activation test below is the
# condition under which the gradients are actually rescaled.
TORCH_CLIP_DENOMINATOR_OFFSET = 1.0e-6


class GradientNormSafeguardMonitor:
    """Clip the global gradient norm at a threshold and record every activation.

    Args:
        maximum_norm: The threshold :math:`c > 0` passed to ``clip_grad_norm_``.
        logger: Logger receiving the WARNING / DEBUG activation reports.
        label: Prefix identifying the run in the log lines (cell and variant).

    Raises:
        ValueError: If ``maximum_norm`` is not a positive finite number.
    """

    def __init__(self, maximum_norm: float, *, logger: logging.Logger, label: str) -> None:
        if not (math.isfinite(maximum_norm) and maximum_norm > 0.0):
            raise ValueError(
                f"maximum_norm must be a positive finite number, received {maximum_norm!r}"
            )
        self.maximum_norm = float(maximum_norm)
        self.logger = logger
        self.label = label
        self.pre_clip_norm_per_update: list[float] = []
        self.activation_count = 0
        self.first_activation_update: int | None = None
        self.smallest_scaling_factor: float | None = None
        self.non_finite_count = 0
        self.first_non_finite_update: int | None = None
        self.largest_pre_clip_norm: float | None = None
        self.update_of_largest_pre_clip_norm: int | None = None

    def clip_and_record(self, parameters: Iterable, update_index: int) -> float:
        """Clip the gradients of ``parameters`` and record the pre-clip norm.

        Args:
            parameters: The model parameters (their ``.grad`` must be populated).
            update_index: The 1-based index of the current parameter update.

        Returns:
            The pre-clip global gradient norm (a Python float).
        """
        import torch

        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=self.maximum_norm)
        pre_clip_norm = float(total_norm)
        self.pre_clip_norm_per_update.append(pre_clip_norm)

        if not math.isfinite(pre_clip_norm):
            self.non_finite_count += 1
            if self.first_non_finite_update is None:
                self.first_non_finite_update = update_index
                self.logger.warning(
                    "[%s] non-finite gradient norm %r at update %d: clip_grad_norm_ "
                    "multiplies the gradients by a non-finite or zero factor",
                    self.label, pre_clip_norm, update_index,
                )
            else:
                self.logger.debug(
                    "[%s] non-finite gradient norm %r at update %d (occurrence %d)",
                    self.label, pre_clip_norm, update_index, self.non_finite_count,
                )
            return pre_clip_norm

        if self.largest_pre_clip_norm is None or pre_clip_norm > self.largest_pre_clip_norm:
            self.largest_pre_clip_norm = pre_clip_norm
            self.update_of_largest_pre_clip_norm = update_index

        scaling_factor = self.maximum_norm / (pre_clip_norm + TORCH_CLIP_DENOMINATOR_OFFSET)
        if scaling_factor < 1.0:
            self.activation_count += 1
            if self.smallest_scaling_factor is None or scaling_factor < self.smallest_scaling_factor:
                self.smallest_scaling_factor = scaling_factor
            if self.first_activation_update is None:
                self.first_activation_update = update_index
                self.logger.warning(
                    "[%s] gradient-norm safeguard bound at update %d: pre-clip norm "
                    "%.6e above the threshold %.3e, gradients multiplied by %.6e",
                    self.label, update_index, pre_clip_norm, self.maximum_norm,
                    scaling_factor,
                )
            else:
                self.logger.debug(
                    "[%s] gradient-norm safeguard bound at update %d: pre-clip norm "
                    "%.6e, factor %.6e (activation %d)",
                    self.label, update_index, pre_clip_norm, scaling_factor,
                    self.activation_count,
                )
        return pre_clip_norm

    def summary(self) -> dict:
        """Scalar record of the safeguard over the updates seen so far.

        ``None`` entries are explicitly empty (no activation, no finite norm, or
        no non-finite norm); they are never replaced by a sentinel value.
        """
        return {
            "gradient_norm_threshold": self.maximum_norm,
            "gradient_norm_updates_recorded": len(self.pre_clip_norm_per_update),
            "gradient_norm_largest_pre_clip": self.largest_pre_clip_norm,
            "gradient_norm_largest_pre_clip_update": self.update_of_largest_pre_clip_norm,
            "gradient_norm_safeguard_activation_count": self.activation_count,
            "gradient_norm_safeguard_first_activation_update": self.first_activation_update,
            "gradient_norm_safeguard_smallest_scaling_factor": self.smallest_scaling_factor,
            "gradient_norm_non_finite_count": self.non_finite_count,
            "gradient_norm_first_non_finite_update": self.first_non_finite_update,
        }

    def log_summary(self) -> None:
        """One INFO line stating whether the safeguard bound during the run."""
        record = self.summary()
        self.logger.info(
            "[%s] gradient-norm safeguard (threshold %.3e) over %d updates: %d "
            "activations (first at update %s), largest pre-clip norm %s at update %s, "
            "%d non-finite norms",
            self.label, self.maximum_norm, record["gradient_norm_updates_recorded"],
            self.activation_count, self.first_activation_update,
            "none" if self.largest_pre_clip_norm is None else f"{self.largest_pre_clip_norm:.6e}",
            self.update_of_largest_pre_clip_norm, self.non_finite_count,
        )
