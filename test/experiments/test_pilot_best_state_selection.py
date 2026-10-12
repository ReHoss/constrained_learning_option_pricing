r"""Unit tests of the fixed validation set of the barrier pilot's ``--best-state-selection validation``.

Covered helpers of ``pilot_down_and_out_put`` (on an untrained model, never on training output):

* ``draw_validation_set`` is a deterministic function of its seed and does not depend on any
  other generator;
* ``validation_loss`` (chunked) equals the weighted loss ``compute_loss`` on the whole set;
* ``validation_loss`` rejects a set size that is not a multiple of the chunk size.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments" / "python_scripts" / "exp_barrier_option"))

pilot = importlib.import_module("pilot_down_and_out_put")

from learning_option_pricing.pricing.corner_similarity_sampling import CornerSimilaritySampling  # noqa: E402

K, B, R, SIGMA, T, S_INF = 1.0, 0.6, 0.03, 0.3, 1.0, 3.0
SAMPLING = CornerSimilaritySampling(barrier=B, truncation_price=S_INF, maturity=T, volatility=SIGMA)


def _untrained_enrichment_model() -> torch.nn.Module:
    return pilot.build_model(K, B, T, 0.0, 0, black_scholes_payoff=True, r=R, sigma=SIGMA, s_inf=S_INF,
                             far_field_dirichlet=True, corner_treatment="enrichment",
                             subtraction_terminal_profile="black_scholes").to(pilot.DEVICE)


def test_validation_set_is_a_deterministic_function_of_its_seed():
    first = pilot.draw_validation_set(SAMPLING, 2 * pilot.VALIDATION_CHUNK_SIZE, seed=7)
    torch.manual_seed(123)  # the global generator must play no role
    second = pilot.draw_validation_set(SAMPLING, 2 * pilot.VALIDATION_CHUNK_SIZE, seed=7)
    other = pilot.draw_validation_set(SAMPLING, 2 * pilot.VALIDATION_CHUNK_SIZE, seed=8)
    for key in ("s", "t", "weights"):
        assert torch.equal(first[key], second[key])
    assert not torch.equal(first["s"], other["s"])


def test_chunked_validation_loss_equals_the_weighted_loss_on_the_whole_set():
    model = _untrained_enrichment_model()
    validation_set = pilot.draw_validation_set(SAMPLING, 2 * pilot.VALIDATION_CHUNK_SIZE, seed=0)
    chunked = pilot.validation_loss(model, validation_set, R, SIGMA)
    whole = float(pilot.compute_loss(model, validation_set["s"].clone().requires_grad_(True),
                                     validation_set["t"].clone().requires_grad_(True), R, SIGMA,
                                     validation_set["weights"]).detach())
    # float32 sums in a different order: equal to a few units in the last place.
    assert chunked == pytest.approx(whole, rel=1e-5)


def test_validation_loss_rejects_a_size_that_is_not_a_multiple_of_the_chunk_size():
    model = _untrained_enrichment_model()
    validation_set = pilot.draw_validation_set(SAMPLING, pilot.VALIDATION_CHUNK_SIZE + 1, seed=0)
    with pytest.raises(ValueError):
        pilot.validation_loss(model, validation_set, R, SIGMA)
