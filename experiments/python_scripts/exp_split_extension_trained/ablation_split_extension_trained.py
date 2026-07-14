r"""Stage-2 trained ablation: split terminal-data extensions on the circle.

Trains the hard-constrained trial solution
``u_hat = (1 - lambda(t)) Phi_theta + Psi`` on the backward evolution problem
``P u = d_t u + A u = 0`` on the strip ``(0, T) x [0, 2*pi)`` with
``u(., T) = g``, for the constant-coefficient generator
``A = nu d_xx + mu d_x + r_0`` of the selected cell, and compares the
terminal-data extensions ``Psi`` of the stage-2 catalogue
(``_split_extension_catalogue.py``).  The residual decomposes as
``P u_hat = R_theta + P Psi`` with the theta-independent forcing ``P Psi``
controlled by the extension; the specification is
``documents/methodology/stage2_trained_ablation_specification.md``.

Analytic-derivative bypass (specification Section 1.4 items 3 and 5).  For
the registry-built extensions the closed-form derivatives are supplied to
``TerminalAnsatz`` and the forcing ``P Psi`` is assembled analytically and
outside the autograd graph; on the first interior batch the analytic and the
autograd assemblies are cross-checked (relative L2 deviation logged, abort
above 1e-3).

Run paths
---------
* single variant (array task):
    ``--cell g2_bernoulli_bandlimited --variant split_diffusion --ablation-dir DIR``
* all variants of one cell (local convenience):
    ``--cell g2_bernoulli_bandlimited``
* regenerate comparison plots without training:
    ``--replot DIR``
* create directory + metadata + per-variant configs and exit (login node):
    ``--cell g2_bernoulli_bandlimited --init-only``
* array-worker contract:
    ``--config-dir DIR/configs --config-name <variant>``

Smoke runs MUST pass ``--debug`` (and are prefixed ``_debug_`` on disk); any
run below ``SMOKE_TEST_NUM_ITERATIONS_THRESHOLD`` iterations without
``--debug`` is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make the sibling catalogue importable whether run as a module or a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _split_extension_catalogue as catalogue  # noqa: E402

logger = logging.getLogger("ablation_split_extension_trained")

SMOKE_TEST_NUM_ITERATIONS_THRESHOLD = 1000

TWO_PI = 2.0 * math.pi

# Evaluation-grid policy (specification Section 3.2): 1024 uniform spatial
# points on [0, 2*pi) times the eleven time slices {0, 0.1 T, ..., T};
# the reference side is evaluated in float64.
EVALUATION_GRID_SIZE = 1024
EVALUATION_TIME_SLICE_COUNT = 11
# Corner window (specification Section 3.3 item 3): circle distance to the
# corner point at most pi / 16.
CORNER_WINDOW_HALF_WIDTH = math.pi / 16.0
# Residual-spectra policy (specification Section 3.4).
SPECTRA_GRID_SIZE = 1024
SPECTRA_TIME_SLICE_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
SPECTRA_RUNNING_MEAN_WINDOW = 7
SPECTRA_IN_BAND_RELATIVE_THRESHOLD = 1.0e-5
SPECTRA_CANCELLATION_THRESHOLD = 0.5
# Upper truncation of the cancellation ratio before the running mean.  This is
# NOT an inert guard: at a wavenumber whose forcing power is of the order of the
# rounding error of the transform the denominator is denormal and the raw ratio
# is astronomically large (values above 1e300 are observed).  The truncation
# bounds the contribution such a wavenumber makes to the seven-point mean of its
# in-band neighbours.  It can change the measured cutoff, so it is reported
# whenever it binds (see the warning below) and it is stated in the report's
# definition of the estimator rather than applied silently.
SPECTRA_CANCELLATION_RATIO_CEILING = 1.5
# Build-time agreement bound between the matched graded extension field and
# its split {d_xx} twin (specification decision D3).
GRADED_MATCHED_AGREEMENT_TOLERANCE = 1.0e-6


# ===========================================================================
# Seeding helpers (master seed -> deterministic role-tagged per-role seeds)
# ===========================================================================

def derive_seed(master_seed: int, role: str) -> int:
    """Deterministically derive a per-role seed from the master seed.

    Identical construction to ``ablation_ansatz_forms.py::derive_seed`` (the
    reference infrastructure pinned by the specification, Section 4): a
    stable blake2b hash of ``"<master_seed>:<role>"``, independent of
    ``PYTHONHASHSEED`` and reproducible across machines.  The role tag is the
    *only* decorrelation key: two variants sharing the master seed receive
    identical model initialisation and sampler trajectories (shared-seed
    policy).
    """
    digest = hashlib.blake2b(
        f"{master_seed}:{role}".encode(), digest_size=8
    ).hexdigest()
    return int(digest, 16) % (2**31 - 1)


# ===========================================================================
# Transient-CUDA retry (gpu_p13 array tasks are occasionally scheduled onto a busy GPU)
# ===========================================================================

def cuda_retry(fn, *, attempts: int = 6, base_delay: float = 10.0):
    """Call ``fn()``, retrying on transient CUDA busy/unavailable errors.

    On shared multi-GPU nodes an array task can touch a GPU that is
    momentarily busy, raising a transient CUDA error at the first device
    access; the condition is waited out with a linear backoff.  Non-CUDA
    errors propagate immediately.
    """
    import time as _time

    import torch

    last_exc = None
    for attempt_index in range(attempts):
        try:
            return fn()
        except RuntimeError as exc:
            message = str(exc).lower()
            transient = "cuda" in message and (
                "busy" in message
                or "unavailable" in message
                or "initialization" in message
            )
            if not transient:
                raise
            last_exc = exc
            delay = base_delay * (attempt_index + 1)
            logger.warning(
                "transient CUDA error (attempt %d/%d): %s; retrying in %.0fs",
                attempt_index + 1, attempts, exc, delay,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _time.sleep(delay)
    raise RuntimeError(
        f"CUDA still unavailable after {attempts} attempts"
    ) from last_exc


# ===========================================================================
# Problem assembly (generator, datum, exact reference)
# ===========================================================================

def build_problem(cell_name: str) -> dict:
    """Return the torch/numpy problem description of one cell.

    Keys: ``cell_name``, ``generator_coefficients`` (order -> coefficient),
    ``terminal_time``, ``datum_kind``, ``band_edge`` (the retained band
    :math:`K_g`), ``cosine_coefficients`` / ``sine_coefficients`` (float64
    arrays over the retained band), ``terminal_datum`` (torch/numpy
    callable ``g(x)``), ``exact_field`` (the exact solution as a
    :class:`PeriodicExtensionField`), ``matched_exponential_rate`` (control
    cell only, else ``None``), ``corner_point``, ``label``.
    """
    import numpy as np

    from learning_option_pricing.pde import (
        bandlimited_bernoulli_cosine_coefficients,
        exact_solution_field,
        make_single_component_sine_cell,
    )

    cell_conf = catalogue.cell_by_name(cell_name)
    generator_coefficients = {
        int(order): float(value)
        for order, value in cell_conf["generator_coefficients"].items()
    }
    terminal_time = float(cell_conf["terminal_time"])

    if cell_conf["datum"] == "bernoulli_bandlimited":
        band_edge = int(cell_conf["truncation_wavenumber"])
        cosine_coefficients = bandlimited_bernoulli_cosine_coefficients(band_edge)
        sine_coefficients = None
        exact_field = exact_solution_field(
            generator_coefficients,
            cosine_coefficients,
            terminal_time=terminal_time,
        )
        matched_exponential_rate = None
        sine_wavenumber = None
        sine_amplitude = None
    elif cell_conf["datum"] == "sine_single_component":
        sine_wavenumber = int(cell_conf["sine_wavenumber"])
        sine_amplitude = float(cell_conf["sine_amplitude"])
        sine_cell = make_single_component_sine_cell(
            diffusivity=generator_coefficients[2],
            wavenumber=sine_wavenumber,
            amplitude=sine_amplitude,
            terminal_time=terminal_time,
        )
        band_edge = sine_wavenumber
        cosine_coefficients = np.zeros(band_edge, dtype=np.float64)
        sine_coefficients = np.zeros(band_edge, dtype=np.float64)
        sine_coefficients[band_edge - 1] = sine_amplitude
        exact_field = sine_cell.exact_solution
        matched_exponential_rate = sine_cell.matched_exponential_rate
    else:  # pragma: no cover - guarded by the catalogue schema
        raise ValueError(f"Unknown datum kind {cell_conf['datum']!r}")

    return {
        "cell_name": cell_name,
        "generator_coefficients": generator_coefficients,
        "terminal_time": terminal_time,
        "datum_kind": cell_conf["datum"],
        "band_edge": band_edge,
        "cosine_coefficients": cosine_coefficients,
        "sine_coefficients": sine_coefficients,
        "sine_wavenumber": sine_wavenumber,
        "sine_amplitude": sine_amplitude,
        "terminal_datum": exact_field.terminal_datum_values,
        "exact_field": exact_field,
        "matched_exponential_rate": matched_exponential_rate,
        "corner_point": float(cell_conf["corner_point"]),
        "label": cell_conf["label"],
    }


class SingleSineWavenumberDatum:
    r"""Exact Fourier coefficients of :math:`g(x) = A \sin(k_0 x)`.

    :math:`\sin(k_0 x) = (e^{i k_0 x} - e^{-i k_0 x}) / (2i)`, so
    :math:`c_{k_0} = -iA/2`, :math:`c_{-k_0} = +iA/2`, and every other
    coefficient vanishes.  Exposes the ``fourier_coefficients`` protocol of
    ``learning_option_pricing.pde.terminal_data_extensions``.
    """

    def __init__(self, wavenumber: int, amplitude: float) -> None:
        self.wavenumber = int(wavenumber)
        self.amplitude = float(amplitude)

    def fourier_coefficients(self, wavenumbers):
        import numpy as np

        wavenumber_array = np.asarray(wavenumbers)
        coefficient_values = np.zeros(wavenumber_array.shape, dtype=np.complex128)
        coefficient_values[wavenumber_array == self.wavenumber] = (
            -0.5j * self.amplitude
        )
        coefficient_values[wavenumber_array == -self.wavenumber] = (
            +0.5j * self.amplitude
        )
        return coefficient_values


def build_closed_form_extension(variant: dict, problem: dict):
    """Closed-form spectral counterpart of a variant's extension.

    Returns the :class:`TerminalDataExtension` instance whose per-wavenumber
    forcing coefficient and closed-form time integrals supply (i) the
    forcing-floor consistency value (specification Section 3.3 item 5) and
    (ii) the exact forcing side of the residual-spectra measurement
    (Section 3.4).  The ``matched_exponential_factor`` variant maps to
    :class:`ExactSolutionExtension`: its extension
    :math:`\\Psi = e^{-\\nu k_0^2 (T-t)} \\sin(k_0 x)` has the exact-solution
    coefficient :math:`e^{(T-t) a(k)} c_k` on the retained band, so the
    forcing coefficient vanishes identically.
    """
    from learning_option_pricing.pde import (
        ConstantCoefficientGenerator,
        ConstantInTimeExtension,
        ConvexRawExtension,
        ExactSolutionExtension,
        GradedGaussianExtension,
        PeriodisedBernoulliDatum,
        SplitSemigroupExtension,
    )

    generator = ConstantCoefficientGenerator(
        coefficients=problem["generator_coefficients"],
        name=problem["cell_name"],
    )
    if problem["datum_kind"] == "bernoulli_bandlimited":
        # On the retained band |k| <= K_g the truncated datum's coefficients
        # coincide exactly with the regularity-index-1 periodised Bernoulli
        # datum: c_k = 1 / (2 pi^2 k^2).
        datum = PeriodisedBernoulliDatum(1)
    else:
        datum = SingleSineWavenumberDatum(
            problem["sine_wavenumber"], problem["sine_amplitude"]
        )
    terminal_time = problem["terminal_time"]

    variant_name = variant["name"]
    if variant_name == "convex_raw":
        return ConvexRawExtension(datum, generator, terminal_time)
    if variant_name == "constant_in_time":
        return ConstantInTimeExtension(datum, generator, terminal_time)
    if variant_name == "split_diffusion":
        return SplitSemigroupExtension(datum, generator, (2,), terminal_time)
    if variant_name == "split_diffusion_advection":
        return SplitSemigroupExtension(datum, generator, (1, 2), terminal_time)
    if variant_name in ("graded_gaussian_matched", "graded_gaussian_mismatched"):
        comparison_diffusivity = (
            float(variant["comparison_diffusivity_ratio"])
            * problem["generator_coefficients"][2]
        )
        return GradedGaussianExtension(
            datum, generator, comparison_diffusivity, terminal_time
        )
    if variant_name in ("exact_solution", "matched_exponential_factor"):
        return ExactSolutionExtension(datum, generator, terminal_time)
    raise KeyError(
        f"No closed-form extension counterpart for variant {variant_name!r}"
    )


def closed_form_forcing_floor(variant: dict, problem: dict) -> float:
    r"""Closed-form Monte-Carlo forcing floor
    :math:`\mathbb{E}[(P\Psi)^2] = \|Lh\|^2_{\mathrm{strip}} / (2\pi T)
    = \tfrac{1}{T} \sum_{0<|k|\le K_g} I_k` at the cell's band edge.
    """
    from learning_option_pricing.pde import (
        symmetric_wavenumber_band,
        total_strip_forcing_squared,
    )

    extension = build_closed_form_extension(variant, problem)
    band = symmetric_wavenumber_band(problem["band_edge"])
    squared_strip_norm = total_strip_forcing_squared(extension, band)
    return squared_strip_norm / (TWO_PI * problem["terminal_time"])


# ===========================================================================
# Ansatz construction
# ===========================================================================

def _assert_graded_matched_agrees_with_split(graded_field, problem) -> float:
    """Build-time consistency assertion of specification decision D3.

    The matched graded extension (``nu_c = nu``) is mathematically identical
    to the split ``{d_xx}`` extension; the two code paths are compared on a
    probe grid over the strip, across the field and its three analytic
    derivatives.  A relative deviation above
    :data:`GRADED_MATCHED_AGREEMENT_TOLERANCE` raises (never a silent pass);
    the measured deviation is logged and returned.
    """
    import numpy as np

    from learning_option_pricing.pde import build_split_diffusion_extension_field

    split_field = build_split_diffusion_extension_field(
        problem["generator_coefficients"],
        problem["cosine_coefficients"],
        sine_coefficients=problem["sine_coefficients"],
        terminal_time=problem["terminal_time"],
    )
    probe_x = np.linspace(0.0, TWO_PI, 257, endpoint=False)[None, :]
    probe_t = np.linspace(0.0, problem["terminal_time"], 9)[:, None]
    worst_relative_deviation = 0.0
    for callable_name in (
        "field", "time_derivative", "space_derivative", "second_space_derivative"
    ):
        graded_values = getattr(graded_field, callable_name)(probe_x, probe_t)
        split_values = getattr(split_field, callable_name)(probe_x, probe_t)
        scale = float(np.max(np.abs(split_values)))
        deviation = float(np.max(np.abs(graded_values - split_values)))
        relative_deviation = deviation / scale if scale > 0.0 else deviation
        worst_relative_deviation = max(worst_relative_deviation, relative_deviation)
    if worst_relative_deviation > GRADED_MATCHED_AGREEMENT_TOLERANCE:
        raise RuntimeError(
            "matched graded extension disagrees with its split {d_xx} twin: "
            f"relative deviation {worst_relative_deviation:.3e} exceeds "
            f"{GRADED_MATCHED_AGREEMENT_TOLERANCE:.0e} (specification D3)."
        )
    logger.info(
        "D3 build-time check: matched graded vs split {d_xx} extension "
        "fields agree to %.3e relative (tolerance %.0e)",
        worst_relative_deviation, GRADED_MATCHED_AGREEMENT_TOLERANCE,
    )
    return worst_relative_deviation


def build_ansatz(variant: dict, problem: dict, hparams: dict, *, model_seed: int):
    """Construct the :class:`TerminalAnsatz` and its extension field.

    Returns ``(model, extension_field)`` where ``extension_field`` is the
    :class:`PeriodicExtensionField` resolved from the variant's string
    registry key (``None`` for the datum-path variants).  The analytic
    derivative callables of the field are supplied to the ansatz
    (specification Section 1.4 item 3); the datum-path variants remain on
    the autograd route.
    """
    import torch

    from learning_option_pricing.models.resnet import ResNet
    from learning_option_pricing.models.terminal_ansatz import (
        TerminalAnsatz,
        make_interpolation_coefficient,
    )
    from learning_option_pricing.pde import EXTENSION_FIELD_REGISTRY

    torch.manual_seed(model_seed)
    network = ResNet(
        d_in=3,  # periodic feature map (cos x, sin x, 2 t / T - 1); decision D5
        d_out=1,
        n=int(hparams["net_width"]),
        M=int(hparams["net_blocks"]),
        L=int(hparams["net_layers_per_block"]),
    )

    terminal_time = problem["terminal_time"]
    if variant["interpolation"] == "linear":
        interp_coeff = make_interpolation_coefficient("linear", T=terminal_time)
    elif variant["interpolation"] == "exponential":
        exponential_rate_gamma = variant["exponential_rate_gamma"]
        if exponential_rate_gamma is None:
            raise ValueError(
                "exponential interpolation requires an explicit "
                "exponential_rate_gamma (specification D10: the library "
                "default belongs to the unit-interval family and would "
                "silently mismatch the matched rate on the circle)."
            )
        matched_rate = problem["matched_exponential_rate"]
        if matched_rate is not None and float(exponential_rate_gamma) != float(
            matched_rate
        ):
            raise ValueError(
                "catalogue exponential_rate_gamma "
                f"{exponential_rate_gamma!r} disagrees with the cell's "
                f"matched rate nu k_0^2 = {matched_rate!r} (specification "
                "D10)."
            )
        interp_coeff = make_interpolation_coefficient(
            "exponential", T=terminal_time, gamma=float(exponential_rate_gamma)
        )
    else:  # pragma: no cover - guarded by the catalogue schema
        raise ValueError(f"Unknown interpolation {variant['interpolation']!r}")

    def periodic_feature_map(xt: torch.Tensor) -> torch.Tensor:
        """Periodic embedding (x, t) -> (cos x, sin x, 2 t / T - 1)."""
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        return torch.cat(
            [torch.cos(x), torch.sin(x), 2.0 * t / terminal_time - 1.0], dim=1
        )

    extension_field = None
    extension_fn = None
    extension_derivative_fns = None
    if variant["extension"] is not None:
        builder = EXTENSION_FIELD_REGISTRY[variant["extension"]]
        comparison_diffusivity = None
        if variant["comparison_diffusivity_ratio"] is not None:
            comparison_diffusivity = (
                float(variant["comparison_diffusivity_ratio"])
                * problem["generator_coefficients"][2]
            )
        extension_field = builder(
            problem["generator_coefficients"],
            problem["cosine_coefficients"],
            sine_coefficients=problem["sine_coefficients"],
            comparison_diffusivity=comparison_diffusivity,
            terminal_time=terminal_time,
        )
        if variant["name"] == "graded_gaussian_matched":
            _assert_graded_matched_agrees_with_split(extension_field, problem)
        extension_fn = extension_field.field
        extension_derivative_fns = extension_field.derivative_callables()

    model = TerminalAnsatz(
        network,
        problem["terminal_datum"],
        interp_coeff,
        form=variant["form"],
        normalizer=periodic_feature_map,
        extension_fn=extension_fn,
        extension_derivative_fns=extension_derivative_fns,
    )
    return model, extension_field


# ===========================================================================
# Sampling
# ===========================================================================

def make_samplers(problem: dict, hparams: dict, *, sampler_seed: int, device):
    """Return interior / terminal collocation samplers (no boundary — D4)."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampler_seed)
    terminal_time = problem["terminal_time"]
    n_interior = int(hparams["n_interior"])
    n_terminal = int(hparams["n_terminal"])

    def _uniform(n):
        return torch.rand(n, generator=generator)

    def sample_interior():
        x = (TWO_PI * _uniform(n_interior)).to(device).requires_grad_(True)
        t = (terminal_time * _uniform(n_interior)).to(device).requires_grad_(True)
        return x, t

    def sample_terminal():
        x = (TWO_PI * _uniform(n_terminal)).to(device)
        t = torch.full((n_terminal,), terminal_time, device=device)
        return x, t

    return sample_interior, sample_terminal


# ===========================================================================
# Training
# ===========================================================================

def train_variant(
    variant, problem, hparams, *, num_iterations, seed, device, log_every
):
    """Train one variant; return ``(model, history, cross_check_deviation)``.

    The startup cross-check (specification Section 1.4 item 5) runs on the
    first interior batch for every variant built with the analytic-derivative
    bypass; the same batch is then consumed as the iteration-1 training
    batch, so the sampler trajectory is identical across all variants of a
    cell (shared-seed policy).  A deviation above 1e-3 aborts the run
    (:class:`RuntimeError` raised by the library guard).
    """
    import torch

    from learning_option_pricing.models.terminal_ansatz import (
        cross_check_extension_forcing_analytic_versus_autograd,
        residual_decomposition,
    )

    generator_coefficients = problem["generator_coefficients"]
    model_seed = derive_seed(seed, "model_init")
    sampler_seed = derive_seed(seed, "sampler")
    model_and_field = cuda_retry(
        lambda: build_ansatz(variant, problem, hparams, model_seed=model_seed)
    )
    model, _ = model_and_field
    model = cuda_retry(lambda: model.to(device))
    sample_interior, sample_terminal = make_samplers(
        problem, hparams, sampler_seed=sampler_seed, device=device
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(hparams["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations
    )

    history = {key: [] for key in (
        "iter", "loss", "loss_pde", "loss_tc",
        "network_energy", "cross_term", "forcing_floor",
        "forcing_velocity", "forcing_diffusion", "forcing_advection",
        "forcing_reaction", "grad_norm", "lr",
    )}
    best_loss = float("inf")
    best_state = None
    best_iter = -1
    cross_check_deviation = None

    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(
        "[%s/%s] training: form=%s interpolation=%s extension=%s params=%d "
        "seeds(model=%d, sampler=%d)",
        problem["cell_name"], variant["name"], variant["form"],
        variant["interpolation"], variant["extension"], n_parameters,
        model_seed, sampler_seed,
    )

    # Periodic checkpointing is deliberately omitted: a single run trains
    # 20000 iterations in about 400 s (well within the array task's 1 h wall
    # limit, on the non-preemptible gpu_p13 QoS), the best-objective state is
    # tracked and restored at the end, and a task that fails is re-run in full
    # by the array. Full state persistence (model, optimiser, scheduler, RNG)
    # would cost more than the run it protects; this matches the reference
    # ansatz-forms runner.
    for it in range(1, num_iterations + 1):
        optimizer.zero_grad()

        x_interior, t_interior = sample_interior()

        if it == 1:
            if model._extension_derivative_fns is not None:
                cross_check_deviation = (
                    cross_check_extension_forcing_analytic_versus_autograd(
                        model, x_interior, t_interior,
                        generator_coefficients=generator_coefficients,
                    )
                )
                logger.info(
                    "[%s] startup cross-check passed: analytic-vs-autograd "
                    "P Psi relative L2 deviation = %.6e (tolerance 1e-3)",
                    variant["name"], cross_check_deviation,
                )
            else:
                logger.info(
                    "[%s] no analytic-derivative bypass (datum-path "
                    "extension); forcing assembled by autograd — startup "
                    "cross-check not applicable",
                    variant["name"],
                )

        decomposition = residual_decomposition(
            model, x_interior, t_interior,
            generator_coefficients=generator_coefficients,
        )
        loss = decomposition["loss"]
        loss.backward()
        # Gradient-norm probe (max_norm = 1e12: a probe, not a clip).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1e12
        )
        optimizer.step()
        scheduler.step()

        loss_value = loss.item()
        if loss_value < best_loss and torch.isfinite(loss).item():
            best_loss = loss_value
            best_iter = it
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        # Dense logging over the first 100 iterations (the transient where
        # the network learns to cancel the forcing), then every log_every.
        if it <= 100 or it % log_every == 0 or it == num_iterations:
            with torch.no_grad():
                x_terminal, t_terminal = sample_terminal()
                u_terminal = model(
                    torch.stack([x_terminal, t_terminal], dim=1)
                ).squeeze(-1)
                terminal_mismatch = (
                    (u_terminal - problem["terminal_datum"](x_terminal)) ** 2
                ).mean().item()
            history["iter"].append(it)
            history["loss"].append(loss_value)
            history["loss_pde"].append(loss_value)  # hard forms: loss == L_pde
            history["loss_tc"].append(terminal_mismatch)
            history["network_energy"].append(
                decomposition["network_energy"].item()
            )
            history["cross_term"].append(decomposition["cross_term"].item())
            history["forcing_floor"].append(
                decomposition["forcing_floor"].item()
            )
            history["forcing_velocity"].append(
                decomposition["forcing_velocity"].item()
            )
            history["forcing_diffusion"].append(
                decomposition["forcing_diffusion"].item()
            )
            history["forcing_advection"].append(
                decomposition["forcing_advection"].item()
            )
            history["forcing_reaction"].append(
                decomposition["forcing_reaction"].item()
            )
            history["grad_norm"].append(float(grad_norm))
            history["lr"].append(scheduler.get_last_lr()[0])
            logger.info(
                "[%s] it=%d loss=%.3e (tc_diag=%.3e | netE=%.3e cross=%.3e "
                "floor=%.3e [vel=%.3e diff=%.3e adv=%.3e reac=%.3e])",
                variant["name"], it, loss_value, terminal_mismatch,
                decomposition["network_energy"].item(),
                decomposition["cross_term"].item(),
                decomposition["forcing_floor"].item(),
                decomposition["forcing_velocity"].item(),
                decomposition["forcing_diffusion"].item(),
                decomposition["forcing_advection"].item(),
                decomposition["forcing_reaction"].item(),
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("[%s] restored best state from iter %d (loss=%.3e)",
                    variant["name"], best_iter, best_loss)

    history["best_iter"] = best_iter
    history["best_loss"] = best_loss
    history["n_parameters"] = n_parameters
    return model, history, cross_check_deviation


def evaluate_best_state_channels(model, problem, hparams, *, evaluation_seed, device):
    """Residual-decomposition channels at the restored best state.

    Measured on a dedicated interior batch drawn from the evaluation-role
    seed (never the training sampler), so the value is comparable across
    variants of a cell.
    """
    import torch

    from learning_option_pricing.models.terminal_ansatz import (
        residual_decomposition,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(evaluation_seed)
    n_interior = int(hparams["n_interior"])
    terminal_time = problem["terminal_time"]
    x = (TWO_PI * torch.rand(n_interior, generator=generator)).to(
        device
    ).requires_grad_(True)
    t = (terminal_time * torch.rand(n_interior, generator=generator)).to(
        device
    ).requires_grad_(True)
    decomposition = residual_decomposition(
        model, x, t, generator_coefficients=problem["generator_coefficients"]
    )
    return {
        "loss_best_state_eval": decomposition["loss"].item(),
        "network_energy_best_state": decomposition["network_energy"].item(),
        "forcing_floor_best_state": decomposition["forcing_floor"].item(),
    }


# ===========================================================================
# Metrics (vs the exact finite component sum, specification Section 3)
# ===========================================================================

def _evaluation_grid():
    import numpy as np

    return np.linspace(0.0, TWO_PI, EVALUATION_GRID_SIZE, endpoint=False)


def _circle_corner_mask(x_grid, corner_point):
    """Boolean mask of the corner window dist(x, x_star) <= pi/16 (circle metric)."""
    import numpy as np

    raw_distance = np.abs(
        np.mod(x_grid - corner_point + math.pi, TWO_PI) - math.pi
    )
    return raw_distance <= CORNER_WINDOW_HALF_WIDTH


def compute_error_metrics(model, problem) -> dict:
    """Relative L2 errors against the exact solution on the evaluation grid.

    The reference side is the exact finite component sum evaluated in
    ``float64`` (specification Section 3.2); the terminal identity
    ``u_hat(., T) == g`` is asserted exactly in floating point for the hard
    forms (Section 3.3 item 4) — a violation raises.
    """
    import numpy as np
    import torch

    device = next(model.parameters()).device
    terminal_time = problem["terminal_time"]
    exact_field = problem["exact_field"]
    x64 = _evaluation_grid()
    x32 = torch.as_tensor(x64, dtype=torch.float32, device=device)
    time_slices = np.linspace(0.0, terminal_time, EVALUATION_TIME_SLICE_COUNT)

    predicted = np.empty((len(time_slices), len(x64)), dtype=np.float64)
    reference = np.empty_like(predicted)
    for slice_index, time_value in enumerate(time_slices):
        t32 = torch.full_like(x32, float(time_value))
        with torch.no_grad():
            u_predicted = model(torch.stack([x32, t32], dim=1)).squeeze(-1)
        predicted[slice_index] = u_predicted.detach().cpu().numpy().astype(
            np.float64
        )
        reference[slice_index] = exact_field.field(
            x64, np.full_like(x64, float(time_value))
        )

    error = predicted - reference
    rel_l2 = float(np.linalg.norm(error) / np.linalg.norm(reference))
    rel_l2_t0 = float(
        np.linalg.norm(error[0]) / np.linalg.norm(reference[0])
    )

    corner_mask = _circle_corner_mask(x64, problem["corner_point"])
    corner_rel_l2_per_slice = [
        float(
            np.linalg.norm(error[j, corner_mask])
            / np.linalg.norm(reference[j, corner_mask])
        )
        for j in range(len(time_slices))
    ]
    rel_l2_corner_t0 = corner_rel_l2_per_slice[0]
    rel_l2_corner_max = float(max(corner_rel_l2_per_slice))

    # Terminal-condition check: exactly zero for every hard form by
    # construction (asserted — never silently accepted).
    tT = torch.full_like(x32, terminal_time)
    with torch.no_grad():
        u_terminal = model(torch.stack([x32, tT], dim=1)).squeeze(-1)
        datum_values = problem["terminal_datum"](x32)
    if not torch.equal(u_terminal, datum_values):
        terminal_mismatch = float(
            (u_terminal - datum_values).norm()
            / (datum_values.norm() + 1e-30)
        )
        raise RuntimeError(
            "terminal identity u_hat(., T) == g violated for a hard form: "
            f"relative L2 mismatch {terminal_mismatch:.3e} (expected exact "
            "floating-point equality; specification Section 3.3 item 4)."
        )
    tc_l2 = 0.0  # asserted exact equality above

    return {
        "rel_l2": rel_l2,
        "rel_l2_t0": rel_l2_t0,
        "rel_l2_corner_t0": rel_l2_corner_t0,
        "rel_l2_corner_max": rel_l2_corner_max,
        "tc_l2": tc_l2,
    }


def _generator_applied_to_datum(problem, x64):
    r"""Closed-form :math:`(A g)(x) = \nu g'' + \mu g' + r_0 g` on the grid.

    Assembled from the analytic spatial derivatives of the exact-solution
    field at :math:`t = T` (where every extension field coincides with the
    datum), in ``float64``.
    """
    import numpy as np

    exact_field = problem["exact_field"]
    coefficients = problem["generator_coefficients"]
    tT = np.full_like(x64, problem["terminal_time"])
    return (
        coefficients[2] * exact_field.second_space_derivative(x64, tT)
        + coefficients.get(1, 0.0) * exact_field.space_derivative(x64, tT)
        + coefficients.get(0, 0.0) * exact_field.field(x64, tT)
    )


def compute_terminal_target(model, problem, variant, extension_field) -> dict:
    r"""Terminal-target check (specification Section 3.5).

    The exact minimiser's terminal profile is
    :math:`\Phi^\star(\cdot, T) = -(P\Psi)(\cdot, T) / d_T'(T)`; for the
    linear factor this is :math:`T\,(P\Psi)(\cdot, T)`.  Per variant:

    * registry extensions: :math:`T` times
      ``PeriodicExtensionField.terminal_forcing_profile``;
    * ``constant_in_time``: :math:`T\,(Ag)`;
    * ``convex_raw`` (hard_convex, linear): :math:`g + T\,(Ag)`;
    * zero-target variants (``exact_solution``,
      ``matched_exponential_factor``): :math:`\Phi^\star(\cdot, T) = 0`,
      and the absolute :math:`L^2` norm is reported instead of a relative
      distance.

    Norms use the continuous convention
    :math:`\|f\|_{L^2(0,2\pi)} = (2\pi\,\mathrm{mean}(f^2))^{1/2}`.
    """
    import numpy as np
    import torch

    device = next(model.parameters()).device
    terminal_time = problem["terminal_time"]
    x64 = _evaluation_grid()
    x32 = torch.as_tensor(x64, dtype=torch.float32, device=device)
    tT = torch.full_like(x32, terminal_time)
    with torch.no_grad():
        phi_terminal = model.free_network(
            torch.stack([x32, tT], dim=1)
        ).squeeze(-1)
    phi_terminal = phi_terminal.detach().cpu().numpy().astype(np.float64)

    is_zero_target = variant["name"] in (
        "exact_solution", "matched_exponential_factor"
    )
    if is_zero_target:
        phi_star = np.zeros_like(x64)
        target_distance = float(
            math.sqrt(TWO_PI * float(np.mean(phi_terminal**2)))
        )
    else:
        if variant["extension"] is not None:
            phi_star = terminal_time * extension_field.terminal_forcing_profile(
                x64
            )
        elif variant["form"] == "hard_constant":
            phi_star = terminal_time * _generator_applied_to_datum(problem, x64)
        elif (
            variant["form"] == "hard_convex"
            and variant["interpolation"] == "linear"
        ):
            datum_values = problem["exact_field"].terminal_datum_values(x64)
            phi_star = datum_values + terminal_time * _generator_applied_to_datum(
                problem, x64
            )
        else:  # pragma: no cover - no stage-2 variant reaches this branch
            raise ValueError(
                f"No closed-form terminal target for variant {variant['name']!r}"
            )
        target_distance = float(
            np.linalg.norm(phi_terminal - phi_star) / np.linalg.norm(phi_star)
        )

    return {
        "phi_theta_terminal": phi_terminal,
        "phi_star_terminal": np.asarray(phi_star, dtype=np.float64),
        "terminal_target_distance": target_distance,
        "terminal_target_is_zero_target": is_zero_target,
    }


def compute_slices(model, problem, terminal_target: dict) -> dict:
    """Tabulate solution slices for torch-free replotting."""
    import numpy as np
    import torch

    device = next(model.parameters()).device
    terminal_time = problem["terminal_time"]
    exact_field = problem["exact_field"]
    x64 = _evaluation_grid()
    x32 = torch.as_tensor(x64, dtype=torch.float32, device=device)

    out = {"x": x64}
    for tag, time_value in (("t0", 0.0), ("tT", terminal_time)):
        t32 = torch.full_like(x32, time_value)
        with torch.no_grad():
            u_predicted = model(torch.stack([x32, t32], dim=1)).squeeze(-1)
        out[f"u_pred_{tag}"] = u_predicted.detach().cpu().numpy().astype(
            np.float64
        )
        out[f"u_ref_{tag}"] = exact_field.field(
            x64, np.full_like(x64, time_value)
        )
    out["g"] = exact_field.terminal_datum_values(x64)
    out["phi_theta_tT"] = terminal_target["phi_theta_terminal"]
    out["phi_star_tT"] = terminal_target["phi_star_terminal"]
    out["terminal_target_is_zero_target"] = np.asarray(
        [terminal_target["terminal_target_is_zero_target"]]
    )
    return out


def compute_spectra(model, problem, variant, closed_form_extension) -> dict:
    r"""Residual frequency decomposition (specification Section 3.4).

    (i) The residual field :math:`r(x, t_s) = (P\hat u)(x, t_s)` is
    evaluated by the training residual assembly on the uniform grid at the
    five time slices; (ii) real FFT with mean removal, normalised so the bin
    at wavenumber :math:`k` estimates the Fourier coefficient of the sampled
    field, powers averaged over the slices; (iii) the forcing side is the
    exact per-wavenumber coefficient
    :math:`\widehat{Lh}(k, t_s)` from the closed-form extension counterpart
    (never an FFT of samples), averaged in power over the same slices.

    For a zero-forcing variant the ratio is undefined: the absolute residual
    power spectrum is saved and the cutoff is recorded as absent.  Out-of-band
    ratio entries are set to zero before the running mean (a conservative
    downward bias at the band edge, documented here; all raw arrays are
    saved so the aggregation can re-derive the estimate).
    """
    import numpy as np
    import torch

    from learning_option_pricing.models.terminal_ansatz import (
        residual_decomposition,
    )

    device = next(model.parameters()).device
    terminal_time = problem["terminal_time"]
    n_grid = SPECTRA_GRID_SIZE
    x64 = np.linspace(0.0, TWO_PI, n_grid, endpoint=False)
    wavenumber_bins = np.fft.rfftfreq(n_grid, d=1.0 / n_grid)

    slice_fractions = np.asarray(SPECTRA_TIME_SLICE_FRACTIONS)
    residual_power_per_slice = np.zeros(
        (len(slice_fractions), len(wavenumber_bins))
    )
    for slice_index, fraction in enumerate(slice_fractions):
        coord = torch.as_tensor(
            x64, dtype=torch.float32, device=device
        ).requires_grad_(True)
        t = torch.full(
            (n_grid,), float(fraction) * terminal_time,
            dtype=torch.float32, device=device,
        ).requires_grad_(True)
        decomposition = residual_decomposition(
            model, coord, t,
            generator_coefficients=problem["generator_coefficients"],
        )
        residual_values = (
            decomposition["residual"].detach().cpu().numpy().astype(np.float64)
        )
        residual_hat = np.fft.rfft(residual_values - residual_values.mean()) / n_grid
        residual_power_per_slice[slice_index] = np.abs(residual_hat) ** 2
    residual_power = residual_power_per_slice.mean(axis=0)

    forcing_power = np.zeros(len(wavenumber_bins))
    positive_wavenumbers = np.arange(1, problem["band_edge"] + 1)
    forcing_power_band = np.zeros(len(positive_wavenumbers))
    for fraction in slice_fractions:
        forcing_coefficients = closed_form_extension.forcing_coefficient(
            positive_wavenumbers, float(fraction) * terminal_time
        )
        forcing_power_band += np.abs(forcing_coefficients) ** 2
    forcing_power_band /= len(slice_fractions)
    forcing_power[1:problem["band_edge"] + 1] = forcing_power_band

    forcing_maximum = float(forcing_power.max())
    forcing_defined = forcing_maximum > 0.0
    if forcing_defined:
        in_band_mask = forcing_power > (
            SPECTRA_IN_BAND_RELATIVE_THRESHOLD * forcing_maximum
        )
        cancellation_ratio = np.where(
            forcing_power > 0.0,
            residual_power / np.where(forcing_power > 0.0, forcing_power, 1.0),
            0.0,
        )
        number_of_truncated_bins = int(
            (cancellation_ratio > SPECTRA_CANCELLATION_RATIO_CEILING).sum()
        )
        if number_of_truncated_bins > 0:
            LOGGER.warning(
                "cancellation-ratio truncation ACTIVE: %d of %d wavenumbers "
                "exceed the ceiling %.3g and are truncated to it; largest raw "
                "ratio %.6g. The truncation is part of the cutoff estimator "
                "and can change the measured k_star -- the raw ratio is saved "
                "so the aggregation can re-derive it without the truncation.",
                number_of_truncated_bins,
                cancellation_ratio.size,
                SPECTRA_CANCELLATION_RATIO_CEILING,
                float(cancellation_ratio.max()),
            )
        clipped_ratio = np.clip(
            cancellation_ratio, 0.0, SPECTRA_CANCELLATION_RATIO_CEILING
        )
        window = SPECTRA_RUNNING_MEAN_WINDOW
        running_mean = np.convolve(
            clipped_ratio, np.ones(window) / window, mode="same"
        )
        k_star = -1
        for bin_index in range(1, len(wavenumber_bins)):
            if in_band_mask[bin_index] and (
                running_mean[bin_index] >= SPECTRA_CANCELLATION_THRESHOLD
            ):
                k_star = int(wavenumber_bins[bin_index])
                break
        k_star_defined = k_star >= 0
    else:
        in_band_mask = np.zeros(len(wavenumber_bins), dtype=bool)
        cancellation_ratio = np.full(len(wavenumber_bins), np.nan)
        running_mean = np.full(len(wavenumber_bins), np.nan)
        k_star = -1
        k_star_defined = False

    return {
        "wavenumber_bins": wavenumber_bins,
        "residual_power": residual_power,
        "residual_power_per_slice": residual_power_per_slice,
        "forcing_power": forcing_power,
        "cancellation_ratio": cancellation_ratio,
        "cancellation_ratio_running_mean": running_mean,
        "in_band_mask": in_band_mask,
        "slice_fractions": slice_fractions,
        "forcing_defined": np.asarray([forcing_defined]),
        "k_star": np.asarray([k_star]),
        "k_star_defined": np.asarray([k_star_defined]),
    }


# ===========================================================================
# Persistence
# ===========================================================================

def save_variant(variant_dir: Path, model, history, metrics, slices, spectra):
    import numpy as np
    import torch

    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), variant_dir / "models" / "model.pt")
    history_arrays = {
        key: np.asarray(value)
        for key, value in history.items()
        if isinstance(value, list)
    }
    np.savez_compressed(variant_dir / "hist.npz", **history_arrays)
    np.savez_compressed(
        variant_dir / "metrics.npz",
        **{key: np.asarray([value]) for key, value in metrics.items()},
    )
    np.savez_compressed(variant_dir / "slices.npz", **slices)
    np.savez_compressed(variant_dir / "spectra.npz", **spectra)


def write_summary(path: Path, payload: dict):
    import yaml

    with open(path, "w") as f:
        yaml.dump(
            payload, f, default_flow_style=False, sort_keys=False,
            width=float("inf"),
        )


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cell", choices=catalogue.cell_names(),
        default="g2_bernoulli_bandlimited",
        help="Training cell: (generator, datum) pair.",
    )
    parser.add_argument(
        "--variant", choices=catalogue.all_variant_names(), default=None,
        help="Single method variant (array-task path). Omit to run all "
             "variants of the cell.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--num-iterations", type=int, default=None,
        help="Override the iteration budget.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--debug", action="store_true",
        help="Mark as a smoke/test run (prefixes output with _debug_).",
    )
    parser.add_argument(
        "--replot", type=str, default=None,
        help="Regenerate comparison plots from an existing run DIR "
             "(no training).",
    )
    parser.add_argument(
        "--init-only", action="store_true",
        help="Create dir + metadata + one config YAML per array task, print "
             "the absolute EXPDIR, then exit (login-node safe).",
    )
    parser.add_argument(
        "--ablation-dir", type=str, default=None,
        help="Shared parent output dir (array path / explicit override).",
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="Folder of per-task YAML configs (array-worker contract).",
    )
    parser.add_argument(
        "--config-name", type=str, default=None,
        help="Basename (no extension) of the YAML config to load.",
    )
    # hyperparameter overrides
    parser.add_argument("--n-interior", type=int, default=None)
    parser.add_argument("--n-terminal", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser


def resolve_hparams(args) -> dict:
    hparams = dict(catalogue.DEFAULT_HPARAMS)
    if args.num_iterations is not None:
        hparams["num_iterations"] = args.num_iterations
    if args.n_interior is not None:
        hparams["n_interior"] = args.n_interior
    if args.n_terminal is not None:
        hparams["n_terminal"] = args.n_terminal
    if args.learning_rate is not None:
        hparams["learning_rate"] = args.learning_rate
    return hparams


def main(argv=None) -> int:
    assert catalogue.RUNNER_SCRIPT_STEM == Path(__file__).stem, (
        f"RUNNER_SCRIPT_STEM={catalogue.RUNNER_SCRIPT_STEM!r} != script stem "
        f"{Path(__file__).stem!r}; rename one so the data folder cannot drift."
    )
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )

    from learning_option_pricing.utils.run_context import script_data_dir

    if args.replot is not None:
        from _split_extension_plots import replot
        replot(Path(args.replot))
        return 0

    # Array-worker contract: a per-task YAML fully specifies the task.
    is_worker = args.config_dir is not None and args.config_name is not None
    config_hparams: dict = {}
    if is_worker:
        import yaml
        config_path = Path(args.config_dir) / f"{args.config_name}.yaml"
        with open(config_path) as f:
            task_config = yaml.safe_load(f)
        args.cell = task_config["cell"]
        args.variant = task_config["variant"]
        args.seed = int(task_config["seed"])
        args.debug = bool(task_config.get("debug", args.debug))
        args.ablation_dir = task_config["ablation_dir"]
        config_hparams = task_config.get("hparams", {})

    hparams = resolve_hparams(args)
    hparams.update(config_hparams)

    # Smoke-test guard: short runs must be flagged.
    if (hparams["num_iterations"] < SMOKE_TEST_NUM_ITERATIONS_THRESHOLD
            and not args.debug):
        raise SystemExit(
            f"--num-iterations {hparams['num_iterations']} is below the "
            f"smoke-test threshold ({SMOKE_TEST_NUM_ITERATIONS_THRESHOLD}); "
            f"pass --debug to flag this as exploratory, or raise the "
            f"iteration count."
        )

    debug_prefix = "_debug_" if args.debug else ""
    if args.ablation_dir is not None:
        ablation_dir = Path(args.ablation_dir)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%fZ")
        ablation_dir = (
            script_data_dir(__file__)
            / (f"{debug_prefix}{timestamp}_{args.cell}"
               f"_iters{hparams['num_iterations']}_seed{args.seed}")
        )
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # Self-contained run log (per-task in array runs to avoid interleaving).
    log_name = f"ablation_{args.variant}.log" if is_worker else "ablation.log"
    log_path = ablation_dir / log_name
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
        "%Y-%m-%dT%H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    cell_conf = catalogue.cell_by_name(args.cell)
    cell_variants = catalogue.variants_for_cell(args.cell)

    # Metadata (written once, at init / local run; workers must not race on
    # it).  This path is torch-free, so it is safe on a cluster login node.
    if not is_worker:
        write_summary(ablation_dir / "metadata.yaml", {
            "command": " ".join(sys.argv),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cell": args.cell,
            "generator_coefficients": {
                int(order): float(value)
                for order, value in cell_conf["generator_coefficients"].items()
            },
            "seed": args.seed,
            "hparams": hparams,
            "label": cell_conf["label"],
            "variants": [v["name"] for v in cell_variants],
        })

    if args.init_only:
        # One config YAML per array task, then the absolute EXPDIR on stdout
        # (logs go to stderr) for the launcher to capture.  No torch import
        # on this path -> safe on the login node.
        import yaml
        configs_dir = ablation_dir / "configs"
        configs_dir.mkdir(exist_ok=True)
        for variant in cell_variants:
            task_config = {
                "cell": args.cell,
                "variant": variant["name"],
                "seed": args.seed,
                "debug": args.debug,
                "ablation_dir": str(ablation_dir.resolve()),
                "hparams": hparams,
            }
            with open(configs_dir / f"{variant['name']}.yaml", "w") as f:
                yaml.dump(task_config, f, sort_keys=False)
        logger.info("init-only: wrote %d task configs to %s",
                    len(cell_variants), configs_dir)
        print(str(ablation_dir.resolve()))
        return 0

    # --- training path: torch is needed from here on ----------------------
    import numpy as np
    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Probe the GPU once up front so a momentarily-busy device is waited out
    # before any training work rather than failing the task mid-build.
    if device.type == "cuda":
        cuda_retry(lambda: torch.zeros(1, device=device))

    model_seed = derive_seed(args.seed, "model_init")
    sampler_seed = derive_seed(args.seed, "sampler")
    evaluation_seed = derive_seed(args.seed, "evaluation")

    logger.info("=" * 72)
    logger.info("STAGE-2 SPLIT-EXTENSION ABLATION (trained, circle)")
    logger.info("=" * 72)
    logger.info("  command:   %s", " ".join(sys.argv))
    logger.info("  python:    %s", sys.version.split()[0])
    logger.info("  numpy:     %s", np.__version__)
    logger.info("  torch:     %s", torch.__version__)
    logger.info("  cuda:      %s (version %s)",
                torch.cuda.is_available(), torch.version.cuda)
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(0)
        logger.info("  gpu:       %s (%.1f GiB)", device_properties.name,
                    device_properties.total_memory / 2**30)
    logger.info("  device:    %s", device)
    logger.info("  cell:      %s", args.cell)
    logger.info("  generator: %s", cell_conf["generator_coefficients"])
    logger.info("  seed:      %d (model_init=%d, sampler=%d, evaluation=%d)",
                args.seed, model_seed, sampler_seed, evaluation_seed)
    logger.info("  hparams:   %s", hparams)
    logger.info("  output:    %s", ablation_dir)
    logger.info("  log:       %s", log_path)

    problem = build_problem(args.cell)

    variants_to_run = (
        [catalogue.variant_by_name(args.cell, args.variant)]
        if args.variant is not None
        else cell_variants
    )
    log_every = max(1, hparams["num_iterations"] // 50)

    summary: dict = {}
    for variant in variants_to_run:
        start_time = time.time()
        model, history, cross_check_deviation = train_variant(
            variant, problem, hparams,
            num_iterations=hparams["num_iterations"],
            seed=args.seed, device=device, log_every=log_every,
        )
        # The extension field of the *trained* ansatz (None for datum-path
        # variants); rebuilt references are avoided — the trained model holds
        # its own field through the derivative callables, but the field
        # object itself is needed for the terminal-forcing profile, so it is
        # rebuilt deterministically here (pure closed forms, no randomness).
        _, extension_field = build_ansatz(
            variant, problem, hparams, model_seed=model_seed
        )
        closed_form_extension = build_closed_form_extension(variant, problem)

        best_state_channels = evaluate_best_state_channels(
            model, problem, hparams,
            evaluation_seed=evaluation_seed, device=device,
        )
        error_metrics = compute_error_metrics(model, problem)
        terminal_target = compute_terminal_target(
            model, problem, variant, extension_field
        )
        slices = compute_slices(model, problem, terminal_target)
        spectra = compute_spectra(
            model, problem, variant, closed_form_extension
        )

        forcing_floor_median_train = float(
            np.median(np.asarray(history["forcing_floor"]))
        )
        forcing_floor_closed_form = closed_form_forcing_floor(variant, problem)

        elapsed = time.time() - start_time
        seconds_per_iteration = elapsed / max(1, hparams["num_iterations"])
        k_star_value = int(spectra["k_star"][0])
        k_star_defined = bool(spectra["k_star_defined"][0])

        metrics = {
            **error_metrics,
            "terminal_target_distance": terminal_target[
                "terminal_target_distance"
            ],
            "terminal_target_is_zero_target": float(
                terminal_target["terminal_target_is_zero_target"]
            ),
            "best_loss": history["best_loss"],
            "best_iter": float(history["best_iter"]),
            **best_state_channels,
            "forcing_floor_median_train": forcing_floor_median_train,
            "forcing_floor_closed_form": forcing_floor_closed_form,
            "cross_check_deviation": (
                cross_check_deviation
                if cross_check_deviation is not None else float("nan")
            ),
            "k_star": float(k_star_value),
            "k_star_defined": float(k_star_defined),
            "n_parameters": float(history["n_parameters"]),
            "wall_time_s": elapsed,
        }

        variant_dir = ablation_dir / f"variant_{variant['name']}"
        save_variant(variant_dir, model, history, metrics, slices, spectra)
        logger.info(
            "[%s] done in %.1fs (%.3f s/iter) | best_loss=%.3e at iter %d | "
            "rel_l2=%.3e rel_l2_t0=%.3e corner_t0=%.3e | floor(median "
            "train)=%.3e vs closed form=%.3e | k_star=%s",
            variant["name"], elapsed, seconds_per_iteration,
            history["best_loss"], history["best_iter"],
            error_metrics["rel_l2"], error_metrics["rel_l2_t0"],
            error_metrics["rel_l2_corner_t0"],
            forcing_floor_median_train, forcing_floor_closed_form,
            k_star_value if k_star_defined else "absent",
        )

        summary[variant["name"]] = {
            **error_metrics,
            "terminal_target_distance": terminal_target[
                "terminal_target_distance"
            ],
            "terminal_target_is_zero_target": terminal_target[
                "terminal_target_is_zero_target"
            ],
            "best_loss": history["best_loss"],
            "best_iter": history["best_iter"],
            **best_state_channels,
            "forcing_floor_median_train": forcing_floor_median_train,
            "forcing_floor_closed_form": forcing_floor_closed_form,
            "cross_check_deviation": cross_check_deviation,
            "k_star": k_star_value if k_star_defined else None,
            "n_parameters": history["n_parameters"],
            "seconds_per_iteration": seconds_per_iteration,
            "wall_time_s": elapsed,
        }

    # In single-variant (array) runs write a per-variant summary to avoid
    # races on a shared file.
    summary_path = (
        ablation_dir / f"summary_{args.variant}.yaml"
        if args.variant is not None
        else ablation_dir / "summary.yaml"
    )
    write_summary(summary_path, summary)
    logger.info("wrote %s", summary_path)

    # Build comparison plots immediately when all variants ran in-process.
    if args.variant is None:
        try:
            from _split_extension_plots import replot
            replot(ablation_dir)
        except Exception as exc:  # plotting must never lose a trained run
            logger.warning("plotting failed (artefacts are saved): %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
