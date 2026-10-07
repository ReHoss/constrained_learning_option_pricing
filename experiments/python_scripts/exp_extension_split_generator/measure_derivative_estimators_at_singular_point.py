r"""Finite differences and automatic differentiation at a first-derivative discontinuity.

The script measures what three derivative estimators return at, and around, a
point :math:`x^\star` where a continuous function has a first-derivative
discontinuity, and compares every measured value with its closed form.

Part A -- difference quotients at :math:`x^\star = 0`, as functions of the step
:math:`h \in (0, +\infty)`, in IEEE double precision.  Six test functions:

* the positive part :math:`\max(x, 0)` and the absolute value :math:`|x|`
  (continuous, piecewise :math:`C^\infty`, one-sided derivatives :math:`a \neq b`);
* the squared positive part :math:`\max(x, 0)^2` (of class :math:`C^1`, increment
  of order :math:`h^2`);
* :math:`\sqrt{|x|}` (Hölder of exponent :math:`1/2`, not Lipschitz at :math:`0`);
* the unit step :math:`H` with :math:`H(0) = 1/2` (a jump of the function itself);
* the exponential :math:`e^{x}` (of class :math:`C^\infty`), the control that
  separates the floating-point cancellation of a difference quotient from the
  effect of the singular point.

For each, the forward, backward and centred first-difference quotients and the
centred second-difference quotient are recorded with their closed-form values.
At a first-derivative discontinuity the centred second-difference quotient equals
:math:`(b - a)/h + O(1)` as :math:`h \to 0`.

Part B -- the discrete second difference of :math:`\max(x - x^\star, 0)` on a
uniform grid of spacing :math:`\Delta`, with :math:`x^\star` placed off the grid.
The second difference is non-zero at the two nodes adjacent to :math:`x^\star`,
with values :math:`(1 - \vartheta)/\Delta` and :math:`\vartheta/\Delta`, and its
sum multiplied by :math:`\Delta` equals the jump :math:`b - a = 1`: a discrete
Dirac mass.

Part C -- automatic differentiation (PyTorch autograd) of five programs that
compute the positive part.  At :math:`x \neq 0` every program returns the
classical derivatives; at :math:`x = 0` each returns its own convention, which
is measured here.  A Monte-Carlo estimate of :math:`\int_{-1}^{1} f''` built from
autograd second derivatives at uniform samples is compared with the grid sum of
second differences (which telescopes to :math:`f'(1) - f'(-1)`) and with the
distributional value :math:`b - a = 1`.

Part D -- the band-limited Bernoulli datum
:math:`g_{K}(x) = \sum_{k=1}^{K} \cos(kx)/(\pi^2 k^2)` of the stage-two ablation.
Autograd returns its second derivative exactly, and the closed form is the
Dirichlet kernel

.. math::

    g_{K}''(x) = \frac{1}{2\pi^2}
      - \frac{1}{2\pi^2}\,\frac{\sin\bigl((K + \tfrac12)x\bigr)}{\sin(x/2)},

the truncation of the distributional second derivative
:math:`g'' = \frac{1}{2\pi^2} - \frac{1}{\pi}\delta_0` of the exact datum on
:math:`(-\pi, \pi]`.  Autograd applied to the closed form of the exact datum
returns the constant :math:`1/(2\pi^2)` at every point.

Reproducibility.  Every array plotted is saved to ``derivative_estimator_arrays.npz``
and every scalar to ``summary.yaml``; ``--replot RUN_DIR`` rebuilds the figures
from those artefacts with no recomputation.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from learning_option_pricing.pde.periodic_spectral_toolbox import (  # noqa: E402
    PeriodisedBernoulliDatum,
)
from learning_option_pricing.utils.figure_layout import finalize_figure  # noqa: E402
from learning_option_pricing.utils.run_context import (  # noqa: E402
    collect_run_metadata,
    find_repo_root,
    init_logging,
    log_parsed_args,
    log_runtime_versions,
    script_data_dir,
    utc_timestamp,
    write_command_txt,
    write_json,
)

logger = logging.getLogger(__name__)

SINGULAR_POINT = 0.0

# Off-grid position of the singular point in Part B, as a fraction of the grid
# spacing: x* = (k + OFF_GRID_FRACTION) * spacing for an integer k.
OFF_GRID_FRACTION = 0.3

# Every closed form of Part D must reproduce the measured autograd value to
# this tolerance; a disagreement aborts the run before any figure is written.
AUTOGRAD_CLOSED_FORM_TOLERANCE = 1.0e-9

# Part C: steps of the centred second difference evaluated at every point of a
# fine grid, sharpness values beta of the regularisation
# softplus_beta(x) = log(1 + e^{beta x}) / beta, and the fine grid itself.
PROFILE_FINITE_DIFFERENCE_STEPS = (0.1, 0.05)
SOFTPLUS_SHARPNESS_VALUES = (20.0, 40.0)
FINE_GRID_NUMBER_OF_POINTS = 200_001

# Fewer Monte-Carlo samples than this is an exploratory run and must carry --debug.
SMOKE_TEST_MONTE_CARLO_SAMPLE_THRESHOLD = 100_000

BERNOULLI_REGULARITY_INDEX = 1

ARRAYS_NAME = "derivative_estimator_arrays.npz"
SUMMARY_NAME = "summary.yaml"
FINITE_DIFFERENCE_FIGURE_NAME = "finite_difference_quotients.png"
AUTOGRAD_FIGURE_NAME = "discrete_dirac_and_autograd.png"
BANDLIMITED_FIGURE_NAME = "bandlimited_datum_second_derivative.png"

TEST_FUNCTION_NAMES = (
    "positive_part",
    "absolute_value",
    "squared_positive_part",
    "square_root_of_absolute_value",
    "unit_step",
    "exponential",
)

TEST_FUNCTION_LABELS = {
    "positive_part": r"$\max(x,0)$",
    "absolute_value": r"$|x|$",
    "squared_positive_part": r"$\max(x,0)^2$",
    "square_root_of_absolute_value": r"$\sqrt{|x|}$",
    "unit_step": r"$H(x)$, $H(0)=\frac{1}{2}$",
    "exponential": r"$e^{x}$ (control, $C^\infty$)",
}

POSITIVE_PART_PROGRAM_NAMES = (
    "torch_relu",
    "torch_clamp_minimum_zero",
    "torch_maximum_with_zero",
    "half_sum_with_absolute_value",
    "torch_where_positive",
)


def derive_seed(master_seed: int, role: str) -> int:
    """Per-role seed from the master seed; same construction as the stage-two runner."""
    digest = hashlib.blake2b(f"{master_seed}:{role}".encode(), digest_size=8).hexdigest()
    return int(digest, 16) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Part A -- difference quotients at the singular point
# ---------------------------------------------------------------------------


def evaluate_test_function(name: str, points: np.ndarray) -> np.ndarray:
    """Values of a test function, in double precision."""
    points = np.asarray(points, dtype=np.float64)
    if name == "positive_part":
        return np.maximum(points, 0.0)
    if name == "absolute_value":
        return np.abs(points)
    if name == "squared_positive_part":
        return np.maximum(points, 0.0) ** 2
    if name == "square_root_of_absolute_value":
        return np.sqrt(np.abs(points))
    if name == "unit_step":
        return np.where(points > 0.0, 1.0, np.where(points < 0.0, 0.0, 0.5))
    if name == "exponential":
        return np.exp(points)
    raise ValueError(f"unknown test function {name!r}")


def closed_form_quotients(name: str, steps: np.ndarray) -> dict[str, np.ndarray]:
    r"""Exact real-arithmetic values of the four quotients at :math:`x^\star = 0`.

    For the exponential, the expressions use ``expm1`` and ``sinh`` so that the
    closed form itself is free of the cancellation the measured quotient suffers.
    """
    steps = np.asarray(steps, dtype=np.float64)
    ones = np.ones_like(steps)
    if name == "positive_part":
        return {"forward": ones, "backward": 0.0 * ones, "centred": 0.5 * ones,
                "second": 1.0 / steps}
    if name == "absolute_value":
        return {"forward": ones, "backward": -ones, "centred": 0.0 * ones,
                "second": 2.0 / steps}
    if name == "squared_positive_part":
        return {"forward": steps, "backward": 0.0 * ones, "centred": 0.5 * steps,
                "second": ones}
    if name == "square_root_of_absolute_value":
        return {"forward": steps ** -0.5, "backward": -(steps ** -0.5),
                "centred": 0.0 * ones, "second": 2.0 * steps ** -1.5}
    if name == "unit_step":
        return {"forward": 0.5 / steps, "backward": 0.5 / steps,
                "centred": 0.5 / steps, "second": 0.0 * ones}
    if name == "exponential":
        return {
            "forward": np.expm1(steps) / steps,
            "backward": -np.expm1(-steps) / steps,
            "centred": np.sinh(steps) / steps,
            "second": (2.0 * np.sinh(0.5 * steps) / steps) ** 2,
        }
    raise ValueError(f"unknown test function {name!r}")


def measured_quotients(name: str, steps: np.ndarray) -> dict[str, np.ndarray]:
    """The four difference quotients at the singular point, computed in float64."""
    centre = evaluate_test_function(name, np.full_like(steps, SINGULAR_POINT))
    right = evaluate_test_function(name, SINGULAR_POINT + steps)
    left = evaluate_test_function(name, SINGULAR_POINT - steps)
    return {
        "forward": (right - centre) / steps,
        "backward": (centre - left) / steps,
        "centred": (right - left) / (2.0 * steps),
        "second": (right - 2.0 * centre + left) / steps**2,
    }


def compute_difference_quotient_arrays(steps: np.ndarray) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"steps": steps}
    for name in TEST_FUNCTION_NAMES:
        measured = measured_quotients(name, steps)
        predicted = closed_form_quotients(name, steps)
        for quotient_name in ("forward", "backward", "centred", "second"):
            arrays[f"measured_{quotient_name}_{name}"] = measured[quotient_name]
            arrays[f"closed_form_{quotient_name}_{name}"] = predicted[quotient_name]
    return arrays


# ---------------------------------------------------------------------------
# Part B -- the discrete Dirac mass on a grid
# ---------------------------------------------------------------------------


def compute_discrete_dirac_arrays(grid_spacings: list[float]) -> dict[str, np.ndarray]:
    r"""Second differences of :math:`\max(x - x^\star, 0)` with :math:`x^\star` off the grid."""
    arrays: dict[str, np.ndarray] = {"grid_spacings": np.asarray(grid_spacings)}
    for index, spacing in enumerate(grid_spacings):
        node_indices = np.arange(-int(round(0.3 / spacing)), int(round(0.3 / spacing)) + 1)
        nodes = node_indices * spacing
        singular_point = OFF_GRID_FRACTION * spacing
        values = np.maximum(nodes - singular_point, 0.0)
        second_difference = np.zeros_like(nodes)
        second_difference[1:-1] = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / spacing**2
        arrays[f"grid_nodes_{index}"] = nodes
        arrays[f"grid_second_difference_{index}"] = second_difference
        arrays[f"grid_mass_{index}"] = np.asarray([second_difference.sum() * spacing])
        arrays[f"grid_singular_point_{index}"] = np.asarray([singular_point])
    return arrays


# ---------------------------------------------------------------------------
# Part C -- automatic differentiation of the positive part
# ---------------------------------------------------------------------------


def positive_part_program(name: str, points: torch.Tensor) -> torch.Tensor:
    if name == "torch_relu":
        return torch.relu(points)
    if name == "torch_clamp_minimum_zero":
        return torch.clamp(points, min=0.0)
    if name == "torch_maximum_with_zero":
        return torch.maximum(points, torch.zeros_like(points))
    if name == "half_sum_with_absolute_value":
        return 0.5 * (points + torch.abs(points))
    if name == "torch_where_positive":
        return torch.where(points > 0.0, points, torch.zeros_like(points))
    raise ValueError(f"unknown program {name!r}")


def autograd_first_and_second_derivatives(
    program, points: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """First and second derivatives by reverse-mode autograd.

    Returns:
        ``(first, second, second_is_structural_zero)``.  The flag is ``True``
        when autograd reports that the first derivative does not depend on the
        input at all (a graph-disconnected derivative); the second derivative is
        then the zero tensor, which is what autograd's ``allow_unused`` contract
        means, and the flag records that this zero is structural.
    """
    points = points.detach().clone().requires_grad_(True)
    values = program(points)
    (first,) = torch.autograd.grad(values.sum(), points, create_graph=True)
    if not first.requires_grad:
        return first.detach(), torch.zeros_like(points), True
    (second,) = torch.autograd.grad(first.sum(), points, allow_unused=True)
    if second is None:
        return first.detach(), torch.zeros_like(points), True
    return first.detach(), second.detach(), False


def compute_autograd_arrays(
    number_of_monte_carlo_samples: int, monte_carlo_seed: int, grid_spacing: float
) -> tuple[dict[str, np.ndarray], dict]:
    probe_points = torch.tensor([-0.5, -1.0e-300, 0.0, 1.0e-300, 0.5], dtype=torch.float64)
    arrays: dict[str, np.ndarray] = {"probe_points": probe_points.numpy()}
    conventions: dict = {}
    for name in POSITIVE_PART_PROGRAM_NAMES:
        first, second, structural = autograd_first_and_second_derivatives(
            lambda points, name=name: positive_part_program(name, points), probe_points
        )
        arrays[f"autograd_first_derivative_{name}"] = first.numpy()
        arrays[f"autograd_second_derivative_{name}"] = second.numpy()
        conventions[name] = {
            "first_derivative_at_zero": float(first[2]),
            "second_derivative_at_zero": float(second[2]),
            "first_derivative_at_minus_one_e_minus_300": float(first[1]),
            "first_derivative_at_plus_one_e_minus_300": float(first[3]),
            "second_derivative_is_graph_disconnected_zero": bool(structural),
        }

    # Profiles on [-1, 1] for the figure (relu program).
    profile_points = torch.linspace(-1.0, 1.0, 2001, dtype=torch.float64)
    first, second, _ = autograd_first_and_second_derivatives(torch.relu, profile_points)
    arrays["profile_points"] = profile_points.numpy()
    arrays["profile_autograd_first_derivative"] = first.numpy()
    arrays["profile_autograd_second_derivative"] = second.numpy()

    # Three estimators of f'' on a fine grid of [-1, 1]: autograd of relu, the
    # centred second difference of step h at every point (a hat of height 1/h),
    # and autograd of the C-infinity regularisation softplus_beta.
    fine_points = torch.linspace(-1.0, 1.0, FINE_GRID_NUMBER_OF_POINTS, dtype=torch.float64)
    fine_points_numpy = fine_points.numpy()
    arrays["fine_points"] = fine_points_numpy
    estimator_integrals: dict = {}
    _, relu_second_fine, _ = autograd_first_and_second_derivatives(torch.relu, fine_points)
    arrays["fine_autograd_relu_second_derivative"] = relu_second_fine.numpy()
    estimator_integrals["autograd_relu"] = float(
        np.trapezoid(relu_second_fine.numpy(), fine_points_numpy))
    for step in PROFILE_FINITE_DIFFERENCE_STEPS:
        hat = (np.maximum(fine_points_numpy + step, 0.0) - 2.0 * np.maximum(fine_points_numpy, 0.0)
               + np.maximum(fine_points_numpy - step, 0.0)) / step**2
        closed_form_hat = np.maximum(step - np.abs(fine_points_numpy), 0.0) / step**2
        hat_deviation = float(np.abs(hat - closed_form_hat).max())
        arrays[f"fine_second_difference_step_{step:g}"] = hat
        estimator_integrals[f"second_difference_step_{step:g}"] = {
            "measured_integral": float(np.trapezoid(hat, fine_points_numpy)),
            "closed_form_integral": 1.0,
            "measured_maximum": float(hat.max()),
            "closed_form_maximum_one_over_step": 1.0 / step,
            "maximum_deviation_from_hat_closed_form": hat_deviation,
        }
    for sharpness in SOFTPLUS_SHARPNESS_VALUES:
        def softplus_program(points, sharpness=sharpness):
            # logaddexp form: no switch to the identity above a threshold, unlike
            # torch.nn.functional.softplus, whose default threshold would replace
            # the program by x (second derivative 0) beyond sharpness * x = 20.
            return torch.logaddexp(torch.zeros_like(points), sharpness * points) / sharpness

        _, softplus_second, _ = autograd_first_and_second_derivatives(softplus_program, fine_points)
        logistic = 1.0 / (1.0 + np.exp(-sharpness * fine_points_numpy))
        closed_form = sharpness * logistic * (1.0 - logistic)
        softplus_deviation = float(np.abs(softplus_second.numpy() - closed_form).max())
        if softplus_deviation > AUTOGRAD_CLOSED_FORM_TOLERANCE * sharpness:
            raise ValueError(
                f"autograd second derivative of softplus (beta = {sharpness}) differs from "
                f"its closed form by {softplus_deviation:.3e}"
            )
        arrays[f"fine_autograd_softplus_second_derivative_{sharpness:g}"] = softplus_second.numpy()
        estimator_integrals[f"autograd_softplus_sharpness_{sharpness:g}"] = {
            "measured_integral": float(np.trapezoid(softplus_second.numpy(), fine_points_numpy)),
            "closed_form_integral_tanh_half_sharpness": math.tanh(0.5 * sharpness),
            "measured_maximum": float(softplus_second.numpy().max()),
            "closed_form_maximum_quarter_sharpness": 0.25 * sharpness,
            "maximum_deviation_from_closed_form": softplus_deviation,
        }

    # Monte-Carlo estimate of the integral of f'' over [-1, 1] from autograd.
    generator = torch.Generator().manual_seed(monte_carlo_seed)
    samples = 2.0 * torch.rand(number_of_monte_carlo_samples, generator=generator,
                               dtype=torch.float64) - 1.0
    number_of_samples_at_singular_point = int((samples == SINGULAR_POINT).sum())
    _, second_at_samples, _ = autograd_first_and_second_derivatives(torch.relu, samples)
    autograd_monte_carlo_integral = float(2.0 * second_at_samples.mean())

    # Grid sum of second differences, x* on a node and x* off the grid.
    number_of_cells = int(round(2.0 / grid_spacing))
    nodes = np.linspace(-1.0, 1.0, number_of_cells + 1)
    grid_integrals = {}
    for placement, shift in (("singular_point_on_node", 0.0),
                             ("singular_point_off_grid", OFF_GRID_FRACTION * grid_spacing)):
        values = np.maximum(nodes - shift, 0.0)
        second_difference = (values[2:] - 2.0 * values[1:-1] + values[:-2]) / grid_spacing**2
        grid_integrals[placement] = float(second_difference.sum() * grid_spacing)

    arrays["integral_estimates"] = np.asarray([
        autograd_monte_carlo_integral,
        grid_integrals["singular_point_on_node"],
        grid_integrals["singular_point_off_grid"],
        1.0,
    ])
    scalars = {
        "autograd_conventions_at_singular_point": conventions,
        "number_of_monte_carlo_samples": int(number_of_monte_carlo_samples),
        "number_of_samples_equal_to_singular_point": number_of_samples_at_singular_point,
        "autograd_monte_carlo_estimate_of_integral_of_second_derivative": autograd_monte_carlo_integral,
        "grid_spacing": float(grid_spacing),
        "grid_sum_estimate_singular_point_on_node": grid_integrals["singular_point_on_node"],
        "grid_sum_estimate_singular_point_off_grid": grid_integrals["singular_point_off_grid"],
        "distributional_value_jump_of_first_derivative": 1.0,
        "estimators_of_second_derivative_on_minus_one_one": estimator_integrals,
    }
    return arrays, scalars


# ---------------------------------------------------------------------------
# Part D -- the band-limited Bernoulli datum
# ---------------------------------------------------------------------------


def exact_datum_torch(points: torch.Tensor) -> torch.Tensor:
    r"""Exact periodised Bernoulli datum, closed form on the principal period.

    With :math:`y \in [-\pi, \pi)` the representative of :math:`x`,
    :math:`g(x) = y^2/(4\pi^2) - |y|/(2\pi) + 1/6`.
    """
    wrapped = torch.remainder(points + math.pi, 2.0 * math.pi) - math.pi
    return wrapped**2 / (4.0 * math.pi**2) - torch.abs(wrapped) / (2.0 * math.pi) + 1.0 / 6.0


def bandlimited_datum_torch(points: torch.Tensor, band_edge: int) -> torch.Tensor:
    """Truncation g_K, with the coefficients taken from the library datum class."""
    datum = PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)
    wavenumbers_numpy = np.arange(1, band_edge + 1)
    coefficients = torch.as_tensor(
        np.real(datum.fourier_coefficients(wavenumbers_numpy)), dtype=torch.float64
    )
    wavenumbers = torch.as_tensor(wavenumbers_numpy, dtype=torch.float64)
    return 2.0 * (coefficients * torch.cos(points[:, None] * wavenumbers)).sum(dim=1)


def dirichlet_kernel_second_derivative(points: np.ndarray, band_edge: int) -> np.ndarray:
    """Closed form of g_K'' through the Dirichlet kernel; its value at 0 is -K/pi^2."""
    points = np.asarray(points, dtype=np.float64)
    half_angle_sine = np.sin(0.5 * points)
    near_zero = np.abs(half_angle_sine) < 1.0e-14
    safe_sine = np.where(near_zero, 1.0, half_angle_sine)
    dirichlet = np.where(near_zero, 2.0 * band_edge + 1.0,
                         np.sin((band_edge + 0.5) * points) / safe_sine)
    return (1.0 - dirichlet) / (2.0 * math.pi**2)


def compute_bandlimited_arrays(
    band_edges: list[int], number_of_window_points: int, number_of_circle_points: int
) -> tuple[dict[str, np.ndarray], dict]:
    window_points = torch.linspace(-math.pi / 8.0, math.pi / 8.0, number_of_window_points,
                                   dtype=torch.float64)
    circle_points = torch.linspace(-math.pi, math.pi, number_of_circle_points + 1,
                                   dtype=torch.float64)[:-1]
    arrays: dict[str, np.ndarray] = {
        "band_edges": np.asarray(band_edges),
        "window_points": window_points.numpy(),
        "circle_points": circle_points.numpy(),
    }
    regular_part = 1.0 / (2.0 * math.pi**2)

    # Exact datum: values against the library, second derivative against 1/(2 pi^2).
    datum = PeriodisedBernoulliDatum(BERNOULLI_REGULARITY_INDEX)
    library_values = np.asarray(datum.pointwise_values(circle_points.numpy()), dtype=float)
    torch_values = exact_datum_torch(circle_points).numpy()
    exact_value_deviation = float(np.abs(library_values - torch_values).max())
    _, exact_second, _ = autograd_first_and_second_derivatives(exact_datum_torch, window_points)
    arrays["exact_datum_autograd_second_derivative"] = exact_second.numpy()
    exact_second_deviation = float(np.abs(exact_second.numpy() - regular_part).max())

    scalars: dict = {
        "exact_datum_closed_form_versus_library_maximum_deviation": exact_value_deviation,
        "exact_datum_autograd_second_derivative_maximum_deviation_from_regular_part":
            exact_second_deviation,
        "regular_part_of_second_derivative": regular_part,
        "jump_of_first_derivative": float(datum.jump_of_rho_derivative),
        "band_edges": {},
    }
    for band_edge in band_edges:
        def program(points, band_edge=band_edge):
            return bandlimited_datum_torch(points, band_edge)

        _, window_second, _ = autograd_first_and_second_derivatives(program, window_points)
        _, circle_second, _ = autograd_first_and_second_derivatives(program, circle_points)
        window_closed_form = dirichlet_kernel_second_derivative(window_points.numpy(), band_edge)
        circle_closed_form = dirichlet_kernel_second_derivative(circle_points.numpy(), band_edge)
        deviation = float(max(np.abs(window_second.numpy() - window_closed_form).max(),
                              np.abs(circle_second.numpy() - circle_closed_form).max()))
        if deviation > AUTOGRAD_CLOSED_FORM_TOLERANCE:
            raise ValueError(
                f"autograd second derivative of g_K (K = {band_edge}) differs from the "
                f"Dirichlet closed form by {deviation:.3e} > {AUTOGRAD_CLOSED_FORM_TOLERANCE:.1e}"
            )
        # Mass of the oscillating part over one period (rectangle rule, exact for a
        # trigonometric polynomial of degree below the number of points).
        spacing = 2.0 * math.pi / number_of_circle_points
        oscillating_mass = float(((circle_second.numpy() - regular_part) * spacing).sum())
        # Spacing of consecutive sign changes of g_K'' - 1/(2 pi^2) in the window,
        # excluding the central lobe.
        oscillating = window_second.numpy() - regular_part
        crossing_indices = np.nonzero(np.diff(np.sign(oscillating)) != 0)[0]
        crossing_points = window_points.numpy()[crossing_indices]
        crossing_points = crossing_points[crossing_points > 0.0]
        measured_half_period = (float(np.diff(crossing_points).mean())
                                if crossing_points.size > 2 else float("nan"))
        _, second_at_singular_point, _ = autograd_first_and_second_derivatives(
            program, torch.zeros(1, dtype=torch.float64))
        arrays[f"bandlimited_window_second_derivative_{band_edge}"] = window_second.numpy()
        arrays[f"bandlimited_circle_second_derivative_{band_edge}"] = circle_second.numpy()
        scalars["band_edges"][int(band_edge)] = {
            "autograd_versus_dirichlet_closed_form_maximum_deviation": deviation,
            "autograd_second_derivative_at_singular_point": float(second_at_singular_point[0]),
            # Closed form 1/(2 pi^2) (1 - (2K + 1)) = -K / pi^2.
            "closed_form_value_at_singular_point": -band_edge / math.pi**2,
            "measured_mass_of_oscillating_part_over_one_period": oscillating_mass,
            "closed_form_mass_jump_of_first_derivative": -1.0 / math.pi,
            "measured_spacing_of_sign_changes_in_window": measured_half_period,
            "closed_form_spacing_pi_over_band_edge_plus_half": math.pi / (band_edge + 0.5),
        }
    return arrays, scalars


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

FINITE_DIFFERENCE_FORMULA = (
    r"Forward $\frac{f(x^\star+h)-f(x^\star)}{h}$, backward $\frac{f(x^\star)-f(x^\star-h)}{h}$, "
    r"centred $\frac{f(x^\star+h)-f(x^\star-h)}{2h}$, second $\frac{f(x^\star+h)-2f(x^\star)+f(x^\star-h)}{h^2}$, "
    r"$x^\star=0$, double precision"
    "\n"
    r"Corner ($a=f'_-(x^\star)\neq b=f'_+(x^\star)$): forward $\to b$, backward $\to a$, centred $\to\frac{a+b}{2}$, "
    r"second $=\frac{b-a}{h}+O(1)$ as $h\to0$"
    "\n"
    r"Grid: $\delta^2_\Delta f(x_j)=\frac{f(x_{j+1})-2f(x_j)+f(x_{j-1})}{\Delta^2}$ for $f(x)=\max(x-x^\star,0)$, "
    r"$x^\star=\vartheta\Delta$, $\vartheta=0.3$; $\Delta\sum_j\delta^2_\Delta f(x_j)=b-a=1$"
)

AUTOGRAD_FORMULA = (
    r"$f(x)=\max(x,0)$; distributional $f''=\delta_0$, so $\int_{-1}^{1}f''=f'(1)-f'(-1)=1$. "
    r"Autograd of $\mathrm{relu}$: $f'=\mathbf{1}_{x>0}$, $f''=0$ on $\mathbb{R}\setminus\{0\}$"
    "\n"
    r"Second difference at every $x$: $\frac{f(x+h)-2f(x)+f(x-h)}{h^2}=\frac{\max(h-|x|,0)}{h^2}$ (height $\frac{1}{h}$, mass $1$); "
    r"$\mathrm{softplus}_\beta(x)=\frac{\log(1+e^{\beta x})}{\beta}$, "
    r"$\mathrm{softplus}_\beta''(x)=\frac{\beta e^{\beta x}}{(1+e^{\beta x})^2}$ (height $\frac{\beta}{4}$, mass $\tanh\frac{\beta}{2}$)"
    "\n"
    r"Monte Carlo $2\,\frac{1}{N}\sum_{i=1}^{N}f''(X_i)$, $X_i\sim\mathcal{U}(-1,1)$; "
    r"grid $\Delta\sum_j\delta^2_\Delta f(x_j)$, $\delta^2_\Delta f(x_j)=\frac{f(x_{j+1})-2f(x_j)+f(x_{j-1})}{\Delta^2}$"
)

BANDLIMITED_FORMULA = (
    r"$g(x)=\sum_{k\geq1}\frac{\cos(kx)}{\pi^2k^2}$, $g''=\frac{1}{2\pi^2}-\frac{1}{\pi}\delta_0$ on $(-\pi,\pi]$;  "
    r"$g_K(x)=\sum_{k=1}^{K}\frac{\cos(kx)}{\pi^2k^2}$"
    "\n"
    r"$g_K''(x)=\frac{1}{2\pi^2}-\frac{1}{2\pi^2}\frac{\sin((K+\frac{1}{2})x)}{\sin(x/2)}$ (Dirichlet kernel), "
    r"$g_K''(0)=-\frac{K}{\pi^2}$, $\int_{-\pi}^{\pi}\left(g_K''-\frac{1}{2\pi^2}\right)=-\frac{1}{\pi}$"
)


def build_finite_difference_figure(arrays: dict, figure_path: Path) -> None:
    steps = np.asarray(arrays["steps"])
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # (a) the corner max(x, 0): finite limits that depend on the stencil.
    axis = axes[0]
    colours = {"forward": "#1f77b4", "backward": "#2ca02c", "centred": "#d62728"}
    for quotient_name, colour in colours.items():
        axis.semilogx(steps, arrays[f"measured_{quotient_name}_positive_part"], "-",
                      color=colour, lw=2.0, label=f"Measured, {quotient_name}")
        axis.semilogx(steps, arrays[f"closed_form_{quotient_name}_positive_part"], "--",
                      color="black", lw=0.9)
    axis.set_xlabel(r"Step $h$")
    axis.set_ylabel(r"First-difference quotient")
    axis.set_title(r"$\max(x,0)$ at $x^\star=0$: limits $b=1$, $a=0$, $\frac{a+b}{2}$", fontsize=9)
    axis.set_ylim(-0.1, 1.1)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)

    # (b) |forward quotient| for the regularity family.
    axis = axes[1]
    family = ("squared_positive_part", "positive_part", "square_root_of_absolute_value", "unit_step")
    palette = plt.cm.viridis(np.linspace(0.05, 0.85, len(family)))
    for name, colour in zip(family, palette):
        axis.loglog(steps, np.abs(arrays[f"measured_forward_{name}"]), "-", color=colour,
                    lw=2.0, label=TEST_FUNCTION_LABELS[name])
        axis.loglog(steps, np.abs(arrays[f"closed_form_forward_{name}"]), "--",
                    color="black", lw=0.9)
    axis.set_xlabel(r"Step $h$")
    axis.set_ylabel(r"$|$Forward quotient$|$")
    axis.set_title(r"Increment of order $h^\alpha$: quotient of order $h^{\alpha-1}$", fontsize=9)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)

    # (c) |second-difference quotient|.
    axis = axes[2]
    family = ("positive_part", "absolute_value", "squared_positive_part",
              "square_root_of_absolute_value", "exponential")
    palette = plt.cm.plasma(np.linspace(0.05, 0.85, len(family)))
    for name, colour in zip(family, palette):
        measured = np.abs(arrays[f"measured_second_{name}"])
        axis.loglog(steps, np.where(measured > 0.0, measured, np.nan), "-", color=colour,
                    lw=2.0, label=TEST_FUNCTION_LABELS[name])
        axis.loglog(steps, np.abs(arrays[f"closed_form_second_{name}"]), "--",
                    color="black", lw=0.9)
    axis.set_xlabel(r"Step $h$")
    axis.set_ylabel(r"$|$Second-difference quotient$|$")
    axis.set_title(r"Second difference: $\frac{b-a}{h}$ at a corner; "
                   r"$e^x$ departs from $1$ by cancellation", fontsize=9)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)

    reference_handle = plt.Line2D([], [], ls="--", color="black", lw=0.9,
                                  label="Closed form (exact arithmetic)")
    legend = figure.legend(handles=[reference_handle], loc="upper center",
                           bbox_to_anchor=(0.5, 0.31), fontsize=8)
    figure.tight_layout(rect=[0, 0.31, 1, 1])
    finalize_figure(figure, figure_path, legends=[legend] + [a.get_legend() for a in axes],
                    formula=FINITE_DIFFERENCE_FORMULA, axes=list(axes))


def build_autograd_figure(arrays: dict, figure_path: Path) -> None:
    figure, axes_grid = plt.subplots(2, 2, figsize=(13.0, 10.0))
    axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 1]]

    # (1, 0) three estimators of f'' near the singular point.
    axis = axes_grid[1, 0]
    fine_points = np.asarray(arrays["fine_points"])
    window = np.abs(fine_points) <= 0.2
    axis.plot(fine_points[window], np.asarray(arrays["fine_autograd_relu_second_derivative"])[window],
              "-", color="#d62728", lw=2.4,
              label=r"Autograd of $\mathrm{relu}$: $0$, mass $0$")
    step_palette = plt.cm.viridis(np.linspace(0.2, 0.6, len(PROFILE_FINITE_DIFFERENCE_STEPS)))
    for step, colour in zip(PROFILE_FINITE_DIFFERENCE_STEPS, step_palette):
        values = np.asarray(arrays[f"fine_second_difference_step_{step:g}"])
        mass = float(np.trapezoid(values, fine_points))
        axis.plot(fine_points[window], values[window], "-", color=colour, lw=1.6,
                  label=rf"Second difference, $h={step:g}$: height {values.max():.4g}, mass {mass:.6f}")
    sharpness_palette = plt.cm.plasma(np.linspace(0.45, 0.8, len(SOFTPLUS_SHARPNESS_VALUES)))
    for sharpness, colour in zip(SOFTPLUS_SHARPNESS_VALUES, sharpness_palette):
        values = np.asarray(arrays[f"fine_autograd_softplus_second_derivative_{sharpness:g}"])
        mass = float(np.trapezoid(values, fine_points))
        axis.plot(fine_points[window], values[window], "-", color=colour, lw=1.6,
                  label=rf"Autograd of $\mathrm{{softplus}}_\beta$, $\beta={sharpness:g}$: "
                        rf"height {values.max():.4g}, mass {mass:.6f}")
    axis.annotate("", xy=(0.0, 21.0), xytext=(0.0, 0.0),
                  arrowprops=dict(arrowstyle="-|>", ls=":", color="grey", lw=1.2))
    axis.text(0.01, 20.0, r"$\delta_0$, mass $1$", color="grey", fontsize=8)
    axis.set_xlabel(r"Point $x$")
    axis.set_ylabel(r"Estimate of $f''(x)$")
    axis.set_title(r"Near $x^\star=0$: a peak of mass $1$, or nothing (autograd of relu)", fontsize=9)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, fontsize=7.5)
    axes.append(axis)

    # (0, 0) discrete Dirac mass.
    axis = axes[0]
    spacings = np.asarray(arrays["grid_spacings"])
    palette = plt.cm.viridis(np.linspace(0.05, 0.75, spacings.size))
    for index, (spacing, colour) in enumerate(zip(spacings, palette)):
        nodes = arrays[f"grid_nodes_{index}"]
        values = arrays[f"grid_second_difference_{index}"]
        mass = float(np.asarray(arrays[f"grid_mass_{index}"]).reshape(-1)[0])
        axis.plot(nodes, values, "-o", color=colour, ms=3.0, lw=1.2,
                  label=rf"$\Delta={spacing:g}$, $\Delta\sum_j\delta^2_\Delta f={mass:.12g}$")
    axis.axvline(0.0, ls=":", color="grey", lw=1.0)
    axis.set_xlim(-0.35, 0.35)
    axis.set_xlabel(r"Grid node $x_j$")
    axis.set_ylabel(r"$\delta^2_\Delta f(x_j)$")
    axis.set_title(r"$\max(x-x^\star,0)$, $x^\star=0.3\Delta$: values $\frac{0.7}{\Delta}$ and $\frac{0.3}{\Delta}$",
                   fontsize=9)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=7.5)

    # (b) autograd profiles of relu.
    axis = axes[1]
    points = arrays["profile_points"]
    axis.plot(points, arrays["profile_autograd_first_derivative"], "-", color="#1f77b4",
              lw=2.0, label=r"Autograd $f'$")
    axis.plot(points, arrays["profile_autograd_second_derivative"], "-", color="#d62728",
              lw=2.0, label=r"Autograd $f''$")
    axis.axvline(0.0, ls=":", color="grey", lw=1.0, label=r"$x^\star=0$")
    axis.set_xlabel(r"Point $x$")
    axis.set_ylabel(r"Derivative")
    axis.set_title(r"Autograd of $\mathrm{relu}$: no value of order $\frac{1}{h}$ anywhere", fontsize=9)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7.5)

    # (c) estimates of the integral of f''.
    axis = axes[2]
    estimates = np.asarray(arrays["integral_estimates"])
    labels = ["Autograd,\nMonte Carlo", "Grid sum,\n$x^\\star$ on a node",
              "Grid sum,\n$x^\\star$ off the grid"]
    axis.bar(np.arange(3), estimates[:3], color=["#d62728", "#1f77b4", "#2ca02c"], width=0.6)
    axis.axhline(estimates[3], ls="--", color="black", lw=1.0,
                 label=r"Distributional value $f'(1)-f'(-1)=1$")
    axis.set_xticks(np.arange(3))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel(r"Estimate of $\int_{-1}^{1}f''$")
    axis.set_title(r"The singular part: lost by autograd, kept by the grid sum", fontsize=9)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=7.5)

    figure.tight_layout(rect=[0, 0.09, 1, 1], h_pad=6.0)
    finalize_figure(figure, figure_path, legends=[a.get_legend() for a in axes],
                    formula=AUTOGRAD_FORMULA, axes=list(axes))


def build_bandlimited_figure(arrays: dict, figure_path: Path) -> None:
    band_edges = [int(k) for k in np.asarray(arrays["band_edges"])]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    palette = plt.cm.viridis(np.linspace(0.75, 0.05, len(band_edges)))

    for axis, key, points_key, title in (
        (axes[0], "window", "window_points",
         r"Window $|x|\leq\pi/8$: oscillation of period $\frac{2\pi}{K+1/2}$, peak $-\frac{K}{\pi^2}$"),
        (axes[1], "circle", "circle_points",
         r"Whole period: the peak concentrates the mass $-\frac{1}{\pi}$ of the Dirac"),
    ):
        points = np.asarray(arrays[points_key])
        for band_edge, colour in zip(band_edges, palette):
            axis.plot(points, arrays[f"bandlimited_{key}_second_derivative_{band_edge}"], "-",
                      color=colour, lw=1.2, label=rf"Autograd $g_K''$, $K={band_edge}$")
        axis.axhline(1.0 / (2.0 * math.pi**2), ls="--", color="black", lw=1.2,
                     label=r"Autograd of exact $g$: $\frac{1}{2\pi^2}$")
        axis.axvline(0.0, ls=":", color="grey", lw=1.0)
        axis.set_xlabel(r"Point $x$")
        axis.set_ylabel(r"Second derivative")
        axis.set_title(title, fontsize=9)
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.22),
                           ncol=4, fontsize=8)
    figure.tight_layout(rect=[0, 0.24, 1, 1])
    finalize_figure(figure, figure_path, legends=[legend], formula=BANDLIMITED_FORMULA,
                    axes=list(axes))


def build_all_figures(arrays: dict, run_directory: Path) -> list[Path]:
    paths = [run_directory / FINITE_DIFFERENCE_FIGURE_NAME,
             run_directory / AUTOGRAD_FIGURE_NAME,
             run_directory / BANDLIMITED_FIGURE_NAME]
    build_finite_difference_figure(arrays, paths[0])
    build_autograd_figure(arrays, paths[1])
    build_bandlimited_figure(arrays, paths[2])
    return paths


def regenerate_figures(run_directory: Path) -> list[Path]:
    """Rebuild every figure from the saved arrays, with no recomputation."""
    arrays_path = run_directory / ARRAYS_NAME
    if not arrays_path.is_file():
        raise FileNotFoundError(f"no saved arrays at {arrays_path}")
    with np.load(arrays_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    return build_all_figures(arrays, run_directory)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finite differences and autograd at a first-derivative discontinuity."
    )
    parser.add_argument("--number-of-steps", type=int, default=45,
                        help="Number of steps h, log-spaced in [1e-12, 1e-1] (default 45).")
    parser.add_argument("--grid-spacings", type=float, nargs="+", default=[0.1, 0.05, 0.025],
                        help="Grid spacings of Part B (default 0.1 0.05 0.025).")
    parser.add_argument("--integral-grid-spacing", type=float, default=1.0e-3,
                        help="Grid spacing of the grid-sum integral estimate of Part C.")
    parser.add_argument("--number-of-monte-carlo-samples", type=int, default=1_000_000,
                        help="Uniform samples of the autograd Monte-Carlo estimate (default 1e6).")
    parser.add_argument("--band-edges", type=int, nargs="+", default=[8, 32, 128],
                        help="Truncation wavenumbers K of Part D (default 8 32 128; 128 is "
                        "the stage-two value).")
    parser.add_argument("--number-of-window-points", type=int, default=8193)
    parser.add_argument("--number-of-circle-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed; only the Monte-Carlo sampler of Part C consumes randomness.")
    parser.add_argument("--debug", action="store_true",
                        help="Prepend '_debug_' to the output folder name.")
    parser.add_argument("--replot", metavar="RUN_DIR", type=str, default=None,
                        help="Rebuild the figures from the saved artefacts of a run directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.replot is not None:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)sZ %(levelname)s [%(name)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%S", force=True)
        run_directory = Path(args.replot).resolve()
        if not run_directory.is_dir():
            parser.error(f"--replot: run directory does not exist: {run_directory}")
        for path in regenerate_figures(run_directory):
            logger.info("Replotted figure from saved artefacts: %s", path)
        return 0

    if (args.number_of_monte_carlo_samples < SMOKE_TEST_MONTE_CARLO_SAMPLE_THRESHOLD
            and not args.debug):
        parser.error(
            f"--number-of-monte-carlo-samples {args.number_of_monte_carlo_samples} is below "
            f"the smoke-test threshold {SMOKE_TEST_MONTE_CARLO_SAMPLE_THRESHOLD}; pass --debug"
        )

    start_wall_clock = time.perf_counter()
    debug_prefix = "_debug_" if args.debug else ""
    config_tag = f"mc{args.number_of_monte_carlo_samples}_K{'-'.join(map(str, args.band_edges))}"
    run_directory = script_data_dir(__file__) / f"{debug_prefix}{utc_timestamp()}_{config_tag}"
    run_directory.mkdir(parents=True, exist_ok=False)
    init_logging(run_dir=run_directory)

    logger.info("Command line: %s", " ".join(sys.argv))
    log_runtime_versions(logger)
    logger.info("Device: cpu (every computation of this script is in float64 on the CPU)")
    log_parsed_args(logger, args)
    monte_carlo_seed = derive_seed(args.seed, "monte_carlo_sampling")
    logger.info("Master seed: %d; derived Monte-Carlo sampler seed: %d", args.seed, monte_carlo_seed)

    repo_root = find_repo_root(Path(__file__))
    metadata = collect_run_metadata(
        run_dir=run_directory, repo_root=repo_root, script_name=Path(__file__).stem,
        command=sys.argv, params=dict(sorted(vars(args).items())),
        extra={"monte_carlo_seed": monte_carlo_seed, "singular_point": SINGULAR_POINT,
               "off_grid_fraction": OFF_GRID_FRACTION},
    )
    write_json(run_directory / "run_metadata.json", metadata)
    write_command_txt(run_directory / "command.txt", sys.argv)

    steps = np.logspace(-12.0, -1.0, args.number_of_steps)
    arrays = compute_difference_quotient_arrays(steps)
    arrays.update(compute_discrete_dirac_arrays(args.grid_spacings))
    autograd_arrays, autograd_scalars = compute_autograd_arrays(
        args.number_of_monte_carlo_samples, monte_carlo_seed, args.integral_grid_spacing)
    arrays.update(autograd_arrays)
    bandlimited_arrays, bandlimited_scalars = compute_bandlimited_arrays(
        args.band_edges, args.number_of_window_points, args.number_of_circle_points)
    arrays.update(bandlimited_arrays)

    # Part A summary: the measured quotients at three representative steps.
    representative_steps = {"1e-2": 1.0e-2, "1e-5": 1.0e-5, "1e-8": 1.0e-8}
    difference_quotient_summary: dict = {}
    for name in TEST_FUNCTION_NAMES:
        difference_quotient_summary[name] = {}
        for step_label, step in representative_steps.items():
            measured = measured_quotients(name, np.asarray([step]))
            predicted = closed_form_quotients(name, np.asarray([step]))
            difference_quotient_summary[name][step_label] = {
                f"{kind}_{quotient}": float(source[quotient][0])
                for quotient in ("forward", "backward", "centred", "second")
                for kind, source in (("measured", measured), ("closed_form", predicted))
            }
    discrete_dirac_summary = {
        f"{spacing:g}": {
            "mass_spacing_times_sum_of_second_differences":
                float(np.asarray(arrays[f"grid_mass_{index}"])[0]),
            "maximum_second_difference": float(np.max(arrays[f"grid_second_difference_{index}"])),
            "closed_form_maximum": (1.0 - OFF_GRID_FRACTION) / spacing,
        }
        for index, spacing in enumerate(args.grid_spacings)
    }

    for name in TEST_FUNCTION_NAMES:
        for step_label, values in difference_quotient_summary[name].items():
            logger.info(
                "Part A | %-30s h=%s | forward %.6e (closed form %.6e) | backward %.6e (%.6e) | "
                "centred %.6e (%.6e) | second %.6e (%.6e)",
                name, step_label, values["measured_forward"], values["closed_form_forward"],
                values["measured_backward"], values["closed_form_backward"],
                values["measured_centred"], values["closed_form_centred"],
                values["measured_second"], values["closed_form_second"])
    for spacing_label, values in discrete_dirac_summary.items():
        logger.info("Part B | spacing %s | spacing x sum of second differences %.15f | "
                    "maximum %.6e (closed form %.6e)", spacing_label,
                    values["mass_spacing_times_sum_of_second_differences"],
                    values["maximum_second_difference"], values["closed_form_maximum"])
    for program_name, convention in autograd_scalars["autograd_conventions_at_singular_point"].items():
        logger.info("Part C | %-30s | f'(0) = %g | f''(0) = %g | f''(0) graph-disconnected: %s | "
                    "f'(-1e-300) = %g | f'(+1e-300) = %g", program_name,
                    convention["first_derivative_at_zero"], convention["second_derivative_at_zero"],
                    convention["second_derivative_is_graph_disconnected_zero"],
                    convention["first_derivative_at_minus_one_e_minus_300"],
                    convention["first_derivative_at_plus_one_e_minus_300"])
    logger.info("Part C | %d uniform samples, %d equal to x* | autograd Monte-Carlo estimate of "
                "int f'' = %.6e | grid sum (x* on node) = %.15f | grid sum (x* off grid) = %.15f | "
                "distributional value = 1",
                autograd_scalars["number_of_monte_carlo_samples"],
                autograd_scalars["number_of_samples_equal_to_singular_point"],
                autograd_scalars["autograd_monte_carlo_estimate_of_integral_of_second_derivative"],
                autograd_scalars["grid_sum_estimate_singular_point_on_node"],
                autograd_scalars["grid_sum_estimate_singular_point_off_grid"])
    logger.info("Part D | exact datum: closed form vs library max deviation %.3e; autograd g'' "
                "max deviation from 1/(2 pi^2) %.3e",
                bandlimited_scalars["exact_datum_closed_form_versus_library_maximum_deviation"],
                bandlimited_scalars[
                    "exact_datum_autograd_second_derivative_maximum_deviation_from_regular_part"])
    for band_edge, values in bandlimited_scalars["band_edges"].items():
        logger.info("Part D | K = %d | autograd vs Dirichlet closed form max deviation %.3e | "
                    "g_K''(0) = %.6f (closed form -K/pi^2 = %.6f) | mass of oscillating part "
                    "%.12f (closed form -1/pi = %.12f) | sign-change spacing %.6f (closed form "
                    "pi/(K+1/2) = %.6f)", band_edge,
                    values["autograd_versus_dirichlet_closed_form_maximum_deviation"],
                    values["autograd_second_derivative_at_singular_point"],
                    values["closed_form_value_at_singular_point"],
                    values["measured_mass_of_oscillating_part_over_one_period"],
                    values["closed_form_mass_jump_of_first_derivative"],
                    values["measured_spacing_of_sign_changes_in_window"],
                    values["closed_form_spacing_pi_over_band_edge_plus_half"])

    np.savez_compressed(run_directory / ARRAYS_NAME, **arrays)
    summary = {
        "part_a_difference_quotients_at_singular_point": difference_quotient_summary,
        "part_b_discrete_dirac_mass": discrete_dirac_summary,
        "part_c_automatic_differentiation": autograd_scalars,
        "part_d_bandlimited_datum": bandlimited_scalars,
    }
    with open(run_directory / SUMMARY_NAME, "w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)
    for path in build_all_figures(arrays, run_directory):
        logger.info("Figure written: %s", path)

    logger.info("Total wall-clock time: %.2f s", time.perf_counter() - start_wall_clock)
    logger.info("Run directory: %s", run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
