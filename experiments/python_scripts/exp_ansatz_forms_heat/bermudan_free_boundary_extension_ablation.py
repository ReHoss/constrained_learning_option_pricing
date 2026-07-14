r"""Bermudan free-boundary extension ablation: which terminal-data extension?

What is measured, and why.
--------------------------

A Bermudan option is priced by a chain of hard-terminal-constraint solves glued at
each exercise date by the pointwise maximum of the payoff and the continuation
value.  The maximum **manufactures** a first-derivative discontinuity --- of
regularity order one whatever the regularity of the payoff --- and places it on the
**free boundary** :math:`\Gamma = \{\payoff = C\}`, a set that is unknown a priori,
moves from stage to stage, and is itself only approximately known.  The extension of
the stage datum into the interior of the strip is what the residual objective sees,
and its choice is what sets the per-stage error injected into the chain.

This script isolates that choice, in the smallest chain in which a free boundary
exists (:math:`M = 2` exercise dates).  The post-exercise stage is reduced to the
**closed-form** Black--Scholes European put, so the exact continuation
:math:`C^{\star}_{1}`, the free boundary :math:`\Gamma_{1}` and the jump
:math:`J_{1}` of the stage datum's first derivative there are all exactly known, and
every diagnostic has an exact reference.  The single trained stage is
:math:`[0, t_{1}]`, with terminal datum
:math:`V_{1} = \max(\payoff, C^{\star}_{1})`.

The five variants are the rows of the theory's ledger
(see :mod:`_bermudan_extension_catalogue`).  Two of the properties that first
suggest themselves as the mark of a good extension **do not discriminate**, and the
ablation is designed to show it:

* *exactness at the slice* (hence a vanishing smoothing bias) is achieved by four of
  the five variants --- it rests only on a semigroup at parameter zero being the
  identity;
* *square-integrability of the forcing* is achieved by the mis-specified split too.

The property that singles out the matched split is **boundedness** of the forcing,
from which follow a finite-variance Monte-Carlo estimator of the objective and a
**bounded target** for the network.  The discriminating measurements are therefore
the growth of :math:`\sup|\mathcal{L} h|` as the slice is approached, the seed
dispersion of the objective, and the deviation from the exact minimiser --- not the
forcing floor.

The generator is the genuine Black--Scholes operator with a strictly positive rate:
on the pure-heat generator of the companion induction experiments the defect part of
the split is zero, the matched split extension coincides with the exact stage
solution, and the ablation would be degenerate.

Disclosure.  In the constant-coefficient Black--Scholes model the principal and
defect parts commute, so the split extension is an explicit shift and discount away
from the exact stage solution (see
:mod:`learning_option_pricing.pde.black_scholes_references`).  The learned solve is
therefore *redundant* in this model, which is a **diagnostic with a known answer**,
not an application: the exact minimiser is available in closed form, which is what
makes every measurement below a falsification test rather than a demonstration.

Reproduction
------------

::

    python bermudan_free_boundary_extension_ablation.py \
        --variant split_matched --seed 0 --num-iterations 20000

    # replot from saved artefacts, no training:
    python bermudan_free_boundary_extension_ablation.py --replot <run_dir>

    # cluster job array (init writes one YAML per (variant, seed)):
    python bermudan_free_boundary_extension_ablation.py --init-only
    python bermudan_free_boundary_extension_ablation.py \
        --config-dir <expdir>/configs --config-name task_000.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bermudan_extension_catalogue as cat  # noqa: E402

logger = logging.getLogger("bermudan_free_boundary_extension")

assert cat.RUNNER_SCRIPT_STEM == Path(__file__).stem, (
    "the catalogue's RUNNER_SCRIPT_STEM has drifted from this script's filename; "
    "the output folder is derived from the filename and the two must agree."
)


# ===========================================================================
# Problem construction
# ===========================================================================


def _exact_continuation_fn():
    """The exact continuation value at t_1: the Black--Scholes European put."""
    import torch

    from learning_option_pricing.pde import black_scholes_put_exact

    def continuation(x: "torch.Tensor") -> "torch.Tensor":
        return black_scholes_put_exact(
            x,
            torch.full_like(x, cat.INTERMEDIATE_EXERCISE_DATE),
            K=cat.STRIKE,
            T=cat.MATURITY,
            volatility=cat.VOLATILITY,
            risk_free_rate=cat.RISK_FREE_RATE,
        )

    return continuation


def _stage_geometry():
    """The trained stage: local time s in [0, T_stage], datum imposed at T_stage."""
    return {
        "stage_terminal_time": cat.INTERMEDIATE_EXERCISE_DATE,
        "generator_coefficients": None,  # filled below (torch-free catalogue)
    }


def build_problem(variant: dict) -> dict:
    """Assemble the stage-1 problem for one variant.

    Returns the terminal datum, the extension field (or ``None``), the analytic
    derivative callables (or ``None``), the exact reference and the free-boundary
    data.  The generator is Black--Scholes; the interior sampler excludes the
    terminal sliver on which the extension's fixed-grid quadrature is unresolved
    (see :mod:`learning_option_pricing.pde.real_line_extension_fields`).
    """
    from learning_option_pricing.pde import (
        GaussianSemigroupExtensionField,
        GradedChenMangasarianExtensionField,
        bermudan_exercise_boundary,
        black_scholes_generator_coefficients,
        constant_chen_mangasarian_datum,
        exact_maximum_datum,
    )

    stage_terminal_time = cat.INTERMEDIATE_EXERCISE_DATE
    continuation = _exact_continuation_fn()

    # --- the free boundary, exactly located -------------------------------
    free_boundary = bermudan_exercise_boundary(
        continuation, K=cat.STRIKE, x_lo=cat.LOG_PRICE_LO, x_hi=cat.LOG_PRICE_HI
    )
    derivative_jump = _datum_derivative_jump(continuation, free_boundary)

    # --- the stage terminal datum -----------------------------------------
    if variant["datum"] == "chen_mangasarian_constant":
        terminal_datum = constant_chen_mangasarian_datum(
            continuation, K=cat.STRIKE, smoothing_scale=cat.CONSTANT_SMOOTHING_SCALE
        )
    elif variant["datum"] == "exact_maximum":
        terminal_datum = exact_maximum_datum(continuation, K=cat.STRIKE)
    else:
        raise KeyError(f"unknown datum kind {variant['datum']!r}")

    # --- the extension ----------------------------------------------------
    extension_field = None
    extension_fn = None
    extension_derivative_fns = None
    own_quadrature_floor = 0.0

    if variant["extension"] == "gaussian_semigroup":
        comparison_volatility = (
            float(variant["comparison_volatility_ratio"]) * cat.VOLATILITY
        )
        extension_field = GaussianSemigroupExtensionField(
            terminal_datum,
            terminal_time=stage_terminal_time,
            comparison_volatility=comparison_volatility,
            y_lo=cat.QUADRATURE_LO,
            y_hi=cat.QUADRATURE_HI,
            n_quad=cat.EXTENSION_QUADRATURE_NODES,
            name=variant["name"],
        )
        extension_fn = extension_field.field
        if variant["analytic_derivatives"]:
            extension_derivative_fns = extension_field.derivative_callables()
        own_quadrature_floor = extension_field.time_to_terminal_floor
    elif variant["extension"] == "graded_chen_mangasarian":
        extension_field = GradedChenMangasarianExtensionField(
            continuation,
            K=cat.STRIKE,
            terminal_time=stage_terminal_time,
            smoothing_scale=cat.CONSTANT_SMOOTHING_SCALE,
            grading_exponent=float(variant["grading_exponent"]),
            name=variant["name"],
        )
        extension_fn = extension_field.field
    elif variant["extension"] is not None:
        raise KeyError(f"unknown extension kind {variant['extension']!r}")

    # The excluded terminal sliver is a SINGLE constant, identical for every variant,
    # so that the samplers do not differ across the ablation axis.  It must dominate
    # this variant's own unresolved-quadrature floor, or the extension would be
    # evaluated where its quadrature cannot resolve the kernel.
    if own_quadrature_floor > cat.EXCLUDED_TERMINAL_SLIVER:
        raise ValueError(
            f"variant {variant['name']!r} has an unresolved-quadrature floor "
            f"{own_quadrature_floor:.3e} exceeding the shared excluded sliver "
            f"{cat.EXCLUDED_TERMINAL_SLIVER:.3e}. Raise EXCLUDED_TERMINAL_SLIVER or "
            "EXTENSION_QUADRATURE_NODES; do not let the sampler differ across "
            "variants, which would make the comparison unfair."
        )

    return {
        "stage_terminal_time": stage_terminal_time,
        "generator_coefficients": black_scholes_generator_coefficients(
            volatility=cat.VOLATILITY, risk_free_rate=cat.RISK_FREE_RATE
        ),
        "terminal_datum": terminal_datum,
        "continuation": continuation,
        "extension_field": extension_field,
        "extension_fn": extension_fn,
        "extension_derivative_fns": extension_derivative_fns,
        "free_boundary": free_boundary,
        "derivative_jump": derivative_jump,
        # The interior sampler excludes the terminal sliver on which the extension's
        # fixed-grid quadrature cannot resolve the Gaussian kernel.  A SINGLE constant,
        # identical for every variant, so the samplers do not differ across the
        # ablation axis; the excluded fraction is logged and written to the metadata.
        "excluded_terminal_sliver": cat.EXCLUDED_TERMINAL_SLIVER,
        "own_quadrature_floor": own_quadrature_floor,
    }


def _datum_derivative_jump(continuation, free_boundary: float) -> float:
    r"""The jump :math:`J = |\partial_x \payoff - \partial_x C|` at the corner."""
    import torch

    from learning_option_pricing.pde import heat_put_payoff

    point = torch.tensor([free_boundary], dtype=torch.float64, requires_grad=True)
    difference = heat_put_payoff(point, cat.STRIKE) - continuation(point)
    (slope,) = torch.autograd.grad(difference, point, torch.ones_like(difference))
    return abs(float(slope))


# ===========================================================================
# Ansatz and training
# ===========================================================================


def derive_seed(master_seed: int, role: str) -> int:
    """Deterministically derive a per-role seed from the master seed."""
    import hashlib

    digest = hashlib.blake2b(
        f"{master_seed}:{role}".encode(), digest_size=8
    ).hexdigest()
    return int(digest, 16) % (2**31 - 1)


def build_ansatz(variant: dict, problem: dict, hparams: dict, *, model_seed: int):
    """Construct the trial solution for one variant."""
    import torch

    from learning_option_pricing.models.resnet import ResNet
    from learning_option_pricing.models.terminal_ansatz import (
        TerminalAnsatz,
        make_interpolation_coefficient,
    )

    torch.manual_seed(model_seed)
    network = ResNet(
        d_in=2,
        d_out=1,
        n=int(hparams["net_width"]),
        M=int(hparams["net_blocks"]),
        L=int(hparams["net_layers_per_block"]),
    )

    stage_terminal_time = problem["stage_terminal_time"]
    interp_coeff = make_interpolation_coefficient(
        variant["interpolation"], T=stage_terminal_time, sigma=cat.VOLATILITY
    )

    x_mid = 0.5 * (cat.LOG_PRICE_LO + cat.LOG_PRICE_HI)
    x_half = 0.5 * (cat.LOG_PRICE_HI - cat.LOG_PRICE_LO)

    def normalizer(xt):
        x = (xt[:, 0:1] - x_mid) / x_half
        t = 2.0 * xt[:, 1:2] / stage_terminal_time - 1.0
        return torch.cat([x, t], dim=1)

    return TerminalAnsatz(
        network,
        problem["terminal_datum"],
        interp_coeff,
        form=variant["form"],
        normalizer=normalizer,
        extension_fn=problem["extension_fn"],
        extension_derivative_fns=problem["extension_derivative_fns"],
    )


def make_interior_sampler(problem: dict, hparams: dict, *, sampler_seed: int, device):
    r"""Interior collocation, excluding the unresolved-quadrature terminal sliver.

    Time is sampled uniformly on
    :math:`[0, T - \tau_{\mathrm{floor}}]`.  The exclusion is applied to every
    variant identically --- so the comparison is fair --- and the excluded fraction
    is logged.  It is required because the extension's fixed-grid quadrature cannot
    resolve the Gaussian kernel closer than :math:`\tau_{\mathrm{floor}}` to the
    slice; sampling there would silently return the un-smoothed (kinked) datum and
    the second-derivative channel of the residual would be wrong.  Note that the
    terminal condition itself is *not* affected: it is structural in the ansatz, not
    sampled.
    """
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampler_seed)
    n_interior = int(hparams["n_interior"])
    stage_terminal_time = problem["stage_terminal_time"]
    time_upper = stage_terminal_time - problem["excluded_terminal_sliver"]

    def sample_interior():
        x = (
            cat.LOG_PRICE_LO
            + (cat.LOG_PRICE_HI - cat.LOG_PRICE_LO)
            * torch.rand(n_interior, generator=generator)
        ).to(device).requires_grad_(True)
        t = (time_upper * torch.rand(n_interior, generator=generator)).to(
            device
        ).requires_grad_(True)
        return x, t

    return sample_interior


def train_variant(variant, problem, hparams, *, num_iterations, seed, device, log_every):
    """Train one variant on the single trained stage; return (model, history)."""
    import torch

    from learning_option_pricing.models.terminal_ansatz import (
        cross_check_extension_forcing_analytic_versus_autograd,
        residual_decomposition,
    )

    generator_coefficients = problem["generator_coefficients"]
    model_seed = derive_seed(seed, "model_init")
    sampler_seed = derive_seed(seed, "sampler")

    model = build_ansatz(variant, problem, hparams, model_seed=model_seed).to(device)
    sample_interior = make_interior_sampler(
        problem, hparams, sampler_seed=sampler_seed, device=device
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "[%s] form=%s datum=%s extension=%s analytic_derivatives=%s params=%d "
        "seeds(model=%d, sampler=%d)",
        variant["name"],
        variant["form"],
        variant["datum"],
        variant["extension"],
        variant["analytic_derivatives"],
        n_params,
        model_seed,
        sampler_seed,
    )
    logger.info(
        "generator (Black--Scholes, log-price): nu=%.6f  (r-nu)=%.6f  -r=%.6f ; "
        "defect part B is of differential order 1",
        generator_coefficients[2],
        generator_coefficients[1],
        generator_coefficients[0],
    )
    logger.info(
        "free boundary x* = %.8f (S* = %.4f); datum derivative jump J = %.8f",
        problem["free_boundary"],
        math.exp(problem["free_boundary"]),
        problem["derivative_jump"],
    )
    logger.info(
        "interior sampler excludes the terminal sliver of width %.3e (%.3f%% of the "
        "stage interval), identically for every variant; this variant's own "
        "unresolved-quadrature floor is %.3e",
        problem["excluded_terminal_sliver"],
        100.0
        * problem["excluded_terminal_sliver"]
        / problem["stage_terminal_time"],
        problem["own_quadrature_floor"],
    )

    # Startup guard: when the analytic derivative bypass is on, the analytic and the
    # autograd assemblies of the extension forcing must agree, or the run is aborted.
    if problem["extension_derivative_fns"] is not None:
        x_guard, t_guard = sample_interior()
        deviation = cross_check_extension_forcing_analytic_versus_autograd(
            model,
            x_guard.detach().reshape(-1, 1),
            t_guard.detach().reshape(-1, 1),
            generator_coefficients=generator_coefficients,
            relative_tolerance=1.0e-3,
        )
        logger.info(
            "analytic-vs-autograd extension-forcing cross-check passed "
            "(relative deviation %.3e)",
            deviation,
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(hparams["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations
    )

    history = {
        key: []
        for key in (
            "iter",
            "loss",
            "network_energy",
            "cross_term",
            "forcing_floor",
            "grad_norm",
            "lr",
        )
    }
    best_loss = math.inf
    best_iter = -1
    best_state = None

    for iteration in range(1, num_iterations + 1):
        optimizer.zero_grad()
        x_f, t_f = sample_interior()
        decomposition = residual_decomposition(
            model, x_f, t_f, generator_coefficients=generator_coefficients
        )
        loss = decomposition["loss"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e12)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss)
        if loss_value < best_loss:
            best_loss = loss_value
            best_iter = iteration
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if iteration % log_every == 0 or iteration == 1:
            history["iter"].append(iteration)
            history["loss"].append(loss_value)
            history["network_energy"].append(float(decomposition["network_energy"]))
            history["cross_term"].append(float(decomposition["cross_term"]))
            history["forcing_floor"].append(float(decomposition["forcing_floor"]))
            history["grad_norm"].append(float(grad_norm))
            history["lr"].append(float(scheduler.get_last_lr()[0]))
            logger.info(
                "iter %6d  loss %.6e  network %.6e  cross %+.6e  floor %.6e  lr %.2e",
                iteration,
                loss_value,
                float(decomposition["network_energy"]),
                float(decomposition["cross_term"]),
                float(decomposition["forcing_floor"]),
                float(scheduler.get_last_lr()[0]),
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_loss"] = best_loss
    history["best_iter"] = best_iter
    logger.info(
        "restored the best-objective state: loss %.6e at iteration %d",
        best_loss,
        best_iter,
    )
    return model, history


# ===========================================================================
# Diagnostics.  Every metric is given by its formula in the report; here each is
# computed against an EXACT reference, since the m = 2 geometry makes one available.
# ===========================================================================


def _relative_error(approx, exact):
    import torch

    return float(torch.linalg.norm(approx - exact) / torch.linalg.norm(exact))


def evaluate(model, variant, problem, hparams) -> dict:
    """Every diagnostic of the ablation, computed against the exact reference."""
    import numpy as np
    import torch

    from learning_option_pricing.models.terminal_ansatz import residual_decomposition
    from learning_option_pricing.pde import (
        bermudan_put_value_exact_black_scholes,
        heat_put_payoff,
    )

    model = model.double().eval()
    generator_coefficients = problem["generator_coefficients"]
    stage_terminal_time = problem["stage_terminal_time"]
    diffusivity = generator_coefficients[2]

    x_eval = torch.linspace(cat.EVALUATION_LO, cat.EVALUATION_HI, 601, dtype=torch.float64)
    results: dict = {}

    # --- 1. slice bias: || Phi(., T) - max(g, C*) ||_inf ------------------
    # Predicted identically zero for every variant whose datum is the exact maximum,
    # and bounded by eps_0 / 2 for the constant-smoothed baseline. A NON-ZERO value
    # for `split_matched` REFUTES Proposition (split extension), part (i).
    with torch.no_grad():
        xt_slice = torch.stack(
            [x_eval, torch.full_like(x_eval, stage_terminal_time)], dim=1
        )
        trial_at_slice = model(xt_slice).reshape(-1)
    exact_datum = torch.maximum(
        heat_put_payoff(x_eval, cat.STRIKE), problem["continuation"](x_eval)
    )
    results["slice_bias_sup"] = float((trial_at_slice - exact_datum).abs().max())

    # --- 2. THE DISCRIMINANT: sup |L h| as the slice is approached ---------
    # Predicted flat in s for the matched split; ~ s^{-1/2} for the mis-specified
    # split; ~ s^{-1} for the linear-graded mollifier; and unbounded for the exact
    # datum with no interior profile.
    forcing_suprema = []
    if problem["extension_fn"] is not None or variant["datum"] == "exact_maximum":
        for time_to_terminal in (0.2, 0.05, 0.0125, 0.003125):
            if time_to_terminal <= problem["own_quadrature_floor"]:
                forcing_suprema.append(
                    {"time_to_terminal": time_to_terminal, "sup_forcing": None}
                )
                continue
            x_probe = torch.linspace(
                cat.EVALUATION_LO, cat.EVALUATION_HI, 4001, dtype=torch.float64
            ).requires_grad_(True)
            t_probe = torch.full_like(
                x_probe, stage_terminal_time - time_to_terminal
            ).requires_grad_(True)
            decomposition = residual_decomposition(
                model,
                x_probe,
                t_probe,
                generator_coefficients=generator_coefficients,
            )
            forcing = decomposition["extension_forcing"].detach().abs()
            forcing_suprema.append(
                {
                    "time_to_terminal": time_to_terminal,
                    "sup_forcing": float(forcing.max()),
                }
            )
    results["forcing_supremum_profile"] = forcing_suprema
    # The measured growth exponent p of sup|L h| ~ s^{-p}: predicted 0 (matched),
    # 1/2 (mis-specified), 1 (graded).
    #
    # TWO estimates are reported, and both are needed. The forcing is the sum of a
    # singular channel and a BOUNDED one, which near the corner have opposite sign and
    # partially cancel; the cancellation is proportionally stronger at the larger s,
    # where the singular channel is smaller. A fit over the whole probe range is
    # therefore contaminated by the bounded channel and understates the exponent, while
    # a fit over the two smallest s -- the asymptotic regime the proposition is about
    # -- does not. Neither estimate replaces the raw profile, which is saved in full.
    def _exponent(entries) -> float | None:
        if len(entries) < 2:
            return None
        first, last = entries[0], entries[-1]
        return float(
            math.log(last["sup_forcing"] / first["sup_forcing"])
            / math.log(first["time_to_terminal"] / last["time_to_terminal"])
        )

    measured = [
        entry for entry in forcing_suprema if entry["sup_forcing"] is not None
    ]
    results["forcing_divergence_exponent_full_range"] = _exponent(measured)
    results["forcing_divergence_exponent_asymptotic"] = _exponent(measured[-2:])
    # Kept under the plain name for the plot and the log line; the asymptotic estimate
    # is the one the proposition predicts.
    results["forcing_divergence_exponent"] = results[
        "forcing_divergence_exponent_asymptotic"
    ]

    # --- 3. the achieved stage objective and its estimator dispersion ------
    # The seed-to-seed and draw-to-draw dispersion of the Monte-Carlo objective is
    # the observable signature of the INFINITE variance predicted for the
    # mis-specified split and for every mollifier grading.
    objective_draws = []
    for draw in range(8):
        sampler = make_interior_sampler(
            problem,
            hparams,
            sampler_seed=derive_seed(1000 + draw, "objective_draw"),
            device=torch.device("cpu"),
        )
        x_d, t_d = sampler()
        decomposition = residual_decomposition(
            model,
            x_d.double(),
            t_d.double(),
            generator_coefficients=generator_coefficients,
        )
        objective_draws.append(float(decomposition["loss"]))
    objective_draws_array = np.asarray(objective_draws)
    results["objective_draws"] = objective_draws
    results["objective_mean"] = float(objective_draws_array.mean())
    results["objective_relative_dispersion"] = float(
        objective_draws_array.std() / max(objective_draws_array.mean(), 1e-300)
    )

    # --- 4. price, Greeks, and the corner windows -------------------------
    x_grad = x_eval.clone().requires_grad_(True)
    xt_inception = torch.stack([x_grad, torch.zeros_like(x_grad)], dim=1)
    price = model(xt_inception).reshape(-1)
    (dprice_dx,) = torch.autograd.grad(
        price, x_grad, torch.ones_like(price), create_graph=True
    )
    (d2price_dx2,) = torch.autograd.grad(
        dprice_dx, x_grad, torch.ones_like(dprice_dx), create_graph=True
    )
    spot = torch.exp(x_eval)
    # Greeks in the price variable: Delta = d_x V / S ; Gamma = (d_xx V - d_x V) / S^2.
    delta_net = (dprice_dx / spot).detach()
    gamma_net = ((d2price_dx2 - dprice_dx) / spot**2).detach()

    exact_reference = bermudan_put_value_exact_black_scholes(
        x_eval,
        torch.tensor(0.0, dtype=torch.float64),
        exercise_times=[cat.INTERMEDIATE_EXERCISE_DATE, cat.MATURITY],
        K=cat.STRIKE,
        volatility=cat.VOLATILITY,
        risk_free_rate=cat.RISK_FREE_RATE,
        y_lo=cat.QUADRATURE_LO,
        y_hi=cat.QUADRATURE_HI,
        n_quad=cat.REFERENCE_QUADRATURE_NODES,
    )
    delta_exact, gamma_exact = _reference_greeks(x_eval)

    results["inception_relative_error"] = _relative_error(
        price.detach(), exact_reference
    )
    results["delta_relative_error"] = _relative_error(delta_net, delta_exact)
    results["gamma_relative_error"] = _relative_error(gamma_net, gamma_exact)

    corner = problem["free_boundary"]
    results["corner_windows"] = []
    for half_width in cat.CORNER_WINDOW_HALF_WIDTHS:
        mask = (x_eval - corner).abs() <= half_width
        if int(mask.sum()) < 3:
            continue
        results["corner_windows"].append(
            {
                "half_width": half_width,
                "n_points": int(mask.sum()),
                "price_relative_error": _relative_error(
                    price.detach()[mask], exact_reference[mask]
                ),
                "delta_relative_error": _relative_error(
                    delta_net[mask], delta_exact[mask]
                ),
                "gamma_relative_error": _relative_error(
                    gamma_net[mask], gamma_exact[mask]
                ),
            }
        )

    # --- 5. the certificate ------------------------------------------------
    # ||e_0||_inf <= sqrt(2) (8 pi nu)^{-1/4} Delta^{1/4} || L Phi ||_{L2}.
    # Predicted to hold (ratio at most one); a ratio ABOVE one refutes the transfer
    # proposition.
    inception_error_sup = float((price.detach() - exact_reference).abs().max())
    strip_area = (cat.LOG_PRICE_HI - cat.LOG_PRICE_LO) * stage_terminal_time
    residual_l2 = math.sqrt(max(results["objective_mean"], 0.0) * strip_area)
    certificate = (
        math.sqrt(2.0)
        * (8.0 * math.pi * diffusivity) ** (-0.25)
        * stage_terminal_time**0.25
        * residual_l2
    )
    results["inception_error_sup"] = inception_error_sup
    results["certificate_bound"] = certificate
    results["certificate_tightness"] = (
        inception_error_sup / certificate if certificate > 0 else None
    )

    # --- 6. the quadrature-floor tally (never silent) ----------------------
    if problem["extension_field"] is not None and hasattr(
        problem["extension_field"], "quadrature_floor_report"
    ):
        results["quadrature_floor_report"] = problem[
            "extension_field"
        ].quadrature_floor_report()

    # --- 7. arrays for replotting -----------------------------------------
    results["_arrays"] = {
        "x": x_eval.numpy(),
        "spot": spot.numpy(),
        "price_net": price.detach().numpy(),
        "price_exact": exact_reference.numpy(),
        "delta_net": delta_net.numpy(),
        "delta_exact": delta_exact.numpy(),
        "gamma_net": gamma_net.numpy(),
        "gamma_exact": gamma_exact.numpy(),
        "payoff": heat_put_payoff(x_eval, cat.STRIKE).numpy(),
    }
    return results


def _reference_greeks(x_eval):
    """Exact Delta and Gamma of the Bermudan reference, by autograd on the reference."""
    import torch

    from learning_option_pricing.pde import bermudan_put_value_exact_black_scholes

    x = x_eval.clone().requires_grad_(True)
    value = bermudan_put_value_exact_black_scholes(
        x,
        torch.tensor(0.0, dtype=torch.float64),
        exercise_times=[cat.INTERMEDIATE_EXERCISE_DATE, cat.MATURITY],
        K=cat.STRIKE,
        volatility=cat.VOLATILITY,
        risk_free_rate=cat.RISK_FREE_RATE,
        y_lo=cat.QUADRATURE_LO,
        y_hi=cat.QUADRATURE_HI,
        n_quad=cat.REFERENCE_QUADRATURE_NODES,
    )
    (dv_dx,) = torch.autograd.grad(
        value, x, torch.ones_like(value), create_graph=True
    )
    (d2v_dx2,) = torch.autograd.grad(
        dv_dx, x, torch.ones_like(dv_dx), create_graph=True
    )
    spot = torch.exp(x_eval)
    return (dv_dx / spot).detach(), ((d2v_dx2 - dv_dx) / spot**2).detach()


# ===========================================================================
# Plots
# ===========================================================================


def plot(arrays, results, variant, out_dir):
    """Rebuild every figure from saved arrays; never recompute."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    spot = arrays["spot"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel 1: price. Solid = trained; dashed = exact reference; dotted = payoff.
    ax = axes[0]
    ax.plot(spot, arrays["price_net"], "-", color=variant["color"], label="Trained")
    ax.plot(spot, arrays["price_exact"], "--", color="black", label="Exact Bermudan")
    ax.plot(spot, arrays["payoff"], ":", color="grey", label="Payoff $(K-s)^+$")
    ax.set_xlabel("Spot $s$")
    ax.set_ylabel("Price at inception")
    ax.set_title("Price")

    # Panel 2: Gamma -- the order at which the corner is felt.
    ax = axes[1]
    ax.plot(spot, arrays["gamma_net"], "-", color=variant["color"], label="Trained")
    ax.plot(spot, arrays["gamma_exact"], "--", color="black", label="Exact")
    ax.set_xlabel("Spot $s$")
    ax.set_ylabel(r"$\Gamma$")
    ax.set_title(r"Second Greek $\Gamma$")

    # Panel 3: THE DISCRIMINANT -- sup|L h| against the distance to the slice.
    ax = axes[2]
    profile = [e for e in results["forcing_supremum_profile"] if e["sup_forcing"]]
    if profile:
        times = np.array([e["time_to_terminal"] for e in profile])
        suprema = np.array([e["sup_forcing"] for e in profile])
        ax.loglog(times, suprema, "-o", color=variant["color"], label="Measured")
        exponent = results.get("forcing_divergence_exponent")
        if exponent is not None:
            ax.loglog(
                times,
                suprema[0] * (times / times[0]) ** (-exponent),
                "--",
                color="black",
                label=rf"Fitted slope $-{exponent:.2f}$",
            )
    ax.set_xlabel(r"Time to the terminal slice $s=T-t$")
    ax.set_ylabel(r"$\sup_x |\mathcal{L}h(x,\,T-s)|$")
    ax.set_title("Extension forcing near the slice")

    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)

    fig.suptitle(variant["label"], fontsize=11)
    formula = (
        r"$\Phi_\theta=\mathrm{d}_T\Psi_\theta+h$;  "
        r"$\mathcal{L}=\partial_t+\nu\partial_{xx}+(r-\nu)\partial_x-r$,  "
        r"$\nu=\sigma^2/2$;  split: $h(\cdot,t)=e^{(T-t)\nu_c\partial_{xx}}"
        r"\max(g,C^\star)$;  "
        r"$\mathcal{L}h=(\nu-\nu_c)\partial_{xx}h+\mathcal{B}h$,  "
        r"$\mathcal{B}=(r-\nu)\partial_x-r$"
    )
    fig.text(0.5, 0.01, formula, ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    fig.savefig(out_dir / "extension_diagnostics.png", dpi=150)
    plt.close(fig)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", choices=cat.variant_names(), default="split_matched")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-iterations", type=int, default=cat.DEFAULT_HPARAMS["num_iterations"]
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument(
        "--debug",
        action="store_true",
        help="prefix the output folder with _debug_ (mandatory for smoke tests)",
    )
    p.add_argument("--replot", type=str, default=None, help="run dir to replot")
    p.add_argument(
        "--init-only",
        action="store_true",
        help="write one YAML per (variant, seed) task and print the ablation dir",
    )
    p.add_argument("--config-dir", type=str, default=None)
    p.add_argument("--config-name", type=str, default=None)
    p.add_argument("--ablation-dir", type=str, default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    from learning_option_pricing.utils.run_context import (
        log_runtime_versions,
        script_data_dir,
        utc_timestamp,
    )

    # --- replot: torch-free, bypasses the smoke guard ----------------------
    if args.replot:
        import numpy as np
        import yaml

        run_dir = Path(args.replot)
        arrays = dict(np.load(run_dir / "diagnostics.npz"))
        with open(run_dir / "metadata.yaml") as f:
            meta = yaml.safe_load(f)
        plot(arrays, meta["results"], cat.variant_by_name(meta["variant"]), run_dir)
        logger.info("replotted %s", run_dir)
        return 0

    # --- init-only: write the per-task configs for the job array -----------
    if args.init_only:
        import yaml

        debug_prefix = "_debug_" if args.debug else ""
        ablation_dir = script_data_dir(__file__) / (
            f"{debug_prefix}{utc_timestamp()}_free_boundary_extension"
        )
        configs_dir = ablation_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        for index, task in enumerate(cat.array_tasks()):
            payload = {
                "variant": task["variant"],
                "seed": task["seed"],
                "num_iterations": args.num_iterations,
                "device": args.device,
                "log_every": args.log_every,
                "debug": args.debug,
                "ablation_dir": str(ablation_dir.resolve()),
            }
            with open(configs_dir / f"task_{index:03d}.yaml", "w") as f:
                yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
        logger.info(
            "wrote %d task configs to %s", len(cat.array_tasks()), configs_dir
        )
        print(str(ablation_dir.resolve()))  # last stdout line: the launcher reads it
        return 0

    # --- config-driven task (the array worker) -----------------------------
    ablation_dir = None
    if args.config_dir and args.config_name:
        import yaml

        with open(Path(args.config_dir) / args.config_name) as f:
            cfg = yaml.safe_load(f)
        args.variant = cfg["variant"]
        args.seed = int(cfg["seed"])
        args.num_iterations = int(cfg["num_iterations"])
        args.device = cfg.get("device", args.device)
        args.log_every = int(cfg.get("log_every", args.log_every))
        args.debug = bool(cfg.get("debug", False))
        ablation_dir = Path(cfg["ablation_dir"])

    # --- smoke-test guard --------------------------------------------------
    if args.num_iterations < cat.SMOKE_TEST_NUM_ITERATIONS_THRESHOLD and not args.debug:
        raise SystemExit(
            f"--num-iterations {args.num_iterations} is below the smoke-test "
            f"threshold ({cat.SMOKE_TEST_NUM_ITERATIONS_THRESHOLD}); pass --debug to "
            "flag this as exploratory, or raise the iteration count."
        )

    import numpy as np
    import torch
    import yaml

    variant = cat.variant_by_name(args.variant)
    hparams = dict(cat.DEFAULT_HPARAMS)
    hparams["num_iterations"] = args.num_iterations
    device = torch.device(args.device)

    debug_prefix = "_debug_" if args.debug else ""
    parent = ablation_dir if ablation_dir is not None else script_data_dir(__file__)
    out_dir = parent / (
        f"{debug_prefix}{utc_timestamp()}_{args.variant}"
        f"_iters{args.num_iterations}_seed{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("command line: %s", " ".join(sys.argv))
    log_runtime_versions(logger)
    logger.info("output directory: %s", out_dir)
    logger.info("hyperparameters: %s", hparams)
    logger.info("master seed: %d", args.seed)

    torch.set_default_dtype(torch.float64)
    problem = build_problem(variant)

    started = time.time()
    model, history = train_variant(
        variant,
        problem,
        hparams,
        num_iterations=args.num_iterations,
        seed=args.seed,
        device=device,
        log_every=args.log_every,
    )
    wall = time.time() - started

    torch.save(model.state_dict(), out_dir / "model.pt")
    logger.info("checkpoint saved: %s", out_dir / "model.pt")

    results = evaluate(model, variant, problem, hparams)
    arrays = results.pop("_arrays")
    np.savez_compressed(out_dir / "diagnostics.npz", **arrays)
    plot(arrays, results, variant, out_dir)

    meta = {
        "variant": args.variant,
        "variant_label": variant["label"],
        "seed": args.seed,
        "num_iterations": args.num_iterations,
        "hparams": hparams,
        "strike": cat.STRIKE,
        "volatility": cat.VOLATILITY,
        "risk_free_rate": cat.RISK_FREE_RATE,
        "maturity": cat.MATURITY,
        "intermediate_exercise_date": cat.INTERMEDIATE_EXERCISE_DATE,
        "free_boundary_log_price": problem["free_boundary"],
        "free_boundary_spot": math.exp(problem["free_boundary"]),
        "datum_derivative_jump": problem["derivative_jump"],
        "excluded_terminal_sliver": problem["excluded_terminal_sliver"],
        "own_quadrature_floor": problem["own_quadrature_floor"],
        "wall_time_s": wall,
        "best_loss": history["best_loss"],
        "best_iter": history["best_iter"],
        "history": {k: v for k, v in history.items() if isinstance(v, list)},
        "results": results,
    }
    with open(out_dir / "metadata.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

    logger.info("=" * 78)
    logger.info("wall-clock %.1f s (%.4f s/iter)", wall, wall / args.num_iterations)
    logger.info("best loss %.6e at iteration %d", history["best_loss"], history["best_iter"])
    logger.info("slice bias (sup)            %.6e", results["slice_bias_sup"])
    logger.info(
        "forcing divergence exponent: asymptotic %s, full-range %s "
        "(predicted: 0 matched, 1/2 mis-specified, 1 graded)",
        results["forcing_divergence_exponent_asymptotic"],
        results["forcing_divergence_exponent_full_range"],
    )
    logger.info("forcing supremum profile: %s", results["forcing_supremum_profile"])
    logger.info(
        "objective relative dispersion over draws %.4f",
        results["objective_relative_dispersion"],
    )
    logger.info("inception relative error   %.6e", results["inception_relative_error"])
    logger.info("Gamma relative error       %.6e", results["gamma_relative_error"])
    logger.info("certificate tightness      %s", results["certificate_tightness"])
    if "quadrature_floor_report" in results:
        logger.info("quadrature floor: %s", results["quadrature_floor_report"])
    logger.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
