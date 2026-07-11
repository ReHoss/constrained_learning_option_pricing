r"""Trial solutions for terminal-condition enforcement.

This module implements the four ansatz forms compared in the boundary-constrained
learning study (report
``2026_01_29_constrained_learning_pde_lehalle_hosseinkhan``).  Writing the free
network as :math:`\Phi_\theta(x, t)`, the interpolation coefficient as
:math:`\lambda(t)` (with :math:`\lambda(T) = 1`, so the network prefactor
:math:`1 - \lambda(t)` vanishes at the terminal time), and the terminal datum as
:math:`g(x)`, the forms are

==================  ==========================================  ==================
``form``            trial solution :math:`\hat u(x, t)`         terminal handling
==================  ==========================================  ==================
``hard_constant``   :math:`(1 - \lambda)\,\Phi_\theta + g`       exact (eq:bermudan-ansatz)
``hard_convex``    :math:`(1 - \lambda)\,\Phi_\theta + \lambda g`  exact (eq:bermudan-ansatz-alt)
``soft_pinn``       :math:`\Phi_\theta`                          penalty in loss
``pure_nn``         :math:`\Phi_\theta`                          none (control)
==================  ==========================================  ==================

The two hard forms differ only in the *terminal-data extension*
:math:`\Psi`: ``hard_constant`` uses the time-constant extension
:math:`\Psi = g`, while ``hard_convex`` damps it as
:math:`\Psi = \lambda(t)\,g`.  ``soft_pinn`` and ``pure_nn`` share the bare
network forward; they are distinguished only by whether the training loop adds
the terminal-mismatch penalty (``soft_pinn``) or omits it entirely (``pure_nn``,
a deliberately non-identifiable control).

**Interpolation coefficient and the source note.**  ``cal_notes/example_heat.tex`` writes the
exponential interpolation coefficient as :math:`b(t) = 1 - e^{-(T-t)}`, which gives
:math:`b(T) = 0` and therefore does *not* enforce the terminal condition by
construction.  The mathematically consistent exponential interpolation
coefficient with :math:`\lambda(T) = 1` (so hard enforcement is preserved) is
:math:`\lambda(t) = e^{-\gamma (T - t)}`; at the eigenvalue-matched rate
:math:`\gamma = \sigma^2 \pi^2 / 2` it reproduces the note's "ideal" single-mode
interpolation coefficient.  :func:`make_interpolation_coefficient` implements this consistent family.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from learning_option_pricing.pde.operators import (
    constant_coefficient_operator,
    constant_coefficient_operator_parts,
)

FORMS = ("hard_constant", "hard_convex", "soft_pinn", "pure_nn")
HARD_FORMS = ("hard_constant", "hard_convex")
SOFT_FORMS = ("soft_pinn", "pure_nn")

INTERPOLATION_KINDS = ("linear", "exponential")

# Exact key set of the analytic-derivative bypass mapping accepted by
# TerminalAnsatz(extension_derivative_fns=...).
EXTENSION_DERIVATIVE_KEYS = ("dt", "dx", "dxx")


# ---------------------------------------------------------------------------
# Interpolation coefficient lambda(t)
# ---------------------------------------------------------------------------

def make_interpolation_coefficient(
    kind: str,
    *,
    T: float,
    t_start: float = 0.0,
    gamma: float | None = None,
    sigma: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    r"""Build the interpolation coefficient :math:`\lambda(t)` with
    :math:`\lambda(T) = 1`.

    The network prefactor in the hard ansatz is :math:`1 - \lambda(t)`, which
    vanishes at :math:`t = T`; for the convex-combination form the extension weight is
    :math:`\lambda(t)`.

    Args:
        kind:    ``"linear"`` -> :math:`\lambda(t) = (t - t_\text{start}) /
                 (T - t_\text{start})` (the report's :math:`t / t_k`);
                 ``"exponential"`` -> :math:`\lambda(t) = e^{-\gamma (T - t)}`.
        T:       Terminal time.
        t_start: Strip start (default 0); only used by the linear branch.
        gamma:   Exponential rate.  Defaults to the eigenvalue-matched
                 :math:`\sigma^2 \pi^2 / 2`, the note's "ideal" rate, when
                 ``None``.
        sigma:   Diffusion scale, used to compute the default ``gamma``.

    Returns:
        A callable mapping a time tensor to :math:`\lambda(t)`.
    """
    if kind == "linear":
        span = T - t_start
        if span <= 0.0:
            raise ValueError(f"Linear interpolation coefficient needs T > t_start; got {T=}, {t_start=}.")

        def _lambda_linear(t: torch.Tensor) -> torch.Tensor:
            return (t - t_start) / span

        return _lambda_linear

    if kind == "exponential":
        rate = gamma if gamma is not None else 0.5 * sigma**2 * math.pi**2
        if rate <= 0.0:
            raise ValueError(f"Exponential interpolation coefficient needs gamma > 0; got {rate}.")

        def _lambda_exp(t: torch.Tensor) -> torch.Tensor:
            return torch.exp(-rate * (T - t))

        return _lambda_exp

    raise ValueError(f"Unknown interpolation-coefficient kind: {kind!r}. Choose from {INTERPOLATION_KINDS}.")


# ---------------------------------------------------------------------------
# Trial solution
# ---------------------------------------------------------------------------

class TerminalAnsatz(nn.Module):
    r"""Trial solution wrapping a free network with a chosen terminal-enforcement form.

    Args:
        network:        Free network :math:`\Phi_\theta`, mapping ``(batch, 2)``
                        inputs ``[x, t]`` to ``(batch, 1)``.
        terminal_datum: Callable ``g(x) -> Tensor`` giving the terminal datum;
                        used only by the hard forms (and only when
                        ``extension_fn`` is not supplied).
        interp_coeff:    Callable ``lambda(t) -> Tensor`` with ``lambda(T) = 1``;
                        used only by the hard forms.  May be ``None`` for the
                        soft forms.
        form:           One of :data:`FORMS`.
        normalizer:     Optional callable applied to the raw ``(batch, 2)`` input
                        before the network (e.g. coordinate rescaling).
        extension_fn:   Optional callable ``Psi_base(x, t) -> Tensor`` giving a
                        time-dependent terminal-data extension on the strip
                        (e.g. a vanishing-bandwidth Chen--Mangasarian smoothing,
                        exact at ``t = T``).  When supplied it replaces the
                        default ``g(x)`` extension for the hard forms; the
                        ``hard_convex`` form then uses ``lambda(t) * Psi_base``.
        extension_derivative_fns:
                        Optional analytic-derivative bypass for the extension:
                        a mapping with the exact keys ``"dt"``, ``"dx"``,
                        ``"dxx"`` to callables ``(x, t) -> Tensor`` returning
                        the closed-form derivatives of :math:`\Psi`
                        (:math:`\partial_t \Psi`, :math:`\partial_x \Psi`,
                        :math:`\partial_{xx} \Psi`), called with the same
                        ``(N, 1)``-shaped tensors as ``extension_fn``.  When
                        supplied, :func:`residual_decomposition` assembles the
                        extension forcing :math:`P\Psi` analytically and
                        outside the autograd graph (``torch.no_grad()``):
                        :math:`P\Psi` is :math:`\theta`-independent, so the
                        loss gradient needs its values, never its graph.
                        Restricted to ``form="hard_constant"`` — the
                        ``hard_convex`` product rule with :math:`\lambda` is
                        not implemented (not needed by any stage-2 variant)
                        and any other form raises :class:`ValueError`.

    Raises:
        ValueError: On an unknown form, a hard form without interpolation
            coefficient or datum/extension, or an
            ``extension_derivative_fns`` mapping supplied with a form other
            than ``"hard_constant"`` or with a key set different from
            ``("dt", "dx", "dxx")``.
    """

    def __init__(
        self,
        network: nn.Module,
        terminal_datum: Callable[[torch.Tensor], torch.Tensor] | None,
        interp_coeff: Callable[[torch.Tensor], torch.Tensor] | None,
        *,
        form: str,
        normalizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        extension_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        extension_derivative_fns: dict[
            str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] | None = None,
    ) -> None:
        if form not in FORMS:
            raise ValueError(f"Unknown form: {form!r}. Choose from {FORMS}.")
        if form in HARD_FORMS and interp_coeff is None:
            raise ValueError(f"form={form!r} requires an interpolation coefficient.")
        if form in HARD_FORMS and terminal_datum is None and extension_fn is None:
            raise ValueError(
                f"form={form!r} requires terminal_datum or extension_fn."
            )
        if extension_derivative_fns is not None:
            if form != "hard_constant":
                raise ValueError(
                    "extension_derivative_fns is implemented for "
                    "form='hard_constant' only (the hard_convex product rule "
                    "with the interpolation coefficient is not implemented); "
                    f"received form={form!r}."
                )
            if set(extension_derivative_fns) != set(EXTENSION_DERIVATIVE_KEYS):
                raise ValueError(
                    "extension_derivative_fns must have the exact keys "
                    f"{EXTENSION_DERIVATIVE_KEYS}, received "
                    f"{sorted(extension_derivative_fns)}."
                )
        super().__init__()
        self.network = network
        self._terminal_datum = terminal_datum
        self._interp_coeff = interp_coeff
        self.form = form
        self.normalizer = normalizer
        self._extension_fn = extension_fn
        self._extension_derivative_fns = extension_derivative_fns

    # -- components -------------------------------------------------------

    def free_network(self, x: torch.Tensor) -> torch.Tensor:
        r"""The bare network :math:`\Phi_\theta(x, t)`, shape ``(batch, 1)``."""
        net_input = self.normalizer(x) if self.normalizer is not None else x
        return self.network(net_input)

    def extension(self, coord: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The terminal-data extension :math:`\Psi(x, t)`.

        With the default (time-constant) extension, ``hard_constant`` ->
        :math:`g(x)` and ``hard_convex`` -> :math:`\lambda(t)\,g(x)`.  When a
        time-dependent ``extension_fn`` :math:`\Psi_{\rm base}(x, t)` is supplied
        it replaces :math:`g(x)`: ``hard_constant`` -> :math:`\Psi_{\rm base}`,
        ``hard_convex`` -> :math:`\lambda(t)\,\Psi_{\rm base}`.  Soft forms ->
        zeros.  ``coord`` and ``t`` are ``(N, 1)`` (or broadcast-compatible) and
        may carry ``requires_grad`` so the extension forcing
        :math:`\mathcal P\Psi` can be differentiated.
        """
        if self.form in SOFT_FORMS:
            return torch.zeros_like(coord)
        base = (self._extension_fn(coord, t) if self._extension_fn is not None
                else self._terminal_datum(coord))
        if self.form == "hard_convex":
            return self._interp_coeff(t) * base
        return base  # hard_constant

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trial solution :math:`\\hat u(x, t)`, shape ``(batch, 1)``.

        Args:
            x: Input ``(batch, 2)`` with columns ``[x, t]``.
        """
        phi = self.free_network(x)
        if self.form in SOFT_FORMS:
            return phi
        coord = x[:, 0:1]
        t = x[:, 1:2]
        lam = self._interp_coeff(t)
        ext = self.extension(coord, t)
        return (1.0 - lam) * phi + ext


# ---------------------------------------------------------------------------
# Residual decomposition (rem:residual-decomposition notation)
# ---------------------------------------------------------------------------

def _resolve_generator_coefficients(
    sigma: float | None,
    generator_coefficients: dict[int, float] | None,
) -> dict[int, float]:
    """Resolve the mutually exclusive ``sigma`` / ``generator_coefficients``.

    Exactly one of the two must be supplied; ``sigma`` alone reproduces the
    historical heat behaviour through ``{2: 0.5 * sigma**2, 1: 0.0, 0: 0.0}``.

    Raises:
        ValueError: If both or neither are supplied.
    """
    if (sigma is None) == (generator_coefficients is None):
        raise ValueError(
            "exactly one of 'sigma' and 'generator_coefficients' must be "
            f"supplied; received sigma={sigma!r} and "
            f"generator_coefficients={generator_coefficients!r}."
        )
    if generator_coefficients is not None:
        return {int(order): float(value) for order, value in generator_coefficients.items()}
    return {2: 0.5 * sigma**2, 1: 0.0, 0: 0.0}


def residual_decomposition(
    ansatz: TerminalAnsatz,
    coord: torch.Tensor,
    t: torch.Tensor,
    sigma: float | None = None,
    *,
    generator_coefficients: dict[int, float] | None = None,
) -> dict[str, torch.Tensor]:
    r"""Split the stage residual into the channels of ``rem:residual-decomposition``.

    For the hard forms the residual of the trial solution decomposes as

    .. math::

        \mathcal P\,\hat u = \underbrace{(1 - \lambda)\,\mathcal P\Phi_\theta
            - \lambda'\,\Phi_\theta}_{R_\theta\ \text{(network contribution)}}
            + \underbrace{\mathcal P\Psi}_{\text{extension forcing}},

    and the stage loss into

    .. math::

        \mathcal L = \mathbb E[R_\theta^2]
            + 2\,\mathbb E[R_\theta\,\mathcal P\Psi]
            + \mathbb E[(\mathcal P\Psi)^2].

    For the soft forms there is no extension; the network *is* the trial
    solution, so ``R = P u`` and the forcing/cross channels are zero.

    Analytic-derivative bypass.  When the ansatz was built with
    ``extension_derivative_fns`` (``hard_constant`` only), the extension
    forcing is assembled **analytically and outside the autograd graph**
    (``torch.no_grad()``) as :math:`P\Psi = \partial_t \Psi +
    c_2\,\partial_{xx}\Psi + c_1\,\partial_x\Psi + c_0\,\Psi` from the
    supplied closed-form derivatives — :math:`P\Psi` is
    :math:`\theta`-independent, so the loss gradient needs its values, never
    its graph.  When absent, the existing autograd route through
    :func:`learning_option_pricing.pde.operators.constant_coefficient_operator_parts`
    is used.

    Args:
        ansatz: The trial solution.
        coord:  Spatial collocation points, shape ``(N,)`` with
                ``requires_grad=True``.
        t:      Time collocation points, shape ``(N,)`` with
                ``requires_grad=True``.
        sigma:  Diffusion scale of the backward heat operator; mutually
                exclusive with ``generator_coefficients`` and equivalent to
                ``generator_coefficients={2: 0.5 * sigma**2, 1: 0.0, 0: 0.0}``
                (the historical behaviour; existing call sites unchanged).
        generator_coefficients: Mapping from differential order to real
                coefficient of the spatial generator, ``{2: nu, 1: mu, 0: r0}``;
                mutually exclusive with ``sigma``.

    Returns:
        A dict of per-point fields and scalar channel energies:
        ``residual`` (:math:`\mathcal P\hat u`), ``network_contribution``
        (:math:`R_\theta`), ``extension_forcing`` (:math:`\mathcal P\Psi`),
        and the scalar means ``loss`` (:math:`\mathbb E[(\mathcal P\hat u)^2]`),
        ``network_energy`` (:math:`\mathbb E[R_\theta^2]`), ``cross_term``
        (:math:`2\,\mathbb E[R_\theta\,\mathcal P\Psi]`), and
        ``forcing_floor`` (:math:`\mathbb E[(\mathcal P\Psi)^2]`).

        The floor is further split by mechanism (the four operator channels
        of :math:`\mathcal P\Psi = \partial_t\Psi + c_2\,\partial_{xx}\Psi
        + c_1\,\partial_x\Psi + c_0\,\Psi`):
        ``forcing_velocity`` (:math:`\mathbb E[(\partial_t\Psi)^2]`),
        ``forcing_diffusion`` (:math:`\mathbb E[(c_2\,\partial_{xx}\Psi)^2]`),
        ``forcing_advection`` (:math:`\mathbb E[(c_1\,\partial_x\Psi)^2]`) and
        ``forcing_reaction`` (:math:`\mathbb E[(c_0\,\Psi)^2]`).  Under the
        ``sigma`` route the advection and reaction channels are identically
        zero.

    Raises:
        ValueError: If both or neither of ``sigma`` and
            ``generator_coefficients`` are supplied.
    """
    resolved_coefficients = _resolve_generator_coefficients(
        sigma, generator_coefficients
    )

    coord_col = coord.unsqueeze(-1)
    t_col = t.unsqueeze(-1)
    xt = torch.cat([coord_col, t_col], dim=1)

    phi = ansatz.free_network(xt).squeeze(-1)
    p_phi = constant_coefficient_operator(phi, coord, t, resolved_coefficients)

    if ansatz.form in SOFT_FORMS:
        residual = p_phi
        network_contribution = p_phi
        extension_forcing = torch.zeros_like(p_phi)
        forcing_velocity = torch.zeros_like(p_phi)
        forcing_diffusion = torch.zeros_like(p_phi)
        forcing_advection = torch.zeros_like(p_phi)
        forcing_reaction = torch.zeros_like(p_phi)
    else:
        lam = ansatz._interp_coeff(t)
        (lam_prime,) = torch.autograd.grad(
            lam, (t,), grad_outputs=torch.ones_like(lam), create_graph=True
        )
        network_contribution = (1.0 - lam) * p_phi - lam_prime * phi

        if ansatz._extension_derivative_fns is not None:
            # Analytic-derivative bypass (hard_constant only, enforced at
            # construction): the theta-independent forcing is assembled from
            # the closed-form derivatives outside the autograd graph.
            derivative_fns = ansatz._extension_derivative_fns
            with torch.no_grad():
                psi_values = ansatz.extension(coord_col, t_col).reshape(coord.shape)
                forcing_velocity = derivative_fns["dt"](coord_col, t_col).reshape(
                    coord.shape
                )
                forcing_diffusion = resolved_coefficients.get(
                    2, 0.0
                ) * derivative_fns["dxx"](coord_col, t_col).reshape(coord.shape)
                forcing_advection = resolved_coefficients.get(
                    1, 0.0
                ) * derivative_fns["dx"](coord_col, t_col).reshape(coord.shape)
                forcing_reaction = resolved_coefficients.get(0, 0.0) * psi_values
                extension_forcing = (
                    forcing_velocity
                    + forcing_diffusion
                    + forcing_advection
                    + forcing_reaction
                )
        else:
            psi = ansatz.extension(coord_col, t_col).squeeze(-1)
            forcing_parts = constant_coefficient_operator_parts(
                psi, coord, t, resolved_coefficients
            )
            forcing_velocity = forcing_parts["velocity"]
            forcing_diffusion = forcing_parts["diffusion"]
            forcing_advection = forcing_parts["advection"]
            forcing_reaction = forcing_parts["reaction"]
            extension_forcing = (
                forcing_velocity
                + forcing_diffusion
                + forcing_advection
                + forcing_reaction
            )
        residual = network_contribution + extension_forcing

    return {
        "residual": residual,
        "network_contribution": network_contribution,
        "extension_forcing": extension_forcing,
        "loss": (residual**2).mean(),
        "network_energy": (network_contribution**2).mean(),
        "cross_term": 2.0 * (network_contribution * extension_forcing).mean(),
        "forcing_floor": (extension_forcing**2).mean(),
        "forcing_velocity": (forcing_velocity**2).mean(),
        "forcing_diffusion": (forcing_diffusion**2).mean(),
        "forcing_advection": (forcing_advection**2).mean(),
        "forcing_reaction": (forcing_reaction**2).mean(),
    }


# ---------------------------------------------------------------------------
# Startup cross-check of the analytic-derivative bypass
# ---------------------------------------------------------------------------

def cross_check_extension_forcing_analytic_versus_autograd(
    ansatz: TerminalAnsatz,
    coord: torch.Tensor,
    t: torch.Tensor,
    sigma: float | None = None,
    *,
    generator_coefficients: dict[int, float] | None = None,
    relative_tolerance: float = 1.0e-3,
) -> float:
    r"""Startup guard: the analytic and the autograd :math:`P\Psi` must agree.

    Both assemblies of the extension forcing are evaluated on the supplied
    batch — the analytic one from ``extension_derivative_fns`` (the training
    route under the bypass) and the autograd one through
    :func:`learning_option_pricing.pde.operators.constant_coefficient_operator`
    on the extension values — and their :math:`L^2` deviation is measured.
    A deviation above ``relative_tolerance`` (default :math:`10^{-3}`, the
    float32 double-backward noise allowance of the stage-2 specification,
    Section 1.4 item 5) raises :class:`RuntimeError`: the guard aborts, it
    never passes silently.

    Normalisation.  The deviation is normalised by
    :math:`\max\bigl(\lVert P\Psi_{\mathrm{autograd}}\rVert_2,
    \lVert\, |\partial_t\Psi| + |c_2\,\partial_{xx}\Psi| +
    |c_1\,\partial_x\Psi| + |c_0\,\Psi|\, \rVert_2\bigr)` — the second term
    is the magnitude of the arithmetic actually performed in the assembly.
    For a non-cancelling forcing this coincides with the plain relative
    :math:`L^2` deviation up to a factor at most the number of channels; for
    a zero-forcing extension (e.g. the exact-solution variant, where the
    channels cancel identically) the plain relative deviation is a ratio of
    round-off terms and is undefined, whereas this normalisation measures the
    agreement against the assembly scale.  When both norms vanish the
    extension is identically zero and the deviation is ``0.0``.

    Args:
        ansatz: A trial solution built with ``extension_derivative_fns``.
        coord:  Spatial batch, shape ``(N,)`` (gradients not required; fresh
                leaf tensors are created internally for the autograd side).
        t:      Time batch, shape ``(N,)``.
        sigma / generator_coefficients: Exactly one must be supplied, as in
                :func:`residual_decomposition`.
        relative_tolerance: Abort threshold on the measured deviation.

    Returns:
        The measured deviation as a float (to be logged by the caller).

    Raises:
        ValueError: If the ansatz has no ``extension_derivative_fns``, or on
            an invalid ``sigma`` / ``generator_coefficients`` combination.
        RuntimeError: If the measured deviation exceeds
            ``relative_tolerance``.
    """
    if ansatz._extension_derivative_fns is None:
        raise ValueError(
            "the cross-check requires an ansatz built with "
            "extension_derivative_fns (the analytic-derivative bypass)."
        )
    resolved_coefficients = _resolve_generator_coefficients(
        sigma, generator_coefficients
    )
    derivative_fns = ansatz._extension_derivative_fns

    # Analytic side (the training route under the bypass).
    with torch.no_grad():
        coord_col = coord.detach().unsqueeze(-1)
        t_col = t.detach().unsqueeze(-1)
        psi_values = ansatz.extension(coord_col, t_col).reshape(coord.shape)
        velocity_channel = derivative_fns["dt"](coord_col, t_col).reshape(coord.shape)
        diffusion_channel = resolved_coefficients.get(2, 0.0) * derivative_fns[
            "dxx"
        ](coord_col, t_col).reshape(coord.shape)
        advection_channel = resolved_coefficients.get(1, 0.0) * derivative_fns[
            "dx"
        ](coord_col, t_col).reshape(coord.shape)
        reaction_channel = resolved_coefficients.get(0, 0.0) * psi_values
        analytic_forcing = (
            velocity_channel
            + diffusion_channel
            + advection_channel
            + reaction_channel
        )
        assembly_scale_norm = torch.linalg.vector_norm(
            velocity_channel.abs()
            + diffusion_channel.abs()
            + advection_channel.abs()
            + reaction_channel.abs()
        )

    # Autograd side, on fresh leaf tensors.
    coord_autograd = coord.detach().clone().requires_grad_(True)
    t_autograd = t.detach().clone().requires_grad_(True)
    psi_autograd = ansatz.extension(
        coord_autograd.unsqueeze(-1), t_autograd.unsqueeze(-1)
    ).reshape(coord_autograd.shape)
    autograd_forcing = constant_coefficient_operator(
        psi_autograd, coord_autograd, t_autograd, resolved_coefficients
    ).detach()

    deviation_norm = torch.linalg.vector_norm(analytic_forcing - autograd_forcing)
    denominator = torch.maximum(
        torch.linalg.vector_norm(autograd_forcing), assembly_scale_norm
    )
    if denominator.item() == 0.0:
        measured_deviation = 0.0
    else:
        measured_deviation = float(deviation_norm / denominator)

    if measured_deviation > relative_tolerance:
        raise RuntimeError(
            "analytic-versus-autograd extension-forcing cross-check failed: "
            f"measured relative L2 deviation {measured_deviation:.6e} exceeds "
            f"the tolerance {relative_tolerance:.1e}; the closed-form "
            "extension derivatives disagree with autograd on this device/"
            "dtype, so the run must not proceed."
        )
    return measured_deviation
