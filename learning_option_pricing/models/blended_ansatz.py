r"""Time-blended trial solutions for terminal-condition enforcement.

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
``hard_blended``    :math:`(1 - \lambda)\,\Phi_\theta + \lambda g`  exact (eq:bermudan-ansatz-alt)
``soft_pinn``       :math:`\Phi_\theta`                          penalty in loss
``pure_nn``         :math:`\Phi_\theta`                          none (control)
==================  ==========================================  ==================

The two hard forms differ only in the *terminal-data extension*
:math:`\Psi`: ``hard_constant`` uses the time-constant extension
:math:`\Psi = g`, while ``hard_blended`` damps it as
:math:`\Psi = \lambda(t)\,g`.  ``soft_pinn`` and ``pure_nn`` share the bare
network forward; they are distinguished only by whether the training loop adds
the terminal-mismatch penalty (``soft_pinn``) or omits it entirely (``pure_nn``,
a deliberately non-identifiable control).

**Blending and the source note.**  ``cal_notes/example_heat.tex`` writes the
exponential blending as :math:`b(t) = 1 - e^{-(T-t)}`, which gives
:math:`b(T) = 0` and therefore does *not* enforce the terminal condition by
construction.  The mathematically consistent exponential interpolation
coefficient with :math:`\lambda(T) = 1` (so hard enforcement is preserved) is
:math:`\lambda(t) = e^{-\gamma (T - t)}`; at the eigenvalue-matched rate
:math:`\gamma = \sigma^2 \pi^2 / 2` it reproduces the note's "ideal" single-mode
blending.  :func:`make_blending` implements this consistent family.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from learning_option_pricing.pde.operators import heat_operator

FORMS = ("hard_constant", "hard_blended", "soft_pinn", "pure_nn")
HARD_FORMS = ("hard_constant", "hard_blended")
SOFT_FORMS = ("soft_pinn", "pure_nn")

BLENDINGS = ("linear", "exponential")


# ---------------------------------------------------------------------------
# Interpolation coefficient lambda(t)
# ---------------------------------------------------------------------------

def make_blending(
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
    vanishes at :math:`t = T`; for the blended form the extension weight is
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
            raise ValueError(f"Linear blending needs T > t_start; got {T=}, {t_start=}.")

        def _lambda_linear(t: torch.Tensor) -> torch.Tensor:
            return (t - t_start) / span

        return _lambda_linear

    if kind == "exponential":
        rate = gamma if gamma is not None else 0.5 * sigma**2 * math.pi**2
        if rate <= 0.0:
            raise ValueError(f"Exponential blending needs gamma > 0; got {rate}.")

        def _lambda_exp(t: torch.Tensor) -> torch.Tensor:
            return torch.exp(-rate * (T - t))

        return _lambda_exp

    raise ValueError(f"Unknown blending kind: {kind!r}. Choose from {BLENDINGS}.")


# ---------------------------------------------------------------------------
# Blended trial solution
# ---------------------------------------------------------------------------

class BlendedTerminalAnsatz(nn.Module):
    r"""Trial solution wrapping a free network with a chosen terminal-enforcement form.

    Args:
        network:        Free network :math:`\Phi_\theta`, mapping ``(batch, 2)``
                        inputs ``[x, t]`` to ``(batch, 1)``.
        terminal_datum: Callable ``g(x) -> Tensor`` giving the terminal datum;
                        used only by the hard forms (and only when
                        ``extension_fn`` is not supplied).
        blending:       Callable ``lambda(t) -> Tensor`` with ``lambda(T) = 1``;
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
                        ``hard_blended`` form then uses ``lambda(t) * Psi_base``.
    """

    def __init__(
        self,
        network: nn.Module,
        terminal_datum: Callable[[torch.Tensor], torch.Tensor] | None,
        blending: Callable[[torch.Tensor], torch.Tensor] | None,
        *,
        form: str,
        normalizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        extension_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if form not in FORMS:
            raise ValueError(f"Unknown form: {form!r}. Choose from {FORMS}.")
        if form in HARD_FORMS and blending is None:
            raise ValueError(f"form={form!r} requires a blending.")
        if form in HARD_FORMS and terminal_datum is None and extension_fn is None:
            raise ValueError(
                f"form={form!r} requires terminal_datum or extension_fn."
            )
        super().__init__()
        self.network = network
        self._terminal_datum = terminal_datum
        self._blending = blending
        self.form = form
        self.normalizer = normalizer
        self._extension_fn = extension_fn

    # -- components -------------------------------------------------------

    def free_network(self, x: torch.Tensor) -> torch.Tensor:
        r"""The bare network :math:`\Phi_\theta(x, t)`, shape ``(batch, 1)``."""
        net_input = self.normalizer(x) if self.normalizer is not None else x
        return self.network(net_input)

    def extension(self, coord: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The terminal-data extension :math:`\Psi(x, t)`.

        With the default (time-constant) extension, ``hard_constant`` ->
        :math:`g(x)` and ``hard_blended`` -> :math:`\lambda(t)\,g(x)`.  When a
        time-dependent ``extension_fn`` :math:`\Psi_{\rm base}(x, t)` is supplied
        it replaces :math:`g(x)`: ``hard_constant`` -> :math:`\Psi_{\rm base}`,
        ``hard_blended`` -> :math:`\lambda(t)\,\Psi_{\rm base}`.  Soft forms ->
        zeros.  ``coord`` and ``t`` are ``(N, 1)`` (or broadcast-compatible) and
        may carry ``requires_grad`` so the extension forcing
        :math:`\mathcal P\Psi` can be differentiated.
        """
        if self.form in SOFT_FORMS:
            return torch.zeros_like(coord)
        base = (self._extension_fn(coord, t) if self._extension_fn is not None
                else self._terminal_datum(coord))
        if self.form == "hard_blended":
            return self._blending(t) * base
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
        lam = self._blending(t)
        ext = self.extension(coord, t)
        return (1.0 - lam) * phi + ext


# ---------------------------------------------------------------------------
# Residual decomposition (rem:residual-decomposition notation)
# ---------------------------------------------------------------------------

def residual_decomposition(
    ansatz: BlendedTerminalAnsatz,
    coord: torch.Tensor,
    t: torch.Tensor,
    sigma: float,
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

    Args:
        ansatz: The trial solution.
        coord:  Spatial collocation points, shape ``(N,)`` with
                ``requires_grad=True``.
        t:      Time collocation points, shape ``(N,)`` with
                ``requires_grad=True``.
        sigma:  Diffusion scale passed to :func:`heat_operator`.

    Returns:
        A dict of per-point fields and scalar channel energies:
        ``residual`` (:math:`\mathcal P\hat u`), ``network_contribution``
        (:math:`R_\theta`), ``extension_forcing`` (:math:`\mathcal P\Psi`),
        and the scalar means ``loss`` (:math:`\mathbb E[(\mathcal P\hat u)^2]`),
        ``network_energy`` (:math:`\mathbb E[R_\theta^2]`), ``cross_term``
        (:math:`2\,\mathbb E[R_\theta\,\mathcal P\Psi]`), and
        ``forcing_floor`` (:math:`\mathbb E[(\mathcal P\Psi)^2]`).
    """
    coord_col = coord.unsqueeze(-1)
    t_col = t.unsqueeze(-1)
    xt = torch.cat([coord_col, t_col], dim=1)

    phi = ansatz.free_network(xt).squeeze(-1)
    p_phi = heat_operator(phi, coord, t, sigma)

    if ansatz.form in SOFT_FORMS:
        residual = p_phi
        network_contribution = p_phi
        extension_forcing = torch.zeros_like(p_phi)
    else:
        lam = ansatz._blending(t)
        (lam_prime,) = torch.autograd.grad(
            lam, (t,), grad_outputs=torch.ones_like(lam), create_graph=True
        )
        network_contribution = (1.0 - lam) * p_phi - lam_prime * phi

        psi = ansatz.extension(coord_col, t_col).squeeze(-1)
        extension_forcing = heat_operator(psi, coord, t, sigma)
        residual = network_contribution + extension_forcing

    return {
        "residual": residual,
        "network_contribution": network_contribution,
        "extension_forcing": extension_forcing,
        "loss": (residual**2).mean(),
        "network_energy": (network_contribution**2).mean(),
        "cross_term": 2.0 * (network_contribution * extension_forcing).mean(),
        "forcing_floor": (extension_forcing**2).mean(),
    }
