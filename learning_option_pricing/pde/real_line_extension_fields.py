r"""Terminal-data extensions on the log-price line, with analytic derivatives.

Setting.  A stage of a Bermudan backward induction occupies the space-time strip
:math:`\mathcal{X} \times [0, T]` in stage-local time, the terminal datum
:math:`V` being imposed at :math:`t = T`.  A *terminal-data extension* is a field
:math:`h` on the whole strip with :math:`h(\cdot, T) = V`; the values for
:math:`t < T` --- the *interior profile* --- are a modelling choice, and they are
the whole of what the residual objective sees.  The trial solution is the
hard-constrained additive form
:math:`\hat u = \mathrm{d}_{T}\,\Psi_{\theta} + h`, so the terminal condition
holds by construction and the objective carries the
:math:`\theta`-independent *extension forcing* :math:`\mathcal{L} h`.

This module supplies the two extensions whose forcing the free-boundary theory
distinguishes, on the **real line** (the periodic counterparts live in
:mod:`learning_option_pricing.pde.periodic_extension_fields`).

Gaussian-semigroup (split) extension.  With the Black--Scholes generator split as
:math:`\mathcal{L}^{X} = \mathcal{A} + \mathcal{B}`,
:math:`\mathcal{A} = \nu\,\partial_{xx}`,
:math:`\mathcal{B} = (r - \nu)\,\partial_{x} - r`, the extension

.. math::

    h(\cdot, t) = e^{(T - t)\,\mathcal{A}_{c}}\, V ,
    \qquad \mathcal{A}_{c} = \nu_{c}\,\partial_{xx} ,

is the Gaussian convolution of the datum at standard deviation
:math:`\sigma_{c}\sqrt{T - t}`.  Because the semigroup at parameter zero is the
identity, :math:`h(\cdot, T) = V` **exactly**, whatever the regularity of
:math:`V` --- in particular when :math:`V = \max(\payoff, C)` has a
first-derivative discontinuity at the free boundary.  Its forcing is

.. math::

    \mathcal{L} h
        = (\nu - \nu_{c})\,\partial_{xx} h
        + \mathcal{B} h ,

which for the **matched** comparison diffusivity :math:`\nu_{c} = \nu` collapses
to :math:`\mathcal{B} h` --- a *first-order* operator applied to a Lipschitz field,
hence **bounded uniformly on the strip**.  For a **mis-specified**
:math:`\nu_{c} \neq \nu` the second-order channel survives; it is
square-integrable but *unbounded*, diverging like :math:`(T-t)^{-1/2}` at the
corner as the terminal slice is approached.

Numerical route, and why the derivatives are analytic.  The convolution is
evaluated on a **fixed** trapezoidal grid in :math:`y`, so

.. math::

    h(x, t) = \sum_{j} w_{j}\, \varphi_{s}(x - y_{j})\, V(y_{j}) ,
    \qquad s = \sigma_{c}\sqrt{T - t} ,

is a finite sum of Gaussians *centred at nodes that do not move with* :math:`x`:
it is therefore :math:`C^{\infty}` in :math:`x`, and the datum's corner is not
reproduced at the quadrature nodes.  (A quadrature whose nodes *do* move with
:math:`x` --- a Gauss--Hermite rule
:math:`\sum_{i} w_{i} V(x + s\,\xi_{i})` --- is a sum of shifted copies of the
kinked datum and would reinstate one Dirac mass per node in
:math:`\partial_{xx} h`, invisible to a collocation loss.  That route is not used
here, and must not be.)

The derivatives are supplied **analytically**, by differentiating the kernel:

.. math::

    \partial_{x}\varphi_{s}(u) = -\frac{u}{s^{2}}\,\varphi_{s}(u) ,
    \qquad
    \partial_{xx}\varphi_{s}(u)
        = \frac{u^{2} - s^{2}}{s^{4}}\,\varphi_{s}(u) ,
    \qquad
    \partial_{t} h = -\nu_{c}\,\partial_{xx} h ,

the last being the heat equation satisfied by :math:`h` in :math:`(x, t)` and
therefore **exact**.  Supplying it analytically is not an optimisation: assembling
:math:`\mathcal{L} h` by automatic differentiation would compute
:math:`\partial_{t} h + \nu\,\partial_{xx} h` as the difference of two quantities
that each diverge like :math:`(T - t)^{-1/2}` near the corner and that cancel to
leading order, which is catastrophic cancellation exactly where the interesting
behaviour is.  With the analytic route the cancellation is performed in closed
form and the matched case returns :math:`\mathcal{B} h` with no subtraction at
all.

The quadrature floor is reported, never silent.  When
:math:`s = \sigma_{c}\sqrt{T-t}` falls below a few grid spacings the fixed grid
cannot resolve the kernel.  At :math:`t = T` exactly the field returns the datum
(that is the terminal identity, and it is exact).  For
:math:`0 < T - t < \tau_{\mathrm{floor}}` the quadrature is unresolved: the field
still returns the datum, but it **warns on first activation, counts every
activation, and records the largest unresolved time-to-terminal**, and
:attr:`GaussianSemigroupExtensionField.quadrature_floor_report` exposes the
tally.  A caller that samples the residual over the whole strip *will* activate
this floor and must either exclude the terminal sliver of width
:math:`\tau_{\mathrm{floor}}` from the interior sampler --- the recommended
course, and what the companion ablation does --- or report the activation count
alongside its results.  A silently-active floor here would regularise the very
corner the study is about.
"""

from __future__ import annotations

import logging
import math

import torch

from learning_option_pricing.pde.heat_references import (
    chen_mangasarian_max,
    heat_put_payoff,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GaussianSemigroupExtensionField",
    "GradedChenMangasarianExtensionField",
    "EXTENSION_FIELD_KINDS",
]

#: The extension kinds this module can build, in the order of the paper's ledger.
EXTENSION_FIELD_KINDS = (
    "gaussian_semigroup",
    "graded_chen_mangasarian",
)

#: Number of grid spacings the Gaussian standard deviation must span for the
#: fixed trapezoidal quadrature to resolve the kernel.  Below this the kernel is
#: narrower than the grid and the convolution degenerates; the value mirrors the
#: guard already used by
#: :func:`learning_option_pricing.pde.heat_references.heat_propagate`.
_QUADRATURE_RESOLUTION_SPACINGS = 5.0


class GaussianSemigroupExtensionField:
    r"""Split extension :math:`h(\cdot, t) = e^{(T-t)\nu_{c}\partial_{xx}} V`.

    Args:
        terminal_datum_fn:      Callable ``V(y) -> Tensor`` on the log-price line.
                                It is evaluated **once per call** on the fixed
                                quadrature grid, so a network-valued datum costs
                                one forward pass of ``n_quad`` points, not
                                ``N * n_quad``.
        terminal_time:          Stage-local terminal time :math:`T` at which the
                                datum is imposed.
        comparison_volatility:  :math:`\sigma_{c}`, so that
                                :math:`\nu_{c} = \sigma_{c}^{2}/2`.  Matched to the
                                model's :math:`\sigma` this is the well-specified
                                split; different from it, the mis-specified one.
        y_lo, y_hi:             Quadrature support.  It must cover the log-price
                                window **padded** by several diffusion lengths
                                :math:`\sigma_{c}\sqrt{T}`, and the datum must be
                                supplied there by its analytic far-field values ---
                                never by zero-padding (the put datum tends to
                                :math:`K`, not to zero, as :math:`x \to -\infty`,
                                so zero-padding inserts a spurious jump of size
                                :math:`K` at the window edge whose convolution
                                produces a first-order source diverging like
                                :math:`(T-t)^{-1/2}`), and never by extrapolating a
                                trained network beyond its training window.
        n_quad:                 Number of quadrature nodes.
        name:                   Identifier used in log messages.
    """

    def __init__(
        self,
        terminal_datum_fn,
        *,
        terminal_time: float,
        comparison_volatility: float,
        y_lo: float,
        y_hi: float,
        n_quad: int = 4000,
        name: str = "gaussian_semigroup",
    ) -> None:
        if comparison_volatility <= 0.0:
            raise ValueError(
                "comparison_volatility must be strictly positive; received "
                f"{comparison_volatility!r}. A vanishing comparison volatility is "
                "the raw (un-smoothed) extension, which is a different object and "
                "is not built by this class."
            )
        if y_hi <= y_lo:
            raise ValueError(f"require y_lo < y_hi; received {y_lo!r}, {y_hi!r}.")
        if n_quad < 2:
            raise ValueError(f"n_quad must be at least 2; received {n_quad!r}.")

        self._terminal_datum_fn = terminal_datum_fn
        self.terminal_time = float(terminal_time)
        self.comparison_volatility = float(comparison_volatility)
        self.comparison_diffusivity = 0.5 * float(comparison_volatility) ** 2
        self.y_lo = float(y_lo)
        self.y_hi = float(y_hi)
        self.n_quad = int(n_quad)
        self.name = name

        self.quadrature_step = (self.y_hi - self.y_lo) / (self.n_quad - 1)
        # The Gaussian standard deviation is sigma_c * sqrt(tau); requiring it to
        # span _QUADRATURE_RESOLUTION_SPACINGS grid steps gives the floor below.
        self.time_to_terminal_floor = (
            _QUADRATURE_RESOLUTION_SPACINGS
            * self.quadrature_step
            / self.comparison_volatility
        ) ** 2

        self._floor_activation_count = 0
        self._floor_evaluation_count = 0
        self._floor_largest_time_to_terminal = 0.0
        self._floor_warned = False

    # -- quadrature -------------------------------------------------------

    def _quadrature_nodes(self, reference: torch.Tensor):
        nodes = torch.linspace(
            self.y_lo,
            self.y_hi,
            self.n_quad,
            dtype=reference.dtype,
            device=reference.device,
        )
        return nodes, self._terminal_datum_fn(nodes).reshape(1, -1)

    def _register_floor(self, time_to_terminal: torch.Tensor) -> torch.Tensor:
        r"""Flag the unresolved-quadrature band, and report it.

        Returns the boolean mask of the points at which the fixed grid cannot
        resolve the kernel, i.e. ``0 < T - t < time_to_terminal_floor``.  Points
        at ``T - t <= 0`` are *not* flagged: there the field is the terminal datum
        by definition, and returning it is exact rather than a truncation.
        """
        unresolved = (time_to_terminal > 0.0) & (
            time_to_terminal < self.time_to_terminal_floor
        )
        self._floor_evaluation_count += int(time_to_terminal.numel())
        count = int(unresolved.sum())
        if count == 0:
            return unresolved

        self._floor_activation_count += count
        largest = float(time_to_terminal[unresolved].max())
        self._floor_largest_time_to_terminal = max(
            self._floor_largest_time_to_terminal, largest
        )
        message = (
            "extension %r: the fixed quadrature grid cannot resolve the Gaussian "
            "kernel at %d of %d evaluated points (time-to-terminal below the floor "
            "%.3e; largest unresolved value %.3e). The terminal datum is returned "
            "there, so its first-derivative discontinuity at the free boundary is "
            "NOT smoothed at those points and the second-derivative channel of the "
            "residual is wrong there. Exclude the terminal sliver of width %.3e "
            "from the interior sampler, or raise n_quad."
        )
        if not self._floor_warned:
            self._floor_warned = True
            logger.warning(
                message,
                self.name,
                count,
                int(time_to_terminal.numel()),
                self.time_to_terminal_floor,
                largest,
                self.time_to_terminal_floor,
            )
        else:
            logger.debug(
                message,
                self.name,
                count,
                int(time_to_terminal.numel()),
                self.time_to_terminal_floor,
                largest,
                self.time_to_terminal_floor,
            )
        return unresolved

    def _convolve(self, coord, t, kernel_fn):
        """Convolve the datum against ``kernel_fn(u, s)`` on the fixed grid."""
        coord = coord.reshape(-1, 1)
        t = t.reshape(-1, 1)
        nodes, datum_values = self._quadrature_nodes(coord)

        time_to_terminal = self.terminal_time - t
        unresolved = self._register_floor(time_to_terminal.detach())

        safe_time = torch.clamp(time_to_terminal, min=self.time_to_terminal_floor)
        standard_deviation = self.comparison_volatility * torch.sqrt(safe_time)
        separation = coord - nodes.reshape(1, -1)
        weights = kernel_fn(separation, standard_deviation)
        value = (weights * datum_values).sum(dim=-1, keepdim=True) * self.quadrature_step

        # At or beyond the terminal slice, and in the unresolved band, the value
        # is the terminal datum itself (exactly at the slice; as the tau -> 0
        # limit, and reported, inside the band).
        at_or_past_terminal = time_to_terminal <= 0.0
        fallback = at_or_past_terminal | unresolved
        return value, fallback, coord

    @staticmethod
    def _kernel(separation, standard_deviation):
        variance = standard_deviation**2
        return torch.exp(-(separation**2) / (2.0 * variance)) / torch.sqrt(
            2.0 * math.pi * variance
        )

    # -- field and its analytic derivatives -------------------------------

    def field(self, coord: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The extension :math:`h(x, t)`, shape ``(N, 1)``."""
        value, fallback, coord_column = self._convolve(coord, t, self._kernel)
        datum_at_query = self._terminal_datum_fn(coord_column.reshape(-1)).reshape(-1, 1)
        return torch.where(fallback, datum_at_query, value)

    def space_derivative(self, coord: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The analytic :math:`\partial_{x} h`, shape ``(N, 1)``."""

        def kernel(separation, standard_deviation):
            return (
                -separation / standard_deviation**2
            ) * self._kernel(separation, standard_deviation)

        value, fallback, coord_column = self._convolve(coord, t, kernel)
        # In the fallback band the extension is the datum, whose derivative is the
        # datum's own (piecewise) derivative; it is obtained by autograd on the
        # datum, which is what a caller sampling there would get anyway.
        fallback_derivative = _autograd_space_derivative(
            self._terminal_datum_fn, coord_column
        )
        return torch.where(fallback, fallback_derivative, value)

    def second_space_derivative(
        self, coord: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        r"""The analytic :math:`\partial_{xx} h`, shape ``(N, 1)``.

        Near the terminal slice this diverges like
        :math:`J\,\varphi_{\sigma_{c}\sqrt{T-t}}(x - x^{\star})`, with :math:`J`
        the jump of the datum's first derivative at the free boundary --- the
        Dirac mass of :math:`\partial_{xx} V` smoothed by the kernel.  The
        divergence is correct and is *not* an artefact: it is exactly the channel
        that the matched split cancels against :math:`\partial_{t} h` and that a
        mis-specified split leaves behind.
        """

        def kernel(separation, standard_deviation):
            variance = standard_deviation**2
            return (
                (separation**2 - variance) / variance**2
            ) * self._kernel(separation, standard_deviation)

        value, fallback, _ = self._convolve(coord, t, kernel)
        # In the unresolved band the second derivative of the raw datum is a Dirac
        # and is not representable; zero is returned and the activation is already
        # counted and warned about by _register_floor.  A caller must not sample
        # there (see the module docstring).
        return torch.where(fallback, torch.zeros_like(value), value)

    def time_derivative(self, coord: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The analytic :math:`\partial_{t} h = -\nu_{c}\,\partial_{xx} h`.

        Exact, by the heat equation
        :math:`\partial_{t} h + \nu_{c}\,\partial_{xx} h = 0` satisfied by the
        semigroup extension.  It is precisely this identity that makes the matched
        split's forcing bounded, and computing it in closed form is what avoids
        the catastrophic cancellation of the automatic-differentiation route.
        """
        return -self.comparison_diffusivity * self.second_space_derivative(coord, t)

    def derivative_callables(self) -> dict:
        """The ``{"dt", "dx", "dxx"}`` mapping consumed by ``TerminalAnsatz``."""
        return {
            "dt": self.time_derivative,
            "dx": self.space_derivative,
            "dxx": self.second_space_derivative,
        }

    # -- reporting --------------------------------------------------------

    def quadrature_floor_report(self) -> dict:
        """Tally of the unresolved-quadrature activations; never silent."""
        evaluated = max(self._floor_evaluation_count, 1)
        return {
            "quadrature_step": self.quadrature_step,
            "time_to_terminal_floor": self.time_to_terminal_floor,
            "activation_count": self._floor_activation_count,
            "evaluation_count": self._floor_evaluation_count,
            "activation_fraction": self._floor_activation_count / evaluated,
            "largest_unresolved_time_to_terminal": (
                self._floor_largest_time_to_terminal
            ),
        }


class GradedChenMangasarianExtensionField:
    r"""Graded mollifier extension
    :math:`h(x, t) = \mathcal{M}^{\mathrm{CM}}_{\varepsilon(t)}(\payoff(x), C(x))`.

    The smoothing scale is graded in time,

    .. math::

        \varepsilon(t)
            = \varepsilon_{0}\,
              \Bigl(\frac{T - t}{T}\Bigr)^{q} ,
        \qquad q \in (0, +\infty) ,

    so that :math:`\varepsilon(T) = 0` and the datum
    :math:`\max(\payoff, C)` is imposed **exactly** at the slice --- the smoothing
    bias of the error recursion therefore vanishes, exactly as it does for the
    split extension.  What separates the two is the forcing:
    :math:`\partial_{xx} h` peaks like
    :math:`\nu\lambda^{2} / (2\varepsilon(t))` at the free boundary, so the strip
    forcing is square-integrable **if and only if** :math:`q \in (1/3, 1)` --- the
    curvature channel needing :math:`q < 1` and the time channel
    :math:`q > 1/3` --- it is **never bounded** for any :math:`q`, and it is
    fourth-power integrable for no :math:`q` at all (the curvature channel needs
    :math:`q < 1/3`, the time channel :math:`q > 3/5`), so the Monte-Carlo
    estimator of the stage objective has infinite variance whatever the grading.
    The linear grading :math:`q = 1` is the marginal, logarithmically divergent
    member.

    Derivatives are taken by automatic differentiation: the field is
    :math:`C^{\infty}` for :math:`t < T`, there is no cancellation to perform (the
    forcing is genuinely large near the slice --- that is the finding), and no
    analytic bypass is supplied.

    Args:
        continuation_fn:  Callable ``C(x) -> Tensor``.
        K:                Strike.
        terminal_time:    Stage-local terminal time :math:`T`.
        smoothing_scale:  :math:`\varepsilon_{0}`.
        grading_exponent: :math:`q`; ``1.0`` is the linear grading.
    """

    def __init__(
        self,
        continuation_fn,
        *,
        K: float,
        terminal_time: float,
        smoothing_scale: float,
        grading_exponent: float = 1.0,
        name: str = "graded_chen_mangasarian",
    ) -> None:
        if smoothing_scale <= 0.0:
            raise ValueError(
                "smoothing_scale must be strictly positive; received "
                f"{smoothing_scale!r}. The limit eps_0 -> 0 is the exact maximum, "
                "which is the raw datum and is not built by this class."
            )
        if grading_exponent <= 0.0:
            raise ValueError(
                "grading_exponent must be strictly positive; received "
                f"{grading_exponent!r}. A vanishing exponent is the constant "
                "mollifier, whose scale does not vanish at the exercise date and "
                "which therefore injects a slice bias."
            )
        self._continuation_fn = continuation_fn
        self.K = float(K)
        self.terminal_time = float(terminal_time)
        self.smoothing_scale = float(smoothing_scale)
        self.grading_exponent = float(grading_exponent)
        self.name = name

    def smoothing_scale_at(self, t: torch.Tensor) -> torch.Tensor:
        r"""The graded scale :math:`\varepsilon(t)`, vanishing at the slice."""
        relative_time_to_terminal = torch.clamp(
            (self.terminal_time - t) / self.terminal_time, min=0.0
        )
        return self.smoothing_scale * relative_time_to_terminal**self.grading_exponent

    def field(self, coord: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r"""The extension :math:`h(x, t)`, shape ``(N, 1)``."""
        coord = coord.reshape(-1, 1)
        t = t.reshape(-1, 1)
        payoff = heat_put_payoff(coord, self.K)
        continuation = self._continuation_fn(coord.reshape(-1)).reshape(-1, 1)
        scale = self.smoothing_scale_at(t)
        # chen_mangasarian_max takes a scalar eps; the graded scale is a field, so
        # the formula is written out with the tensor scale.
        return 0.5 * (
            payoff
            + continuation
            + torch.sqrt((payoff - continuation) ** 2 + scale**2)
        )


def _autograd_space_derivative(function, coord_column: torch.Tensor) -> torch.Tensor:
    """First spatial derivative of ``function`` at ``coord_column`` by autograd."""
    with torch.enable_grad():
        point = coord_column.detach().clone().requires_grad_(True)
        value = function(point.reshape(-1)).reshape(-1, 1)
        (derivative,) = torch.autograd.grad(
            value, point, grad_outputs=torch.ones_like(value), create_graph=False
        )
    return derivative.detach()


def constant_chen_mangasarian_datum(continuation_fn, *, K: float, smoothing_scale: float):
    r"""The constant-scale smoothed datum
    :math:`\mathcal{M}^{\mathrm{CM}}_{\varepsilon_{0}}(\payoff, C)`.

    This is the received baseline: the scale does **not** vanish at the exercise
    date, so the datum actually imposed at the slice is not the maximum and the
    error recursion charges a bias :math:`\smoothMaxBias_{k} \in
    [0, \varepsilon_{0}/2]` at every interior exercise date.  Returned as a plain
    ``V(x)`` callable, to be used as the ``terminal_datum`` of the trial solution
    (no interior profile is chosen: the extension is the datum itself, weighted by
    the interpolation coefficient of the convex form).
    """

    def datum(x: torch.Tensor) -> torch.Tensor:
        return chen_mangasarian_max(
            heat_put_payoff(x, K), continuation_fn(x), smoothing_scale
        )

    return datum


def exact_maximum_datum(continuation_fn, *, K: float):
    r"""The exact glued datum :math:`V = \max(\payoff, C)`.

    No mollification: the datum imposed at the slice is the maximum itself, so the
    smoothing bias of the error recursion vanishes.  Its first derivative jumps at
    the free boundary :math:`\{\payoff = C\}`, of regularity order one *whatever*
    the regularity of the payoff --- the maximum manufactures the corner.
    """

    def datum(x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(heat_put_payoff(x, K), continuation_fn(x))

    return datum
