# Spectral forcing floor and split terminal-data extensions on the circle

This note documents the periodic spectral-forcing study implemented by
`learning_option_pricing/pde/periodic_spectral_toolbox.py`,
`learning_option_pricing/pde/terminal_data_extensions.py`, and the three
experiment scripts under
`experiments/python_scripts/exp_extension_split_generator/`.  It records the
setting, the exact datum classes, the generator catalogue, the extension
catalogue with every closed-form time integral, the predictions P1–P4 (with
P4 in its corrected form), and the artefact contract of each script.

The propositions instantiated here are Propositions 3–7 of the methodology
report "On boundary-constrained learning of partial differential equations"
(repository `2026_01_29_constrained_learning_pde_lehalle_hosseinkhan`, file
`boundary_constrained_learning_problem.tex`): Proposition 3 ("Regularity
governs spectral decay", now stated for every integer regularity order
$\rho \ge 0$), Proposition 4 ("Elliptic amplification"), Proposition 5
("Forcing floor under mollification"), Proposition 6 ("Ideal-filter model"),
and Proposition 7 ("Split-generator extension removes the floor").  The
report is stated on the real line with the Fourier transform; the study
below is its instantiation on the circle with Fourier coefficients, where
every quantity admits an exact closed form.

## 1. Periodic setting and Fourier convention

All objects are defined on the circle $[0, 2\pi)$.  For a function
$f \in L^2(0, 2\pi)$ and an integer wavenumber $k \in \mathbb{Z}$, the
Fourier coefficient of $f$ at $k$ is

$$
c_k = \frac{1}{2\pi} \int_0^{2\pi} f(x)\, e^{-ikx}\, dx,
\qquad
f(x) = \sum_{k \in \mathbb{Z}} c_k e^{ikx},
$$

so that Parseval's identity reads
$\|f\|_{L^2(0,2\pi)}^2 = 2\pi \sum_{k \in \mathbb{Z}} |c_k|^2$.

Measurement policy.  Every measurement of the study is evaluated from the
exact analytic Fourier coefficients over integer wavenumber arrays
(vectorised `complex128`); the fast Fourier transform of sampled values is
never used as a measurement.  FFT synthesis exists only to plot a datum on a
grid (`synthesise_datum_on_grid`), with the band truncation stated in its
docstring.

## 2. Terminal-datum classes

### 2.1 Periodised Bernoulli datum (`PeriodisedBernoulliDatum`)

Let $\rho \in \{0, 1, 2\}$ be the regularity index and let $B_n$ denote the
$n$-th Bernoulli polynomial.  The datum is defined by
$g(x) = B_{\rho+1}(x / 2\pi)$ on $[0, 2\pi)$, extended periodically.  It has
a single break point at $x^\star = 0$: the datum belongs to $C^{\rho-1}$
(piecewise polynomial), its $\rho$-th derivative jumps at $x^\star$, and its
exact Fourier coefficients are

$$
c_0 = 0,
\qquad
c_k = -\frac{(\rho+1)!}{(2\pi i k)^{\rho+1}} \quad (k \ne 0),
$$

so that $|c_k| = (\rho+1)! / (2\pi |k|)^{\rho+1}$ holds exactly, with no
remainder term.  The jump of the $\rho$-th derivative at the break point is

$$
J = -\frac{(\rho+1)!}{(2\pi)^{\rho}} .
$$

Explicitly: $\rho = 0$ gives $J = -1$; $\rho = 1$ gives
$J = -2/(2\pi) = -1/\pi \approx -0.318$; $\rho = 2$ gives
$J = -6/(2\pi)^2 = -3/(2\pi^2) \approx -0.152$.

### 2.2 Square-wave datum (`SquareWaveDatum`)

The square wave equals $+1$ on $(0, \pi)$ and $-1$ on $(\pi, 2\pi)$, with
exact coefficients

$$
c_0 = 0,
\qquad
c_k = \frac{2}{i \pi k} \ (k \text{ odd}),
\qquad
c_k = 0 \ (k \text{ even}).
$$

Its regularity index is $\rho = 0$ and it has two break points: the datum
jumps by $+2$ at $x = 0$ and by $-2$ at $x = \pi$.  This is the
multi-singularity extension case: the single-break-point floor prediction of
Section 5.2 does not apply verbatim, so the square-wave floor curve is
measured only, with no prediction line.

## 3. Constant-coefficient generator and its symbol

A constant-coefficient spatial generator
$A = \sum_j a_j \partial_x^j$, with real coefficients $a_j$ and
differential orders $j$ ranging over a finite set of non-negative integers,
acts on the spectral component $e^{ikx}$ as multiplication by the Fourier
symbol

$$
a(k) = \sum_j a_j (ik)^j \in \mathbb{C}.
$$

The maximal order $2p$ must be even; the principal constant is
$a_0 = \lim_{|k| \to \infty} |a(k)| / k^{2p} = |a_{2p}|$.

Dissipativity is validated, never silently enforced: before a symbol (or a
subset symbol) is exponentiated into a semigroup, the bound
$\max_k \operatorname{Re} a(k) \le 10^{-12}$ must hold over the working
band, and a violation raises `ValueError` reporting the offending wavenumber
and real part.

Symbol splitting.  For a subset $A$ of the generator's differential orders,
the split is $a(k) = a_A(k) + b(k)$, where
$a_A(k) = \sum_{j \in A} a_j (ik)^j$ is the subset symbol and
$b(k) = a(k) - a_A(k)$ is the defect symbol.  The defect order is
$m = \max\{ j \notin A \}$ (with $m = -\infty$ when the subset exhausts
every order, in which case $b = 0$ identically).

Named generators of the study:

| Tag | Name | Symbol $a(k)$ | Half order $p$ | Principal constant $a_0$ |
|-----|------|---------------|----------------|--------------------------|
| G1 | advection–diffusion–reaction | $-0.7 k^2 + 1.3\,ik - 0.4$ | $1$ | $0.7$ |
| G2 | Black–Scholes log-price ($\sigma = 0.5$, $r = 0.03$) | $-0.125 k^2 - 0.095\,ik - 0.03$ | $1$ | $0.125$ |
| G3 | biharmonic–advection–reaction | $-0.05 k^4 + 1.3\,ik - 0.4$ | $2$ | $0.05$ |

For G2 the coefficients arise from the Black–Scholes generator in the
log-price coordinate,
$\tfrac{\sigma^2}{2} \partial_{xx} + (r - \tfrac{\sigma^2}{2}) \partial_x - r$.
For G3, $(ik)^4 = k^4$, so the dissipative order-4 coefficient is negative.

## 4. Extension catalogue and closed-form time integrals

A terminal datum $g$ with coefficients $c_k$ is extended from the terminal
slice $t = T$ (the study uses $T = 1$) into the strip
$(0, T) \times [0, 2\pi)$ by an analytic extension $h$ with per-wavenumber
coefficient $\hat h(k, t)$.  The forcing of the extension under the
generator with symbol $a$ is

$$
\widehat{Lh}(k, t) = \partial_t \hat h(k, t) + a(k)\, \hat h(k, t),
$$

and the squared strip norm of the forcing is (Parseval)

$$
\|Lh\|^2_{L^2((0,T);\,L^2(0,2\pi))}
= 2\pi \sum_{0 < |k| \le K_{\max}} \int_0^T |\widehat{Lh}(k, t)|^2\, dt .
$$

Every extension class exposes a closed-form value of the per-wavenumber time
integral $\int_0^T |\widehat{Lh}(k, t)|^2 dt$; no time quadrature is used
anywhere.  Let $\varphi$ be the function defined by
$\varphi(z) = (e^{zT} - 1)/z$ for $z \ne 0$ and $\varphi(0) = T$ (the
$z \to 0$ limit, implemented as an explicit branch on the exact zero, never
through a denominator epsilon).

| Extension (class) | Coefficient $\hat h(k, t)$ | Forcing $\widehat{Lh}(k, t)$ | Closed-form $\int_0^T \lvert\widehat{Lh}(k, t)\rvert^2 dt$ |
|---|---|---|---|
| Convex raw (`ConvexRawExtension`) | $(t/T)\, c_k$ | $c_k \bigl( \tfrac{1}{T} + \tfrac{t}{T} a(k) \bigr)$ | $\lvert c_k\rvert^2 \bigl( \lvert\alpha\rvert^2 T + \operatorname{Re}(\overline{\alpha}\beta) T^2 + \lvert\beta\rvert^2 T^3/3 \bigr)$, where $\alpha = 1/T$ and $\beta = a(k)/T$ |
| Constant-in-time (`ConstantInTimeExtension`) | $c_k$ | $a(k)\, c_k$ | $T\, \lvert a(k)\rvert^2 \lvert c_k\rvert^2$ |
| Split semigroup (`SplitSemigroupExtension`) | $e^{(T-t)\, a_A(k)}\, c_k$ | $b(k)\, \hat h(k, t)$ | $\lvert b(k)\rvert^2 \lvert c_k\rvert^2\, \varphi\bigl(2 \operatorname{Re} a_A(k)\bigr)$ |
| Graded Gaussian (`GradedGaussianExtension`), comparison diffusivity $\nu_c \ge 0$ | $e^{-(T-t)\, \nu_c k^2}\, c_k$ | $\bigl( a(k) + \nu_c k^2 \bigr)\, \hat h(k, t)$ | $\lvert a(k) + \nu_c k^2\rvert^2 \lvert c_k\rvert^2\, \varphi(-2 \nu_c k^2)$ |
| Exact solution (`ExactSolutionExtension`) | $e^{(T-t)\, a(k)}\, c_k$ | $0$ identically | $0$ |

The convex raw extension arises from the linear terminal-distance factor
$d_T(t) = 1 - t/T$ (so $1 - d_T(t) = t/T$ and $d_T'(t) = -1/T$); it is the
trial-solution form analysed in the ideal-filter model (Proposition 6 of the
methodology report).  The graded Gaussian extension with matched comparison
diffusivity $\nu_c = \nu$ (where $\nu$ denotes the generator's order-2
coefficient) coincides with the split extension of subset
$A = \{\partial_{xx}\}$; the mismatched case uses $\nu_c = \nu/2$.

## 5. Predictions P1–P4

### 5.1 P1 — split forcing identity, and P1b — Gaussian-smoothed put

Prediction P1 instantiates Proposition 7(i) of the methodology report: for
the split semigroup extension,

$$
\widehat{Lh}(k, t)
= \partial_t \hat h(k, t) + a(k)\, \hat h(k, t)
= b(k)\, \hat h(k, t),
$$

the forcing reduces to the defect symbol applied to the extension
coefficient.  The verification (`verify_split_identity.py`) measures the
relative residual
$\max_k |(\partial_t \hat h + a \hat h) - b \hat h| / \max_k |b \hat h|$
at $t \in \{0, T/2, 0.99\,T\}$ for six (generator, subset) configurations,
with tolerance $10^{-13}$, and checks that a central difference in time
converges to the analytic $\partial_t \hat h$ with a fitted log–log slope in
$[1.9, 2.1]$ (order 2).

Prediction P1b is the real-line Black–Scholes instance: the matched-variance
Gaussian smoothing of the put payoff,

$$
h(x, \tau) = K\,\Phi(d) - e^{x + \sigma^2 \tau / 2}\, \Phi(d - \sigma\sqrt{\tau}),
\qquad
d = \frac{\ln K - x}{\sigma\sqrt{\tau}},
\qquad
\tau = T - t,
$$

with $\Phi$ the standard normal cumulative distribution function, solves the
heat equation in $\tau$, so the second-order part of the Black–Scholes
operator cancels: $\partial_t h + \tfrac{\sigma^2}{2} \partial_{xx} h = 0$.
The cancellation is verified with `torch.autograd` in `float64` to a
relative residual of at most $10^{-10}$, normalised by the grid-wide scale
$\max(\max |\partial_t h|, \max |\tfrac{\sigma^2}{2} \partial_{xx} h|)$
(the two terms vanish simultaneously along the zero-crossing curve of
$\partial_{xx} h$, where a pointwise quotient is an ill-conditioned $0/0$
form); a complementary pointwise relative residual restricted to the
well-scaled grid points is recorded in the summary.

### 5.2 P2 — operator-channel floor prediction

Prediction P2 instantiates Propositions 3–5 of the methodology report on the
circle.  The band-truncated operator-channel floor of a generator with
symbol $a$ acting on a datum with coefficients $c_k$ is

$$
\mathrm{floor}(K) = 2\pi \sum_{0 < |k| \le K} |a(k)|^2 |c_k|^2 .
$$

For a periodised Bernoulli datum (single break point, regularity index
$\rho$, jump $J$ of the $\rho$-th derivative) and a generator of maximal
order $2p$ with principal constant $a_0$, the floor prediction is: with
growth exponent $e = 4p - 2\rho - 1$,

* if $e > 0$:
  $\mathrm{floor}(K) = C_{\mathrm{pred}}\, K^{e}\, (1 + o(1))$ as
  $K \to \infty$, with $C_{\mathrm{pred}} = a_0^2 J^2 / (\pi e)$;
* if $e < 0$: the floor saturates to its finite total sum (a plateau is
  reported instead of a slope).

The measurement grid crosses {G1, G3} with $\rho \in \{0, 1, 2\}$ (expected
exponents $e = 3, 1, -1$ for G1 and $e = 7, 5, 3$ for G3), adds (G2,
$\rho = 1$) with $e = 1$, and includes the square-wave datum on G1 as the
multi-singularity case, measured only.

### 5.3 P3 — forcing spectra at fixed time

With $\rho = 1$ ($|c_k| = 2/(2\pi k)^2$ exactly) the near-terminal
envelopes of the forcing modulus $|\widehat{Lh}(k, t)|$ are:

* constant-in-time and convex raw as $t \to T$: flat (white), value
  $a_0 |c_k| k^2 = a_0/(2\pi^2)$, the convex raw scaled by $t/T$;
* split of defect order $m$: $|b_m| |c_k| k^m \propto k^{m-\rho-1}$,
  decaying for $m \le \rho$;
* graded Gaussian mismatched ($\nu_c = \nu/2$): flat, with the power
  reduced by the factor $((\nu-\nu_c)/\nu)^2 = 0.25$ relative to
  constant-in-time;
* exact solution: identically zero (asserted exactly in floating point, not
  to a tolerance).

At $t < T$ the split / graded moduli equal the near-terminal envelope
multiplied by the semigroup factor $e^{-(T-t)\nu k^2}$ (respectively
$e^{-(T-t)\nu_c k^2}$); with the crossover wavenumber
$k_c = ((T-t)\nu)^{-1/2}$, the departure of the measured curve below the
envelope for $k \ge k_c$ is this factor, not a violation of the envelope
prediction.

### 5.4 P4 — total strip forcing versus the band edge (corrected)

The squared strip norm
$2\pi \sum_{0<|k|\le K_{\max}} \int_0^T |\widehat{Lh}(k,t)|^2 dt$ is
tabulated at the band edges $K_{\max} \in \{2^{12}, 2^{16}, 2^{20}\}$.  The
prediction, in its corrected form, is:

* unbounded (linear in $K_{\max}$) growth for the constant-in-time and
  convex raw extensions;
* convergence for the splits with defect order $m \le \rho$, for the
  matched graded extension ($\nu_c = \nu$), **and for the mismatched graded
  extension** ($\nu_c = \nu/2$);
* identically zero for the exact solution.

Convergence for the mismatched graded extension follows from the closed
form.  The per-wavenumber time integral is
$|a(k) + \nu_c k^2|^2 |c_k|^2\, \varphi(-2\nu_c k^2)$, and as
$|k| \to \infty$ one has
$\varphi(-2\nu_c k^2) = (2\nu_c k^2)^{-1} (1 + o(1))$ and
$|a(k) + \nu_c k^2|^2 = (\nu - \nu_c)^2 k^4 (1 + o(1))$, hence

$$
\int_0^T |\widehat{Lh}(k, t)|^2\, dt
= \frac{(\nu - \nu_c)^2}{2\nu_c}\, |c_k|^2 k^2\, (1 + o(1))
= O(k^{-2})
\qquad (|k| \to \infty)
$$

for $\rho = 1$ ($|c_k|^2 \propto k^{-4}$).  The tail is summable, so the
strip forcing converges as $K_{\max} \to \infty$.

**Slice versus strip.**  The flat forcing spectrum of the mismatched graded
extension (Section 5.3) holds only on the exact terminal slice $t = T$ — a
null set in time.  For $t < T$ the forcing modulus at wavenumber $k$ is
suppressed by the factor $e^{-(T-t)\nu_c k^2}$: the flat spectral component
is present only within a temporal boundary layer of width
$(2\nu_c k^2)^{-1}$ adjacent to the terminal slice, and the time integral
weights it by exactly that width.  The fixed-time $L^2(0, 2\pi)$ norm of the
forcing at $t = T$ therefore grows without bound in the band edge (the P3
statement), while the strip norm converges (the P4 statement); the two must
not be conflated.

**A refuted draft prediction, as a worked example of the verification
pass.**  An earlier draft of P4 predicted linear divergence in $K_{\max}$
for the mismatched graded extension, arguing from its flat terminal-slice
spectrum — the conflation described above.  The prediction was refuted on
both of the independent channels that the verification pass maintains: the
closed-form time integral (the $O(k^{-2})$ tail derived above) and the
measurement (the measured classification of the strip-forcing table is
`convergent` for both generators; observed in every run of
`measure_forcing_spectra.py` to date).  The corrected prediction
(`convergent`) is now encoded in `PREDICTED_STRIP_CLASSIFICATION`, with the
derivation and the history recorded in a code comment; the disagreement
mechanism (the `agreement` field of `summary.yaml` and the `[DISAGREEMENT]`
tag in the run log) remains in place for future use.  The episode is the
intended behaviour of the measurement policy: a predicted classification is
recorded next to the measured one, and a disagreement is flagged, never
suppressed.

## 6. Scripts, command-line synopses, artefact contracts

All three scripts follow the repository conventions: `--seed` (master seed,
recorded for the run-log contract; every computation is deterministic),
`--debug` (prepends `_debug_` to the output folder name; mechanically
required below the smoke-test thresholds), and `--replot RUN_DIR` (rebuilds
every figure from the saved artefacts alone, without recomputation).
Output directories are derived from the script filename via
`script_data_dir(__file__)`.

### 6.1 `verify_split_identity.py` (P1, P1b)

```
python verify_split_identity.py
    [--maximum-wavenumber 512]
    [--finite-difference-maximum-wavenumber 64]
    [--finite-difference-min-exponent 4]
    [--finite-difference-max-exponent 10]
    [--put-log-price-points 41]
    [--put-time-to-maturity-points 21]
    [--seed 0] [--debug] [--replot RUN_DIR]
```

Artefacts: `run_metadata.json`, `summary.yaml` (all residual tables and
PASS/FAIL outcomes), `split_identity_residuals.npz`,
`finite_difference_convergence.npz`, `gaussian_smoothed_put_fields.npz`,
`finite_difference_convergence.png`, `run.log`.  Exit code 0 if and only if
every check passes.

### 6.2 `measure_forcing_floor_scaling.py` (P2)

```
python measure_forcing_floor_scaling.py
    [--log2-minimum-band-edge 7]
    [--log2-maximum-band-edge 22]
    [--log2-fit-minimum-band-edge 10]
    [--chunk-length 524288]
    [--seed 0] [--debug] [--replot RUN_DIR]
```

Artefacts: `floor_curves.npz` (dyadic band edges and one floor curve per
grid cell), `summary.yaml` (per-cell fitted slopes, predicted exponents and
constants, measured-over-predicted ratios, cross-check discrepancies),
`run_metadata.json`, `command.txt`, `run.log`,
`forcing_floor_scaling.png`.  The chunked accumulation is cross-checked
against the library function `operator_channel_floor` at the smallest band
edge; a relative discrepancy above $10^{-12}$ raises `ValueError`.

### 6.3 `measure_forcing_spectra.py` (P3, P4)

```
python measure_forcing_spectra.py
    [--spectrum-max-wavenumber 4096]
    [--strip-band-edges 4096 65536 1048576]
    [--ratio-wavenumber 1048576]
    [--near-terminal-fraction 0.99]
    [--seed 0] [--debug] [--replot RUN_DIR]
```

Artefacts: `forcing_spectra_measurements.npz` (every measured curve and
every predicted envelope), `summary.yaml` (P4 strip-forcing table with
`measured_classification`, `predicted_classification`, `agreement` per
extension and generator; mismatch power ratio; matched-graded/split
coincidence), `run_metadata.json`, `command.txt`, `run.log`, the two spectra
figures `forcing_spectra__<generator>.png`, and
`strip_forcing_vs_band_edge.png`.  The "Measured: ..." sentence of the
strip-forcing formula box is composed at plot time from the saved
`measured_classification` fields, so the caption restates the artefact
rather than a hard-coded expectation.

## 7. Library modules and tests

* `learning_option_pricing/pde/periodic_spectral_toolbox.py` — datum
  classes, `ConstantCoefficientGenerator` (symbol, splits, dissipativity
  validation, semigroup multiplier), named generator factories, the
  operator-channel floor and its predicted exponent and constant.  Tested by
  `test/pde/test_periodic_spectral_toolbox.py`.
* `learning_option_pricing/pde/terminal_data_extensions.py` — the extension
  catalogue of Section 4 with the closed-form time integrals and the strip
  norm.  Tested by `test/pde/test_terminal_data_extensions.py` (split
  identity to machine precision, closed forms against dense trapezoid
  quadrature, mismatch power ratio, growth-versus-convergence contrast,
  dissipativity refusals).
* `learning_option_pricing/utils/figure_layout.py` — shared figure-layout
  helpers (formula box below the axes, layout checks, clip-safe saving),
  used by every figure of the study through thin `_figure_layout.py`
  compatibility shims in the experiment directories.  Tested by
  `test/utils/test_figure_layout.py`.
