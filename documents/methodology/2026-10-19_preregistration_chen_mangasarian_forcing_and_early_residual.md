# Pre-registration — forcing energy of the Chen–Mangasarian extension, and early-time residual of the trained fields

**Written 2026-10-19, before any computation of this study.** The author asked which analytical results distinguish the algebraic-kernel (Chen–Mangasarian) extension from the retained evolution, and then to run the two tests proposed for it (« yes go ahead », 2026-10-19). No value of this study has been computed before this document. Times are paper times, with the datum at $t=0$; the runner's backward convention places the datum at $t=T$, so a paper time $\tau T$ is the runner slice fraction $1-\tau$.

## 1. Questions

1. **Forcing energy.** The Chen–Mangasarian extension convolves the datum with the kernel $\kappa_\varepsilon(y)=\varepsilon^2/(2(y^2+\varepsilon^2)^{3/2})$, of Fourier multiplier $\varphi(z)=zK_1(z)$, at the width $\varepsilon(t)=\varepsilon_0t/T$, with $\varepsilon_0=\sqrt{2\nu T}$ and $\nu=(|\gamma_{2p}|T)^{1/p}/T$. For $t\le T/(\varepsilon_0|k|)$ it leaves the component of wavenumber $k$ almost unchanged, so that its forcing there is $-\lambda_{\mathcal L^\Omega,k}\hat g_k$, that of the raw constant extension. The derivation of 2026-10-19 gives, for the datum with $|\hat g_k|=1/(2\pi^2k^2)$ and the band-truncated forcing energy $E(K)=2\pi\sum_{0<|k|\le K}\int_0^T|\widehat{\mathcal Lh}(k,t)|^2\,\mathrm dt$,
$$E(K)=\frac{\gamma_{2p}^2\,I}{\pi^3\varepsilon_0}\sum_{k\le K}k^{4p-5}\,(1+o(1)),\qquad I=\int_0^\infty(zK_1(z))^2\,\mathrm dz=\frac{3\pi^2}{32},$$
as $K\to\infty$. This study measures $E(K)$ in closed form to test this asymptotic law.
2. **Early-time residual.** At $t=0$ the Chen–Mangasarian extension has the initial time derivative $0$, so its residual tends to $-\mathcal L^\Omega g$, whereas the retained evolution has the initial time derivative $Ag$. This study measures the residual of the saved trained fields at early paper times, to test whether the residual of the trained Chen–Mangasarian fields is concentrated there relative to the highest-order split.

## 2. Design

**Cells.** The two circle cells of the paper: `g2_bernoulli_full` ($G_1=0.125\,\partial_{xx}-0.095\,\partial_x-0.03$, $p=1$, $\varepsilon_0=0.5$) and `g3_bernoulli_full` ($G_2=-0.05\,\partial_{xxxx}+1.3\,\partial_x-0.4$, $p=2$, $\varepsilon_0=\sqrt{2\sqrt{0.05}}\simeq0.669$), $T=1$, datum `PeriodisedBernoulliDatum(1)`. The extensions are built by the runner's `build_closed_form_extension`, so that they are the objects of the experiments.

**Part 1 (closed form, no network).** $E(K)$ at the dyadic band edges $K=2^j$, $4\le j\le20$, for the Chen–Mangasarian extension, the highest-order split and the constant-in-time extension, on both cells. The Chen–Mangasarian time integral is evaluated in the scaled variable $z=\varepsilon_0|k|(T-t)/T$ on panels refined at $z=0$, where the integrand has a $z\ln z$ term; the previous evaluation (256 Gauss–Legendre nodes in $t$) does not resolve the time scale $T/(\varepsilon_0|k|)$ at large $|k|$. The new quadrature is checked against adaptive quadrature at several wavenumbers before any value is saved.

**Part 2 (saved trained fields, no retraining).** The saved `model.pt` of the three seeds of: the Chen–Mangasarian extension (`full_datum_remaining_choices_2026-10-12`), the highest-order split and the constant-in-time extension (`full_datum_sampling_2026-10-08`), on both cells. For each run, the production spectra routine `compute_spectra` is called at one slice at a time, at the paper times $\tau\in\{0.001,0.01,0.03,0.1,0.5\}$. Measured per run and slice: the residual power of the trained field and the closed-form forcing power, each summed over the low band $1\le k\le8$ and the high band $32\le k\le256$.

**Compute.** Jean Zay `prepost` (non-billed), account `akz@cpu`. No training and no gradient step.

## 3. Predictions

**Part 1.**
- **F1 (order two).** For the Chen–Mangasarian extension on `g2_bernoulli_full`, each dyadic increment $E(2^{j+1})-E(2^j)$, $15\le j\le19$, is positive and lies within $2\,\%$ of $\ln2\cdot\gamma_2^2I/(\pi^3\varepsilon_0)\simeq6.464\times10^{-4}$.
- **F2 (order four).** For the Chen–Mangasarian extension on `g3_bernoulli_full`, the least-squares log–log slope of $E(K)$ over $2^{16}\le K\le2^{20}$ lies within $0.05$ of $4$, and $E(2^{20})/(C_2\,2^{80})$ lies within $2\,\%$ of $1$, with $C_2=\gamma_4^2I/(4\pi^3\varepsilon_0)\simeq2.789\times10^{-5}$.
- **F3 (control).** For the highest-order split, the relative increment over the last dyadic step is below $10^{-3}$ on both cells.
- **F4 (control).** For the constant-in-time extension, the log–log slope over $2^{16}\le K\le2^{20}$ lies within $0.05$ of $4p-3$ (Corollary A.1 of the paper): $1$ on `g2`, $5$ on `g3`.

**Part 2.** Let $R(\tau)$ be the ratio of the median over seeds of the high-band residual power of the trained Chen–Mangasarian fields to that of the trained highest-order splits, at the paper time $\tau$.
- **S1 (closed-form check of the initial time derivative).** At $\tau=0.001$, the high-band forcing power of the Chen–Mangasarian extension lies within $10\,\%$ of that of the constant-in-time extension, on both cells.
- **S2.** $R(0.001)>R(0.1)$ on both cells.
- **S3.** $R(0.001)\ge10$ on both cells.

No prediction is made on the cancellation ratio (residual over forcing) of the trained fields; it is reported.

**Admissibility.** A run of part 2 is admissible when its `model.pt` loads into the rebuilt ansatz and every reported power is finite. A run that is not admissible is reported with its seed and excluded.

## 4. Outcome

To be written after the job.
