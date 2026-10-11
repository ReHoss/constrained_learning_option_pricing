# Pre-registration — threshold of finite forcing energy in the half order $p$ and the remainder order $m_B$

**Written 2026-10-11, before any run of the measurement script of this study.** On 2026-10-11 the author asked to « tester les combinaisons de seuils de $m_B$ et $p$ », and noted that the relation between $p$ and $m_B$ is less direct at $p=3$ than at $p=2$. The only computations made before this document are the unit tests of commit `17bc1ff`. They evaluate five cells of the grid below for $p=3$, at band edges $2^{14}$ and $2^{18}$, to check the code. Their values are not results of this study.

## 1. Question

Let $A$ be a parabolic retained operator with constant coefficients and let $B=\mathcal L^X-A$ be its remainder, of order $m_B$, with $s_B=\max(m_B-p,0)$.
- Proposition 4.2(iii) of the paper bounds the forcing energy $E_h$ by $\lVert g\rVert_{H^{s_B}}^2$.
- Proposition 4.3(i) states that, for $m_B>p$, $E_h<+\infty$ if and only if $g\in H^{s_B}$.

Consider a datum that belongs to $H^s$ exactly for $s<\sigma$, with $\sigma\in(0,+\infty)$. The two propositions then give
$$E_h<+\infty\iff m_B-p<\sigma .$$
A truncation of $\mathcal L^X$ that keeps its term of order $2p$ has $m_B\le2p-1$. Every such truncation therefore gives finite forcing energy if and only if $p-1<\sigma$.
- For the datum of the paper, $\sigma=3/2$. Every truncation is then admissible at $p\in\{1,2\}$.
- At $p=3$, the truncation that retains only the term of order six has $m_B=5$ and $s_B=2>3/2$, so it gives infinite forcing energy.

The study tests this relation between $p$, $m_B$ and $\sigma$ for every truncation at $p\in\{1,2,3\}$ and three data. The test is in closed form, without training.

## 2. Design

**Convention.** The library and the runner use the backward convention $\partial_tu+\mathcal L^Xu=0$, with the datum at $t=T$. Under $t\mapsto T-t$ this is the forward problem of the paper, $\partial_tu=\mathcal L^Xu$ with $u(\cdot,0)=g$, with the same operator and the same forcing energy. The domain is $\mathbb T=\mathbb R/(2\pi\mathbb Z)$ and $T=1$.

**Generators.** For $p\in\{1,2,3\}$, the generator $\mathcal L^X_p=\sum_{j=0}^{2p}\gamma_{\mathcal L^X,j}\,\partial^j/\partial x^j$ has a coefficient of modulus one at every order:
$$\gamma_{\mathcal L^X,j}=1\ \ (j\ \text{odd}),\qquad \gamma_{\mathcal L^X,j}=(-1)^{j/2+1}\ \ (j\ \text{even}),$$
so that $\gamma_{\mathcal L^X,j}(ik)^j=-k^j$ for every even $j$. Explicitly:
- $\mathcal L^X_1=\partial^2_{xx}+\partial_x-1$;
- $\mathcal L^X_2=-\partial^4_{xxxx}+\partial^3/\partial x^3+\partial^2_{xx}+\partial_x-1$;
- $\mathcal L^X_3=\partial^6/\partial x^6+\partial^5/\partial x^5-\partial^4_{xxxx}+\partial^3/\partial x^3+\partial^2_{xx}+\partial_x-1$.

Every even-order term is dissipative. Every subset of orders that contains $2p$ therefore defines a parabolic retained operator. The coefficients have equal modulus so that the relative corrections to the leading behaviour of $|\lambda_{B,k}|^2$ are of order $k^{-2}$ with a constant of order one. The predicted exponents below do not depend on the coefficients.

**Extensions.** For every $m\in\{0,\dots,2p-1\}$, the truncation $A_{p,m}=\sum_{j>m}\gamma_{\mathcal L^X,j}\,\partial^j/\partial x^j$ is retained. Its remainder $B=\sum_{j\le m}\gamma_{\mathcal L^X,j}\,\partial^j/\partial x^j$ has order $m_B=m$, and the extension is $h=e^{tA_{p,m}}g$. The constant-in-time extension $h=g$ is added as a reference: it is the empty subset, with $m_B=2p$ and no smoothing. Truncations with $m\le p$ satisfy $m_B\le p$, the case covered for every continuous datum by Proposition 4.3(ii).

**Data.** The data are $g_\alpha(x)=\sum_{k\ge1}\cos(kx)/(\pi^2k^\alpha)$ for $\alpha\in\{5/4,2,3\}$ (class `PowerLawCosineDatum`).
- For $\alpha>1$ the series converges absolutely, so $g_\alpha\in C(\mathbb T;\mathbb R)\cap L^2(\mathbb T;\mathbb R)$, the hypothesis of Propositions 4.2 and 4.3.
- With the Sobolev norm $\lVert g\rVert_{H^s}^2=2\pi\sum_k(1+|k|)^{2s}|\hat g_k|^2$, $g_\alpha\in H^s(\mathbb T)$ exactly for $s<\sigma=\alpha-1/2$. This gives $\sigma\in\{3/4,3/2,5/2\}$.
- $\alpha=2$ is the datum of the paper.
- $\alpha=5/4$ is a continuous datum outside $H^1(\mathbb T)$, which witnesses the converse of Proposition 4.3(ii) at $m_B=p+1$.

**Measured quantity.** The band-truncated forcing energy, in closed form, is
$$E(K)=2\pi\sum_{0<|k|\le K}|\lambda_{B,k}|^2\,|\hat g_k|^2\int_0^T\bigl|e^{t\lambda_{A,k}}\bigr|^2\,\mathrm dt .$$
It is the forcing energy of the extension of the band-limited datum $\sum_{1\le k\le K}\cos(kx)/(\pi^2k^\alpha)$. The time integral of each wavenumber is evaluated exactly (library function `exponential_time_integral_factor`). By Parseval's identity and monotone convergence, $E_h(g_\alpha)=\lim_{K\to\infty}E(K)$. Hence $E_h(g_\alpha)<+\infty$ exactly when $E(K)$ is bounded.

**Grid and computation.**
- Band edges $K=2^4,\dots,2^{22}$.
- $3$ data $\times(3+5+7)$ extensions $=45$ cells.
- Script: `experiments/python_scripts/exp_extension_split_generator/measure_forcing_energy_threshold_grid.py`.
- Jean Zay, partition `prepost` (non-billed CPU).
- A smoke run with `--debug` and $K\le2^{12}$ precedes the run; its values are not results.
- The computation is deterministic and consumes no random number generator.

## 3. Predictions

**Exponent.** The leading behaviour of the summand is elementary to derive (docstring of `predicted_split_forcing_energy_exponent`). As $|k|\to\infty$:
- $|\lambda_{B,k}|^2=\gamma_{\mathcal L^X,m_B}^2k^{2m_B}(1+O(k^{-2}))$;
- for a truncation, $\int_0^T|e^{t\lambda_{A,k}}|^2\mathrm dt=(2k^{2p})^{-1}(1+O(k^{-2}))$;
- for $h=g$, this time integral equals $T$.

It follows that $E(K)$ is bounded if $e<0$. If $e>0$, then $E(K)=C\,K^{e}(1+o(1))$ as $K\to\infty$, with
$$e=2(m_B-s_A-\sigma),\qquad s_A=p\ \text{(truncation)},\quad s_A=0\ (h=g),$$
and the constant $C$ given by `predicted_split_forcing_energy_constant`. For a truncation, $e<0$ is the condition $s_B<\sigma$ of Section 1. Since $\sigma\notin\mathbb Z$, the case $e=0$ does not occur in the grid.

**Predicted exponents $e$** (rows $p$, columns $m_B$; the last entry of each row is $h=g$):

$\alpha=5/4$, $\sigma=3/4$:

| $p$ | $m_B=0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ |
|---|---|---|---|---|---|---|---|
| $1$ | $-3.5$ | $-1.5$ | $2.5$ ($h=g$) | | | | |
| $2$ | $-5.5$ | $-3.5$ | $-1.5$ | $0.5$ | $6.5$ ($h=g$) | | |
| $3$ | $-7.5$ | $-5.5$ | $-3.5$ | $-1.5$ | $0.5$ | $2.5$ | $10.5$ ($h=g$) |

$\alpha=2$, $\sigma=3/2$ (datum of the paper):

| $p$ | $m_B=0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ |
|---|---|---|---|---|---|---|---|
| $1$ | $-5$ | $-3$ | $1$ ($h=g$) | | | | |
| $2$ | $-7$ | $-5$ | $-3$ | $-1$ | $5$ ($h=g$) | | |
| $3$ | $-9$ | $-7$ | $-5$ | $-3$ | $-1$ | $1$ | $9$ ($h=g$) |

$\alpha=3$, $\sigma=5/2$:

| $p$ | $m_B=0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ |
|---|---|---|---|---|---|---|---|
| $1$ | $-7$ | $-5$ | $-1$ ($h=g$) | | | | |
| $2$ | $-9$ | $-7$ | $-5$ | $-3$ | $3$ ($h=g$) | | |
| $3$ | $-11$ | $-9$ | $-7$ | $-5$ | $-3$ | $-1$ | $7$ ($h=g$) |

These tables imply the following pattern for the truncations:
- for $\sigma=3/4$, the first growing truncation is $m_B=p+1$ at $p\in\{2,3\}$, and no truncation grows at $p=1$;
- for $\sigma=3/2$, the only growing truncation is $p=3$, $m_B=5$;
- for $\sigma=5/2$, no truncation grows.

The constant-in-time extension grows in every cell except $p=1$, $\alpha=3$, where $g_3\in H^2$ gives $\mathcal L^Xg_3\in L^2$.

**Pre-registered criteria**, with $K_{\max}=2^{22}$ and the relative increment $\iota=(E(K_{\max})-E(K_{\max}/2))/E(K_{\max})$:
- **R1 (boundedness).** In every cell with $e<0$, $\iota<10^{-3}$. In every cell with $e>0$, $\iota>10^{-1}$; the predicted value is $1-2^{-e}$, which is at least $1-2^{-1/2}\simeq0.29$.
- **R2 (rate).** In every cell with $e>0$:
  - the log–log slope fitted over $K\in[2^{16},2^{22}]$ lies within $5\times10^{-2}$ of $e$;
  - $E(K_{\max})/(C\,K_{\max}^{e})$ lies within $2\times10^{-2}$ of $1$.

The tolerances reflect the slowest predicted convergence, which occurs in the cells with $e=1/2$. There, $E(K)-C\,K^{1/2}$ tends to a constant, so the relative deviation decays like $K^{-1/2}$.

**Checks that abort the run.** Every energy value must be finite. The chunked accumulation at $K=2^4$ must equal the library function `total_strip_forcing_squared` to a relative tolerance of $10^{-12}$.

**Claim strength.** Propositions 4.2(iii) and 4.3 are proved in the paper. The exponent and the constant are the elementary asymptotics stated above, which the paper does not state. The measurement tests the threshold that the propositions imply, and the rate that the asymptotics give, on finite band edges.

## 4. Outcome

Written 2026-10-11 after Jean Zay job 913797 (`prepost`, account `akz@cpu`, revision `0ece300`, clean tree).
- The job ran the unit tests of the prediction functions (65 passed), a smoke run with `--debug` at $K\le2^{12}$, then the pre-registered run.
- The smoke run gave R2 in 11 of its 12 growing cells. It is not a result of the study.
- The pre-registered run took $110$ s.
- Run directory: `data/measure_forcing_energy_threshold_grid/2026-10-11-12-32-38-947956Z_log2K22/`, copied to the laptop. Its figures `forcing_energy_against_band_edge.png` and `threshold_table.png` were rebuilt from the saved artefacts with `--replot` after a correction of the position of the threshold marker in the table.

**Checks.** Every energy value is finite. The largest relative discrepancy between the chunked accumulation and `total_strip_forcing_squared` at $K=2^4$ is $\simeq2.5\times10^{-16}$.

**R1 holds in the 45 cells.**
- In the 33 cells with $e<0$, the largest relative increment over the last dyadic step is $\iota\simeq5.8\times10^{-8}$ ($\alpha=2$, $p=2$, $m_B=3$, $e=-1$). In 26 of these cells, $\iota=0$ in double precision: the summands with $2^{21}<|k|\le2^{22}$ fall below the rounding resolution of the accumulated sum.
- In the 12 cells with $e>0$, the smallest increment is $\iota\simeq0.292$ ($\alpha=5/4$, $p=3$, $m_B=4$, $e=1/2$), against the predicted $1-2^{-1/2}\simeq0.293$.

**R2 holds in the 12 cells with $e>0$.** The largest deviation of the fitted slope from $e$ is $\simeq2.5\times10^{-3}$, and the largest deviation of $E(K_{\max})/(C\,K_{\max}^{e})$ from $1$ is $\simeq1.6\times10^{-3}$. Both occur at $\alpha=5/4$, $p=3$, $m_B=4$, the cell with the slowest predicted convergence.

| $\alpha$ | $p$ | $m_B$ | Extension | $e$ | Fitted slope | $E(K_{\max})/(C\,K_{\max}^e)$ |
|---|---|---|---|---|---|---|
| $5/4$ | $1$ | $2$ | $h=g$ | $2.5$ | $\simeq2.5000$ | $\simeq1.0000$ |
| $5/4$ | $2$ | $3$ | truncation | $0.5$ | $\simeq0.4997$ | $\simeq1.0002$ |
| $5/4$ | $2$ | $4$ | $h=g$ | $6.5$ | $\simeq6.5000$ | $\simeq1.0000$ |
| $5/4$ | $3$ | $4$ | truncation | $0.5$ | $\simeq0.4975$ | $\simeq1.0016$ |
| $5/4$ | $3$ | $5$ | truncation | $2.5$ | $\simeq2.5000$ | $\simeq1.0000$ |
| $5/4$ | $3$ | $6$ | $h=g$ | $10.5$ | $\simeq10.5000$ | $\simeq1.0000$ |
| $2$ | $1$ | $2$ | $h=g$ | $1$ | $\simeq1.0000$ | $\simeq1.0000$ |
| $2$ | $2$ | $4$ | $h=g$ | $5$ | $\simeq5.0000$ | $\simeq1.0000$ |
| $2$ | $3$ | $5$ | truncation | $1$ | $\simeq1.0000$ | $\simeq1.0000$ |
| $2$ | $3$ | $6$ | $h=g$ | $9$ | $\simeq9.0000$ | $\simeq1.0000$ |
| $3$ | $2$ | $4$ | $h=g$ | $3$ | $\simeq3.0000$ | $\simeq1.0000$ |
| $3$ | $3$ | $6$ | $h=g$ | $7$ | $\simeq7.0000$ | $\simeq1.0000$ |

**Pattern of the truncations** (measured, and identical to the predicted pattern of Section 3):
- with the datum of the paper ($\alpha=2$), every truncation gives a bounded $E(K)$ at $p\in\{1,2\}$. At $p=3$ the truncations with $m_B\le4$ are bounded, and the truncation retaining only the term of order six ($m_B=5$, $s_B=2$) grows linearly in $K$;
- with $\alpha=5/4$, a continuous datum outside $H^1(\mathbb T)$, the bounded truncations are exactly those with $m_B\le p$, the case of Proposition 4.3(ii), at every $p\in\{1,2,3\}$;
- with $\alpha=3$, every truncation is bounded at $p\le3$, and the constant-in-time extension is bounded at $p=1$.
