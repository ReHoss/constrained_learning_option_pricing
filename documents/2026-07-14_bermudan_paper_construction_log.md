# Construction log — the Bermudan paper (2026-07-14)

Record of how `bermudan_chain_of_hard_constraints.tex` was built: the split from the
hard-constraint paper, the theorem that was attempted, the adversarial refutation of
it, and the adjudication that produced the theorem actually published. Written so the
construction can be re-read later; it is not part of either paper.

Documents (Overleaf submodule
`latex_documents/reports/2026_01_29_constrained_learning_pde_lehalle_hosseinkhan/`):

- `boundary_constrained_learning_problem.tex` — the hard-constraint paper, amputated of its Section 4.
- `bermudan_chain_of_hard_constraints.tex` — **new**, the Bermudan paper.
- `bermudan_option_pricing.tex` — the probabilistic/PDE formalisation note (**cited, not absorbed**).
- `cal_notes/`, `notation.sty` — signed by C.-A. Lehalle. **Untouched.**

---

## 1. The split

Section 4 of the hard-constraint paper (639 lines, `sec:bermudan-iteration` and
everything under it) was excised and became Sections 1–2 of the new paper.

### 1.1 What MOVED (from Section 4 into Section 3 of the hard-constraint paper)

The critical knot: `def:terminal-data-extension` was *defined* inside the Bermudan
Section 4 but *referenced* by the hard-constraint Section 3 in four places
(lines 287, 407, 796, 1558 of the original) — including by Proposition 2
(`prop:coefficient-nobias`, bias-freeness), which is foundational to the main paper.
Removing Section 4 would have left four undefined references in a load-bearing
proposition.

Moved into Section 3, de-indexed from the stage-$k$ form to the general strip form:

| Object | Old home | New home |
|---|---|---|
| `def:terminal-data-extension` | Bermudan §4 | hard-constraint §3, before `eq:additive-form` |
| `rem:extension-not-innocuous` | Bermudan §4 | hard-constraint §3, right after the definition |
| `rem:alternative-ansatz` | Bermudan §4 | hard-constraint §3, after `eq:convex-form` (recast: the pre-multiplied ansatz is the additive one under the relabelling $h \mapsto (1-d)h$) |

The Bermudan paper now **cites** the definition (as "Definition 2 of the methodology
report") rather than owning it.

### 1.2 Cross-references repaired

**Into Section 4 (broke on removal), all fixed in the hard-constraint paper:**

- `Section~\ref{sec:bermudan-iteration}` (line 274, the operator convention $\mathcal{L}=\partial_t + \mathcal{L}^X$) → replaced by a self-contained statement of the convention in place.
- `Definition~\ref{def:terminal-data-extension}` ×4 → resolved by the move above.

**Out of Section 4 (into the hard-constraint paper), all recast as prose citations
to the methodology report in the new paper** — it must compile standalone:
`def:boundary-constrained-model` (×3), `sec:learning-models`, `def:distance-to-boundary`,
`sec:coefficient-theory` (×2), `ex:l2-residual-loss`, `sec:stylised-facts`.
The new paper restates each object it needs in `\S1.2` ("Constructions inherited from
the methodology report") with its own labels.

**Result: 0 undefined references in both documents.**

### 1.3 Macros

Migrated out of the hard-constraint preamble into the new paper: `\heatSemigroup` ($S$),
`\exerciseOperator` ($\Pi$), `\stageError` ($e$), `\solveError` ($\zeta$),
`\smoothMaxBias` ($\omega$). A comment records where they went.

### 1.4 The symbol conflict, decided

`\defectOrder` carried the comment *"bare $m$ denotes the number of exercise dates in
the Bermudan section"*. That is a collision: $m$ was doing duty both as a **regularity
order** (always object-subscripted: $m_g$, $m_B$, $m_{V_k}$) and as a **count**.

**Decision.** The base letter $m$ is reserved exclusively for the concept *regularity /
differential order*, always specialised by the object's own symbol in subscript. The
number of exercise dates gets its own glyph, $M$ (`\numberExerciseDates`) — which is
already what `bermudan_option_pricing.tex` uses for the exercise-set cardinality, so
the corpus stays consistent. The collocation count becomes bare $N$ (was $N_F$, a word
subscript, proscribed). Stated as `rem:notation-M` in the paper; the stale comment was
deleted from the hard-constraint preamble.

### 1.5 `bermudan_option_pricing.tex`: cited, not absorbed

**Decision, with reasons.** It is already a standalone, compiling document with its own
authors and abstract. Absorbing it would create a second, divergent copy of the same
probabilistic formalisation. Its scope is also narrower than the new paper's (a *single*
early-exercise date, with existence/uniqueness in the piecewise-regular class left open),
so folding "work to do" into a paper whose thesis is a theorem would weaken it. The new
paper cites it as the *formalisation note* and restates only its deterministic
consequence (the backward dynamic program, the location of the regularity loss).

---

## 2. The theorem that was attempted (§3(iv) as posed)

> The split extension imposes $V_k$ exactly at the terminal slice, mollifying only the
> interior, so the smoothing-bias term $\tfrac12\sum_{k<M}\varepsilon(t_k)$ **disappears**
> and the bound tightens to $\|e_0\|_\infty \le \sum_k \|\zeta_{k-1}\|_\infty$. The split
> is *the* construction achieving both $\omega_k=0$ and a finite stage forcing.

I derived it, then had it attacked by **two independent refuters** (one analysis lens,
one numerical lens — the second wrote NumPy scripts and measured).

---

## 3. Objections, and the adjudication

Every contested step was re-derived by hand before ruling. Verdicts are mine.

| # | Objection | Verdict | Consequence |
|---|---|---|---|
| 1 | **The uniqueness claim is FALSE.** A graded Chen–Mangasarian mollifier with exponent $p \in (1/3,1)$ has $\omega_k=0$ *and* finite $L^2$ forcing. So does the mis-specified split. | **CONFIRMED — fatal** | Re-derived: curvature channel $\int(\partial_{xx}M_\varepsilon)^2 \sim \tfrac{3\pi}{32}\lambda^3/\varepsilon \sim s^{-p}$ (finite iff $p<1$); time channel $\sim s^{3p-2}$ (finite iff $p>1/3$). Uniqueness in the $L^2$ column is gone. |
| 2 | **The correct discriminant is BOUNDEDNESS, not $L^2$-finiteness.** No CM grading gives finite $L^4$ (curvature needs $p<1/3$, time needs $p>3/5$ — empty). The split has $\|Lh\|_\infty<\infty$ hence every $L^q$. | **CONFIRMED** | Re-derived independently. This is a *stronger* theorem than the one I had. Finite $L^4$ ⟺ finite variance of the collocation estimator — which is exactly what the project's conventions care about. **Uniqueness restored, in the right column.** |
| 3 | **Mechanism:** the split doesn't make the singular second-order channel integrable — it **cancels it exactly** ($\partial_t h + Ah = 0$ identically). Every other member leaves it uncancelled and merely arranges convergence. | **CONFIRMED** | Became `rem:split-is-mollifier`. The split *is* a graded Gaussian mollifier of exponent $1/2$; opposing "mollifier" to "split" was wrong. |
| 4 | **CIRCULARITY.** $[A,B]=0$ for constant-coefficient BS, so $e^{sL^X}=e^{sA}e^{sB}$ with $(e^{sB}\phi)(x)=e^{-rs}\phi(x+(r-\nu)s)$. Hence $u_k(x,t) = e^{-rs}h_k(x+(r-\nu)s,t)$: anything that can evaluate $h_k$ evaluates the **exact** stage solution at the same cost. The neural solve is redundant. | **CONFIRMED — fatal to the framing** | Verified by hand. Became `prop:conjugacy` + `rem:conjugacy`, disclosed **prominently** (abstract + open questions). The constant-coefficient Bermudan is a *diagnostic with a known answer*, not an application; the construction is non-vacuous exactly where $[A,B]\ne 0$ (local vol, stochastic rates, baskets). |
| 5 | **The bound needs $L^\infty(\mathbb R)$, not $L^\infty(\mathcal X)$.** $S_k$ is the free-space operator; it is not an operator on $L^\infty(\mathcal X)$, so the telescoping cannot be closed on the training window. | **CONFIRMED** | The companion results report wrote $L^\infty(\mathcal{X})$ — that is an error. One-symbol correction, proof untouched, disclosed in `rem:whole-line` with the two routes that close the gap. |
| 6 | **The missing link:** nothing connected $\|\zeta\|$ to the quantity actually minimised. | **CONFIRMED — serious omission** | Re-derived via backward Duhamel: $\|\zeta_{k-1}\|_\infty \le \sqrt2 (8\pi\nu)^{-1/4}\Delta_k^{1/4}\|L\Phi^{(k)}\|_{L^2}$. Became `prop:loss-transfer`, and with it the **certificate** $\|e_0\|_\infty \le \sqrt2(8\pi\nu)^{-1/4}\sum_k \Delta_k^{1/4}\|L\Phi^{(k)}\|_{L^2}$. *This is where finiteness of the forcing earns its keep — the only place.* It is now the paper's headline: the two mechanisms genuinely fuse. |
| 7 | **Where does $\omega_k$ come from?** From one algebraic step: inserting $\pm\Pi_k(C^\star_k)$. It measures the *gluing map at the exact continuation*, not the continuation. | **CONFIRMED (not an objection)** | So contamination of $V_k$ by $\zeta$ is genuinely absorbed by the *inherited* term. The "does the split really impose $V_k$ exactly even when $V_k$ is itself approximate?" question is answered **yes** — and it is spelled out inside the proof of `prop:split-exactness`(i). |
| 8 | **Jensen / maximisation bias.** $\max(g,\cdot)$ is convex ⇒ the glued datum is upward-biased in expectation; $S_k$ is positive ⇒ biases do **not** cancel across stages. | **CONFIRMED — presentation defect** | The $L^\infty$ bound survives (it bounds *magnitude*), but "the induction does not amplify" invited the conflation with "no bias". Became `rem:no-statistical-unbiasedness`, cited to Longstaff–Schwartz and Glasserman–Yu. |
| 9 | **Exercise region approximated.** With a root-found region the extra term is $\sum_k \Lambda_k \delta_k$; and a bracketing root-finder can *miss a whole component* if $\{g=C_k\}$ isn't a single crossing — then the term is $O(1)$, not $O(\delta)$. | **CONFIRMED — partial** | The bound survives with one extra injection per date. `rem:region-approximation`. The multiple-crossing failure mode is flagged as requiring a hypothesis or a sign-change sweep. |
| 10 | **The raw extension's $F_k=+\infty$ is INVISIBLE to the loss.** Measured: MC estimate converges to $1.779\times10^{-2}$ with 0.13 % seed spread. The Dirac lives on a null set; autodiff through `max` returns the piecewise value. | **CONFIRMED — serious** | "Divergent ≠ large" and the loss cannot see it. Forced me to find the *operative* defect, which is stronger: `prop:raw-unbounded` — the target $\Psi^\star$ has a first-derivative discontinuity of amplitude $\tfrac{1-d_k}{d_k}J_k \to \infty$. **A representation defect, proven.** |
| 11 | **Bounded target.** From bounded forcing, Duhamel gives $\|w\|_\infty \le s\,\Theta_k$ and $d_k = s/t_k$, hence $\|\Psi^\star_k\|_\infty \le t_k \Theta_k$. | **My own derivation, prompted by #10** | Became `prop:split-exactness`(iii). *Note*: the methodology report's Proposition 2 (bias-freeness) cannot be invoked here — its hypothesis "$\partial_t h$ bounded" **fails** for a corner datum. The Duhamel route needs only the bounded forcing and is the right one. |
| 12 | **Zero-padding the datum destroys everything.** $V_k \to K$ as $x\to-\infty$, not 0. Measured: $\sup|Bh|$ goes $0.0455 \to 46.47$ as $\tau: 10^{-2}\to10^{-6}$, diverging as $1/\sqrt\tau$; the padding artefact **exceeds** the mis-specification signal. | **CONFIRMED — fatal for the implementation** | My original truncation remark was far too complacent. Rewritten as `rem:far-field`: extend analytically ($g=K-e^x$ left of $\Gamma_k$, $0$ right), never zero, never by extrapolating the network; margin $\ge 5\sigma\sqrt{\Delta_k}$. |
| 13 | **Fixed-node quadrature reinstates the Dirac.** $\hat h = \sum_i w_i V_k(x+\sigma\sqrt s\,\xi_i)$ is a sum of *shifted kinked copies* → one Dirac per node, silent to autodiff. | **CONFIRMED — serious** | `rem:quadrature`: split the integral at $x^\star_k$ (left branch closed-form). **This requires root-finding after all**, so the "pointwise max needs no root" convenience is *not* realised by a faithful implementation. Stated, not hidden. |
| 14 | **$O(\Delta)$ vs $O(\sqrt\Delta)$ is not an observable.** Measured crossover $\Delta^\star = 6.9\times10^{-5}$; at $\Delta=0.05$ the discrimination is a factor **1.247**, not an order of magnitude. | **CONFIRMED — fatal as a diagnostic** | The asymptotic proposition is true and kept; the *diagnostic* was removed from the experimental spec and replaced. Added `rem:crossover` with the analytic $\Delta^\star$ and the requirement to check $\Delta_k<\Delta^\star$ first. |
| 15 | **$V_k$ bounded + Lipschitz on $\mathbb R$ is an architectural hypothesis** (a ReLU net grows linearly ⇒ $\|V_k\|_\infty=\infty$ ⇒ the bound is vacuous). | **CONFIRMED** | Became `ass:standing`, stated as a hypothesis rather than assumed silently. |
| 16 | The $\omega \le \varepsilon/2$ bound is not sharp; the propagated bias is $O(\varepsilon^2\log(1/\varepsilon))$ as $\varepsilon\to0$. | **CONFIRMED, but does not bite** | Re-derived: $\|\omega\|_{L^1} \approx \tfrac{\varepsilon^2}{2\lambda}\log(1/\varepsilon)$. It is an $\varepsilon\to0$ statement. At the $\varepsilon_0=2$ actually used, the *measured* bias is 0.566–0.857 against the bound 1 — the bound is nearly attained and the motivation stands **empirically**. Recorded; no change to the claim. |
| 17 | The linear-grading divergence constant is $3\pi/32$ (both refuters, independently; numerically $1.00009$). | **CONFIRMED** | My draft had hedged the prefactor. Now stated exactly in `prop:graded-forcing`. |

### The statement actually published

`prop:split-exactness`, with `prop:raw-unbounded`, `prop:graded-forcing`,
`prop:misspecification`, `prop:conjugacy`, and the four-column ledger
`tab:ledger`:

- **(i) Exactness at the slice** ⇒ $\omega_k = 0$ ⇒ bound tightens to $\sum_k\|\zeta_{k-1}\|_\infty$.
  **Does NOT single out the split** — four candidates achieve it. Said so explicitly.
- **(ii) Bounded forcing** $\|\mathcal{L}h_k\|_{L^\infty} \le |r-\nu|\,\mathrm{Lip}(V_k) + r\|V_k\|_\infty$
  ⇒ every $L^q$ ⇒ **finite-variance** MC estimator. **This is what singles it out.**
  Square-integrability does *not* (three candidates have it).
- **(iii) Bounded target** $\|\Psi^\star_k\|_\infty \le t_k\Theta_k$, against
  `prop:raw-unbounded`'s diverging kink.
- **Disclosed:** the conjugacy makes the whole thing redundant in constant-coefficient BS.

Explicitly **not** claimed: minimality; statistical unbiasedness; that the mollified and
split chains' $\zeta$ are comparable (different algorithms).

---

## 4. Decisions taken

1. **Kept `prop:bermudan-recursion` verbatim** (statement + proof) except one symbol:
   $L^\infty(\mathcal{X}) \to L^\infty(\mathbb{R})$. The original space is *false*
   (objection #5) and shipping a knowingly-false statement was not an option. Disclosed
   in `rem:whole-line`.
2. **Copied the four Bermudan figures** from the code repo's report into the Overleaf
   `figures/`. They were not there.
3. **Added 4 bib entries** (Longstaff–Schwartz 2001, Lord et al. 2008 CONV, Fang–Oosterlee
   2009 COS, Glasserman–Yu 2004) — required by the register rule that a claim about the
   state of practice needs a citation. Both the maximisation-bias remark and the
   conjugacy disclosure make such claims.
4. **The trained ablation is SPECIFIED, NOT RUN.** Every measured slot of
   `tab:trained-spec` is a dash. The "predicted" block is typographically separated.
   No inferred value is reported as measured.

## 5. What was NOT done

- The trained ablation (Section 5.3 of the new paper) is not implemented.
- The construction is not demonstrated where $[A,B]\ne 0$ — which, per objection #4, is
  the only place it is not redundant. **This is the principal open item.**
- No SLURM job id exists for the Bermudan runs (only the oracle run, array 178747); the
  paper says "no scheduler job identifier is recorded" rather than inventing one.
