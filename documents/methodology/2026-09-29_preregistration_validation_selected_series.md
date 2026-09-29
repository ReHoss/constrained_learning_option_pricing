# Pre-registration — the validation-selected model series (all trained cells)

**Written 2026-09-29, before implementation and before any run of this series.** Decided by the
author ("je suis d'accord ... lance la nouvelle campagne de calculs"). It replaces, for every trained
cell, the retention rule of the earlier runs, and produces the single model series the paper
reports (`claude-workspace-rho/research-methodology/writing/paper-writing-rules.md`, §5bis).

## 1. Why

Every earlier run retained the state of lowest **training** mini-batch loss, with no validation
part, and saved the post-step state while comparing the pre-step loss. §5bis names this as the
defect to avoid. It was measured to matter: the ratio of the residual on an independent batch
to the minimum training mini-batch loss ranges from 1.2–2.1 (split, convex, constant-in-time) to
3–15 (Chen–Mangasarian) and 281 (graded Gaussian on $G_3$). For the non-split extensions the
retained iterations coincide across cells that share the sampler seed, so the selection is set
by the sampler's luckiest batch (manifest `documents/methodology/2026-09-29_variable_coefficient_campaign_manifest.yaml`).

## 2. The three parts (same distribution)

All three parts are drawn from the uniform distribution on the cylinder
$Q=\mathbb T\times(0,T)$; a change of distribution is deferred to a later study.

- **Training part.** A fresh batch of $4096$ points at each iteration (sampler-role seed), as
  before. It is what the optimiser sees.
- **Validation part (selects).** A fixed set of $16384$ points drawn once, with its own derived
  seed (role `validation`), independent of the training and evaluation seeds. Every $100$
  iterations, and at the last iteration, the mean squared residual of the current parameters,
  $\mathcal V(\theta)=\frac{1}{16384}\sum_{i}(\mathcal P\hat u_\theta(x_i,t_i))^2$, is evaluated on it
  (in chunks of $4096$ points, which changes the arithmetic order but not the quantity). The
  retained state is the argmin of $\mathcal V$ over these evaluations; the saved parameters are
  exactly the evaluated ones (no offset); a tie keeps the earlier iteration.
- **Evaluation part (scored once, selects nothing).** Unchanged: the relative $L^2$ error
  $\delta_\Gamma$ on the fixed grid of $1024\times11$ points against the exact solution (constant
  coefficients) or the Fourier–Galerkin reference (variable coefficients; $N=512$ at order 2,
  $N=256$ at order 4); the residual on an independent evaluation batch of $4096$ points
  (evaluation-role seed); the residual spectra at the five time slices.

The selection criterion (a residual) differs from the reported error (a distance to the
solution). This is inherent: selecting on the distance to the solution would use the evaluation
part, and the variable-coefficient cells have no closed-form solution. It is stated in the paper.

## 3. What is reported

- Reported: $\delta_\Gamma$ (with its values at $t=0$ and near the break point), and the residual on
  the evaluation batch.
- Recorded, not reported as performance: the validation residual at the retained state (the
  value of the selection criterion) and the minimum training mini-batch loss (a diagnostic).
- The new runs write the keys `retained_iter`, `validation_residual_at_retained_state`,
  `best_training_batch_loss`, `best_training_batch_iter`; they do **not** write the keys
  `best_loss` and `best_iter` of the earlier runs, so that the two retention rules can never be
  mixed in an aggregation. The aggregator uses the evaluation-batch residual
  (`loss_best_state_eval`) wherever a trained residual is displayed, for both series.

## 4. Campaign

Unchanged: network, budget ($20000$ iterations), optimiser and schedule, collocation sizes, the
master seeds $\{0,1,2\}$, and every variant of every cell.

| Cells | Variants per cell | Hardware |
|---|---|---|
| $G_1$, $G_2$ | 9 (stage-2 V1–V7 and the two Chen–Mangasarian variants) | Jean Zay V100 |
| heat control | 2 | Jean Zay V100 |
| $G_3$ | 7 | Jean Zay V100 |
| LV2 ($\varepsilon=0.25,0.75$), LV4 ($\varepsilon=0.25,0.75$) | 4 | Adastra MI250 |

$129$ tasks. Each cell runs on one hardware; all variable-coefficient cells share the MI250, which
removes the cross-hardware caveat of the earlier LV4 amplitude comparison.

**Separation of the series.** The new run directories are written under
`data/ablation_split_extension_trained/series_validation_selected/` and aggregated with
`--data-root` pointing there. The earlier series stays where it is and is used only in the
measurement note, for the comparison of the two retention rules (per variant: ratio of the new to
the old $\delta_\Gamma$, and the retained iterations). That comparison does not enter the paper.

## 5. Analysis

The comparisons and falsification conditions already pre-registered apply unchanged to the new
series: stage-2 specification (split versus the other extensions), §8–9 of the stage-2
specification (Chen–Mangasarian, fourth order), and
`2026-09-29_preregistration_variable_coefficient_split.md` §6 (freezing point, dose–response).
Quartiles and ranges over three seeds are described, not tested.

## 6. Not included

A variation of the datum band $K$ in training (a candidate test of the growth predictions) is not
part of this series; it awaits a decision.
