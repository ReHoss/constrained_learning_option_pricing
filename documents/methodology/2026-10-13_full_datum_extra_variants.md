# Full datum — the other variants of the paper's earlier comparison (descriptive, not reported)

**Written 2026-10-13, before the runs.** The author decided on 2026-10-13 to report, in the boundary-split paper, the six choices of $h$ of the CMAP talk on the full datum $g$ (raw constant, transported datum, highest-order split, highest- and first-order split, Chen–Mangasarian with $\alpha=1$, full-operator control), and to remove the other variants from the paper, while asking that these other variants be computed on $g$ as well (« les calculer au cas où mais je veux les retirer comme pour les slides »).

**Runs.** Cells `g2_bernoulli_full` (Black–Scholes, the paper's $G_2$) and `g3_bernoulli_full` (order four, the paper's $G_3$), with the protocol of the validation-selected series and seeds $\{0,1,2\}$:
- on `g2_bernoulli_full`: `convex_raw` (raw linear extension, on the unprojected datum), `graded_gaussian_mismatched`, `graded_chen_mangasarian_narrow` ($\alpha=1/2$);
- on `g3_bernoulli_full`: `convex_raw`, `graded_gaussian_width_matched`.

That is $15$ tasks on Jean Zay V100 (`akz@v100`, `qos_gpu-t3`), after a smoke test on `qos_gpu-dev` with `--debug`.

**No prediction is registered.** These runs are descriptive: they are not reported in the paper nor in the talk, and no claim rests on them. The line-source metrics, a closed form of the constant-in-time extension only, are not recorded for `convex_raw`.

## Outcome

To be written after the run.
