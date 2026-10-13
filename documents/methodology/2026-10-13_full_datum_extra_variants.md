# Full datum — the other variants of the paper's earlier comparison (descriptive, not reported)

**Written 2026-10-13, before the runs.** The author decided on 2026-10-13 to report, in the boundary-split paper, the six choices of $h$ of the CMAP talk on the full datum $g$ (raw constant, transported datum, highest-order split, highest- and first-order split, Chen–Mangasarian with $\alpha=1$, full-operator control), and to remove the other variants from the paper, while asking that these other variants be computed on $g$ as well (« les calculer au cas où mais je veux les retirer comme pour les slides »).

**Runs.** Cells `g2_bernoulli_full` (Black–Scholes, the paper's $G_2$) and `g3_bernoulli_full` (order four, the paper's $G_3$), with the protocol of the validation-selected series and seeds $\{0,1,2\}$:
- on `g2_bernoulli_full`: `convex_raw` (raw linear extension, on the unprojected datum), `graded_gaussian_mismatched`, `graded_chen_mangasarian_narrow` ($\alpha=1/2$);
- on `g3_bernoulli_full`: `convex_raw`, `graded_gaussian_width_matched`.

That is $15$ tasks on Jean Zay V100 (`akz@v100`, `qos_gpu-t3`), after a smoke test on `qos_gpu-dev` with `--debug`.

**No prediction is registered.** These runs are descriptive: they are not reported in the paper nor in the talk, and no claim rests on them. The line-source metrics, a closed form of the constant-in-time extension only, are not recorded for `convex_raw`.

## Outcome

Written 2026-10-13 after Jean Zay array 1025143 (smoke test 1025141) completed at revision `6a99c83`; summaries copied to `data/ablation_split_extension_trained/full_datum_extra_variants_2026-10-13/` (no checkpoint files). The 15 runs are admissible (finite quantities, exact initial-value check, no safeguard activation, no non-finite norm). Wall times $400$ to $935$ s (Black–Scholes), $1680$ to $1897$ s (order four).

Median relative error $\delta_\Gamma$ on the full datum (measured, 3 seeds), with the value on $g_{128}$ of the paper's validation-selected series where it exists:

| Cell | Extension | Seeds 0, 1, 2 | Median on $g$ | Median on $g_{128}$ (paper) |
|---|---|---|---|---|
| Black–Scholes | raw linear (`convex_raw`) | $0.08277$, $0.08273$, $0.08277$ | $\simeq0.0828$ | $\simeq1.21\times10^{-3}$ |
| Black–Scholes | graded Gaussian | $4.79\times10^{-4}$, $6.65\times10^{-4}$, $5.09\times10^{-4}$ | $\simeq5.09\times10^{-4}$ | $\simeq5.40\times10^{-4}$ |
| Black–Scholes | Chen–Mangasarian, $\alpha=1/2$ | $3.55\times10^{-3}$, $7.98\times10^{-3}$, $2.91\times10^{-3}$ | $\simeq3.55\times10^{-3}$ | $\simeq1.97\times10^{-3}$ |
| Order four | raw linear (`convex_raw`) | $0.0882$, $0.0880$, $0.0877$ | $\simeq0.0880$ | $\simeq3.80$ |
| Order four | graded Gaussian | $0.237$, $0.126$, $0.116$ | $\simeq0.126$ | $\simeq0.175$ |

The $g_{128}$ medians are read from `RHB_notes/paper/generated/individual_results.csv` (validation-selected series). These runs are not reported in the paper nor in the talk.
