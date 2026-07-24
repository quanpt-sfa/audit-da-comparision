# Chapter 4 validated-results audit

## Source bundle

- Input supplied by the authorised data user: `manuscript_results.zip`
- SHA-256: `2bb41a09fc81db43eb412a2442d7c6723592679fd9cc7fe09f05653e4ebef4c0`
- Uploaded execution metadata: seed 20260723; 2,000 bootstrap draws; 2,000 simulation draws.
- The bundle did not include the raw `accrual_panel.csv.gz`. Case-level paired data were reconstructed from `rq2_direct_cases.csv`; model-level validation used the uploaded accrual and attribution case tables.

## Why the aggregate outputs were not inserted unchanged

A Methods-to-results audit identified material mismatches in the uploaded runner outputs. The Chapter 4 values were therefore reconstructed from the available case-level tables under the locked protocol:

1. `icb_industry` was absent, so the uploaded industry-FE and industry-slope cells were not estimated. The validated reconstruction uses the fully populated ICB Level-1 field (`icb_l1`).
2. The uploaded accrual estimator used an ordinary intercept in addition to inverse assets. The validated reconstruction follows the Methods and uses no ordinary intercept.
3. The uploaded attribution table used a three-part additive approximation. The validated reconstruction applies the exact two-component fixed-reference Shapley decomposition over the four required states.
4. Direct switching was recomputed with outcome-specific complete cases; a CFO-sign switch requires both values to be non-zero and to have opposite signs.
5. Signed-shift reassignment was recomputed within fiscal-year cells and, for DA, within model--fiscal-year cells.
6. Applied regressions were recomputed with short-term debt scaled by beginning assets, a reconstructed current ratio, one-percent winsorisation of continuous controls, and ICB Level-1 fixed effects.

## Execution limits retained in the manuscript

- Big Four status is absent, so the four Big Four applied-consequence specifications are not estimated.
- The optional concentration and near-zero files were absent, so no new supplemental-channel inference is reported.
- These omissions do not enter the three confirmatory intersection--union rules.

The populated Results source is split across `04_results_part0.tex`, `04_results_part1.tex`, and `04_results_part2.tex`. Pending-rerun language has been removed only where a validated result is available.
