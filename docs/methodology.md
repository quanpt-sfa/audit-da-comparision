# Paired Bayesian DA signal gate

## Objective

The first gate asks whether paired preliminary and audited financial statements contain a measurable discretionary-accrual transition signal after propagating normal-accrual benchmark uncertainty. Audit-firm effects and opinion calibration are deliberately excluded until this engagement-level signal passes.

## Identification target

For firm-year `i,t`, the pipeline estimates a joint posterior for pre- and post-audit DA using one rolling engine trained only on audited history through `t-1`. Parameter, model, and firm-intercept draws are shared across the two versions. Benchmark-error correlation is not point-identified, so results are reported over a prespecified `rho` grid.

The Bayesian engine estimates the normal-accrual reference point; the observed pre/post accounting data determine movement. Three references are reported:

1. version-specific inputs;
2. pre-audit inputs for both versions;
3. audited inputs for both versions.

## Candidate models

The default set contains Jones, Modified Jones, Kothari performance-adjusted, and a nonlinear Modified Jones specification. Out-of-sample stacking weights are estimated from the immediately preceding audited year.

## Core posterior metrics

- signed DA shift;
- absolute abnormality reduction;
- probability of economically meaningful improvement;
- probability of deterioration;
- normalization, partial correction, overshoot, deterioration, and no-material-movement probabilities;
- posterior signal-to-noise ratio.

## Interpretation boundary

The signal is booked ex-post accrual intervention. It does not separately identify detection, waived adjustments, deterrence, confirmation, or overall audit quality. Firm-level auditor modelling is permitted only after benchmark and sensitivity gates pass.
