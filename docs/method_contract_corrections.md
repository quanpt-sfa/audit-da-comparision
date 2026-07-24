# Locked Results method contract

This note governs the Chapter 4 completion run and supersedes any legacy result checkpoint that does not expose the contract fields below.

## 1. Jones-family intercept

The Jones, Modified Jones, Kothari-style, and nonlinear Modified Jones regressions do not include an ordinary constant in addition to the inverse-assets regressor. The locked design is therefore estimated with:

```python
StandardScaler(with_mean=False)
LinearRegression(fit_intercept=False)
```

Predictors may be divided by their estimated standard deviations for numerical stability, but they must not be mean-centred. Mean centring with `fit_intercept=False` would imply an effective ordinary intercept in the original feature space. `inv_assets` remains a substantive scale regressor. Under the no-global-intercept industry-FE sensitivity, all industry indicators are retained rather than dropping a reference industry.

The same contract now applies to `src/audit_da/baseline.py` and the Chapter 4 completion architectures.

## 2. Attribution estimand

The locked attribution is an exact three-player Shapley decomposition of absolute discretionary-accrual reduction across:

1. PAT movement;
2. CFO movement;
3. benchmark residual movement.

The third player absorbs the remainder between the estimated signed DA shift and the accounting identity movements attributed to PAT and CFO. Depending on the reference construction and winsorisation, it may include benchmark-input, target-boundary, or residual transformation movement. It must not be silently discarded or reallocated to PAT or CFO.

Every generated case must satisfy the Shapley efficiency identity within numerical tolerance:

```text
phi_pat + phi_cfo + phi_benchmark = |DA_pre| - |DA_post|
```

The confirmatory contrast remains `abs(phi_cfo) - abs(phi_pat)`; the benchmark player is retained to make that comparison well-defined rather than forcing a false two-component identity.

## 3. Signed-shift reassignment

The symmetric-sign benchmark retains observed absolute shift magnitudes and independently randomises their directions. The signed-shift-reassignment benchmark preserves the observed signed-shift distribution inside each fiscal year and independently permutes shifts only among observations from that year.

For DA, tasks are already separated by model and reference benchmark. The effective reassignment cell is therefore:

```text
model x benchmark x fiscal_year
```

For direct CFO switching, the reassignment cell is fiscal year.

## Executable controls

Run the static and runtime audit:

```powershell
python .\scripts\32_audit_method_contract.py `
  --config .\config\results_completion.yaml `
  --skip-existing-outputs
```

Run a clean corrected Chapter 4 workflow:

```powershell
python .\scripts\33_run_method_corrected_results.py `
  --config .\config\results_completion.yaml `
  --clean `
  --workers 31 `
  --simulation-batch-size 32
```

Resume is permitted only through the guarded wrapper:

```powershell
python .\scripts\33_run_method_corrected_results.py `
  --config .\config\results_completion.yaml `
  --resume `
  --workers 31 `
  --simulation-batch-size 32
```

The wrapper audits existing architecture, attribution, and randomisation checkpoints before allowing resume, and audits the completed bundle again after execution.
