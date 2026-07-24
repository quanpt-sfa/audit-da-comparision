# Manuscript Results completion workflow

> **Authoritative submission workflow:** scripts `34`--`35` generate core
> Chapter 4 results. Script `36` generates supplemental diagnostics separately.
> Scripts `31`--`33` and `artifacts/manuscript_results/` remain only for auditing
> superseded outputs.

## Core inputs

```text
data/processed/accrual_panel.csv.gz
data/processed/accrual_panel_unrestricted.csv.gz
```

The locked panel supplies test issuer-years. The unrestricted panel supplies
historical audited estimation observations. Both are filtered to nonfinancial
firms by the final runner.

The source window is 2015--2025. Because the models use one-year lagged inputs:

- 2015 supplies lag support only;
- 2016 is the first model-complete training year;
- 2017 is the first model-based test year;
- direct preliminary/audited comparisons retain 2016.

## Core method contract

The authoritative run uses:

- no ordinary intercept;
- predictor scaling without mean centring;
- `inv_assets` as a substantive scale regressor;
- historical-only outcome winsorisation;
- raw current-state `ta_scaled`;
- exact two-player PAT--CFO fixed-reference attribution;
- a common complete-case direct switching population;
- signed-shift reassignment within fiscal year;
- paired-difference primary applied tests;
- fully interacted stacked sensitivity models;
- one unique signed-DA change test per focal variable.

The complete specification is in `docs/final_results_contract_v4.md`.

## Core validation and run

```powershell
pytest -q `
  tests\test_final_results_contract.py `
  tests\test_method_contract.py `
  tests\test_switching_complete_case.py `
  tests\test_results_completion.py `
  tests\test_results_parallel.py
```

```powershell
python .\scripts\34_audit_final_results_contract.py `
  --config .\config\results_completion.yaml `
  --skip-existing-outputs
```

```powershell
python .\scripts\35_run_final_results.py `
  --config .\config\results_completion.yaml `
  --clean `
  --workers 31 `
  --simulation-batch-size 32 `
  2>&1 | Tee-Object .\artifacts\chapter4_final_v4.log
```

Core outputs are written to:

```text
artifacts/manuscript_results_final/
```

## Supplemental diagnostics

No external concentration or near-zero result files are accepted. The committed
producer is:

```powershell
pytest -q tests\test_supplemental_diagnostics.py

python .\scripts\36_run_supplemental_diagnostics.py `
  --config .\config\supplemental_diagnostics.yaml
```

It uses the processed panel and either the cached mapped CFS line-item table or
the raw long financial-statement file. Outputs are written to:

```text
artifacts/supplemental_diagnostics/
```

See `docs/supplemental_diagnostics.md`.

## Manuscript integration

Do not update or merge Chapter 4 from the legacy output directory. Regenerate
Chapter 4, Discussion, Conclusion, and the LaTeX project only after both the
core post-run audit and supplemental manifest pass.
