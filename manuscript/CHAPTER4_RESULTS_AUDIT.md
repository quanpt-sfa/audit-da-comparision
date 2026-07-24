# Chapter 4 results audit — superseded pending rerun

## Current status

The previously populated Chapter 4 values must not be cited or treated as final. They were produced before the panel-construction audit established that:

1. ICB Level-1 existed but the completion runner requested the nonexistent `icb_industry` alias;
2. `audit_firm` and `audit_opinion` existed in `bctc_audit_annual_long.csv` but were omitted from `accrual_panel.csv.gz`;
3. financial firms had not been explicitly excluded from every Chapter 4 analysis;
4. the reused symbols `VSM` and `VTS` required legal-entity canonicalisation (`VSMS` and `VTSC` for the securities firms);
5. the applied-consequence runner had not propagated `big4_flag` and ICB Level-1 consistently.

The branch now contains a corrected pipeline. A new Chapter 4 may be populated only from a full run that first rebuilds the enriched master panel and then passes the nonfinancial analysis-sample contract.

## Required execution sequence

```powershell
python .\scripts\01_build_panel.py `
  --config .\config\signal_gate.yaml

Remove-Item .\artifacts\manuscript_results `
  -Recurse -Force -ErrorAction SilentlyContinue

python .\scripts\31_complete_manuscript_results.py `
  --config .\config\results_completion.yaml `
  --workers 31 `
  --simulation-batch-size 32
```

The Results runner now refuses an unenriched panel and rejects checkpoints that do not match `analysis_sample_contract.json`.

## Data contract after correction

The master panel retains all entities and includes:

- `icb_l1` and the remaining ICB levels;
- `financial_flag`, `analysis_eligible`, and `exclusion_reason`;
- `auditor_name_raw`, `auditor_brand`, `auditor_group`, and `big4_flag`;
- `audit_opinion_raw`, `audit_opinion_group`, and opinion status fields.

Financial firms remain in the master panel for traceability but are excluded from all core Chapter 4 calculations. Container Miền Trung remains `VSM`; Chứng khoán VSM is `VSMS`; Gạch Ngói Từ Sơn remains `VTS`; Chứng khoán Việt Thành is `VTSC`.

## Superseded bundle

The earlier uploaded `manuscript_results.zip` had SHA-256 `2bb41a09fc81db43eb412a2442d7c6723592679fd9cc7fe09f05653e4ebef4c0`. Its aggregate results and the manuscript values derived from it are retained only as an audit trail and must be replaced after the corrected full rerun.
