# audit-da-comparision

Pipeline for testing whether pre-audit and post-audit discretionary accruals contain a measurable paired Bayesian transition signal.

## Current scope

This branch implements the engagement-level signal gate only:

1. profile the long financial-statement file;
2. extract and standardize accrual variables;
3. build paired unaudited/audited firm-year observations using audited lag values;
4. fit rolling Bayesian normal-accrual models with firm-level partial pooling;
5. estimate stacking weights from prior-year audited validation data;
6. generate shared posterior pre/post draws under three reference benchmarks;
7. evaluate signed shifts, normalization probabilities, overshooting, deterioration, sensitivity to benchmark-error correlation and error-scale ratios, and go/no-go criteria;
8. compare the Bayesian results with OLS Jones-family baselines.

Audit-firm effects, switcher networks, and opinion calibration are intentionally deferred until the paired signal passes.

## Data

The input is expected to use the schema of `financial_statement_full_long.csv.gz`, including:

- `issuer_ticker`, `raw_exchange`, `fiscal_year`;
- `audit_status` (`unaudited`/`audited`);
- `statement_family`, `scope`, `source_item_id`;
- `value_numeric` and matching-quality flags.

Raw and generated data are excluded from Git.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .[dev]
```

## Run

```bash
python scripts/run_pipeline.py \
  --config config/signal_gate.yaml \
  --input /path/to/financial_statement_full_long.csv.zip
```

`--input` must point to a file (`.zip`, `.csv`, or `.csv.gz`), not a folder.

Windows example:

```powershell
python .\scripts\run_pipeline.py --input data\raw\financial_statement_full_long.csv.gz
```

Individual stages:

```bash
python scripts/00_profile_input.py --input /path/to/file.zip
python scripts/01_build_panel.py --input /path/to/file.zip
python scripts/02_run_signal_gate.py
python scripts/03_run_baselines.py
```

For `00_profile_input.py` and `01_build_panel.py`, `--input` also must be a file path (`.zip`, `.csv`, or `.csv.gz`).

## Tests

```bash
pytest
```

See [`docs/methodology.md`](docs/methodology.md) for the estimand and interpretation limits.
