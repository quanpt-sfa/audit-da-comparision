# audit-da-comparision

Research pipeline for testing whether pre-audit and post-audit discretionary accruals contain a measurable paired transition signal, and for evaluating whether that signal is consistent with cash-flow classification shifting and related audit-regime heterogeneity.

## Overview

The repository now contains four connected workflow layers:

1. a core paired discretionary-accrual signal gate;
2. non-Bayesian OLS baseline diagnostics;
3. post-baseline transition, falsification, decomposition, and calibration diagnostics;
4. observed cash-flow-statement reclassification validation, including auditor-regime, auditor-switch, exchange, and COVID robustness analyses.

The codebase is organized as a `src/` layout package with script entry points under [scripts](scripts) and workflow configuration under [config](config).

## Data expectations

The main raw input is a long-format financial-statement file compatible with `financial_statement_full_long.csv.gz`. The pipeline expects fields such as:

- `issuer_ticker`, `raw_exchange`, `fiscal_year`
- `audit_status` (`unaudited` / `audited`)
- `statement_family`, `scope`, `source_item_id`, `item_name_raw`
- `value_numeric`
- matching and eligibility flags such as `identity_match_status`, `retrospective_eligible`, and `prospective_flag`

Some advanced workflows also require auditor-source data, configured in [config/auditor_regime.yaml](config/auditor_regime.yaml).

Raw data, generated artifacts, logs, and local environment outputs are intentionally excluded from Git.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Core signal-gate workflow

Run the complete paired-DA pipeline:

```bash
python scripts/run_pipeline.py \
  --config config/signal_gate.yaml \
  --input /path/to/financial_statement_full_long.csv.gz
```

`--input` must point to a file (`.zip`, `.csv`, or `.csv.gz`), not a directory.

Windows example:

```powershell
python .\scripts\run_pipeline.py --input data\raw\financial_statement_full_long.csv.gz
```

This runner executes:

1. [scripts/00_profile_input.py](scripts/00_profile_input.py)
2. [scripts/01_build_panel.py](scripts/01_build_panel.py)
3. [scripts/02_run_signal_gate.py](scripts/02_run_signal_gate.py)
4. [scripts/03_run_baselines.py](scripts/03_run_baselines.py)

Typical outputs include:

- `artifacts/input_profile.json`
- `data/processed/accrual_panel.csv.gz`
- `artifacts/paired_da_posterior.csv.gz`
- `artifacts/rolling_fold_diagnostics.csv`
- `artifacts/SIGNAL_GATE_REPORT.md`
- `artifacts/ols_baselines.csv.gz`

## Additional workflows

OLS baseline diagnostics:

```bash
python scripts/04_analyze_ols_baselines.py --config config/ols_diagnostics.yaml
```

Post-baseline diagnostics and falsification suite:

```bash
python scripts/run_next_diagnostics.py --config config/next_diagnostics.yaml
```

Observed CFS shifting validation and completion gates:

```bash
python scripts/run_cfs_shifting_validation.py --config config/cfs_shifting_validation.yaml
```

Auditor-regime heterogeneity analysis:

```bash
python scripts/22_analyze_auditor_regime.py --config config/auditor_regime.yaml
python scripts/23_write_auditor_regime_report.py --config config/auditor_regime.yaml
```

The larger validation workflow also includes:

- detailed CFS line-item mapping and reconciliation
- CFS deep-dive and uncertainty-bridge analysis
- auditor-switch event-study and dynamic-DiD checks
- yearly AUC heterogeneity reporting
- exchange and COVID robustness checks
- TT200 time-contract enforcement and completion gates

## Configuration

Key workflow configs live in [config](config):

- [config/signal_gate.yaml](config/signal_gate.yaml)
- [config/ols_diagnostics.yaml](config/ols_diagnostics.yaml)
- [config/next_diagnostics.yaml](config/next_diagnostics.yaml)
- [config/cfs_shifting_validation.yaml](config/cfs_shifting_validation.yaml)
- [config/auditor_regime.yaml](config/auditor_regime.yaml)
- [config/cfs_regime_robustness.yaml](config/cfs_regime_robustness.yaml)

## Tests

```bash
pytest
```

## Documentation

Methodology and workflow-specific notes are documented in [docs/methodology.md](docs/methodology.md) and related files under [docs](docs).
