#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from audit_da.baseline import run_ols_baselines
from audit_da.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-Bayesian DA transition baselines")
    parser.add_argument("--config", default="config/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    panel_path = resolve_path(args.config, config["paths"]["processed_panel"])
    panel = pd.read_csv(panel_path, compression="gzip" if str(panel_path).endswith(".gz") else None, low_memory=False)
    results = run_ols_baselines(panel, config)
    output = resolve_path(args.config, config["paths"].get("baseline_output", "artifacts/ols_baselines.csv.gz"))
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False, compression="gzip" if str(output).endswith(".gz") else None)
    summary = results.groupby(["model", "benchmark"], observed=True)["reduction"].agg(["count", "mean", "median"])
    print(f"Wrote {output}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
