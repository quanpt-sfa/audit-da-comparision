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

from audit_da.config import load_config, resolve_path
from audit_da.report import build_signal_report, save_report
from audit_da.signal import run_signal_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling paired Bayesian DA signal analysis")
    parser.add_argument("--config", default="config/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    panel_path = resolve_path(args.config, config["paths"]["processed_panel"])
    panel = pd.read_csv(panel_path, compression="gzip" if str(panel_path).endswith(".gz") else None, low_memory=False)
    posterior, folds = run_signal_gate(panel, config)
    posterior_path = resolve_path(args.config, config["paths"]["posterior_output"])
    fold_path = resolve_path(args.config, config["paths"]["fold_output"])
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    posterior.to_csv(posterior_path, index=False, compression="gzip" if str(posterior_path).endswith(".gz") else None)
    folds.to_csv(fold_path, index=False)
    report = build_signal_report(panel, posterior, config)
    report_path = resolve_path(args.config, config["paths"]["report_output"])
    save_report(report, report_path)
    print(f"Wrote posterior metrics: {posterior_path}")
    print(f"Wrote fold diagnostics: {fold_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
