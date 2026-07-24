#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.panel_metadata import select_analysis_sample  # noqa: E402
from audit_da.predictive_validity import PredictiveValiditySettings  # noqa: E402
from audit_da.predictive_validity_parallel import (  # noqa: E402
    run_predictive_validity_parallel,
)
from audit_da.results_completion.core import output_hash  # noqa: E402


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def stage(message: str) -> None:
    print(f"[predictive-validity] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the quarterly-aggregate and audited annual reporting states "
            "using earnings persistence, future-CFO informativeness, CFO persistence, "
            "an earnings/CFO horse race, and accrual-quality robustness."
        )
    )
    parser.add_argument(
        "--config",
        default="config/predictive_validity.yaml",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the configured output directory before running.",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=None,
        help="Override issuer-cluster bootstrap draws, mainly for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Parallel bootstrap workers. The workflow has five independent bootstrap "
            "jobs, so values above 5 do not add useful parallelism."
        ),
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=None,
        help="Vectorized cluster-resampling draws per worker batch.",
    )
    parser.add_argument(
        "--minimum-train-rows",
        type=int,
        default=None,
        help="Override the minimum expanding-window training rows.",
    )
    parser.add_argument(
        "--aq-minimum-train-rows",
        type=int,
        default=None,
        help="Override the minimum leave-one-year-out accrual-quality training rows.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = PredictiveValiditySettings(**config.get("settings", {}))
    runtime = dict(config.get("runtime", {}))
    workers = max(
        1,
        int(
            args.workers
            if args.workers is not None
            else runtime.get("workers", 5)
        ),
    )
    bootstrap_batch_size = max(
        1,
        int(
            args.bootstrap_batch_size
            if args.bootstrap_batch_size is not None
            else runtime.get("bootstrap_batch_size", 128)
        ),
    )
    if args.bootstrap_draws is not None:
        settings = replace(settings, bootstrap_draws=max(1, args.bootstrap_draws))
    if args.minimum_train_rows is not None:
        settings = replace(
            settings,
            minimum_train_rows=max(1, args.minimum_train_rows),
        )
    if args.aq_minimum_train_rows is not None:
        settings = replace(
            settings,
            aq_minimum_train_rows=max(1, args.aq_minimum_train_rows),
        )

    panel_path = resolve(config_path, config["paths"]["panel_input"])
    output_dir = resolve(config_path, config["paths"]["output_dir"])
    if not panel_path.exists():
        raise FileNotFoundError(f"Processed panel missing: {panel_path}")
    if output_dir.exists():
        if not args.clean:
            raise FileExistsError(
                f"Output directory exists: {output_dir}. Rerun with --clean."
            )
        shutil.rmtree(output_dir)

    stage(f"loading paired reporting-state panel: {panel_path}")
    master = pd.read_csv(panel_path, low_memory=False)
    panel, sample_manifest = select_analysis_sample(
        master,
        config.get("sample", {}),
    )
    stage(
        f"selected nonfinancial sample: {len(panel):,} rows, "
        f"{panel[['issuer_ticker', 'fiscal_year']].drop_duplicates().shape[0]:,} firm-years"
    )
    stage(
        "runtime: "
        f"bootstrap_workers={min(workers, 5)}, "
        f"bootstrap_draws={settings.bootstrap_draws:,}, "
        f"batch_size={bootstrap_batch_size}"
    )

    tables = run_predictive_validity_parallel(
        panel,
        settings,
        workers=workers,
        bootstrap_batch_size=bootstrap_batch_size,
        progress=stage,
    )
    required_nonempty = [
        "predictive_validity_cases",
        "predictive_validity_coefficients",
        "predictive_validity_oos_predictions",
        "predictive_validity_oos_summary",
        "predictive_validity_oos_state_differences",
        "accrual_quality_cases",
        "accrual_quality_coefficients",
        "accrual_quality_crossfit_cases",
        "accrual_quality_summary",
        "accrual_quality_state_differences",
    ]
    empty = [name for name in required_nonempty if tables[name].empty]
    if empty:
        raise ValueError(f"Predictive-validity outputs unexpectedly empty: {empty}")

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "config": str(config_path),
        "panel_input": str(panel_path),
        "master_rows": int(len(master)),
        "analysis_rows": int(len(panel)),
        "analysis_issuer_years": int(
            panel[["issuer_ticker", "fiscal_year"]].drop_duplicates().shape[0]
        ),
        "analysis_key_sha256": output_hash(
            panel[["issuer_ticker", "fiscal_year", "audit_status"]]
        ),
        "settings": settings.__dict__,
        "runtime": {
            "workers_requested": workers,
            "bootstrap_workers_effective": min(workers, 5),
            "bootstrap_batch_size": bootstrap_batch_size,
        },
        "sample_selection": sample_manifest.to_dict(orient="records"),
        "outputs": {},
    }

    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["outputs"][name] = {
            "path": str(path),
            "rows": int(len(frame)),
            "sha256": output_hash(frame),
        }
        stage(f"wrote {path} ({len(frame):,} rows)")

    manifest_path = output_dir / "predictive_validity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    stage(f"complete: {manifest_path}")


if __name__ == "__main__":
    main()
