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
from audit_da.results_completion.applied import supplemental_inference  # noqa: E402
from audit_da.results_completion.applied_unique import applied_consequence_tables  # noqa: E402
from audit_da.results_completion.confirmatory import confirmatory_summary  # noqa: E402
from audit_da.results_completion.core import (  # noqa: E402
    CompletionSettings, output_hash, sample_exclusion_manifest, write_outputs,
)
from audit_da.results_completion.final_contract import (  # noqa: E402
    final_contract_sha256, validate_final_contract,
)
from audit_da.results_completion.method_locked import randomisation_benchmarks  # noqa: E402
from audit_da.results_completion.method_v2 import (  # noqa: E402
    benchmark_movement_diagnostic,
    build_attribution_cases,
    estimate_accrual_architectures,
)
from audit_da.results_completion.parallel import attribution_tables, switching_tables  # noqa: E402
from audit_da.results_completion.switching import direct_revision_tables  # noqa: E402
from audit_da.results_completion.switching_complete_case import (  # noqa: E402
    profit_gate_sensitivity, switching_cases,
)
from audit_da.results_completion.time_shift_two_player import time_shift_benchmarks  # noqa: E402


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def stage(message: str) -> None:
    print(f"[final-results] {message}", flush=True)


def _validate_panel(panel: pd.DataFrame, config: dict, label: str) -> None:
    required = list(config.get("panel_contract", {}).get("required_columns", []))
    required += ["issuer_ticker", "fiscal_year", "audit_status"]
    missing = sorted(set(required) - set(panel.columns))
    if missing:
        raise ValueError(f"{label} panel missing columns: {missing}")


def _panel_contract(
    master_analysis: pd.DataFrame,
    analysis_panel: pd.DataFrame,
    master_training: pd.DataFrame,
    training_panel: pd.DataFrame,
    analysis_manifest: pd.DataFrame,
    training_manifest: pd.DataFrame,
) -> dict[str, object]:
    keys = ["issuer_ticker", "fiscal_year", "audit_status"]
    return {
        "master_analysis_rows": len(master_analysis),
        "analysis_rows": len(analysis_panel),
        "analysis_issuer_years": analysis_panel[["issuer_ticker", "fiscal_year"]].drop_duplicates().shape[0],
        "analysis_key_sha256": output_hash(analysis_panel[keys]),
        "master_training_rows": len(master_training),
        "training_rows": len(training_panel),
        "training_issuer_years": training_panel[["issuer_ticker", "fiscal_year"]].drop_duplicates().shape[0],
        "training_key_sha256": output_hash(training_panel[keys]),
        "analysis_sample_manifest": analysis_manifest.to_dict(orient="records"),
        "training_sample_manifest": training_manifest.to_dict(orient="records"),
    }


def _load_required_supplemental(
    config_path: Path,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = config.get("required_supplemental_paths", {})
    expected = {"concentration_input", "near_zero_input"}
    missing_keys = sorted(expected - set(required))
    if missing_keys:
        raise ValueError(f"Final contract missing supplemental config keys: {missing_keys}")
    loaded: dict[str, pd.DataFrame] = {}
    for name in sorted(expected):
        path = resolve(config_path, required[name])
        if not path.exists():
            raise FileNotFoundError(f"Required supplemental input absent: {name}={path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Required supplemental input empty: {name}={path}")
        loaded[name] = frame
    return loaded["concentration_input"], loaded["near_zero_input"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final Results pipeline under contract v2"
    )
    parser.add_argument("--config", default="config/results_completion.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--simulation-batch-size", type=int, default=None)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete final_output_dir before running.",
    )
    parser.add_argument(
        "--include-supplemental",
        action="store_true",
        help=(
            "Run line-item concentration and near-zero-CFO supplemental diagnostics. "
            "Core Results do not require raw CFS data or supplemental inputs."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = validate_final_contract(config)
    settings = CompletionSettings(**config.get("settings", {}))
    if args.workers is not None:
        settings = replace(settings, parallel_workers=max(1, args.workers))
    if args.simulation_batch_size is not None:
        settings = replace(
            settings,
            simulation_batch_size=max(1, args.simulation_batch_size),
        )

    paths = config["paths"]
    analysis_path = resolve(config_path, paths["analysis_panel_input"])
    training_path = resolve(config_path, paths["training_panel_input"])
    output_dir = resolve(config_path, paths["final_output_dir"])
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Final output directory exists: {output_dir}. Use the guarded clean runner."
            )
        shutil.rmtree(output_dir)

    stage(f"loading locked analysis panel: {analysis_path}")
    master_analysis = pd.read_csv(analysis_path, low_memory=False)
    stage(f"loading unrestricted training panel: {training_path}")
    master_training = pd.read_csv(training_path, low_memory=False)
    _validate_panel(master_analysis, config, "analysis")
    _validate_panel(master_training, config, "training")

    analysis_panel, analysis_selection = select_analysis_sample(
        master_analysis, config.get("sample", {})
    )
    training_panel, training_selection = select_analysis_sample(
        master_training, config.get("sample", {})
    )
    panel_contract = _panel_contract(
        master_analysis, analysis_panel,
        master_training, training_panel,
        analysis_selection, training_selection,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "analysis_training_contract.json").write_text(
        json.dumps(panel_contract, indent=2), encoding="utf-8"
    )
    (output_dir / "final_method_contract.json").write_text(
        json.dumps({
            "contract": contract,
            "sha256": final_contract_sha256(contract),
            "run_scope": (
                "core_plus_supplemental" if args.include_supplemental else "core"
            ),
        }, indent=2),
        encoding="utf-8",
    )

    industry = config.get("columns", {}).get("industry", "icb_l1")

    stage("estimating final accrual architectures")
    if config.get("models"):
        accrual, estimation_manifest = estimate_accrual_architectures(
            analysis_panel, training_panel, settings,
            models=config["models"], industry_column=industry,
        )
    else:
        accrual, estimation_manifest = estimate_accrual_architectures(
            analysis_panel, training_panel, settings,
            industry_column=industry,
        )

    stage("building exact two-player fixed-reference attribution")
    attribution_cases = build_attribution_cases(accrual, analysis_panel, settings)
    tables: dict[str, pd.DataFrame] = {
        "accrual_architecture_cases": accrual,
        "accrual_estimation_manifest": estimation_manifest,
        "rq1_attribution_cases": attribution_cases,
        "rq1_version_specific_benchmark_movement": benchmark_movement_diagnostic(
            accrual, analysis_panel, settings
        ),
    }
    tables.update(direct_revision_tables(analysis_panel, settings))
    tables.update(attribution_tables(attribution_cases, settings, progress=stage))

    stage("constructing common complete-case switching populations")
    direct, model_cases = switching_cases(accrual, analysis_panel, settings)
    tables["rq2_direct_cases"] = direct
    tables["rq2_model_cases"] = model_cases
    tables.update(switching_tables(direct, model_cases, settings, progress=stage))
    tables["rq2_profit_gate_sensitivity"] = profit_gate_sensitivity(
        direct, model_cases, settings
    )
    tables["rq2_randomisation"] = randomisation_benchmarks(
        direct, model_cases, settings, progress=stage
    )

    stage("running two-player time-shift diagnostics")
    tables["rq1_time_shift_benchmarks"] = time_shift_benchmarks(
        attribution_cases, analysis_panel, settings, progress=stage
    )

    stage("estimating applied models with unique test families")
    applied_full, applied_unique, applied_manifest = applied_consequence_tables(
        accrual, analysis_panel, settings
    )
    tables["applied_consequence_full"] = applied_full
    tables["applied_consequence_unique_tests"] = applied_unique
    tables["applied_consequence_manifest"] = applied_manifest

    if args.include_supplemental:
        stage("running requested supplemental diagnostics")
        concentration, near_zero = _load_required_supplemental(config_path, config)
        tables["supplemental_inference"] = supplemental_inference(
            concentration, near_zero, settings
        )
        if tables["supplemental_inference"].empty:
            raise ValueError("Required supplemental inputs produced no diagnostic rows")
    else:
        stage("skipping supplemental diagnostics; core scope requested")

    tables["confirmatory_family_summary"] = confirmatory_summary(
        tables["rq1_attribution_matrix"],
        tables["rq2_switch_summary"],
        tables["rq2_randomisation"],
    )

    analysis_exclusions = sample_exclusion_manifest(analysis_panel, accrual, settings)
    tables["sample_exclusion_manifest"] = pd.concat([
        analysis_selection.assign(population="analysis"),
        training_selection.assign(population="training"),
        analysis_exclusions.assign(population="analysis"),
    ], ignore_index=True, sort=False)

    metadata = {
        "config": str(config_path),
        "analysis_panel_input": str(analysis_path),
        "training_panel_input": str(training_path),
        "final_contract_sha256": final_contract_sha256(contract),
        "run_scope": "core_plus_supplemental" if args.include_supplemental else "core",
        "seed": settings.seed,
        "bootstrap_draws": settings.bootstrap_draws,
        "simulation_draws": settings.simulation_draws,
        "parallel_workers": settings.parallel_workers,
        "simulation_batch_size": settings.simulation_batch_size,
        "analysis_key_sha256": panel_contract["analysis_key_sha256"],
        "training_key_sha256": panel_contract["training_key_sha256"],
    }
    stage(f"writing {len(tables)} final result tables")
    write_outputs(tables, output_dir, metadata)
    stage(f"complete: {output_dir}")


if __name__ == "__main__":
    main()
