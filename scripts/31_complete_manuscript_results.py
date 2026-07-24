from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.panel_metadata import select_analysis_sample  # noqa: E402
from audit_da.results_completion import (  # noqa: E402
    CompletionSettings,
    applied_consequence_tables,
    attribution_tables,
    build_attribution_cases,
    confirmatory_summary,
    direct_revision_tables,
    estimate_accrual_architectures,
    output_hash,
    profit_gate_sensitivity,
    randomisation_benchmarks,
    sample_exclusion_manifest,
    supplemental_inference,
    switching_cases,
    switching_tables,
    time_shift_benchmarks,
    write_outputs,
)


RESUME_TABLES = (
    "accrual_architecture_cases",
    "accrual_estimation_manifest",
    "rq1_attribution_cases",
    "direct_revision_cases",
    "direct_revision_symmetric",
    "direct_revision_asymmetric",
    "rq1_attribution_matrix",
    "rq1_signed_quadrants",
    "rq2_direct_cases",
    "rq2_model_cases",
    "rq2_switch_summary",
    "rq2_switch_magnitudes",
    "rq2_jaccard",
)
SAMPLE_CONTRACT = "analysis_sample_contract.json"


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def stage(message: str) -> None:
    print(f"[results-completion] {message}", flush=True)


def load_resume_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    missing = [name for name in RESUME_TABLES if not (output_dir / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot resume because required checkpoint tables are missing: "
            + ", ".join(missing)
        )
    tables: dict[str, pd.DataFrame] = {}
    for name in RESUME_TABLES:
        path = output_dir / f"{name}.csv"
        tables[name] = pd.read_csv(path)
        stage(f"loaded checkpoint {name} ({len(tables[name]):,} rows)")
    return tables


def load_or_compute_checkpoint(
    name: str,
    output_dir: Path,
    resume: bool,
    compute,
) -> pd.DataFrame:
    path = output_dir / f"{name}.csv"
    if resume and path.exists():
        frame = pd.read_csv(path)
        stage(f"loaded heavy checkpoint {name} ({len(frame):,} rows)")
        return frame
    frame = compute()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    stage(f"wrote heavy checkpoint {name} ({len(frame):,} rows)")
    return frame


def _validate_panel_contract(panel: pd.DataFrame, config: dict) -> None:
    required = list(config.get("panel_contract", {}).get("required_columns", []))
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(
            "Processed panel is not enriched for the locked Chapter 4 run. "
            f"Missing columns: {missing}. Rebuild it with scripts/01_build_panel.py."
        )
    for column in ("icb_l1", "financial_flag"):
        if column in required and panel[column].isna().all():
            raise ValueError(f"Required panel column is entirely missing: {column}")


def _analysis_contract(
    master_panel: pd.DataFrame,
    analysis_panel: pd.DataFrame,
    sample_manifest: pd.DataFrame,
) -> dict[str, object]:
    key_columns = [
        column
        for column in ("issuer_ticker", "fiscal_year", "audit_status")
        if column in analysis_panel
    ]
    return {
        "master_rows": len(master_panel),
        "analysis_rows": len(analysis_panel),
        "master_issuer_years": master_panel[["issuer_ticker", "fiscal_year"]]
        .drop_duplicates()
        .shape[0],
        "analysis_issuer_years": analysis_panel[["issuer_ticker", "fiscal_year"]]
        .drop_duplicates()
        .shape[0],
        "analysis_key_sha256": output_hash(analysis_panel[key_columns]),
        "sample_manifest": sample_manifest.to_dict(orient="records"),
    }


def _write_or_validate_analysis_contract(
    output_dir: Path,
    contract: dict[str, object],
    resume: bool,
) -> None:
    path = output_dir / SAMPLE_CONTRACT
    if resume:
        if not path.exists():
            raise FileNotFoundError(
                "Cannot resume old checkpoints because the nonfinancial analysis-sample "
                f"contract is missing: {path}. Run without --resume."
            )
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "Resume checkpoints were produced from a different analysis sample. "
                "Delete artifacts/manuscript_results or run without --resume."
            )
        stage("validated nonfinancial analysis-sample contract")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        stage(f"wrote nonfinancial analysis-sample contract: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete manuscript Results outputs required by the locked research design"
    )
    parser.add_argument("--config", default="config/results_completion.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse architecture, attribution, switching, and completed heavy-stage "
            "CSV checkpoints already written to output_dir."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override process workers. Use 31 on a 32-core workstation.",
    )
    parser.add_argument(
        "--simulation-batch-size",
        type=int,
        default=None,
        help="Override vectorized simulation batch size; 32 is memory-conservative.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = CompletionSettings(**config.get("settings", {}))
    if args.workers is not None:
        settings = replace(settings, parallel_workers=max(1, args.workers))
    if args.simulation_batch_size is not None:
        settings = replace(
            settings, simulation_batch_size=max(1, args.simulation_batch_size)
        )

    panel_path = resolve(config_path, config["paths"]["panel_input"])
    output_dir = resolve(config_path, config["paths"]["output_dir"])

    stage(f"loading master panel: {panel_path}")
    master_panel = pd.read_csv(panel_path)
    stage(f"master panel loaded ({len(master_panel):,} rows)")
    _validate_panel_contract(master_panel, config)
    panel, financial_exclusion_manifest = select_analysis_sample(
        master_panel, config.get("sample", {})
    )
    excluded_rows = len(master_panel) - len(panel)
    stage(
        f"nonfinancial analysis panel selected ({len(panel):,} rows; "
        f"{excluded_rows:,} rows excluded)"
    )
    contract = _analysis_contract(master_panel, panel, financial_exclusion_manifest)
    _write_or_validate_analysis_contract(output_dir, contract, args.resume)

    stage(
        f"parallel settings: workers={settings.parallel_workers or 'auto'}, "
        f"batch_size={settings.simulation_batch_size}, "
        f"BLAS_threads_per_worker={settings.blas_threads_per_worker}"
    )

    if args.resume:
        stage(f"resuming from CSV checkpoints in {output_dir}")
        tables = load_resume_tables(output_dir)
        accrual = tables["accrual_architecture_cases"]
        attribution_cases = tables["rq1_attribution_cases"]
        direct = tables["rq2_direct_cases"]
        model_cases = tables["rq2_model_cases"]
    else:
        stage("estimating accrual architectures")
        if config.get("models"):
            accrual, estimation_manifest = estimate_accrual_architectures(
                panel,
                settings,
                models=config["models"],
                industry_column=config.get("columns", {}).get("industry", "icb_l1"),
            )
        else:
            accrual, estimation_manifest = estimate_accrual_architectures(
                panel,
                settings,
                industry_column=config.get("columns", {}).get("industry", "icb_l1"),
            )
        stage(f"accrual architectures complete ({len(accrual):,} rows)")

        stage("building Shapley attribution cases")
        attribution_cases = build_attribution_cases(accrual, panel, settings)
        tables = {
            "accrual_architecture_cases": accrual,
            "accrual_estimation_manifest": estimation_manifest,
            "rq1_attribution_cases": attribution_cases,
        }
        tables.update(direct_revision_tables(panel, settings))
        stage("running vectorized issuer-cluster attribution inference")
        tables.update(attribution_tables(attribution_cases, settings, progress=stage))

        stage("constructing RQ2 switching cases")
        direct, model_cases = switching_cases(accrual, panel, settings)
        tables["rq2_direct_cases"] = direct
        tables["rq2_model_cases"] = model_cases
        stage("running vectorized issuer-cluster switching inference")
        tables.update(switching_tables(direct, model_cases, settings, progress=stage))

        output_dir.mkdir(parents=True, exist_ok=True)
        for name in RESUME_TABLES:
            tables[name].to_csv(output_dir / f"{name}.csv", index=False)
        stage("wrote base resume checkpoints")

    stage("running profit-gate threshold sensitivity")
    tables["rq2_profit_gate_sensitivity"] = load_or_compute_checkpoint(
        "rq2_profit_gate_sensitivity",
        output_dir,
        args.resume,
        lambda: profit_gate_sensitivity(direct, model_cases, settings),
    )

    stage("running vectorized RQ2 randomisation benchmarks")
    tables["rq2_randomisation"] = load_or_compute_checkpoint(
        "rq2_randomisation",
        output_dir,
        args.resume,
        lambda: randomisation_benchmarks(
            direct, model_cases, settings, progress=stage
        ),
    )

    analysis_manifest = sample_exclusion_manifest(panel, accrual, settings)
    tables["sample_exclusion_manifest"] = pd.concat(
        [financial_exclusion_manifest, analysis_manifest], ignore_index=True, sort=False
    )

    stage("running vectorized RQ1 time-shift donor benchmarks")
    tables["rq1_time_shift_benchmarks"] = load_or_compute_checkpoint(
        "rq1_time_shift_benchmarks",
        output_dir,
        args.resume,
        lambda: time_shift_benchmarks(
            attribution_cases, panel, settings, progress=stage
        ),
    )

    stage("running all applied-consequence comparisons")
    applied_path = output_dir / "applied_consequence_full.csv"
    applied_manifest_path = output_dir / "applied_consequence_manifest.csv"
    if args.resume and applied_path.exists() and applied_manifest_path.exists():
        applied = pd.read_csv(applied_path)
        applied_manifest = pd.read_csv(applied_manifest_path)
        stage("loaded applied-consequence checkpoints")
    else:
        applied, applied_manifest = applied_consequence_tables(accrual, panel, settings)
        output_dir.mkdir(parents=True, exist_ok=True)
        applied.to_csv(applied_path, index=False)
        applied_manifest.to_csv(applied_manifest_path, index=False)
        stage("wrote applied-consequence checkpoints")
    tables["applied_consequence_full"] = applied
    tables["applied_consequence_manifest"] = applied_manifest

    optional = config.get("optional_paths", {})
    concentration = near_zero = None
    if optional.get("concentration_input"):
        path = resolve(config_path, optional["concentration_input"])
        if path.exists():
            concentration = pd.read_csv(path)
    if optional.get("near_zero_input"):
        path = resolve(config_path, optional["near_zero_input"])
        if path.exists():
            near_zero = pd.read_csv(path)
    tables["supplemental_inference"] = supplemental_inference(
        concentration, near_zero, settings
    )
    tables["confirmatory_family_summary"] = confirmatory_summary(
        tables["rq1_attribution_matrix"],
        tables["rq2_switch_summary"],
        tables["rq2_randomisation"],
    )

    stage(f"writing {len(tables)} result tables and deterministic hashes")
    write_outputs(
        tables,
        output_dir,
        {
            "config": str(config_path),
            "panel_input": str(panel_path),
            "seed": settings.seed,
            "bootstrap_draws": settings.bootstrap_draws,
            "simulation_draws": settings.simulation_draws,
            "parallel_workers": settings.parallel_workers,
            "simulation_batch_size": settings.simulation_batch_size,
            "blas_threads_per_worker": settings.blas_threads_per_worker,
            "master_panel_rows": len(master_panel),
            "analysis_panel_rows": len(panel),
            "financial_rows_excluded": excluded_rows,
            "analysis_key_sha256": contract["analysis_key_sha256"],
            "resumed": args.resume,
        },
    )
    stage(f"complete: wrote {len(tables)} result tables to {output_dir}")


if __name__ == "__main__":
    main()
