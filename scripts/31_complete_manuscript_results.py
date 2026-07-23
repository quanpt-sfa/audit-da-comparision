from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.results_completion import (  # noqa: E402
    CompletionSettings,
    attribution_tables,
    build_attribution_cases,
    confirmatory_summary,
    direct_revision_tables,
    estimate_accrual_architectures,
    profit_gate_sensitivity,
    randomisation_benchmarks,
    switching_cases,
    switching_tables,
    write_outputs,
    sample_exclusion_manifest,
    time_shift_benchmarks,
    applied_consequence_tables,
    supplemental_inference,
)


def resolve(config_path: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (config_path.parent.parent / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete manuscript Results outputs required by the locked research design")
    parser.add_argument("--config", default="config/results_completion.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = CompletionSettings(**config.get("settings", {}))
    panel_path = resolve(config_path, config["paths"]["panel_input"])
    output_dir = resolve(config_path, config["paths"]["output_dir"])
    panel = pd.read_csv(panel_path)

    if config.get("models"):
        accrual, estimation_manifest = estimate_accrual_architectures(
            panel, settings, models=config["models"],
            industry_column=config.get("columns", {}).get("industry", "icb_industry"),
        )
    else:
        accrual, estimation_manifest = estimate_accrual_architectures(
            panel, settings,
            industry_column=config.get("columns", {}).get("industry", "icb_industry"),
        )
    attribution_cases = build_attribution_cases(accrual, panel, settings)
    tables = {
        "accrual_architecture_cases": accrual,
        "accrual_estimation_manifest": estimation_manifest,
        "rq1_attribution_cases": attribution_cases,
    }
    tables.update(direct_revision_tables(panel, settings))
    tables.update(attribution_tables(attribution_cases, settings))
    direct, model_cases = switching_cases(accrual, panel, settings)
    tables["rq2_direct_cases"] = direct
    tables["rq2_model_cases"] = model_cases
    tables.update(switching_tables(direct, model_cases, settings))
    tables["rq2_profit_gate_sensitivity"] = profit_gate_sensitivity(direct, model_cases, settings)
    tables["rq2_randomisation"] = randomisation_benchmarks(direct, model_cases, settings)
    tables["sample_exclusion_manifest"] = sample_exclusion_manifest(panel, accrual, settings)
    tables["rq1_time_shift_benchmarks"] = time_shift_benchmarks(attribution_cases, panel, settings)
    applied, applied_manifest = applied_consequence_tables(accrual, panel, settings)
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
    tables["supplemental_inference"] = supplemental_inference(concentration, near_zero, settings)
    tables["confirmatory_family_summary"] = confirmatory_summary(
        tables["rq1_attribution_matrix"], tables["rq2_switch_summary"], tables["rq2_randomisation"]
    )
    write_outputs(tables, output_dir, {
        "config": str(config_path), "panel_input": str(panel_path), "seed": settings.seed,
        "bootstrap_draws": settings.bootstrap_draws, "simulation_draws": settings.simulation_draws,
    })
    print(f"Wrote {len(tables)} result tables to {output_dir}")


if __name__ == "__main__":
    main()
