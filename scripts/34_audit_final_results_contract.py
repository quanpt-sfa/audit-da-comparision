#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.panel_metadata import select_analysis_sample  # noqa: E402
from audit_da.results_completion.core import DEFAULT_MODELS  # noqa: E402
from audit_da.results_completion.final_contract import (  # noqa: E402
    LOCKED_FINAL_CONTRACT,
    final_contract_sha256,
    validate_final_contract,
)
from audit_da.results_completion.method_v2 import (  # noqa: E402
    _fit_model_no_intercept,
    _shapley_two,
)
from audit_da.results_completion.method_locked import _within_cell_permutations  # noqa: E402


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _strict_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        bad = sorted(series.loc[mapped.isna()].astype(str).unique())[:10]
        raise ValueError(f"Cannot parse boolean checkpoint column {column}: {bad}")
    return mapped.astype(bool)


def _runtime_checks() -> dict[str, object]:
    training = pd.DataFrame({
        "inv_assets": [0.01, 0.02, 0.03, 0.04, 0.05],
        "drev_scaled": [0.2, 0.1, -0.1, 0.3, -0.2],
        "ta_scaled": [0.03, 0.01, -0.02, 0.05, -0.03],
    })
    scaler, model, _ = _fit_model_no_intercept(
        training, ["inv_assets", "drev_scaled"]
    )
    no_intercept = (
        model.fit_intercept is False
        and scaler.with_mean is False
        and float(model.intercept_) == 0.0
    )
    da_pre = np.array([0.20, -0.10, 0.05])
    pat = np.array([0.02, -0.01, 0.01])
    cfo = np.array([-0.03, 0.04, -0.02])
    phi_pat, phi_cfo = _shapley_two(da_pre, pat, cfo)
    reduction = np.abs(da_pre) - np.abs(da_pre + pat + cfo)
    shapley_error = float(np.max(np.abs(phi_pat + phi_cfo - reduction)))
    values = np.array([10.0, 11.0, 20.0, 21.0])
    years = np.array([2020, 2020, 2021, 2021])
    permutations = _within_cell_permutations(
        values, years, 100, np.random.default_rng(7)
    )
    within_year = all(
        np.all(
            np.sort(permutations[:, indices], axis=1)
            == np.sort(values[indices])
        )
        for indices in (np.array([0, 1]), np.array([2, 3]))
    )
    checks = {
        "no_ordinary_intercept": bool(no_intercept),
        "no_feature_centering": bool(not scaler.with_mean),
        "two_player_shapley_efficiency": shapley_error <= 1.0e-12,
        "two_player_shapley_max_error": shapley_error,
        "within_fiscal_year_reassignment": bool(within_year),
    }
    required = (
        "no_ordinary_intercept",
        "no_feature_centering",
        "two_player_shapley_efficiency",
        "within_fiscal_year_reassignment",
    )
    if not all(checks[key] for key in required):
        raise AssertionError(f"Final runtime contract failed: {checks}")
    return checks


def _audit_supplemental_inputs(
    config_path: Path,
    config: dict,
    *,
    required: bool,
) -> dict[str, object]:
    configured = config.get("required_supplemental_paths", {})
    result: dict[str, object] = {
        "required": required,
        "status": "PASS" if required else "NOT_REQUIRED_FOR_CORE",
        "paths": {},
    }
    for key in ("concentration_input", "near_zero_input"):
        value = configured.get(key)
        if value is None:
            if required:
                raise ValueError(f"Required supplemental path not configured: {key}")
            result["paths"][key] = {
                "configured": False,
                "exists": False,
            }
            continue
        path = resolve(config_path, value)
        exists = path.exists()
        nonempty = bool(exists and path.stat().st_size > 0)
        result["paths"][key] = {
            "configured": True,
            "path": str(path),
            "exists": exists,
            "nonempty": nonempty,
        }
        if required and not exists:
            raise FileNotFoundError(
                f"Required supplemental input missing: {key}={path}"
            )
        if required and not nonempty:
            raise ValueError(f"Required supplemental input empty: {key}={path}")
    return result


def _audit_inputs(
    config_path: Path,
    config: dict,
    *,
    require_supplemental: bool,
) -> dict[str, object]:
    paths = config["paths"]
    analysis_path = resolve(config_path, paths["analysis_panel_input"])
    training_path = resolve(config_path, paths["training_panel_input"])
    if not analysis_path.exists():
        raise FileNotFoundError(f"Locked analysis panel missing: {analysis_path}")
    if not training_path.exists():
        raise FileNotFoundError(f"Unrestricted training panel missing: {training_path}")

    settings = config["settings"]
    contract = config["final_method_contract"]
    source_start = int(settings["source_start_year"])
    training_start = int(settings["training_start_year"])
    test_start = int(settings["test_start_year"])
    lag_support_years = int(contract["lag_support_years"])
    if training_start != source_start + lag_support_years:
        raise ValueError(
            "Usable training start must follow the source support year by the "
            f"locked lag count: source={source_start}, lag={lag_support_years}, "
            f"training={training_start}"
        )
    if test_start != training_start + 1:
        raise ValueError(
            "The first model test year must follow the first usable training year: "
            f"training={training_start}, test={test_start}"
        )

    master_training = pd.read_csv(training_path, low_memory=False)
    training, _ = select_analysis_sample(master_training, config.get("sample", {}))
    audited_label = str(settings["audited_label"])
    source_rows = training.loc[
        training.fiscal_year.eq(source_start)
        & training.audit_status.eq(audited_label)
    ].copy()
    if source_rows.empty:
        raise ValueError(
            f"Training panel has no audited lag-support observations for {source_start}"
        )
    start_rows = training.loc[
        training.fiscal_year.eq(training_start)
        & training.audit_status.eq(audited_label)
    ].copy()
    if start_rows.empty:
        raise ValueError(
            f"Training panel has no audited estimation observations for {training_start}"
        )

    models = config.get("models") or DEFAULT_MODELS
    minimum_rows = int(settings["min_train_rows"])
    complete_counts: dict[str, int] = {}
    for model_name, features in models.items():
        needed = ["ta_scaled", *list(features)]
        missing = sorted(set(needed) - set(start_rows.columns))
        if missing:
            raise ValueError(
                f"Training panel missing model columns for {model_name}: {missing}"
            )
        complete = (
            start_rows[needed]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=needed)
        )
        complete_counts[str(model_name)] = int(len(complete))
        if len(complete) < minimum_rows:
            raise ValueError(
                f"Model {model_name} has only {len(complete)} complete audited rows "
                f"in usable training start year {training_start}; required={minimum_rows}"
            )

    return {
        "analysis_panel": str(analysis_path),
        "training_panel": str(training_path),
        "source_start_year": source_start,
        "source_start_role": "lag_support_only",
        "source_start_audited_rows": int(len(source_rows)),
        "training_start_year": training_start,
        "training_start_audited_rows": int(len(start_rows)),
        "training_start_complete_rows_by_model": complete_counts,
        "model_test_start_year": test_start,
        "direct_comparison_start_year": source_start + lag_support_years,
        "supplemental": _audit_supplemental_inputs(
            config_path,
            config,
            required=require_supplemental,
        ),
    }


def _audit_outputs(
    output_dir: Path,
    *,
    settings: dict,
    require_supplemental: bool,
) -> dict[str, object]:
    required_files = {
        "accrual_estimation_manifest.csv",
        "rq1_attribution_cases.csv",
        "rq2_direct_cases.csv",
        "rq2_switch_summary.csv",
        "rq2_randomisation.csv",
        "applied_consequence_full.csv",
        "applied_consequence_unique_tests.csv",
        "confirmatory_family_summary.csv",
    }
    if require_supplemental:
        required_files.add("supplemental_inference.csv")
    missing = sorted(
        name for name in required_files if not (output_dir / name).exists()
    )
    if missing:
        raise FileNotFoundError(f"Final output bundle incomplete: {missing}")

    estimation = pd.read_csv(output_dir / "accrual_estimation_manifest.csv")
    estimated = estimation.loc[estimation.status.eq("estimated")].copy()
    if estimated.empty:
        raise ValueError("No estimated architecture rows")
    if _strict_bool(estimated["ordinary_intercept"], "ordinary_intercept").any():
        raise ValueError("Architecture used an ordinary intercept")
    if _strict_bool(estimated["feature_centering"], "feature_centering").any():
        raise ValueError("Architecture mean-centred predictors")
    if _strict_bool(estimated["current_outcome_clipped"], "current_outcome_clipped").any():
        raise ValueError("Current test outcomes were clipped")

    expected_training_start = int(settings["training_start_year"])
    expected_model_test_start = int(settings["test_start_year"])
    pooled = estimated.loc[estimated.architecture.eq("pooled")].copy()
    if pooled.empty:
        raise ValueError("No pooled architecture rows were estimated")
    model_years = (
        pooled.groupby("model", observed=True)
        .agg(
            first_test_year=("test_year", "min"),
            earliest_training_year=("train_min_year", "min"),
        )
        .reset_index()
    )
    bad_test = model_years.loc[
        model_years.first_test_year.ne(expected_model_test_start)
    ]
    if not bad_test.empty:
        raise ValueError(
            "Model-based outputs do not begin in the configured first test year: "
            f"expected={expected_model_test_start}; observed="
            f"{bad_test.to_dict(orient='records')}"
        )
    bad_training = model_years.loc[
        model_years.earliest_training_year.ne(expected_training_start)
    ]
    if not bad_training.empty:
        raise ValueError(
            "Historical estimation omitted the configured usable training start year: "
            f"expected={expected_training_start}; observed="
            f"{bad_training.to_dict(orient='records')}"
        )

    attribution = pd.read_csv(output_dir / "rq1_attribution_cases.csv")
    if attribution.empty:
        raise ValueError("Attribution output empty")
    if not attribution.attribution_player_count.eq(2).all():
        raise ValueError("Attribution is not two-player")
    if not attribution.attribution_estimand.eq(
        LOCKED_FINAL_CONTRACT["attribution_estimand"]
    ).all():
        raise ValueError("Attribution uses another estimand")
    if attribution.benchmark.eq("version_specific").any():
        raise ValueError("Version-specific rows entered fixed-reference attribution")
    if float(attribution.phi_benchmark.abs().max()) > 1.0e-12:
        raise ValueError("Compatibility benchmark component is non-zero")
    efficiency = (
        attribution.phi_pat + attribution.phi_cfo - attribution.reduction
    ).abs()
    if float(efficiency.max()) > 1.0e-10:
        raise ValueError("Two-player Shapley efficiency failed")
    if float(attribution.benchmark_move.abs().max()) > 1.0e-10:
        raise ValueError("PAT and CFO do not exhaust fixed-reference movement")

    direct = pd.read_csv(output_dir / "rq2_direct_cases.csv")
    direct_required = [
        "pat_pre", "pat_post", "cfo_pre", "cfo_post",
        "ta_scaled_pre", "ta_scaled_post", "lag_assets_pre",
    ]
    if direct[direct_required].isna().any().any():
        raise ValueError("Direct switching output contains incomplete cases")
    expected_direct_start = int(settings["source_start_year"]) + 1
    observed_direct_start = int(pd.to_numeric(direct.fiscal_year).min())
    if observed_direct_start != expected_direct_start:
        raise ValueError(
            "Direct reporting-state comparisons should retain the first lag-usable "
            f"year: expected={expected_direct_start}, observed={observed_direct_start}"
        )
    summary = pd.read_csv(output_dir / "rq2_switch_summary.csv")
    direct_summary = summary.loc[summary.model.eq("direct")]
    if not direct_summary.denominator.eq(len(direct)).all():
        raise ValueError("Direct switching summaries use different denominators")

    applied = pd.read_csv(output_dir / "applied_consequence_full.csv")
    unique = pd.read_csv(output_dir / "applied_consequence_unique_tests.csv")
    if unique.difference_test_id.duplicated().any():
        raise ValueError("Applied unique-test IDs duplicated")
    if float(applied.estimand_alignment_error.max()) > 1.0e-10:
        raise ValueError("Stacked and paired applied estimands misaligned")
    signed_unique = unique.loc[unique.outcome.eq("signed_da")]
    if len(signed_unique) != 3:
        raise ValueError("Expected one signed-DA test per focal variable")

    supplemental_rows = 0
    supplemental_path = output_dir / "supplemental_inference.csv"
    if supplemental_path.exists():
        supplemental = pd.read_csv(supplemental_path)
        supplemental_rows = int(len(supplemental))
    if require_supplemental and supplemental_rows == 0:
        raise ValueError("Supplemental inference empty")

    return {
        "output_dir": str(output_dir),
        "scope": "core_plus_supplemental" if require_supplemental else "core",
        "estimated_architectures": int(len(estimated)),
        "model_year_contract": model_years.to_dict(orient="records"),
        "model_test_start_year": expected_model_test_start,
        "usable_training_start_year": expected_training_start,
        "direct_comparison_start_year": observed_direct_start,
        "attribution_rows": int(len(attribution)),
        "direct_switching_rows": int(len(direct)),
        "unique_applied_tests": int(len(unique)),
        "supplemental_rows": supplemental_rows,
        "max_shapley_error": float(efficiency.max()),
        "max_applied_alignment_error": float(applied.estimand_alignment_error.max()),
    }


def run_audit(
    config_path: Path,
    check_outputs: bool = True,
    require_supplemental: bool = False,
) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = validate_final_contract(config)
    output_dir = resolve(config_path, config["paths"]["final_output_dir"])
    result = {
        "status": "PASS",
        "scope": "core_plus_supplemental" if require_supplemental else "core",
        "contract": contract,
        "contract_sha256": final_contract_sha256(contract),
        "runtime_checks": _runtime_checks(),
        "inputs": _audit_inputs(
            config_path,
            config,
            require_supplemental=require_supplemental,
        ),
        "outputs": _audit_outputs(
            output_dir,
            settings=config["settings"],
            require_supplemental=require_supplemental,
        ) if check_outputs else {"status": "SKIPPED"},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "full" if require_supplemental else "core"
    audit_path = output_dir / f"final_results_audit_{suffix}.json"
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["audit_path"] = str(audit_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the final Results contract")
    parser.add_argument("--config", default="config/results_completion.yaml")
    parser.add_argument("--skip-existing-outputs", action="store_true")
    parser.add_argument(
        "--include-supplemental",
        action="store_true",
        help=(
            "Also require and audit line-item concentration and near-zero-CFO "
            "supplemental inputs/outputs. Core audit does not require raw CFS data."
        ),
    )
    args = parser.parse_args()
    result = run_audit(
        Path(args.config).resolve(),
        check_outputs=not args.skip_existing_outputs,
        require_supplemental=args.include_supplemental,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
