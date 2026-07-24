#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.results_completion.architecture import _shapley_three  # noqa: E402
from audit_da.results_completion.method_contract import (  # noqa: E402
    LOCKED_METHOD_CONTRACT,
    method_contract_sha256,
    validate_method_contract,
)
from audit_da.results_completion.method_locked import (  # noqa: E402
    _fit_model_no_intercept,
    _within_cell_permutations,
)


def _resolve(config_path: Path, value: str) -> Path:
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
    training = pd.DataFrame(
        {
            "inv_assets": [0.01, 0.02, 0.03, 0.04, 0.05],
            "drev_scaled": [0.2, 0.1, -0.1, 0.3, -0.2],
            "ta_scaled": [0.03, 0.01, -0.02, 0.05, -0.03],
        }
    )
    scaler, model, _ = _fit_model_no_intercept(
        training, ["inv_assets", "drev_scaled"]
    )
    intercept_pass = (
        model.fit_intercept is False
        and scaler.with_mean is False
        and float(model.intercept_) == 0.0
    )

    da_pre = np.array([0.20, -0.10, 0.05])
    pat = np.array([0.02, -0.01, 0.01])
    cfo = np.array([-0.03, 0.04, -0.02])
    benchmark = np.array([0.01, -0.02, 0.005])
    phi_pat, phi_cfo, phi_benchmark = _shapley_three(
        da_pre, pat, cfo, benchmark
    )
    total_move = pat + cfo + benchmark
    reduction = np.abs(da_pre) - np.abs(da_pre + total_move)
    shapley_error = float(
        np.max(np.abs(phi_pat + phi_cfo + phi_benchmark - reduction))
    )
    shapley_pass = shapley_error <= 1.0e-12

    values = np.array([10.0, 11.0, 20.0, 21.0])
    cells = np.array([2020, 2020, 2021, 2021])
    permutations = _within_cell_permutations(
        values, cells, 100, np.random.default_rng(7)
    )
    cell_pass = all(
        np.all(
            np.sort(permutations[:, indices], axis=1)
            == np.sort(values[indices])
        )
        for indices in (np.array([0, 1]), np.array([2, 3]))
    )

    checks = {
        "no_ordinary_intercept": bool(intercept_pass),
        "no_feature_centering": bool(scaler.with_mean is False),
        "three_player_shapley_efficiency": bool(shapley_pass),
        "three_player_shapley_max_error": shapley_error,
        "within_fiscal_year_reassignment": bool(cell_pass),
    }
    required = (
        "no_ordinary_intercept",
        "no_feature_centering",
        "three_player_shapley_efficiency",
        "within_fiscal_year_reassignment",
    )
    if not all(checks[key] for key in required):
        raise AssertionError(f"Runtime method-contract check failed: {checks}")
    return checks


def _audit_existing_outputs(output_dir: Path) -> dict[str, object]:
    audit: dict[str, object] = {"output_dir": str(output_dir)}

    estimation_path = output_dir / "accrual_estimation_manifest.csv"
    if estimation_path.exists():
        estimation = pd.read_csv(estimation_path)
        required = {
            "status",
            "ordinary_intercept",
            "feature_centering",
            "scale_regressor",
        }
        missing = sorted(required - set(estimation.columns))
        if missing:
            raise ValueError(
                "Existing architecture checkpoint predates the locked method contract; "
                f"missing columns: {missing}"
            )
        estimated = estimation.loc[estimation["status"].eq("estimated")].copy()
        if estimated.empty:
            raise ValueError("Architecture checkpoint has no estimated specifications")
        if _strict_bool(
            estimated["ordinary_intercept"], "ordinary_intercept"
        ).any():
            raise ValueError("Existing checkpoint used an ordinary intercept")
        if _strict_bool(
            estimated["feature_centering"], "feature_centering"
        ).any():
            raise ValueError("Existing checkpoint centred Jones predictors")
        if not estimated["scale_regressor"].eq("inv_assets").all():
            raise ValueError("Existing checkpoint has an invalid scale regressor")
        audit["architecture_checkpoint"] = "PASS"
        audit["estimated_architecture_rows"] = int(len(estimated))
    else:
        audit["architecture_checkpoint"] = "NOT_PRESENT"

    attribution_path = output_dir / "rq1_attribution_cases.csv"
    if attribution_path.exists():
        attribution = pd.read_csv(attribution_path)
        required = {
            "phi_pat",
            "phi_cfo",
            "phi_benchmark",
            "reduction",
            "attribution_estimand",
            "attribution_player_count",
        }
        missing = sorted(required - set(attribution.columns))
        if missing:
            raise ValueError(
                "Existing attribution checkpoint predates the locked estimand; "
                f"missing columns: {missing}"
            )
        if attribution.empty:
            raise ValueError("Attribution checkpoint is empty")
        error = (
            attribution[["phi_pat", "phi_cfo", "phi_benchmark"]].sum(axis=1)
            - attribution["reduction"]
        ).abs()
        if float(error.max()) > 1.0e-10:
            raise ValueError("Existing Shapley checkpoint violates efficiency")
        if not attribution["attribution_player_count"].eq(3).all():
            raise ValueError("Existing attribution checkpoint is not three-player")
        if not attribution["attribution_estimand"].eq(
            LOCKED_METHOD_CONTRACT["attribution_estimand"]
        ).all():
            raise ValueError("Existing attribution checkpoint uses another estimand")
        audit["attribution_checkpoint"] = "PASS"
        audit["attribution_max_efficiency_error"] = float(error.max())
    else:
        audit["attribution_checkpoint"] = "NOT_PRESENT"

    randomisation_path = output_dir / "rq2_randomisation.csv"
    if randomisation_path.exists():
        randomisation = pd.read_csv(randomisation_path)
        required = {"benchmark", "reassignment_cell", "cell_count"}
        missing = sorted(required - set(randomisation.columns))
        if missing:
            raise ValueError(
                "Existing randomisation checkpoint predates within-year reassignment; "
                f"missing columns: {missing}"
            )
        signed = randomisation.loc[
            randomisation["benchmark"].eq("signed_shift_reassignment")
        ]
        if signed.empty:
            raise ValueError("Randomisation checkpoint has no signed reassignment rows")
        if not signed["reassignment_cell"].eq("fiscal_year").all():
            raise ValueError("Signed shifts were not reassigned within fiscal year")
        if not pd.to_numeric(signed["cell_count"], errors="coerce").ge(1).all():
            raise ValueError("Randomisation checkpoint has invalid fiscal-year cells")
        audit["randomisation_checkpoint"] = "PASS"
        audit["signed_reassignment_rows"] = int(len(signed))
    else:
        audit["randomisation_checkpoint"] = "NOT_PRESENT"

    return audit


def run_audit(config_path: Path, check_outputs: bool = True) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contract = validate_method_contract(config)
    output_dir = _resolve(config_path, config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "method_contract_audit.json"
    result = {
        "status": "PASS",
        "audit_path": str(audit_path),
        "contract": contract,
        "contract_sha256": method_contract_sha256(contract),
        "locked_reference": LOCKED_METHOD_CONTRACT,
        "runtime_checks": _runtime_checks(),
        "existing_outputs": (
            _audit_existing_outputs(output_dir)
            if check_outputs
            else {"status": "SKIPPED"}
        ),
    }
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the locked Jones, Shapley, and randomisation contract"
    )
    parser.add_argument("--config", default="config/results_completion.yaml")
    parser.add_argument(
        "--skip-existing-outputs",
        action="store_true",
        help="Validate code and config without inspecting prior result checkpoints.",
    )
    args = parser.parse_args()
    result = run_audit(
        Path(args.config).resolve(),
        check_outputs=not args.skip_existing_outputs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
