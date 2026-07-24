from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

LOCKED_METHOD_CONTRACT: dict[str, Any] = {
    "version": "results-method-contract-v1",
    "jones_ordinary_intercept": False,
    "jones_feature_centering": False,
    "jones_scale_regressor": "inv_assets",
    "attribution_estimand": "three_player_pat_cfo_benchmark_residual",
    "attribution_players": ["pat", "cfo", "benchmark_residual"],
    "signed_shift_reassignment_cell": "fiscal_year",
}


def validate_method_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(config.get("method_contract", {}))
    missing = [key for key in LOCKED_METHOD_CONTRACT if key not in supplied]
    mismatches = {
        key: {"expected": expected, "observed": supplied.get(key)}
        for key, expected in LOCKED_METHOD_CONTRACT.items()
        if key in supplied and supplied.get(key) != expected
    }
    if missing or mismatches:
        raise ValueError(
            "Results method contract mismatch. "
            f"Missing={missing}; mismatches={mismatches}"
        )
    return {key: supplied[key] for key in LOCKED_METHOD_CONTRACT}


def method_contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(contract), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
