from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


LOCKED_FINAL_CONTRACT: dict[str, Any] = {
    "version": "final-results-contract-v2",
    "analysis_population": "locked_nonfinancial_panel",
    "training_population": "unrestricted_nonfinancial_history",
    "require_training_start_year": True,
    "jones_ordinary_intercept": False,
    "jones_feature_centering": False,
    "jones_scale_regressor": "inv_assets",
    "current_test_outcome_clipping": False,
    "attribution_estimand": "two_player_pat_cfo_fixed_reference",
    "attribution_players": ["pat", "cfo"],
    "attribution_benchmarks": ["audited_reference", "pre_reference"],
    "direct_switching_population": "common_complete_case",
    "signed_shift_reassignment_cell": "fiscal_year",
    "applied_primary_test": "paired_difference",
    "stacked_state_slopes": "fully_interacted",
    "signed_da_difference_family": "one_unique_test_per_focal",
    "supplemental_inputs": "required",
}


def validate_final_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    supplied = dict(config.get("final_method_contract", {}))
    missing = [key for key in LOCKED_FINAL_CONTRACT if key not in supplied]
    mismatches = {
        key: {"expected": expected, "observed": supplied.get(key)}
        for key, expected in LOCKED_FINAL_CONTRACT.items()
        if key in supplied and supplied.get(key) != expected
    }
    if missing or mismatches:
        raise ValueError(
            "Final Results contract mismatch. "
            f"Missing={missing}; mismatches={mismatches}"
        )
    return {key: supplied[key] for key in LOCKED_FINAL_CONTRACT}


def final_contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(contract), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
