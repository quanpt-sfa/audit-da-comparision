from __future__ import annotations

import pandas as pd

from audit_da.diag_cfs_proxy_validation import _apply_scale_scope_waiver


def test_scale_scope_waiver_changes_status_not_sample_counts() -> None:
    original = pd.DataFrame(
        {
            "sample_restriction": [
                "exclude_scale_scope_flags",
                "nonfinancial_only",
                "analysis_core",
            ],
            "status": [
                "NOT_EVALUATED",
                "EVALUATED",
                "PARTIALLY_EVALUATED",
            ],
            "reason": [
                "No configured scale/scope flag columns found",
                "financial_flag; known_share=1.000000",
                "Unavailable restrictions: scale_scope",
            ],
            "model_rows": [7008, 7008, 7008],
            "firm_years": [7008, 7008, 7008],
            "share_model_rows": [1.0, 1.0, 1.0],
            "lag_assets_floor": [float("nan"), float("nan"), 10.0],
        }
    )
    tables = {"cfs_proxy_sample_restriction_status": original.copy()}
    settings = {
        "sample_restrictions": {
            "require_scale_scope_screening": False,
            "scale_scope_waiver_reason": "Same source, unit, scope, and period.",
        }
    }

    result = _apply_scale_scope_waiver(tables, settings)
    status = result["cfs_proxy_sample_restriction_status"].set_index(
        "sample_restriction"
    )

    assert status.loc["exclude_scale_scope_flags", "status"] == "WAIVED_BY_DESIGN"
    assert status.loc["analysis_core", "status"] == "EVALUATED"
    assert status.loc["analysis_core", "model_rows"] == 7008
    assert status.loc["analysis_core", "firm_years"] == 7008
