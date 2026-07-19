from __future__ import annotations

import pandas as pd

from audit_da.cfs_completion import (
    completion_gate_status,
    core_reconciliation_outputs,
    history_incremental_comparison,
    restrict_estimation_panel,
)


def test_estimation_panel_excludes_financial_unknown_and_nonlisted() -> None:
    panel = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "fiscal_year": [2024] * 5,
            "audit_status": ["unaudited"] * 5,
            "raw_exchange": ["HOSE", "HOSE", "HNX", "OTC", "UPCOM"],
            "lag_assets": [100.0, 100.0, 100.0, 100.0, 0.0],
            "financial_flag": [False, True, pd.NA, False, False],
        }
    )
    eligible, status = restrict_estimation_panel(
        panel,
        {
            "estimation_sample": {
                "require_nonfinancial": True,
                "require_listed": True,
                "listed_exchanges": ["HOSE", "HNX", "UPCOM"],
                "require_valid_ticker": True,
                "require_positive_lag_assets": True,
            }
        },
    )
    assert eligible["issuer_ticker"].tolist() == ["AAA"]
    assert status.loc[0, "eligible_rows"] == 1
    assert status.loc[0, "status"] == "EVALUATED"


def test_history_nested_comparison_uses_identical_common_all_sample() -> None:
    rows = []
    for model, auc, ap, lift in [
        ("earnings_working_capital", 0.70, 0.30, 3.0),
        ("earnings_working_capital_history", 0.72, 0.32, 3.2),
    ]:
        rows.append(
            {
                "proxy_model": model,
                "outcome": "any_candidate",
                "sample_mode": "common_all_models",
                "sample_restriction": "analysis_core",
                "rows": 100,
                "positives": 20,
                "prevalence": 0.20,
                "auc": auc,
                "average_precision": ap,
                "top_decile_lift": lift,
            }
        )
    comparison = history_incremental_comparison(
        pd.DataFrame(rows),
        {
            "history_nested_comparison": {
                "base_model": "earnings_working_capital",
                "nested_model": "earnings_working_capital_history",
                "sample_mode": "common_all_models",
                "sample_restriction": "analysis_core",
            }
        },
    )
    assert len(comparison) == 1
    assert comparison.loc[0, "base_rows"] == comparison.loc[0, "nested_rows"]
    assert round(comparison.loc[0, "delta_auc_nested_minus_base"], 6) == 0.02
    assert round(comparison.loc[0, "delta_ap_nested_minus_base"], 6) == 0.02


def test_core_reconciliation_uses_only_common_primary_keys() -> None:
    line_items = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "AAA", "BBB", "BBB"],
            "fiscal_year": [2024] * 4,
            "audit_status": ["unaudited", "audited"] * 2,
            "selected_statement_family": ["cash_flow_indirect"] * 4,
            "cff_borrowing_proceeds": [0.0, 10.0, 0.0, 100.0],
        }
    )
    observed = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB"],
            "fiscal_year": [2024, 2024],
            "cfs_resolution": [
                "identity_consistent_offsetting_reclassification_candidate",
                "identity_consistent_offsetting_reclassification_candidate",
            ],
            "offset_channel_pattern": ["cff_dominant", "cff_dominant"],
            "cfo_adjustment_direction": [
                "audited_cfo_decrease",
                "audited_cfo_decrease",
            ],
            "delta_cfi_scaled": [0.0, 0.0],
            "delta_cff_scaled": [0.1, 1.0],
        }
    )
    panel = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB"],
            "fiscal_year": [2024, 2024],
            "audit_status": ["audited", "audited"],
            "lag_assets": [100.0, 100.0],
        }
    )
    primary = pd.DataFrame(
        {
            "issuer_ticker": ["AAA"],
            "fiscal_year": [2024],
        }
    )
    all_models = primary.copy()
    outputs = core_reconciliation_outputs(
        line_items,
        observed,
        panel,
        primary,
        all_models,
        {
            "concept_sections": {"cff_borrowing_proceeds": "financing"},
            "candidate_label": "identity_consistent_offsetting_reclassification_candidate",
        },
    )
    cases = outputs[
        "cfs_line_item_reconciliation_cases_common_primary_core"
    ]
    assert cases["issuer_ticker"].nunique() == 1
    assert cases["issuer_ticker"].iloc[0] == "AAA"


def test_completion_status_has_no_pdf_verification_gate() -> None:
    status = completion_gate_status(
        pd.DataFrame([{"status": "EVALUATED"}]),
        pd.DataFrame([{"outcome": "any_candidate"}]),
        pd.DataFrame([{"issuer_ticker": "AAA", "fiscal_year": 2024}]),
    )
    assert "pdf_verification_manifest" not in set(status["gate"])
    assert set(status["gate"]) == {
        "nonfinancial_estimation_sample",
        "nested_history_incremental_test",
        "common_primary_core_reconciliation",
        "scale_scope_screening",
    }
