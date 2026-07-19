from __future__ import annotations

import pandas as pd

from audit_da.cfs_completion import (
    build_pdf_verification_manifest,
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


def _reconciliation_cases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issuer_ticker": ["DBT", "AAA", "BBB", "CCC"],
            "fiscal_year": [2024, 2024, 2024, 2024],
            "section": ["financing", "financing", "investing", "investing"],
            "cfs_resolution": [
                "identity_consistent_offsetting_reclassification_candidate"
            ]
            * 4,
            "offset_channel_pattern": [
                "cff_dominant",
                "cff_dominant",
                "cfi_dominant",
                "cfi_dominant",
            ],
            "cfo_adjustment_direction": [
                "audited_cfo_increase",
                "audited_cfo_decrease",
                "audited_cfo_increase",
                "audited_cfo_increase",
            ],
            "aggregate_section_change_scaled": [0.2, 0.3, -0.2, -0.15],
            "mapped_line_change_sum_scaled": [0.7, 0.3, -0.2, -0.15],
            "reconciliation_residual_scaled": [0.5, 0.0, 0.0, 0.0],
            "mapped_share_of_aggregate": [3.5, 1.0, 1.0, 1.0],
            "mapped_share_within_80_120pct": [False, True, True, True],
            "mapped_concepts_available": [3, 3, 3, 3],
            "dominant_line_item": [
                "cff_borrowing_proceeds",
                "cff_borrowing_proceeds",
                "cfi_ppe_purchase",
                "cfi_loans_advanced",
            ],
            "dominant_line_item_change_scaled": [0.7, 0.3, -0.2, -0.15],
        }
    )


def test_pdf_manifest_contains_forced_and_mechanism_cases() -> None:
    manifest = build_pdf_verification_manifest(
        _reconciliation_cases(),
        {
            "candidate_label": "identity_consistent_offsetting_reclassification_candidate",
            "pdf_verification": {
                "quotas": {
                    "cff_down_borrowing": 2,
                    "cfi_up_ppe": 2,
                    "cfi_up_loans": 2,
                    "reconciliation_outliers": 2,
                },
                "force_cases": [
                    {
                        "issuer_ticker": "DBT",
                        "fiscal_year": 2024,
                        "reason": "forced",
                    }
                ],
            },
        },
    )
    assert set(["DBT", "AAA", "BBB", "CCC"]).issubset(
        set(manifest["issuer_ticker"])
    )
    assert manifest.loc[manifest["issuer_ticker"].eq("DBT"), "verification_priority"].min() == 0
    assert (manifest["verification_result"] == "PENDING").all()


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
