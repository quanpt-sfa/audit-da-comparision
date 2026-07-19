from __future__ import annotations

import pandas as pd

from audit_da.cfs_line_reconcile import line_item_reconciliation
from audit_da.diag_cfs_proxy_validation import (
    classify_cfs_item,
    compile_item_rules,
    rolling_expected_cfo_proxies,
    validate_proxy_predictions,
)


def test_extended_item_mapping_is_unambiguous() -> None:
    rules = compile_item_rules(
        {
            "line_item_rules": [
                {
                    "concept": "cff_lease_principal_payments",
                    "section": "financing",
                    "include": [r"tra[_ ]no.*thue[_ ]tai[_ ]chinh"],
                },
                {
                    "concept": "cff_debt_repayments",
                    "section": "financing",
                    "include": [r"tra[_ ]no.*vay"],
                    "exclude": [r"thue[_ ]tai[_ ]chinh"],
                },
            ]
        }
    )
    concept, section, count = classify_cfs_item(
        "cash_flow_indirect__tien_chi_tra_no_thue_tai_chinh",
        "Tiền chi trả nợ thuê tài chính",
        "cash_flow_indirect",
        rules,
    )
    assert (concept, section, count) == (
        "cff_lease_principal_payments",
        "financing",
        1,
    )


def _prediction_rows() -> pd.DataFrame:
    rows = []
    residuals = [2.0, 1.0, -1.0, -2.0]
    for model in [
        "raw_cfo_level",
        "within_year_cfo_percentile",
        "expected_model",
    ]:
        for ticker, residual in zip(["A", "B", "C", "D"], residuals):
            rows.append(
                {
                    "issuer_ticker": ticker,
                    "fiscal_year": 2024,
                    "raw_exchange": "HOSE",
                    "lag_assets": 100.0,
                    "pre_cfo_scaled": residual,
                    "proxy_model": model,
                    "proxy_family": (
                        "expected_cfo_model"
                        if model == "expected_model"
                        else "simple_baseline"
                    ),
                    "expected_cfo_scaled": 0.0,
                    "abnormal_cfo_proxy": residual,
                }
            )
    return pd.DataFrame(rows)


def _observed_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "issuer_ticker": ["A", "B", "C", "D"],
            "fiscal_year": [2024] * 4,
            "cfs_resolution": [
                "identity_consistent_offsetting_reclassification_candidate",
                "identity_consistent_other",
                "identity_consistent_other",
                "identity_consistent_offsetting_reclassification_candidate",
            ],
            "delta_cfo_scaled": [-0.10, 0.0, 0.0, 0.10],
            "offset_channel_pattern": [
                "cff_dominant",
                "mixed",
                "mixed",
                "cfi_dominant",
            ],
        }
    )


def _settings() -> dict:
    return {
        "material_cfo_threshold": 0.05,
        "proxy_models": {"expected_model": []},
        "common_sample_models": [
            "raw_cfo_level",
            "within_year_cfo_percentile",
            "expected_model",
        ],
        "sample_restrictions": {
            "listed_exchanges": ["HOSE"],
            "lag_assets_floor_quantile": 0.0,
        },
    }


def test_outcome_specific_scores_use_both_tails() -> None:
    result = validate_proxy_predictions(
        _prediction_rows(), _observed_rows(), _settings()
    )
    table = result["cfs_shifting_proxy_validation"]
    expected = table[
        table["proxy_model"].eq("expected_model")
        & table["sample_mode"].eq("common_models")
        & table["sample_restriction"].eq("analysis_core")
    ]
    any_row = expected[expected["outcome"].eq("any_candidate")].iloc[0]
    down_row = expected[
        expected["outcome"].eq("cff_down_candidate")
    ].iloc[0]
    up_row = expected[expected["outcome"].eq("cfi_up_candidate")].iloc[0]
    assert any_row["score_rule"] == "absolute"
    assert any_row["auc"] == 1.0
    assert down_row["auc"] == 1.0
    assert up_row["auc"] == 1.0


def test_common_sample_removes_model_specific_extra_rows() -> None:
    predictions = _prediction_rows()
    extra = predictions.iloc[[0]].copy()
    extra["issuer_ticker"] = "E"
    extra["proxy_model"] = "expected_model"
    predictions = pd.concat([predictions, extra], ignore_index=True)
    observed = pd.concat(
        [
            _observed_rows(),
            pd.DataFrame(
                {
                    "issuer_ticker": ["E"],
                    "fiscal_year": [2024],
                    "cfs_resolution": ["identity_consistent_other"],
                    "delta_cfo_scaled": [0.0],
                    "offset_channel_pattern": ["mixed"],
                }
            ),
        ],
        ignore_index=True,
    )
    result = validate_proxy_predictions(predictions, observed, _settings())
    table = result["cfs_shifting_proxy_validation"]
    expected = table[
        table["proxy_model"].eq("expected_model")
        & table["outcome"].eq("any_candidate")
        & table["sample_restriction"].eq("analysis_core")
    ]
    assert expected.loc[
        expected["sample_mode"].eq("model_available"), "rows"
    ].iloc[0] == 5
    assert expected.loc[
        expected["sample_mode"].eq("common_models"), "rows"
    ].iloc[0] == 4


def test_fold_metrics_include_robust_tail_diagnostics() -> None:
    rows = []
    for year in range(2015, 2025):
        for index in range(120):
            assets = 100.0 + index
            revenue = 80.0 + index * 0.2
            cfo = 0.1 * assets + 0.02 * revenue
            if year == 2024 and index == 119:
                cfo = 50_000.0
            rows.append(
                {
                    "issuer_ticker": f"F{index:03d}",
                    "fiscal_year": year,
                    "audit_status": "unaudited",
                    "raw_exchange": "HOSE",
                    "lag_assets": assets,
                    "cfo": cfo,
                    "pat": 5.0,
                    "revenue": revenue,
                    "drev": 1.0,
                    "drec": 0.5,
                    "inv_assets": 1.0 / assets,
                    "loss": 0.0,
                }
            )
    predictions, folds = rolling_expected_cfo_proxies(
        pd.DataFrame(rows),
        {
            "minimum_test_year": 2023,
            "maximum_test_year": 2024,
            "minimum_train_rows": 500,
            "proxy_models": {
                "sales": [
                    "inv_assets",
                    "pre_revenue_scaled",
                    "pre_drev_scaled",
                ]
            },
        },
    )
    assert not predictions.empty
    row = folds.query(
        "fiscal_year == 2024 and proxy_model == 'sales'"
    ).iloc[0]
    for column in [
        "winsorized_rmse",
        "rmse_ex_top_1pct",
        "mae",
        "median_absolute_error",
        "p99_absolute_error",
        "maximum_error_issuer",
    ]:
        assert column in folds.columns
    assert row["winsorized_rmse"] < row["rmse"]
    assert row["maximum_error_issuer"] == "F119"


def test_line_contributors_are_candidate_only() -> None:
    line_items = pd.DataFrame(
        {
            "issuer_ticker": ["A", "A", "B", "B"],
            "fiscal_year": [2024] * 4,
            "audit_status": ["unaudited", "audited"] * 2,
            "selected_statement_family": ["cash_flow_indirect"] * 4,
            "cff_borrowing_proceeds": [0.0, 10.0, 0.0, 100.0],
        }
    )
    observed = pd.DataFrame(
        {
            "issuer_ticker": ["A", "B"],
            "fiscal_year": [2024, 2024],
            "cfs_resolution": [
                "identity_consistent_offsetting_reclassification_candidate",
                "identity_consistent_other",
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
            "issuer_ticker": ["A", "B"],
            "fiscal_year": [2024, 2024],
            "audit_status": ["audited", "audited"],
            "lag_assets": [100.0, 100.0],
        }
    )
    output = line_item_reconciliation(
        line_items,
        observed,
        panel,
        {
            "concept_sections": {
                "cff_borrowing_proceeds": "financing"
            }
        },
    )
    candidate = output["cfs_line_item_top_contributors"]
    all_rows = output["cfs_line_item_top_contributors_all"]
    assert candidate["rows"].sum() == 1
    assert all_rows["rows"].sum() == 2
