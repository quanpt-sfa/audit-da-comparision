from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from audit_da.cfs_item_map import classify_cfs_item, compile_item_rules
from audit_da.cfs_proxy_validate_samples import validate_proxy_predictions_dual_common
from audit_da.icb_industry import attach_icb_industry, load_icb_industry


def test_icb_loader_detects_vietnamese_schema_and_financials(tmp_path: Path) -> None:
    path = tmp_path / "bctc_industry_icb.csv"
    pd.DataFrame(
        {
            "Mã CK": ["AAA", "BBB"],
            "Ngành cấp 1": ["Công nghiệp", "Tài chính"],
            "Mã ngành ICB": [2000, 8000],
        }
    ).to_csv(path, index=False, encoding="utf-8-sig")

    mapping, status = load_icb_industry(path, {"financial_icb_prefixes": ["8"]})
    assert status.loc[0, "ticker_column"] == "Mã CK"
    assert mapping.set_index("issuer_ticker").loc["AAA", "financial_flag"] == 0
    assert mapping.set_index("issuer_ticker").loc["BBB", "financial_flag"] == 1

    panel = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB", "CCC"],
            "fiscal_year": [2024, 2024, 2024],
            "audit_status": ["unaudited"] * 3,
        }
    )
    merged, join_status, unmatched = attach_icb_industry(panel, mapping)
    assert join_status.loc[0, "matched_tickers"] == 2
    assert unmatched["issuer_ticker"].tolist() == ["CCC"]
    assert merged.loc[merged["issuer_ticker"].eq("CCC"), "financial_flag"].isna().all()


def test_extended_cff_and_interest_rules_match_realistic_labels() -> None:
    config = yaml.safe_load(Path("config/cfs_shifting_validation.yaml").read_text(encoding="utf-8"))
    rules = compile_item_rules(config["cfs_shifting_validation"])
    examples = {
        "Tiền trả nợ gốc vay": "cff_debt_repayments",
        "Tiền thanh toán vốn gốc đi thuê tài chính": "cff_lease_principal_payments",
        "Tiền trả lại vốn góp cho các chủ sở hữu, mua lại cổ phiếu": "cff_share_repurchases",
        "Tiền lãi đã nhận": "cfi_interest_received",
        "Thu lãi và cổ tức": "cfi_interest_dividends_received",
    }
    for label, expected in examples.items():
        concept, section, count = classify_cfs_item(
            f"cash_flow_indirect__{label}", label, "cash_flow_indirect", rules
        )
        assert count == 1, (label, concept, count)
        assert concept == expected
        assert section in {"investing", "financing"}


def _prediction_rows() -> pd.DataFrame:
    primary = [
        "sales_level_only",
        "roychowdhury_sales",
        "earnings_conditioned",
        "earnings_working_capital",
        "raw_cfo_level",
        "within_year_cfo_percentile",
    ]
    rows = []
    for ticker, residual in [("AAA", 0.2), ("BBB", -0.2)]:
        for model in primary:
            rows.append(
                {
                    "issuer_ticker": ticker,
                    "fiscal_year": 2024,
                    "raw_exchange": "HOSE",
                    "lag_assets": 100.0,
                    "pre_cfo_scaled": residual,
                    "proxy_model": model,
                    "proxy_family": "test",
                    "expected_cfo_scaled": 0.0,
                    "abnormal_cfo_proxy": residual,
                    "financial_flag": False,
                }
            )
    rows.append(
        {
            "issuer_ticker": "AAA",
            "fiscal_year": 2024,
            "raw_exchange": "HOSE",
            "lag_assets": 100.0,
            "pre_cfo_scaled": 0.2,
            "proxy_model": "firm_history_deviation",
            "proxy_family": "test",
            "expected_cfo_scaled": 0.0,
            "abnormal_cfo_proxy": 0.2,
            "financial_flag": False,
        }
    )
    return pd.DataFrame(rows)


def test_primary_and_all_model_common_samples_are_both_reported() -> None:
    predictions = _prediction_rows()
    observed = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB"],
            "fiscal_year": [2024, 2024],
            "cfs_resolution": [
                "identity_consistent_offsetting_reclassification_candidate",
                "identity_consistent_offsetting_reclassification_candidate",
            ],
            "delta_cfo_scaled": [-0.10, 0.10],
            "offset_channel_pattern": ["cff_dominant", "cfi_dominant"],
        }
    )
    primary_models = [
        "sales_level_only", "roychowdhury_sales", "earnings_conditioned",
        "earnings_working_capital", "raw_cfo_level", "within_year_cfo_percentile",
    ]
    all_model_names = primary_models + ["firm_history_deviation"]
    settings = {
        "material_cfo_threshold": 0.05,
        "common_primary_models": primary_models,
        "common_all_models": all_model_names,
        "sample_restrictions": {
            "listed_exchanges": ["HOSE"],
            "lag_assets_floor_quantile": 0.0,
        },
    }
    tables = validate_proxy_predictions_dual_common(predictions, observed, settings)
    status = tables["cfs_common_sample_status"].set_index("sample_mode")
    assert status.loc["common_primary_models", "common_firm_years"] == 2
    assert status.loc["common_all_models", "common_firm_years"] == 1

    primary = tables["cfs_shifting_proxy_common_primary_core_cases"]
    all_models = tables["cfs_shifting_proxy_common_all_core_cases"]
    assert primary[["issuer_ticker", "fiscal_year"]].drop_duplicates().shape[0] == 2
    assert all_models[["issuer_ticker", "fiscal_year"]].drop_duplicates().shape[0] == 1
    assert set(primary["proxy_model"].unique()) == set(primary_models)
    assert "firm_history_deviation" not in set(primary["proxy_model"])
    assert set(all_models["proxy_model"].unique()) == set(all_model_names)

    comparison = tables["cfs_common_sample_metric_comparison"]
    assert not comparison.empty
    assert (comparison["firm_year_rows_lost"] >= 0).all()
