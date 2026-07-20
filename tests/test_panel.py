import csv

import pandas as pd

from audit_da.panel import build_accrual_features, profile_input


def _config():
    return {
        "analysis_window": {
            "source_start_year": 2015,
            "source_end_year": 2025,
            "training_start_year": 2015,
            "test_start_year": 2016,
            "test_end_year": 2025,
        },
        "input": {
            "audited_label": "audited",
            "minimum_year": 2015,
            "maximum_year": 2025,
        },
        "panel": {
            "minimum_lag_assets": 1,
            "primary_total_accruals": "cash_flow",
        },
    }


def _paired_rows(years):
    rows = []
    for year in years:
        for status in ["audited", "unaudited"]:
            rows.append(
                {
                    "issuer_ticker": "AAA",
                    "raw_exchange": "HOSE",
                    "fiscal_year": year,
                    "audit_status": status,
                    "scope": "consolidated",
                    "assets": 90 + 10 * (year - 2014),
                    "revenue": 75 + 5 * (year - 2014),
                    "receivables": 10,
                    "ppe_gross": 30,
                    "pat": 8,
                    "cfo_indirect": 6,
                    "current_assets": 60,
                    "cash": 5,
                    "current_liabilities": 30,
                    "short_term_debt": 4,
                    "tax_payable": 2,
                    "depreciation": 3,
                }
            )
    return rows


def test_build_accrual_features_uses_audited_lag_for_both_versions():
    panel = build_accrual_features(
        pd.DataFrame(_paired_rows([2020, 2021])), _config()
    )
    current = panel[panel.fiscal_year == 2021]
    assert len(current) == 2
    assert (current["lag_assets"] == 150).all()
    assert (current["ta_scaled"] == (2.0 / 150.0)).all()


def test_pre_2015_rows_cannot_supply_any_lookback_to_2015():
    panel = build_accrual_features(
        pd.DataFrame(_paired_rows([2014, 2015, 2016])), _config()
    )

    assert panel["fiscal_year"].min() == 2015
    first_year = panel[panel["fiscal_year"].eq(2015)]
    second_year = panel[panel["fiscal_year"].eq(2016)]

    assert first_year["lag_assets"].isna().all()
    assert first_year["drev"].isna().all()
    assert first_year["drec"].isna().all()
    assert first_year["ta_scaled"].isna().all()
    assert (second_year["lag_assets"] == 100).all()
    assert (second_year["drev"] == 5).all()


def test_input_profile_excludes_every_year_before_2015(tmp_path):
    path = tmp_path / "input.csv"
    fieldnames = [
        "issuer_ticker",
        "fiscal_year",
        "audit_status",
        "statement_family",
        "scope",
        "unit",
        "identity_match_status",
        "retrospective_eligible",
        "prospective_flag",
        "source_item_id",
        "item_name_raw",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for year in [2014, 2015, 2016, 2026]:
            writer.writerow(
                {
                    "issuer_ticker": "AAA",
                    "fiscal_year": year,
                    "audit_status": "audited",
                    "statement_family": "balance_sheet",
                    "scope": "consolidated",
                    "unit": "VND",
                    "identity_match_status": "exact",
                    "retrospective_eligible": "1",
                    "prospective_flag": "0",
                    "source_item_id": "assets",
                    "item_name_raw": "Assets",
                }
            )

    profile = profile_input(path, minimum_year=2015, maximum_year=2025)
    assert profile["rows"] == 2
    assert profile["year_min"] == 2015
    assert profile["year_max"] == 2016
    assert profile["rows_excluded_outside_window"] == 2
