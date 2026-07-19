import pandas as pd

from audit_da.panel import build_accrual_features


def test_build_accrual_features_uses_audited_lag_for_both_versions():
    rows = []
    for year in [2020, 2021]:
        for status in ["audited", "unaudited"]:
            rows.append({
                "issuer_ticker": "AAA", "raw_exchange": "HOSE", "fiscal_year": year,
                "audit_status": status, "scope": "consolidated", "assets": 100 + 10 * (year - 2020),
                "revenue": 80 + 5 * (year - 2020), "receivables": 10, "ppe_gross": 30,
                "pat": 8, "cfo_indirect": 6, "current_assets": 60, "cash": 5,
                "current_liabilities": 30, "short_term_debt": 4, "tax_payable": 2,
                "depreciation": 3,
            })
    config = {
        "input": {"audited_label": "audited"},
        "panel": {"minimum_lag_assets": 1, "primary_total_accruals": "cash_flow"},
    }
    panel = build_accrual_features(pd.DataFrame(rows), config)
    current = panel[panel.fiscal_year == 2021]
    assert len(current) == 2
    assert (current["lag_assets"] == 100).all()
    assert (current["ta_scaled"] == 0.02).all()
