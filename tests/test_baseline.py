import pandas as pd

from audit_da.baseline import run_ols_baselines


def test_ols_baseline_runs_on_paired_panel():
    rows = []
    for year in range(2015, 2021):
        for firm_index in range(20):
            for status, shift in [("audited", 0.0), ("unaudited", 0.01)]:
                rows.append({
                    "issuer_ticker": f"F{firm_index}", "firm_id": f"F{firm_index}",
                    "fiscal_year": year, "audit_status": status,
                    "ta_scaled": 0.1 + 0.001 * firm_index + shift,
                    "inv_assets": 0.01, "drev_scaled": 0.1, "drev_drec_scaled": 0.08,
                    "ppe_scaled": 0.3, "roa": 0.05, "loss": 0.0, "drev_drec_sq": 0.0064,
                })
    config = {
        "input": {"audited_label": "audited", "unaudited_label": "unaudited"},
        "signal": {"minimum_test_year": 2019, "maximum_test_year": 2020, "benchmarks": ["version_specific"]},
        "models": {"minimum_train_rows": 50, "candidate_models": {"mj": ["inv_assets", "drev_drec_scaled", "ppe_scaled"]}},
    }
    result = run_ols_baselines(pd.DataFrame(rows), config)
    assert not result.empty
    assert set(result["fiscal_year"]) == {2019, 2020}
