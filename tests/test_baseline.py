import pandas as pd

from audit_da.baseline import run_ols_baselines


def _panel_rows():
    rows = []
    for year in range(2014, 2021):
        for firm_index in range(20):
            for status, shift in [("audited", 0.0), ("unaudited", 0.01)]:
                rows.append(
                    {
                        "issuer_ticker": f"F{firm_index}",
                        "firm_id": f"F{firm_index}",
                        "fiscal_year": year,
                        "audit_status": status,
                        "ta_scaled": 0.1 + 0.001 * firm_index + shift,
                        "inv_assets": 0.01,
                        "drev_scaled": 0.1,
                        "drev_drec_scaled": 0.08,
                        "ppe_scaled": 0.3,
                        "roa": 0.05,
                        "loss": 0.0,
                        "drev_drec_sq": 0.0064,
                    }
                )
    return pd.DataFrame(rows)


def _config():
    return {
        "analysis_window": {
            "source_start_year": 2015,
            "source_end_year": 2025,
            "training_start_year": 2015,
            "test_start_year": 2016,
            "test_end_year": 2020,
        },
        "input": {
            "audited_label": "audited",
            "unaudited_label": "unaudited",
        },
        "signal": {
            "minimum_test_year": 2016,
            "maximum_test_year": 2020,
            "benchmarks": ["version_specific"],
        },
        "models": {
            "training_start_year": 2015,
            "minimum_train_rows": 20,
            "candidate_models": {
                "mj": ["inv_assets", "drev_drec_scaled", "ppe_scaled"]
            },
        },
    }


def test_ols_baseline_runs_on_paired_panel():
    result = run_ols_baselines(_panel_rows(), _config())
    assert not result.empty
    assert set(result["fiscal_year"]) == {2016, 2017, 2018, 2019, 2020}


def test_ols_training_never_uses_pre_tt200_rows():
    result = run_ols_baselines(_panel_rows(), _config())
    assert result["training_min_year"].min() == 2015
    assert result["training_start_year_contract"].eq(2015).all()
    assert result["test_start_year_contract"].eq(2016).all()
    assert not result["fiscal_year"].eq(2015).any()
