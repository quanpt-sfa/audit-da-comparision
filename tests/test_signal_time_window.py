import pandas as pd

from audit_da.signal import run_signal_gate


def test_bayesian_signal_never_uses_2014_and_handles_first_tt200_fold():
    rows = []
    for year in [2014, 2015, 2016]:
        for firm_index in range(40):
            statuses = ["audited"] if year < 2016 else ["audited", "unaudited"]
            for status in statuses:
                rows.append(
                    {
                        "issuer_ticker": f"F{firm_index:03d}",
                        "firm_id": f"F{firm_index:03d}",
                        "raw_exchange": "HOSE",
                        "fiscal_year": year,
                        "audit_status": status,
                        "ta_scaled": (
                            100.0
                            if year == 2014
                            else 0.10
                            + 0.001 * firm_index
                            + (0.01 if status == "unaudited" else 0.0)
                        ),
                        "inv_assets": 0.01 + 0.00001 * firm_index,
                    }
                )

    config = {
        "analysis_window": {
            "source_start_year": 2015,
            "source_end_year": 2016,
            "training_start_year": 2015,
            "test_start_year": 2016,
            "test_end_year": 2016,
        },
        "input": {
            "audited_label": "audited",
            "unaudited_label": "unaudited",
            "minimum_year": 2015,
            "maximum_year": 2016,
        },
        "models": {
            "training_start_year": 2015,
            "minimum_train_rows": 30,
            "minimum_validation_rows": 10,
            "posterior_draws": 20,
            "random_seed": 17,
            "candidate_models": {"simple": ["inv_assets"]},
        },
        "signal": {
            "minimum_test_year": 2016,
            "maximum_test_year": 2016,
            "benchmarks": ["version_specific"],
            "rho_grid": [1.0],
            "error_sd_ratio_grid": [1.0],
            "delta_grid": [0.0],
        },
        "panel": {"winsor_lower": 0.01, "winsor_upper": 0.99},
    }

    posterior, folds = run_signal_gate(pd.DataFrame(rows), config)

    assert set(posterior["fiscal_year"]) == {2016}
    assert posterior["training_min_year"].eq(2015).all()
    assert posterior["training_start_year_contract"].eq(2015).all()
    assert folds["training_min_year"].eq(2015).all()
    assert folds["stacking_weight_mode"].eq(
        "equal_weight_no_prevalidation_history"
    ).all()


def test_bayesian_signal_skips_2016_when_2015_lag_features_are_unavailable():
    rows = []
    for year in [2014, 2015, 2016, 2017]:
        for firm_index in range(40):
            statuses = ["audited"] if year < 2016 else ["audited", "unaudited"]
            for status in statuses:
                ta_scaled = (
                    100.0
                    if year == 2014
                    else float("nan")
                    if year == 2015
                    else 0.10
                    + 0.001 * firm_index
                    + (0.01 if status == "unaudited" else 0.0)
                )
                inv_assets = (
                    0.001
                    if year == 2014
                    else float("nan")
                    if year == 2015
                    else 0.01 + 0.00001 * firm_index
                )
                rows.append(
                    {
                        "issuer_ticker": f"F{firm_index:03d}",
                        "firm_id": f"F{firm_index:03d}",
                        "raw_exchange": "HOSE",
                        "fiscal_year": year,
                        "audit_status": status,
                        "ta_scaled": ta_scaled,
                        "inv_assets": inv_assets,
                    }
                )

    config = {
        "analysis_window": {
            "source_start_year": 2015,
            "source_end_year": 2017,
            "training_start_year": 2015,
            "test_start_year": 2016,
            "test_end_year": 2017,
        },
        "input": {
            "audited_label": "audited",
            "unaudited_label": "unaudited",
            "minimum_year": 2015,
            "maximum_year": 2017,
        },
        "models": {
            "training_start_year": 2015,
            "minimum_train_rows": 30,
            "minimum_validation_rows": 10,
            "posterior_draws": 20,
            "random_seed": 23,
            "candidate_models": {"simple": ["inv_assets"]},
        },
        "signal": {
            "minimum_test_year": 2016,
            "maximum_test_year": 2017,
            "benchmarks": ["version_specific"],
            "rho_grid": [1.0],
            "error_sd_ratio_grid": [1.0],
            "delta_grid": [0.0],
        },
        "panel": {"winsor_lower": 0.01, "winsor_upper": 0.99},
    }

    posterior, folds = run_signal_gate(pd.DataFrame(rows), config)

    assert set(posterior["fiscal_year"]) == {2017}
    skipped = folds[folds["fiscal_year"].eq(2016)].iloc[0]
    assert skipped["status"] == "INSUFFICIENT_FINITE_TRAINING"
    assert skipped["finite_train_rows_simple"] == 0
    assert skipped["training_min_year"] == 2015
    effective = folds[folds["fiscal_year"].eq(2017)].iloc[0]
    assert effective["status"] == "OK"
    assert effective["finite_train_rows_simple"] == 40
