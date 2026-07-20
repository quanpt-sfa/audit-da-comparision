from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.auditor_switch_dynamic_did import run_switch_dynamic_did
from audit_da.auditor_switch_event_study import run_switch_event_study


CANDIDATE = "identity_consistent_offsetting_reclassification_candidate"


def _synthetic_switch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    auditor_rows: list[dict] = []
    years = range(2018, 2023)

    specifications: list[tuple[str, str, int]] = []
    specifications += [(f"D{i:03d}", "DOWNGRADE", 1) for i in range(12)]
    specifications += [(f"U{i:03d}", "UPGRADE", 1) for i in range(12)]
    specifications += [(f"B{i:03d}", "STABLE_BIG4", 0) for i in range(30)]
    specifications += [(f"N{i:03d}", "STABLE_NON_BIG4", 0) for i in range(30)]

    for ticker, regime, treated in specifications:
        for year in years:
            if regime == "DOWNGRADE":
                auditor_group = "BIG4" if year < 2020 else "NON_BIG4"
                correction = 0.18 if year >= 2020 else 0.0
            elif regime == "UPGRADE":
                auditor_group = "NON_BIG4" if year < 2020 else "BIG4"
                correction = 0.18 if year < 2020 else 0.0
            elif regime == "STABLE_BIG4":
                auditor_group = "BIG4"
                correction = 0.0
            else:
                auditor_group = "NON_BIG4"
                correction = 0.0

            is_candidate = correction > 0
            rows.append(
                {
                    "issuer_ticker": ticker,
                    "fiscal_year": year,
                    "raw_exchange": "HOSE",
                    "industry_name": "INDUSTRIAL",
                    "lag_assets_common": 100.0,
                    "pre_cfo_scaled": 0.05,
                    "cfs_resolution": CANDIDATE if is_candidate else "other",
                    "delta_cfo_scaled": -correction,
                    "offset_channel_pattern": (
                        "cff_dominant" if is_candidate else "mixed"
                    ),
                    "synthetic_treated": treated,
                }
            )
            auditor_rows.append(
                {
                    "issuer_ticker": ticker,
                    "fiscal_year": year,
                    "auditor_group": auditor_group,
                    "auditor_brand": (
                        "KPMG" if auditor_group == "BIG4" else "LOCAL"
                    ),
                    "auditor_firm_year_status": "EXACT_ONE_NAME",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(auditor_rows)


def _settings() -> dict:
    return {
        "switch_event_study": {
            "pre_periods": 2,
            "post_periods": 2,
            "reference_event_time": -1,
            "one_event_per_direction": True,
            "exact_match_columns": ["raw_exchange", "industry_name"],
            "minimum_controls_per_event": 5,
            "minimum_events": 5,
            "outcomes": ["any_candidate", "cff_down_candidate", "signed_cfo_correction"],
        },
        "dynamic_did": {
            "horizons": [-2, 0, 1, 2],
            "outcomes": ["any_candidate", "cff_down_candidate", "signed_cfo_correction"],
            "minimum_events": 5,
            "minimum_overlap_treated": 999,
            "minimum_overlap_controls": 999,
            "bootstrap_repetitions": 20,
            "bootstrap_seed": 77,
        },
    }


def test_clean_switch_event_study_recovers_directional_cff_pattern() -> None:
    cases, auditors = _synthetic_switch_data()
    tables = run_switch_event_study(cases, auditors, _settings())

    diagnostics = tables["cfs_auditor_switch_event_diagnostics"]
    primary = diagnostics[diagnostics["primary_event"]]
    assert len(primary) == 24
    assert set(primary["switch_direction"]) == {"UPGRADE", "DOWNGRADE"}

    stacked = tables["cfs_auditor_switch_stacked_sample"]
    assert stacked["event_id"].nunique() == 24
    assert stacked["event_time"].min() == -2
    assert stacked["event_time"].max() == 2

    estimates = tables["cfs_auditor_switch_event_study"]
    downgrade = estimates[
        estimates["switch_direction"].eq("DOWNGRADE")
        & estimates["outcome"].eq("cff_down_candidate")
        & estimates["event_time"].eq(0)
    ]
    upgrade = estimates[
        estimates["switch_direction"].eq("UPGRADE")
        & estimates["outcome"].eq("cff_down_candidate")
        & estimates["event_time"].eq(0)
    ]
    assert len(downgrade) == 1
    assert len(upgrade) == 1
    assert downgrade.iloc[0]["estimate"] > 0.50
    assert upgrade.iloc[0]["estimate"] < -0.50


def test_switch_dynamic_did_recovers_switcher_stayer_contrasts() -> None:
    cases, auditors = _synthetic_switch_data()
    event_tables = run_switch_event_study(cases, auditors, _settings())
    did_tables = run_switch_dynamic_did(
        event_tables["cfs_auditor_switch_stacked_sample"], _settings()
    )

    estimates = did_tables["cfs_auditor_switch_dynamic_did"]
    downgrade = estimates[
        estimates["switch_direction"].eq("DOWNGRADE")
        & estimates["outcome"].eq("cff_down_candidate")
        & estimates["horizon"].eq(0)
    ]
    upgrade = estimates[
        estimates["switch_direction"].eq("UPGRADE")
        & estimates["outcome"].eq("cff_down_candidate")
        & estimates["horizon"].eq(0)
    ]
    assert len(downgrade) == 1
    assert len(upgrade) == 1
    assert downgrade.iloc[0]["estimate"] > 0.50
    assert upgrade.iloc[0]["estimate"] < -0.50
    assert did_tables["cfs_auditor_switch_dynamic_did_status"].loc[0, "status"] == "PASS"
