from __future__ import annotations

import pandas as pd

from audit_da.cfs_target_finalize import (
    CANDIDATE_LABEL,
    finalize_cfs_target_tables,
)
from audit_da.diag_cfs_identity import cfs_identity_tables


def _panel_2017_candidate() -> pd.DataFrame:
    rows = []
    for status, pat, cfo, cfi in [
        ("unaudited", 10.0, 0.0, 0.0),
        ("audited", 10.0, 10.0, -10.0),
    ]:
        net = cfo + cfi
        rows.append(
            {
                "issuer_ticker": "AAA",
                "fiscal_year": 2017,
                "audit_status": status,
                "raw_exchange": "HOSE",
                "lag_assets": 100.0,
                "pat": pat,
                "cfo": cfo,
                "cfi": cfi,
                "cff": 0.0,
                "net_cash_change": net,
                "cash_begin_cfs": 20.0,
                "fx_effect": 0.0,
                "cash_end_cfs": 20.0 + net,
                "cash": 20.0 + net,
                "ta_scaled": (pat - cfo) / 100.0,
                "assets": 110.0,
                "ta_source": "cash_flow",
            }
        )
    return pd.DataFrame(rows)


def _baseline_starting_2018() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "issuer_ticker": "ZZZ",
                "fiscal_year": 2018,
                "model": "modified_jones",
                "benchmark": "audited_reference",
                "da_pre": 0.0,
                "da_post": 0.0,
                "signed_shift": 0.0,
                "reduction": 0.0,
                "raw_ta_shift": 0.0,
            }
        ]
    )


def _settings() -> dict:
    return {
        "absolute_tolerance_vnd": 1e-8,
        "scaled_tolerance": 1e-10,
        "offset_closure_scaled_tolerance": 1e-10,
        "primary_model": "modified_jones",
        "primary_benchmark": "audited_reference",
        "material_cfo_threshold": 0.05,
        "cfo_to_pat_ratio_threshold": 5.0,
        "trim_fraction": 0.01,
    }


def test_2017_target_does_not_depend_on_baseline_year_coverage() -> None:
    tables = cfs_identity_tables(
        _panel_2017_candidate(), _baseline_starting_2018(), _settings()
    )
    before = tables["cfs_identity_cases"].iloc[0]

    # This reproduces the original defect: decomposition fields are absent and
    # the candidate condition silently evaluates False.
    assert pd.isna(before["abs_delta_cfo"])
    assert before["cfs_resolution"] == "identity_consistent_other"

    finalized = finalize_cfs_target_tables(tables, _settings())
    after = finalized["cfs_identity_cases"].iloc[0]

    assert after["cfs_resolution"] == CANDIDATE_LABEL
    assert after["target_primitives_source"] == "PAIRED_STATEMENTS_DIRECT"
    assert bool(after["target_primitives_complete"])
    assert not bool(after["decomposition_available"])
    assert after["abs_delta_cfo"] == 0.10
    assert after["cfo_to_pat_abs_ratio"] > 1_000_000

    coverage = finalized["cfs_target_input_coverage"].iloc[0]
    assert coverage["fiscal_year"] == 2017
    assert coverage["candidate_rows"] == 1
    assert coverage["target_complete_rows"] == 1
    assert coverage["decomposition_available_rows"] == 0
