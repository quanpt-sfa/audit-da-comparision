from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit_da.diag_cfs_deep_dive import (
    audit_quality_tables,
    build_offset_channel_panel,
    chronic_reclassifier_tables,
    common_sample_anchor_table,
)


def identity_cases() -> pd.DataFrame:
    rows = []
    for year, delta_cfo, delta_cfi, delta_cff in [
        (2018, 0.10, -0.09, -0.01),
        (2019, -0.12, 0.02, 0.10),
        (2020, 0.08, -0.07, -0.01),
        (2021, -0.09, 0.01, 0.08),
    ]:
        rows.append({
            "issuer_ticker": "AAA",
            "fiscal_year": year,
            "cfs_resolution": "identity_consistent_offsetting_reclassification_candidate",
            "delta_cfo_scaled": delta_cfo,
            "delta_cfi_scaled": delta_cfi,
            "delta_cff_scaled": delta_cff,
            "delta_fx_effect_scaled": 0.0,
            "delta_cfs_cash_change_scaled": 0.0,
            "reduction": 0.02,
            "abs_delta_cfo": abs(delta_cfo),
            "abs_delta_pat": 0.001,
            "lag_assets_common": 100.0,
            "cfo_pre": 10.0,
            "cfo_post": 10.0 + 100.0 * delta_cfo,
        })
    return pd.DataFrame(rows)


def test_offset_channel_and_bidirectional_chronicity() -> None:
    settings = {
        "minimum_year": 2018,
        "dominant_offset_share_threshold": 0.60,
        "chronic_min_candidate_years": 4,
        "chronic_candidate_share": 0.75,
        "direction_consistency_threshold": 0.80,
    }
    panel = build_offset_channel_panel(identity_cases(), settings)
    assert panel.loc[panel["fiscal_year"].eq(2018), "offset_channel_pattern"].iloc[0] == "cfi_dominant"
    assert np.allclose(panel["offset_reconstruction_error"], 0.0)
    profiles = chronic_reclassifier_tables(panel, settings)["chronic_reclassifier_profiles"]
    row = profiles.loc[profiles["issuer_ticker"].eq("AAA")].iloc[0]
    assert bool(row["chronic_reclassifier"])
    assert row["direction_type"] == "bidirectional"


def test_common_sample_anchor_preserves_sign_reversal() -> None:
    alignment = pd.DataFrame({
        "issuer_ticker": ["A", "B"],
        "fiscal_year": [2020, 2020],
        "model": ["modified_jones", "modified_jones"],
        "benchmark": ["audited_reference", "audited_reference"],
        "component": ["cfo", "cfo"],
        "cfo_component_movement": [-0.10, 0.10],
        "da_pre": [0.15, -0.15],
        "da_pre_balance_sheet_anchor": [-0.15, 0.15],
        "cfo_reduction_cashflow_anchor": [0.10, 0.10],
        "cfo_reduction_balance_sheet_anchor": [-0.10, -0.10],
    })
    result = common_sample_anchor_table(alignment, {"trim_fraction": 0.0})
    assert result["mean_reduction_cashflow_anchor"].iloc[0] > 0
    assert result["mean_reduction_balance_sheet_anchor"].iloc[0] < 0
    assert result["common_rows"].iloc[0] == 2


def test_missing_audit_metadata_is_not_evaluated(tmp_path: Path) -> None:
    data = pd.DataFrame({
        "issuer_ticker": ["A"],
        "fiscal_year": [2020],
        "candidate": [True],
        "audited_cfo_decrease": [1.0],
        "abs_delta_cfo": [0.1],
        "reduction": [0.02],
    })
    tables = audit_quality_tables(data, tmp_path / "missing.csv", {})
    assert tables["audit_quality_status"]["status"].iloc[0] == "NOT_EVALUATED"
