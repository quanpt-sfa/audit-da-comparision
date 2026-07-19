from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.diag_cfs_identity import cfs_identity_tables
from audit_da.diag_cfo_tilt import cfo_tilt_tables
from audit_da.diag_component_placebo import component_placebo_tables


def _panel_and_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for index in range(30):
        ticker = "2300323118" if index == 0 else f"A{index:02d}"
        year = 2024
        lag_assets = 100.0
        pat_pre = 10.0
        pat_post = 10.0 if index < 20 else 12.0
        cfo_pre = 0.0
        cfo_post = 10.0 if index < 20 else 0.0
        cfi_pre = 0.0
        cfi_post = -10.0 if index < 20 else 0.0
        cff_pre = cff_post = 0.0
        begin_cash = 10.0
        fx = 0.0
        net_pre = cfo_pre + cfi_pre + cff_pre
        net_post = cfo_post + cfi_post + cff_post
        end_pre = begin_cash + net_pre + fx
        end_post = begin_cash + net_post + fx

        for status, pat, cfo, cfi, cff, net, end in [
            ("unaudited", pat_pre, cfo_pre, cfi_pre, cff_pre, net_pre, end_pre),
            ("audited", pat_post, cfo_post, cfi_post, cff_post, net_post, end_post),
        ]:
            ta = pat - cfo
            panel_rows.append({
                "issuer_ticker": ticker,
                "fiscal_year": year,
                "audit_status": status,
                "raw_exchange": "HOSE",
                "lag_assets": lag_assets,
                "pat": pat,
                "cfo": cfo,
                "cfi": cfi,
                "cff": cff,
                "net_cash_change": net,
                "cash_begin_cfs": begin_cash,
                "fx_effect": fx,
                "cash_end_cfs": end,
                "cash": end,
                "ta_scaled": ta / lag_assets,
                # The independent balance-sheet anchor points in the opposite
                # direction for the CFO-dominant cases.
                "ta_balance_sheet": -10.0 if index < 20 else ta,
                "assets": 110.0,
                "ta_source": "cash_flow",
            })

        nda = 0.0
        da_pre = (pat_pre - cfo_pre) / lag_assets - nda
        da_post = (pat_post - cfo_post) / lag_assets - nda
        for model in ["modified_jones"]:
            for benchmark in ["audited_reference"]:
                baseline_rows.append({
                    "issuer_ticker": ticker,
                    "fiscal_year": year,
                    "model": model,
                    "benchmark": benchmark,
                    "da_pre": da_pre,
                    "da_post": da_post,
                    "signed_shift": da_post - da_pre,
                    "reduction": abs(da_pre) - abs(da_post),
                    "raw_ta_shift": da_post - da_pre,
                })

    return pd.DataFrame(panel_rows), pd.DataFrame(baseline_rows)


def test_identity_consistent_offsetting_cfo_change_is_candidate() -> None:
    panel, baseline = _panel_and_baseline()
    tables = cfs_identity_tables(panel, baseline, {
        "absolute_tolerance_vnd": 1e-8,
        "scaled_tolerance": 1e-10,
        "offset_closure_scaled_tolerance": 1e-10,
        "primary_model": "modified_jones",
        "primary_benchmark": "audited_reference",
        "material_cfo_threshold": .05,
        "cfo_to_pat_ratio_threshold": 5.0,
        "trim_fraction": .01,
    })
    cases = tables["cfs_identity_cases"]
    assert cases["cfs_full_pass_pre"].all()
    assert cases["cfs_full_pass_post"].all()
    assert (
        cases.loc[cases["abs_delta_cfo"].ge(.05), "cfs_resolution"]
        == "identity_consistent_offsetting_reclassification_candidate"
    ).all()
    assert tables["invalid_ticker_cases"]["ticker_numeric_only"].any()


def test_cfo_alignment_is_positive_only_on_cashflow_anchor() -> None:
    panel, baseline = _panel_and_baseline()
    tables = component_placebo_tables(baseline, panel, {
        "models": ["modified_jones"],
        "benchmarks": ["audited_reference"],
        "components": ["pat", "cfo"],
        "conditioning_bins": {
            "within": ["fiscal_year"],
            "size_bins": 2,
            "abs_da_pre_bins": 2,
            "abs_component_bins": 2,
        },
        "minimum_stratum_size": 5,
        "permutations": 20,
        "trim_fraction": .01,
        "random_seed": 42,
    })
    anchor = tables["component_anchor_diagnostics"]
    cfo = anchor[anchor["component"].eq("cfo")].iloc[0]
    assert cfo["mean_reduction_cashflow_anchor"] > 0
    assert cfo["mean_reduction_balance_sheet_anchor"] < 0
    assert tables["component_placebo_summary"]["all_outputs_finite"].all()


def test_cfo_tilt_reports_negative_non_cfo_subgroup() -> None:
    panel, baseline = _panel_and_baseline()
    tables = cfo_tilt_tables(baseline, panel, {
        "asset_pre_post_gap_threshold": .05,
        "asset_growth_multiple_threshold": 5.0,
        "small_lag_assets_quantile": .01,
        "cfo_to_pat_ratio_grid": [5.0],
        "reduction_tolerance_grid": [0.0],
        "trim_fraction": .01,
        "primary_model": "modified_jones",
        "primary_benchmark": "audited_reference",
        "candidate_abs_cfo_threshold": .05,
        "persistence_cfo_to_pat_threshold": 5.0,
    })
    summary = tables["cfo_tilt_summary"]
    assert set(summary["dominance_group"]) == {
        "all", "cfo_dominant", "not_cfo_dominant"
    }
    persistence = tables["cfo_candidate_persistence"]
    assert (persistence["candidate_years"] >= 1).all()
