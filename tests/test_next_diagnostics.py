from __future__ import annotations

import pandas as pd

from audit_da.next_diagnostics import directional_placebo, family_discordance, sign_transition_tables, ta_source_audit


def make_panel() -> pd.DataFrame:
    rows = []
    for ticker, year, pre_ta, post_ta in [("A", 2024, 0.10, 0.04), ("B", 2024, -0.08, -0.02), ("C", 2024, 0.05, -0.05), ("D", 2024, -0.03, 0.06)]:
        for status, ta in [("unaudited", pre_ta), ("audited", post_ta)]:
            rows.append({
                "issuer_ticker": ticker, "fiscal_year": year, "audit_status": status, "raw_exchange": "HOSE",
                "ta_source": "cash_flow", "total_accruals": ta * 100, "ta_cashflow": ta * 100,
                "ta_balance_sheet": ta * 100, "pat": ta * 100 + 10, "cfo": 10, "lag_assets": 100,
                "ta_scaled": ta, "roa": 0.1, "revenue": 100, "receivables": 20, "ppe": 30,
                "firm_id": ticker, "inv_assets": 0.01, "drev_scaled": 0.02,
                "drev_drec_scaled": 0.01, "ppe_scaled": 0.3, "loss": 0.0, "drev_drec_sq": 0.0001,
            })
    return pd.DataFrame(rows)


def make_baseline() -> pd.DataFrame:
    records = []
    values = {"A": (0.10, 0.04), "B": (-0.08, -0.02), "C": (0.05, -0.05), "D": (-0.03, 0.06)}
    shifts = {"A": -0.06, "B": 0.06, "C": -0.10, "D": 0.09}
    models = ["jones", "modified_jones", "kothari", "nonlinear_modified_jones"]
    for ticker, (pre, post) in values.items():
        for model in models:
            adjustment = 0.0
            if model in ["kothari", "nonlinear_modified_jones"] and ticker in ["C", "D"]:
                adjustment = 0.03 if ticker == "C" else -0.03
            p = post + adjustment
            records.append({
                "issuer_ticker": ticker, "fiscal_year": 2024, "model": model,
                "benchmark": "audited_reference", "da_pre": pre, "da_post": p,
                "signed_shift": shifts[ticker] + adjustment, "reduction": abs(pre) - abs(p),
                "raw_ta_shift": shifts[ticker],
            })
    return pd.DataFrame(records)


def test_ta_source_formula_identity() -> None:
    summary, pair = ta_source_audit(make_panel())
    assert summary["max_abs_identity_error"].max() == 0
    assert not pair["ta_source_mismatch"].any()


def test_sign_flip_is_exposed_when_reduction_zero() -> None:
    flips = sign_transition_tables(make_baseline(), [0.0], [0.0, 0.005])["sign_flip_summary"]
    hidden = flips[(flips["flip_category"] == "strict_flips_hidden_inside_R_near_zero") & (flips["reduction_delta"] == 0.0)]
    assert hidden["count"].max() >= 1


def test_centered_placebo_is_directional_under_symmetric_noise() -> None:
    expanded, expanded_panel = [], []
    for k in range(30):
        b = make_baseline(); b["issuer_ticker"] = b["issuer_ticker"] + str(k); expanded.append(b)
        p = make_panel(); p["issuer_ticker"] = p["issuer_ticker"] + str(k); expanded_panel.append(p)
    summary, _ = directional_placebo(
        pd.concat(expanded, ignore_index=True), pd.concat(expanded_panel, ignore_index=True),
        ["jones"], ["audited_reference"], ["fiscal_year", "raw_exchange"], 10, 100, 0.01, 42,
    )
    centered = summary[summary["placebo_type"].isin(["centered_permutation", "symmetric_sign"])]
    assert (centered["placebo_mean"] <= 0).all()


def test_family_discordance_identifies_opposite_families() -> None:
    tables = family_discordance(
        make_baseline(), make_panel(),
        {"classic": ["jones", "modified_jones"], "performance": ["kothari", "nonlinear_modified_jones"]},
        [0.0],
    )
    assert not tables["family_discordance_summary"].empty
    assert tables["family_discordance_cases"]["any_family_discordance"].any()
