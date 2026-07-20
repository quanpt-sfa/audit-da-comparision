from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.auditor_regime import (
    auditor_interaction_models,
    auditor_switch_diagnostics,
    classify_auditor_name,
    cluster_bootstrap_differences,
    load_auditor_firm_year,
    prepare_auditor_analysis_sample,
    stratified_auditor_metrics,
)


def test_big4_name_normalization() -> None:
    cases = {
        "Công ty TNHH Deloitte Việt Nam": ("BIG4", "DELOITTE"),
        "PricewaterhouseCoopers (Vietnam) Ltd.": ("BIG4", "PWC"),
        "Công ty TNHH Ernst & Young Việt Nam": ("BIG4", "EY"),
        "KPMG Limited": ("BIG4", "KPMG"),
        "Công ty TNHH Kiểm toán A&C": ("NON_BIG4", "cong ty tnhh kiem toan a c"),
    }
    for raw, expected in cases.items():
        mapped = classify_auditor_name(raw)
        assert (mapped["auditor_group"], mapped["auditor_brand"]) == expected


def test_loader_detects_column_and_filters_to_audited_consolidated(tmp_path) -> None:
    source = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "AAA", "BBB", "CCC"],
            "fiscal_year": [2024, 2024, 2024, 2024],
            "audit_status": ["audited", "audited", "audited", "unaudited"],
            "scope": ["consolidated"] * 4,
            "auditing_company_name": ["Deloitte", "Deloitte", "A&C", "KPMG"],
        }
    ).to_csv(source, index=False)
    firm_year, mapping, status = load_auditor_firm_year(
        [source], {}, audited_label="audited", required_scope="consolidated"
    )
    assert status.loc[0, "auditor_name_column"] == "auditing_company_name"
    assert set(firm_year["issuer_ticker"]) == {"AAA", "BBB"}
    assert firm_year.loc[firm_year["issuer_ticker"].eq("AAA"), "auditor_group"].iloc[0] == "BIG4"
    assert firm_year.loc[firm_year["issuer_ticker"].eq("BBB"), "auditor_group"].iloc[0] == "NON_BIG4"
    assert not mapping.empty


def _sample(n: int = 600) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    ticker = [f"F{i:04d}" for i in range(n)]
    group = np.where(np.arange(n) % 2 == 0, "BIG4", "NON_BIG4")
    abnormal = rng.normal(size=n)
    logit = (
        -1.8
        + 0.7 * np.abs(abnormal)
        + 0.15 * (group == "BIG4")
        + 0.35 * np.abs(abnormal) * (group == "BIG4")
    )
    probability = 1 / (1 + np.exp(-logit))
    outcome = rng.binomial(1, probability)
    cases = pd.DataFrame(
        {
            "issuer_ticker": ticker,
            "fiscal_year": 2024,
            "proxy_model": "earnings_working_capital",
            "sample_mode": "common_primary_models",
            "sample_restriction": "analysis_core",
            "abnormal_cfo_proxy": abnormal,
            "lag_assets": np.exp(rng.normal(25, 1, n)),
            "pre_cfo_scaled": rng.normal(0.05, 0.1, n),
            "raw_exchange": np.where(np.arange(n) % 3 == 0, "HOSE", "HNX"),
            "industry_name": np.where(np.arange(n) % 4 == 0, "A", "B"),
            "any_candidate": outcome,
            "audited_cfo_decrease": rng.binomial(1, probability * 0.6),
            "audited_cfo_increase": rng.binomial(1, probability * 0.5),
            "cff_down_candidate": rng.binomial(1, probability * 0.35),
            "cfi_up_candidate": rng.binomial(1, probability * 0.30),
        }
    )
    auditor = pd.DataFrame(
        {
            "issuer_ticker": ticker,
            "fiscal_year": 2024,
            "auditor_name_raw": np.where(group == "BIG4", "Deloitte", "A&C"),
            "auditor_name_normalized": np.where(group == "BIG4", "deloitte", "a c"),
            "auditor_brand": np.where(group == "BIG4", "DELOITTE", "a c"),
            "auditor_group": group,
            "big4_flag": np.where(group == "BIG4", 1.0, 0.0),
            "auditor_name_status": np.where(
                group == "BIG4", "MAPPED_BIG4", "MAPPED_NON_BIG4"
            ),
            "auditor_firm_year_status": "EXACT_ONE_NAME",
        }
    )
    return cases, auditor


def test_metrics_and_bootstrap_use_identical_model_sample() -> None:
    cases, auditor = _sample()
    sample, coverage = prepare_auditor_analysis_sample(cases, auditor, {})
    metrics, differences = stratified_auditor_metrics(sample, {})
    bootstrap = cluster_bootstrap_differences(
        sample, {"bootstrap_repetitions": 20, "bootstrap_seed": 7}
    )
    assert set(coverage["auditor_group"]) == {"BIG4", "NON_BIG4"}
    assert set(metrics["auditor_group"]).issuperset({"BIG4", "NON_BIG4"})
    assert len(differences) == 5
    assert set(bootstrap["metric"]) == {
        "delta_prevalence",
        "delta_auc",
        "delta_ap",
        "delta_lift",
    }


def test_interaction_model_reports_clustered_focal_terms() -> None:
    cases, auditor = _sample(800)
    sample, _ = prepare_auditor_analysis_sample(cases, auditor, {})
    result = auditor_interaction_models(
        sample,
        {
            "outcomes": ["any_candidate"],
            "minimum_interaction_rows": 300,
            "minimum_interaction_positives": 20,
            "fixed_effects": ["raw_exchange", "industry_name"],
        },
    )
    focal = result[result["focal_term"].eq(True)]
    assert set(focal["term"]) == {"score_z", "big4", "score_x_big4"}
    assert focal["cluster_se"].notna().all()
    assert set(focal["status"]).issubset({"CONVERGED", "MAX_ITER"})


def test_switch_diagnostics_counts_direction() -> None:
    mapping = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "AAA", "BBB", "BBB"],
            "fiscal_year": [2023, 2024, 2023, 2024],
            "auditor_group": ["NON_BIG4", "BIG4", "BIG4", "NON_BIG4"],
            "auditor_brand": ["a c", "PWC", "EY", "bdo"],
        }
    )
    events, summary = auditor_switch_diagnostics(mapping)
    assert len(events) == 2
    assert set(summary["switch_type"]) == {
        "NON_BIG4_TO_BIG4",
        "BIG4_TO_NON_BIG4",
    }
