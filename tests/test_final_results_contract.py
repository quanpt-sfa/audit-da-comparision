from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.results_completion.applied_unique import (
    _fully_interacted_stacked,
    _unique_test_id,
)
from audit_da.results_completion.core import CompletionSettings, _cluster_ols
from audit_da.results_completion.final_contract import (
    LOCKED_FINAL_CONTRACT,
    validate_final_contract,
)
from audit_da.results_completion.method_v2 import (
    _predictor_bounds,
    _shapley_two,
    build_attribution_cases,
    estimate_accrual_architectures,
)


def test_final_contract_is_exact() -> None:
    config = {"final_method_contract": dict(LOCKED_FINAL_CONTRACT)}
    assert validate_final_contract(config) == LOCKED_FINAL_CONTRACT


def test_current_outcome_is_not_in_predictor_bounds() -> None:
    bounds = {
        "ta_scaled": (-0.1, 0.1),
        "inv_assets": (0.01, 0.05),
    }
    assert _predictor_bounds(bounds) == {
        "inv_assets": (0.01, 0.05)
    }


def test_two_player_shapley_efficiency() -> None:
    da_pre = np.array([0.2, -0.1, 0.05])
    pat = np.array([0.02, -0.01, 0.01])
    cfo = np.array([-0.03, 0.04, -0.02])
    phi_pat, phi_cfo = _shapley_two(da_pre, pat, cfo)
    reduction = np.abs(da_pre) - np.abs(da_pre + pat + cfo)
    np.testing.assert_allclose(phi_pat + phi_cfo, reduction, atol=1.0e-12)


def _analysis_panel() -> pd.DataFrame:
    common = {
        "issuer_ticker": "TEST",
        "fiscal_year": 2016,
        "lag_assets": 1.0,
        "inv_assets": 1.0,
        "icb_l1": "Industrials",
    }
    return pd.DataFrame([
        {
            **common,
            "audit_status": "unaudited",
            "ta_scaled": 100.0,
            "pat": 100.0,
            "cfo": 0.0,
        },
        {
            **common,
            "audit_status": "audited",
            "ta_scaled": -100.0,
            "pat": -100.0,
            "cfo": 0.0,
        },
    ])


def _training_panel() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "issuer_ticker": "A",
            "fiscal_year": 2015,
            "audit_status": "audited",
            "ta_scaled": 0.1,
            "inv_assets": 1.0,
            "icb_l1": "Industrials",
        },
        {
            "issuer_ticker": "B",
            "fiscal_year": 2015,
            "audit_status": "audited",
            "ta_scaled": 0.2,
            "inv_assets": 2.0,
            "icb_l1": "Industrials",
        },
    ])


def test_separate_history_and_raw_current_outcome() -> None:
    settings = CompletionSettings(
        training_start_year=2015,
        test_start_year=2016,
        test_end_year=2016,
        min_train_rows=2,
        min_industry_rows=2,
        bootstrap_draws=10,
        simulation_draws=10,
    )
    accrual, _ = estimate_accrual_architectures(
        _analysis_panel(),
        _training_panel(),
        settings,
        models={"jones": ["inv_assets"]},
        industry_column="icb_l1",
    )
    audited_reference = accrual.loc[
        accrual.architecture.eq("pooled")
        & accrual.benchmark.eq("audited_reference")
    ].iloc[0]
    assert audited_reference.train_min_year == 2015
    assert audited_reference.current_outcome_clipped is False
    assert np.isclose(audited_reference.signed_shift, -200.0)

    attribution = build_attribution_cases(
        accrual, _analysis_panel(), settings
    )
    assert attribution.attribution_player_count.eq(2).all()
    assert attribution.benchmark_move.abs().max() < 1.0e-10
    np.testing.assert_allclose(
        attribution.phi_pat + attribution.phi_cfo,
        attribution.reduction,
        atol=1.0e-10,
    )


def test_fully_interacted_stacked_matches_paired_difference() -> None:
    focal = np.arange(1.0, 9.0)
    x = np.column_stack([np.ones(len(focal)), focal])
    y_pre = 1.0 + 2.0 * focal
    y_aud = 3.0 + 5.0 * focal
    clusters = np.array([f"C{i}" for i in range(len(focal))])
    b_stack, _, _ = _fully_interacted_stacked(
        y_pre, y_aud, x, clusters
    )
    b_diff, _, _ = _cluster_ols(
        y_aud - y_pre, x, clusters
    )
    np.testing.assert_allclose(
        b_stack[x.shape[1] :], b_diff, atol=1.0e-10
    )


def test_signed_da_test_id_is_not_replicated_by_model() -> None:
    assert _unique_test_id(
        "jones", "loss", "signed_da"
    ) == _unique_test_id(
        "modified_jones", "loss", "signed_da"
    )
    assert _unique_test_id(
        "jones", "loss", "high_da"
    ) != _unique_test_id(
        "modified_jones", "loss", "high_da"
    )
