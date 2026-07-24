from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audit_da.results_completion.architecture import _shapley_three
from audit_da.results_completion.core import CompletionSettings
from audit_da.results_completion.method_contract import (
    LOCKED_METHOD_CONTRACT,
    validate_method_contract,
)
from audit_da.results_completion.method_locked import (
    _fit_model_no_intercept,
    _within_cell_permutations,
    randomisation_benchmarks,
)


def test_locked_contract_requires_exact_values() -> None:
    config = {"method_contract": dict(LOCKED_METHOD_CONTRACT)}
    assert validate_method_contract(config) == LOCKED_METHOD_CONTRACT

    invalid = {"method_contract": dict(LOCKED_METHOD_CONTRACT)}
    invalid["method_contract"]["jones_ordinary_intercept"] = True
    with pytest.raises(ValueError, match="method contract mismatch"):
        validate_method_contract(invalid)


def test_jones_fit_has_no_ordinary_or_centering_intercept() -> None:
    training = pd.DataFrame(
        {
            "inv_assets": [0.01, 0.02, 0.03, 0.04, 0.05],
            "drev_scaled": [-0.2, 0.0, 0.1, 0.2, 0.3],
            "ta_scaled": [-0.03, 0.01, 0.02, 0.04, 0.06],
        }
    )
    scaler, model, residual_sd = _fit_model_no_intercept(
        training, ["inv_assets", "drev_scaled"]
    )
    assert scaler.with_mean is False
    assert model.fit_intercept is False
    assert float(np.asarray(model.intercept_)) == 0.0
    assert np.isfinite(residual_sd)


def test_three_player_shapley_reconstructs_absolute_da_reduction() -> None:
    da_pre = np.array([0.20, -0.10, 0.05, -0.02])
    pat = np.array([0.02, -0.01, 0.01, 0.00])
    cfo = np.array([-0.03, 0.04, -0.02, 0.01])
    benchmark = np.array([0.01, -0.02, 0.005, -0.005])
    phi_pat, phi_cfo, phi_benchmark = _shapley_three(
        da_pre, pat, cfo, benchmark
    )
    reduction = np.abs(da_pre) - np.abs(
        da_pre + pat + cfo + benchmark
    )
    np.testing.assert_allclose(
        phi_pat + phi_cfo + phi_benchmark,
        reduction,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_signed_shifts_are_permuted_only_within_fiscal_year() -> None:
    values = np.array([10.0, 11.0, 12.0, 20.0, 21.0, 22.0])
    years = np.array([2020, 2020, 2020, 2021, 2021, 2021])
    draws = _within_cell_permutations(
        values, years, 200, np.random.default_rng(9)
    )
    for indices in (np.array([0, 1, 2]), np.array([3, 4, 5])):
        expected = np.sort(values[indices])
        for row in draws[:, indices]:
            np.testing.assert_array_equal(np.sort(row), expected)


def test_randomisation_output_records_fiscal_year_cells() -> None:
    direct = pd.DataFrame(
        {
            "issuer_ticker": ["A", "B", "C", "D"],
            "fiscal_year": [2020, 2020, 2021, 2021],
            "cfo_pre": [-2.0, -1.0, 1.0, 2.0],
            "cfo_post": [1.0, -0.5, -1.0, 3.0],
            "cfo_sign_switch": [True, False, True, False],
        }
    )
    model_cases = pd.DataFrame(
        {
            "issuer_ticker": ["A", "B", "C", "D"],
            "fiscal_year": [2020, 2020, 2021, 2021],
            "model": ["jones"] * 4,
            "benchmark": ["audited_reference"] * 4,
            "da_pre": [-0.2, -0.1, 0.1, 0.2],
            "signed_shift": [0.3, 0.05, -0.2, 0.0],
            "da_sign_switch": [True, False, True, False],
        }
    )
    settings = CompletionSettings(
        simulation_draws=20,
        simulation_batch_size=4,
        parallel_workers=1,
    )
    result = randomisation_benchmarks(direct, model_cases, settings)
    signed = result.loc[result.benchmark.eq("signed_shift_reassignment")]
    assert signed.reassignment_cell.eq("fiscal_year").all()
    assert signed.cell_count.eq(2).all()
