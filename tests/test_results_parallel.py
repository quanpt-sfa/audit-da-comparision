from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.results_completion.core import CompletionSettings, cluster_bootstrap_1d
from audit_da.results_completion.parallel import (
    _randomisation_worker,
    randomisation_benchmarks,
)
from audit_da.results_completion.time_shift import (
    _donor_indices,
    _simulate_time_shift_task,
    _vectorized_shapley_contrast,
    time_shift_benchmarks,
)


def test_cluster_bootstrap_1d_reproducible_mean_and_median() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    clusters = np.array(["A", "A", "B", "B", "C"], dtype=object)
    for statistic in ("mean", "median"):
        first = cluster_bootstrap_1d(
            values,
            clusters,
            statistic=statistic,
            draws=100,
            seed=7,
            batch_size=16,
        )
        second = cluster_bootstrap_1d(
            values,
            clusters,
            statistic=statistic,
            draws=100,
            seed=7,
            batch_size=16,
        )
        assert first == second
        assert np.isfinite(first["ci_low"])


def test_vectorized_shapley_returns_one_contrast_per_draw() -> None:
    da = np.array([0.2, -0.1, 0.3])
    pat = np.array([[0.01, 0.02, -0.03], [0.03, -0.01, 0.02]])
    cfo = np.array([[-0.02, 0.01, 0.04], [0.01, 0.02, -0.01]])
    benchmark = np.array([0.005, -0.002, 0.001])
    result = _vectorized_shapley_contrast(da, pat, cfo, benchmark)
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


def test_donor_indices_exclude_self_when_group_has_multiple_rows() -> None:
    rng = np.random.default_rng(4)
    groups = [np.array([0, 1, 2]), np.array([3, 4])]
    for design in (
        "cyclic_within_issuer",
        "independent_within_issuer",
        "same_year_peer",
    ):
        donors = _donor_indices(groups, 5, 20, rng, design)
        assert donors.shape == (20, 5)
        assert np.all(donors != np.arange(5)[None, :])


def test_time_shift_task_is_reproducible() -> None:
    task = {
        "model": "jones",
        "architecture": "pooled",
        "benchmark": "audited_reference",
        "donor_design": "independent_within_issuer",
        "da_pre": np.array([0.2, 0.1, -0.1, -0.2]),
        "pat_move": np.array([0.01, 0.02, 0.03, 0.04]),
        "cfo_move": np.array([-0.03, -0.02, -0.01, -0.04]),
        "benchmark_move": np.array([0.0, 0.001, 0.0, -0.001]),
        "observed": 0.01,
        "groups": [np.array([0, 1]), np.array([2, 3])],
        "draws": 40,
        "batch_size": 8,
        "blas_threads": 1,
        "seed": 11,
    }
    assert _simulate_time_shift_task(task) == _simulate_time_shift_task(task)


def test_randomisation_worker_is_reproducible() -> None:
    task = {
        "outcome": "da_sign",
        "model": "jones",
        "benchmark_model": "audited_reference",
        "values_pre": np.array([-0.2, -0.1, 0.1, 0.2]),
        "shifts": np.array([0.3, 0.05, -0.2, 0.0]),
        "observed_switch": 0.5,
        "draws": 50,
        "batch_size": 10,
        "blas_threads": 1,
        "seed": 9,
    }
    assert _randomisation_worker(task) == _randomisation_worker(task)


def _panel() -> pd.DataFrame:
    rows = []
    for issuer, industry in (("A", "I1"), ("B", "I1"), ("C", "I2")):
        for year in (2020, 2021):
            for state in ("unaudited", "audited"):
                rows.append(
                    {
                        "issuer_ticker": issuer,
                        "fiscal_year": year,
                        "audit_status": state,
                        "icb_industry": industry,
                    }
                )
    return pd.DataFrame(rows)


def _cases() -> pd.DataFrame:
    rows = []
    for model in ("jones", "modified_jones"):
        for benchmark in ("audited_reference", "pre_reference"):
            for issuer_index, issuer in enumerate(("A", "B", "C")):
                for year in (2020, 2021):
                    rows.append(
                        {
                            "issuer_ticker": issuer,
                            "fiscal_year": year,
                            "model": model,
                            "architecture": "pooled",
                            "benchmark": benchmark,
                            "da_pre": 0.1 + issuer_index * 0.01,
                            "pat_move": 0.01 * (year - 2019),
                            "cfo_move": -0.02 * (issuer_index + 1),
                            "benchmark_move": 0.001,
                            "component_contrast": 0.01,
                        }
                    )
    return pd.DataFrame(rows)


def test_time_shift_process_pool_is_reproducible() -> None:
    settings = CompletionSettings(
        test_start_year=2020,
        test_end_year=2021,
        simulation_draws=20,
        parallel_workers=2,
        simulation_batch_size=4,
    )
    first = time_shift_benchmarks(_cases(), _panel(), settings)
    second = time_shift_benchmarks(_cases(), _panel(), settings)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12


def test_randomisation_process_pool_is_reproducible() -> None:
    direct = pd.DataFrame(
        {
            "cfo_pre": [-2.0, -1.0, 1.0, 2.0],
            "cfo_post": [1.0, -0.5, -1.0, 3.0],
            "cfo_sign_switch": [True, False, True, False],
        }
    )
    model = pd.DataFrame(
        {
            "model": ["jones"] * 4,
            "benchmark": ["audited_reference"] * 4,
            "da_pre": [-0.2, -0.1, 0.1, 0.2],
            "signed_shift": [0.3, 0.05, -0.2, 0.0],
            "da_sign_switch": [True, False, True, False],
        }
    )
    settings = CompletionSettings(
        simulation_draws=20,
        parallel_workers=2,
        simulation_batch_size=4,
    )
    first = randomisation_benchmarks(direct, model, settings)
    second = randomisation_benchmarks(direct, model, settings)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
