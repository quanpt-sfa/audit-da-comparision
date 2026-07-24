from __future__ import annotations

import numpy as np
import pandas as pd

from audit_da.results_completion import (
    CompletionSettings, _adjust_pvalues, _midrank_against_reference,
    cluster_bootstrap, direct_revision_tables, paired_panel,
)


def _panel() -> pd.DataFrame:
    rows = []
    for issuer, offset in [("A", 0.0), ("B", 1.0), ("C", 2.0)]:
        for year in [2020, 2021]:
            for state, mul in [("unaudited", 1.0), ("audited", 1.1)]:
                assets = 100.0 + 10 * offset
                pat = (5 + year - 2020 + offset) * mul
                cfo = (4 + offset) * (1.0 if state == "unaudited" else 0.8)
                rows.append({"issuer_ticker": issuer, "fiscal_year": year, "audit_status": state,
                             "pat": pat, "cfo": cfo, "lag_assets": assets,
                             "ta_scaled": (pat-cfo)/assets})
    return pd.DataFrame(rows)


def test_paired_panel_and_direct_tables() -> None:
    settings = CompletionSettings(test_start_year=2020, test_end_year=2021, bootstrap_draws=50, simulation_draws=50)
    panel = _panel()
    assert len(paired_panel(panel, settings)) == 6
    tables = direct_revision_tables(panel, settings)
    assert set(tables) == {"direct_revision_cases", "direct_revision_symmetric", "direct_revision_asymmetric"}
    assert len(tables["direct_revision_symmetric"]) == 24


def test_midrank_uses_reference_distribution() -> None:
    rank = _midrank_against_reference(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0, 2.0, 4.0]))
    assert np.allclose(rank, [0.125, 0.5, 0.75])


def test_cluster_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame({"issuer_ticker": ["A", "A", "B", "B", "C"], "x": [1, 2, 3, 4, 5]})
    stat = lambda z: float(z.x.mean())
    a = cluster_bootstrap(frame, stat, draws=100, seed=7, null=0.0)
    b = cluster_bootstrap(frame, stat, draws=100, seed=7, null=0.0)
    assert a == b and a["estimate"] == 3.0


def test_multiple_testing_adjustments_are_monotone() -> None:
    p = [0.01, 0.03, 0.2]
    holm = _adjust_pvalues(p, "holm")
    bh = _adjust_pvalues(p, "bh")
    assert np.all((holm >= 0) & (holm <= 1))
    assert np.all((bh >= 0) & (bh <= 1))
    assert holm[0] <= holm[1] <= holm[2]


def test_sample_manifest_and_supplemental_inference() -> None:
    from audit_da.results_completion import sample_exclusion_manifest, supplemental_inference
    settings = CompletionSettings(test_start_year=2020, test_end_year=2021, bootstrap_draws=50, simulation_draws=50)
    panel = _panel()
    accrual = pd.DataFrame({"issuer_ticker": ["A", "B"], "fiscal_year": [2020, 2020],
        "model": ["jones", "jones"], "architecture": ["pooled", "pooled"],
        "benchmark": ["audited_reference", "audited_reference"]})
    manifest = sample_exclusion_manifest(panel, accrual, settings)
    assert "paired_state_population" in set(manifest.stage)
    concentration = pd.DataFrame({"issuer_ticker": ["A", "A", "B", "C"], "excess_nhhi": [-.1, .2, -.3, .1]})
    result = supplemental_inference(concentration, None, settings)
    assert result.loc[0, "diagnostic"] == "excess_nhhi"


def test_write_outputs_hashes_summary_tables(tmp_path) -> None:
    from audit_da.results_completion import write_outputs
    write_outputs({"summary": pd.DataFrame({"metric": ["b", "a"], "value": [2.0, 1.0]})}, tmp_path, {"seed": 1})
    assert (tmp_path / "summary.csv").exists()
    manifest = (tmp_path / "results_completion_manifest.json").read_text()
    assert "sha256" in manifest
