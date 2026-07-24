from __future__ import annotations

import json

import pandas as pd

from audit_da.results_completion import output_hash, write_outputs


def test_output_hash_accepts_mixed_type_fiscal_year_and_is_row_order_invariant() -> None:
    frame = pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "AAA", "BBB"],
            "fiscal_year": [2020, "pooled", 2021],
            "value": [1.0, 2.0, 3.0],
        }
    )
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

    assert output_hash(frame) == output_hash(shuffled)


def test_write_outputs_hashes_mixed_type_summary_table(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "model": ["jones", "jones"],
            "fiscal_year": [2020, "pooled"],
            "estimate": [0.1, 0.2],
        }
    )

    write_outputs({"summary": frame}, tmp_path, {"seed": 1})

    manifest = json.loads(
        (tmp_path / "results_completion_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["outputs"]["summary"]["rows"] == 2
    assert len(manifest["outputs"]["summary"]["sha256"]) == 64
