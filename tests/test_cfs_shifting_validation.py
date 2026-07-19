from __future__ import annotations

import pandas as pd

from audit_da.diag_cfs_proxy_validation import (
    classify_cfs_item,
    compile_item_rules,
    validate_proxy_predictions,
)


def test_conservative_item_mapping() -> None:
    rules = compile_item_rules({"line_item_rules": [
        {"concept": "cff_borrowing_proceeds", "section": "financing", "include": [r"tien[_ ]thu.*di[_ ]vay"]},
        {"concept": "cfi_ppe_purchase", "section": "investing", "include": [r"tien[_ ]chi.*mua[_ ]sam.*tscd"]},
    ]})
    concept, section, count = classify_cfs_item(
        "cash_flow_indirect__1_tien_thu_tu_di_vay",
        "Tiền thu từ đi vay",
        "cash_flow_indirect",
        rules,
    )
    assert (concept, section, count) == ("cff_borrowing_proceeds", "financing", 1)


def test_proxy_validation_distinguishes_directional_outcome() -> None:
    predictions = pd.DataFrame({
        "issuer_ticker": ["A", "B", "C", "D"],
        "fiscal_year": [2024] * 4,
        "raw_exchange": ["HOSE"] * 4,
        "pre_cfo_scaled": [0.2, 0.1, -0.1, -0.2],
        "proxy_model": ["m"] * 4,
        "expected_cfo_scaled": [0.0] * 4,
        "abnormal_cfo_proxy": [2.0, 1.0, -1.0, -2.0],
        "proxy_rank_within_year": [1.0, .75, .5, .25],
    })
    cases = pd.DataFrame({
        "issuer_ticker": ["A", "B", "C", "D"],
        "fiscal_year": [2024] * 4,
        "cfs_resolution": [
            "identity_consistent_offsetting_reclassification_candidate",
            "identity_consistent_other",
            "identity_consistent_other",
            "identity_consistent_other",
        ],
        "delta_cfo_scaled": [-0.10, 0.0, 0.0, 0.0],
        "offset_channel_pattern": ["cff_dominant", "mixed", "mixed", "mixed"],
    })
    result = validate_proxy_predictions(predictions, cases, {"material_cfo_threshold": .05})
    row = result["cfs_shifting_proxy_validation"].query("outcome == 'cff_down_candidate'").iloc[0]
    assert row["auc"] == 1.0
    assert row["top_decile_lift"] == 4.0
