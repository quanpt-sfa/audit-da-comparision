from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_da.bctc_auditor_source import (
    is_bctc_audit_annual_long,
    load_bctc_audit_annual_long,
)


SOURCE_COLUMNS = [
    "issuer_ticker",
    "source_ticker_raw",
    "firm_name_raw",
    "exchange_raw",
    "year",
    "period_type",
    "statement_scope",
    "audit_status",
    "audit_indicator",
    "audit_value_raw",
    "audit_opinion_raw",
    "audit_firm_raw",
    "source_file",
    "source_column",
    "source_header_raw",
]


def test_verified_bctc_metadata_contract(tmp_path: Path) -> None:
    source = tmp_path / "bctc_audit_annual_long.csv"
    rows = [
        [
            "AAA", "AAA", "AAA Corp", "HOSE", 2024, "annual", "Hợp nhất",
            "audited", "audit_opinion", "Chấp nhận toàn phần",
            "Chấp nhận toàn phần", None, "aaa.xlsx", 1, "header",
        ],
        [
            "AAA", "AAA", "AAA Corp", "HOSE", 2024, "annual", "Hợp nhất",
            "audited", "audit_firm", "Kiểm toán Deloitte VN",
            None, "Kiểm toán Deloitte VN", "aaa.xlsx", 2, "header",
        ],
        [
            "BBB", "BBB", "BBB Corp", "HNX", 2024, "annual", "Hợp nhất",
            "audited", "audit_firm", "Aasc., Ltd",
            None, "Aasc., Ltd", "bbb.xlsx", 2, "header",
        ],
        # Must be excluded because this is not the verified audit-firm row.
        [
            "CCC", "CCC", "CCC Corp", "UPCOM", 2024, "annual", "Hợp nhất",
            "audited", "audit_opinion", "Ngoại trừ",
            "Ngoại trừ", None, "ccc.xlsx", 1, "header",
        ],
    ]
    pd.DataFrame(rows, columns=SOURCE_COLUMNS).to_csv(source, index=False)

    assert is_bctc_audit_annual_long(source)
    firm_year, mapping, status = load_bctc_audit_annual_long(source)

    groups = firm_year.set_index("issuer_ticker")["auditor_group"].to_dict()
    assert groups == {"AAA": "BIG4", "BBB": "NON_BIG4"}
    assert len(mapping) == 2
    assert status.loc[0, "source_schema"] == "BCTC_AUDIT_ANNUAL_LONG_V1"
    assert status.loc[0, "source_contract_status"] == "PASS"
    assert status.loc[0, "source_rows"] == 4
    assert status.loc[0, "audit_firm_rows"] == 2
    assert status.loc[0, "audit_firm_value_mismatches"] == 0


def test_multiple_auditors_for_one_firm_year_remain_ambiguous(tmp_path: Path) -> None:
    source = tmp_path / "bctc_audit_annual_long.csv"
    base = {
        "issuer_ticker": "VSM",
        "source_ticker_raw": "VSM",
        "firm_name_raw": "VSM Corp",
        "exchange_raw": "HOSE",
        "year": 2024,
        "period_type": "annual",
        "statement_scope": "Hợp nhất",
        "audit_status": "audited",
        "audit_indicator": "audit_firm",
        "audit_opinion_raw": None,
        "source_file": "vsm.xlsx",
        "source_column": 1,
        "source_header_raw": "header",
    }
    rows = []
    for name in ["Aasc., Ltd", "Kiểm Toán Và Tư Vấn Tâm An"]:
        row = dict(base)
        row["audit_value_raw"] = name
        row["audit_firm_raw"] = name
        rows.append(row)
    pd.DataFrame(rows, columns=SOURCE_COLUMNS).to_csv(source, index=False)

    firm_year, _, status = load_bctc_audit_annual_long(source)

    assert len(firm_year) == 1
    assert firm_year.loc[0, "auditor_group"] == "AMBIGUOUS"
    assert firm_year.loc[0, "auditor_firm_year_status"] == (
        "AMBIGUOUS_MULTIPLE_AUDITORS"
    )
    assert status.loc[0, "ambiguous_firm_years"] == 1
    assert status.loc[0, "duplicate_firm_year_rows"] == 1


def test_contract_rejects_missing_verified_column(tmp_path: Path) -> None:
    source = tmp_path / "broken.csv"
    pd.DataFrame({"issuer_ticker": ["AAA"], "year": [2024]}).to_csv(
        source, index=False
    )

    assert not is_bctc_audit_annual_long(source)
