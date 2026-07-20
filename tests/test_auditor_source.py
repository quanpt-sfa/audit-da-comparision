from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_da.auditor_source import (
    discover_auditor_sources,
    load_auditor_firm_year_resilient,
)


def test_wide_audit_firm_name_column_is_detected(tmp_path: Path) -> None:
    source = tmp_path / "audit_wide.csv"
    pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB"],
            "fiscal_year": [2024, 2024],
            "audit_firm_name": [
                "Công ty TNHH Deloitte Việt Nam",
                "Công ty TNHH Kiểm toán AASC",
            ],
        }
    ).to_csv(source, index=False)

    firm_year, mapping, status = load_auditor_firm_year_resilient(
        [source], {}, required_scope=None
    )

    assert firm_year.set_index("issuer_ticker").loc["AAA", "auditor_group"] == "BIG4"
    assert firm_year.set_index("issuer_ticker").loc["BBB", "auditor_group"] == "NON_BIG4"
    assert status.loc[0, "source_schema"] == "WIDE_AUDITOR_COLUMN"
    assert not mapping.empty


def test_long_audit_indicator_rows_are_detected(tmp_path: Path) -> None:
    source = tmp_path / "bctc_audit_annual_long.csv"
    pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "AAA", "BBB", "BBB"],
            "year": [2024, 2024, 2024, 2024],
            "audit_indicator": [
                "audit_firm",
                "audit_opinion",
                "audit_firm",
                "audit_opinion",
            ],
            "audit_value": [
                "Công ty TNHH KPMG",
                "Chấp nhận toàn phần",
                "Công ty TNHH Kiểm toán A&C",
                "Chấp nhận toàn phần",
            ],
        }
    ).to_csv(source, index=False)

    firm_year, _, status = load_auditor_firm_year_resilient(
        [source], {}, required_scope=None
    )

    groups = firm_year.set_index("issuer_ticker")["auditor_group"].to_dict()
    assert groups == {"AAA": "BIG4", "BBB": "NON_BIG4"}
    assert status.loc[0, "source_schema"] == "LONG_AUDIT_INDICATOR"
    assert status.loc[0, "indicator_column"] == "audit_indicator"
    assert status.loc[0, "auditor_name_column"] == "audit_value"


def test_missing_sources_return_not_evaluated_instead_of_raising(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    firm_year, mapping, status = load_auditor_firm_year_resilient(
        [missing], {"required": True}, required_scope=None
    )

    assert firm_year.empty
    assert mapping.empty
    assert not status.empty
    assert set(status["overall_status"]) == {"NOT_EVALUATED"}


def test_source_discovery_finds_historical_audit_table(tmp_path: Path) -> None:
    source = tmp_path / "data" / "audit" / "bctc_audit_annual_long.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "issuer_ticker,year,audit_indicator,audit_value\n",
        encoding="utf-8",
    )

    found = discover_auditor_sources(
        tmp_path,
        [],
        ["data/**/bctc_audit_annual_long.csv"],
    )

    assert found == [source.resolve()]
