from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_da.auditor_source_safe import load_auditor_firm_year_safe


def test_zero_rows_after_status_scope_filter_retries_without_filters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bctc_audit_annual_long.csv"
    pd.DataFrame(
        {
            "issuer_ticker": ["AAA", "BBB"],
            "fiscal_year": [2024, 2024],
            "audit_firm_name": [
                "Công ty TNHH Deloitte Việt Nam",
                "Công ty TNHH Kiểm toán AASC",
            ],
            # These source-specific values do not equal the CFS labels
            # `audited` and `consolidated`, reproducing the reported crash.
            "audit_status": ["annual_audit", "annual_audit"],
            "scope": ["hop_nhat", "hop_nhat"],
        }
    ).to_csv(source, index=False)

    firm_year, mapping, status = load_auditor_firm_year_safe(
        [source],
        {},
        audited_label="audited",
        required_scope="consolidated",
    )

    groups = firm_year.set_index("issuer_ticker")["auditor_group"].to_dict()
    assert groups == {"AAA": "BIG4", "BBB": "NON_BIG4"}
    assert not mapping.empty
    assert status.loc[0, "filter_strategy"] == (
        "RETRY_WITHOUT_AUDIT_STATUS_OR_SCOPE"
    )
    assert "KeyError" in status.loc[0, "initial_error"]


def test_terminal_loader_error_returns_not_evaluated(tmp_path: Path) -> None:
    source = tmp_path / "broken.csv"
    source.write_bytes(b"\xff\xfe\x00\x00")

    firm_year, mapping, status = load_auditor_firm_year_safe(
        [source],
        {},
    )

    assert firm_year.empty
    assert mapping.empty
    assert not status.empty
    assert set(status["overall_status"]) == {"NOT_EVALUATED"}
