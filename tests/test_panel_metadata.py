from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_da.bctc_auditor_source import canonicalize_entity_ticker
from audit_da.panel_metadata import enrich_panel_metadata, select_analysis_sample


def _base_panel() -> pd.DataFrame:
    rows = []
    for ticker in ("VSM", "VSMS", "VTS", "VTSC"):
        for state in ("unaudited", "audited"):
            rows.append(
                {
                    "issuer_ticker": ticker,
                    "fiscal_year": 2024,
                    "audit_status": state,
                    "raw_exchange": "TEST",
                    "scope": "consolidated",
                }
            )
    return pd.DataFrame(rows)


def _industry_file(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "issuer_ticker": "VSM",
                "fiscal_year": 2024,
                "firm_name_raw": "Công ty Cổ phần Container Miền Trung",
                "icb_l1": "Công nghiệp",
                "icb_l2": "Vận tải",
            },
            {
                "issuer_ticker": "VSM",
                "fiscal_year": 2024,
                "firm_name_raw": "Công ty Cổ phần Chứng khoán VSM",
                "icb_l1": "Tài chính",
                "icb_l2": "Dịch vụ tài chính",
            },
            {
                "issuer_ticker": "VTS",
                "fiscal_year": 2024,
                "firm_name_raw": "Công ty Cổ phần Gạch Ngói Từ Sơn",
                "icb_l1": "Xây dựng và Vật liệu",
                "icb_l2": "Vật liệu xây dựng",
            },
            {
                "issuer_ticker": "VTS",
                "fiscal_year": 2024,
                "firm_name_raw": "Công ty Cổ phần Chứng khoán Việt Thành",
                "icb_l1": "Tài chính",
                "icb_l2": "Dịch vụ tài chính",
            },
        ]
    ).to_csv(path, index=False)


def _audit_file(path: Path) -> None:
    columns = [
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
    entities = [
        ("VSM", "Công ty Cổ phần Container Miền Trung", "Deloitte Việt Nam", "Chấp nhận toàn phần"),
        ("VSM", "Công ty Cổ phần Chứng khoán VSM", "AASC", "Ý kiến ngoại trừ"),
        ("VTS", "Công ty Cổ phần Gạch Ngói Từ Sơn", "AASC", "Chấp nhận toàn phần"),
        ("VTS", "Công ty Cổ phần Chứng khoán Việt Thành", "KPMG Việt Nam", "Từ chối ra ý kiến"),
    ]
    rows = []
    for ticker, firm_name, auditor, opinion in entities:
        rows.append(
            [
                ticker,
                ticker,
                firm_name,
                "TEST",
                2024,
                "annual",
                "Hợp nhất",
                "audited",
                "audit_firm",
                auditor,
                None,
                auditor,
                f"{ticker}_firm.xlsx",
                1,
                "header",
            ]
        )
        rows.append(
            [
                ticker,
                ticker,
                firm_name,
                "TEST",
                2024,
                "annual",
                "Hợp nhất",
                "audited",
                "audit_opinion",
                opinion,
                opinion,
                None,
                f"{ticker}_opinion.xlsx",
                2,
                "header",
            ]
        )
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_full_legal_names_resolve_reused_symbols() -> None:
    assert canonicalize_entity_ticker(
        "VSM", "Công ty Cổ phần Chứng khoán VSM"
    ) == "VSMS"
    assert canonicalize_entity_ticker(
        "VTS", "Công ty Cổ phần Chứng khoán Việt Thành"
    ) == "VTSC"
    assert canonicalize_entity_ticker(
        "VSM", "Công ty Cổ phần Container Miền Trung"
    ) == "VSM"
    assert canonicalize_entity_ticker(
        "VTS", "Công ty Cổ phần Gạch Ngói Từ Sơn"
    ) == "VTS"


def test_enrichment_preserves_master_and_excludes_financial_firms(
    tmp_path: Path,
) -> None:
    industry = tmp_path / "industry.csv"
    audit = tmp_path / "audit.csv"
    _industry_file(industry)
    _audit_file(audit)

    enriched, statuses = enrich_panel_metadata(
        _base_panel(),
        industry_path=industry,
        audit_metadata_path=audit,
        unknown_industry_policy="exclude",
    )

    assert len(enriched) == 8
    assert set(enriched.issuer_ticker) == {"VSM", "VSMS", "VTS", "VTSC"}
    assert enriched.groupby("issuer_ticker").financial_flag.first().to_dict() == {
        "VSM": False,
        "VSMS": True,
        "VTS": False,
        "VTSC": True,
    }
    assert enriched.groupby("issuer_ticker").big4_flag.first().to_dict() == {
        "VSM": 1.0,
        "VSMS": 0.0,
        "VTS": 0.0,
        "VTSC": 1.0,
    }
    assert enriched.groupby("issuer_ticker").audit_opinion_group.first().to_dict() == {
        "VSM": "unmodified",
        "VSMS": "qualified",
        "VTS": "unmodified",
        "VTSC": "disclaimer",
    }
    assert set(statuses) == {"industry", "audit"}

    selected, manifest = select_analysis_sample(
        enriched,
        {
            "exclude_financial_firms": True,
            "unknown_industry_policy": "exclude",
        },
    )
    assert set(selected.issuer_ticker) == {"VSM", "VTS"}
    assert len(selected) == 4
    excluded = manifest.set_index("stage").loc["excluded_financial_firms"]
    assert excluded["rows"] == 4
    assert excluded["issuer_years"] == 2
