from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "01_build_panel.py"
SPEC = importlib.util.spec_from_file_location("build_panel_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_source(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


def test_legal_name_contract_is_preferred(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    _write_source(
        source,
        [
            {
                "issuer_ticker": "VSM",
                "raw_exchange": "OTC",
                "firm_name_raw": "Công ty Cổ phần Chứng khoán VSM",
            }
        ],
    )
    status = MODULE._validate_financial_source_identity(source)
    assert status["status"] == "PASS"
    assert status["mode"] == "legal_name"
    assert status["legal_name_column"] == "firm_name_raw"


def test_listed_exchange_fallback_accepts_quarantined_source(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    _write_source(
        source,
        [
            {"issuer_ticker": "VSM", "raw_exchange": "HNX"},
            {"issuer_ticker": "VTS", "raw_exchange": "UPCoM"},
            {"issuer_ticker": "AAA", "raw_exchange": "HOSE"},
        ],
    )
    status = MODULE._validate_financial_source_identity(source)
    assert status["status"] == "PASS_LISTED_ONLY"
    assert status["mode"] == "listed_exchange_fallback"
    assert status["collision_rows_checked"] == 2


def test_listed_exchange_fallback_rejects_otc_collision(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    _write_source(
        source,
        [
            {"issuer_ticker": "VSM", "raw_exchange": "HNX"},
            {"issuer_ticker": "VSM", "raw_exchange": "OTC"},
        ],
    )
    with pytest.raises(ValueError, match="Resolve or quarantine"):
        MODULE._validate_financial_source_identity(source)


def test_missing_exchange_and_legal_name_is_blocked(tmp_path: Path) -> None:
    source = tmp_path / "source.csv.gz"
    _write_source(source, [{"issuer_ticker": "VSM"}])
    with pytest.raises(ValueError, match="fallback columns are missing"):
        MODULE._validate_financial_source_identity(source)
