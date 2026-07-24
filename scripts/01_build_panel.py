#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.config import load_config, resolve_path
from audit_da.io import materialize_csv_gz
from audit_da.panel import build_and_save_panel
from audit_da.panel_metadata import enrich_panel_metadata


LEGAL_NAME_COLUMNS = ("firm_name_raw", "firm_name", "company_name")
REUSED_TICKERS = {"VSM", "VTS"}
LISTED_EXCHANGE_MARKERS = ("HOSE", "HSX", "HNX", "UPCOM")


def _optional_path(
    config_path: str,
    config: dict,
    key: str,
    override: str | None,
) -> Path | None:
    if override:
        return Path(override).resolve()
    value = config.get("paths", {}).get(key)
    return resolve_path(config_path, value) if value else None


def _normalise_ticker(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    return re.sub(r"\.(?:HO|HN|UPCOM)$", "", text)


def _is_listed_exchange(value: object) -> bool:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    token = re.sub(r"[^A-Z0-9]+", "", text)
    return any(marker in token for marker in LISTED_EXCHANGE_MARKERS)


def _validate_financial_source_identity(path: Path) -> dict[str, object]:
    """Validate VSM/VTS identity without requiring legal names unnecessarily.

    Preferred mode uses a legal-name column and lets the panel extractor apply the
    entity-name overrides. Older financial-statement extracts omit legal names but
    already quarantine the two OTC securities entities. Those files are accepted
    only when every remaining VSM/VTS row has a recognised listed exchange.
    """
    compression = "gzip" if str(path).endswith(".gz") else "infer"
    columns = list(pd.read_csv(path, compression=compression, nrows=0).columns)
    legal_name_column = next(
        (column for column in LEGAL_NAME_COLUMNS if column in columns), None
    )
    if legal_name_column is not None:
        return {
            "mode": "legal_name",
            "legal_name_column": legal_name_column,
            "collision_rows_checked": 0,
            "status": "PASS",
        }

    required = {"issuer_ticker", "raw_exchange"}
    missing = sorted(required - set(columns))
    if missing:
        raise ValueError(
            "The financial-statement source has no legal-name column and cannot "
            "prove that reused VSM/VTS symbols were resolved because these fallback "
            f"columns are missing: {missing}."
        )

    collision_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        compression=compression,
        usecols=["issuer_ticker", "raw_exchange"],
        chunksize=250_000,
        low_memory=False,
    ):
        tickers = chunk["issuer_ticker"].map(_normalise_ticker)
        selected = chunk.loc[tickers.isin(REUSED_TICKERS)].copy()
        if selected.empty:
            continue
        selected["issuer_ticker"] = tickers.loc[selected.index]
        collision_frames.append(selected.drop_duplicates())

    if not collision_frames:
        return {
            "mode": "listed_exchange_fallback",
            "legal_name_column": "",
            "collision_rows_checked": 0,
            "status": "PASS_NO_REUSED_TICKERS",
        }

    collision_rows = pd.concat(collision_frames, ignore_index=True).drop_duplicates()
    listed = collision_rows["raw_exchange"].map(_is_listed_exchange)
    unsafe = collision_rows.loc[~listed].copy()
    if not unsafe.empty:
        preview = unsafe.head(20).to_dict(orient="records")
        raise ValueError(
            "The financial-statement source omits legal names and still contains "
            "VSM/VTS rows whose exchange does not prove that they are the listed "
            "entities. Resolve or quarantine these rows before pivoting. "
            f"Unsafe identities: {preview}"
        )

    return {
        "mode": "listed_exchange_fallback",
        "legal_name_column": "",
        "collision_rows_checked": int(len(collision_rows)),
        "status": "PASS_LISTED_ONLY",
        "collision_identities": collision_rows.sort_values(
            ["issuer_ticker", "raw_exchange"]
        ).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract, enrich, and standardize the paired accrual panel"
    )
    parser.add_argument("--config", default="config/signal_gate.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--industry", default=None)
    parser.add_argument("--audit-metadata", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = (
        Path(args.input)
        if args.input
        else resolve_path(args.config, config["paths"]["input"])
    )
    local = materialize_csv_gz(
        input_path, resolve_path(args.config, "data/raw")
    )
    identity_status = _validate_financial_source_identity(local)
    print("Financial-source identity contract:")
    print(pd.DataFrame([identity_status]).to_string(index=False))

    output = resolve_path(args.config, config["paths"]["processed_panel"])
    industry_path = _optional_path(
        args.config, config, "industry_input", args.industry
    )
    audit_path = _optional_path(
        args.config, config, "audit_metadata_input", args.audit_metadata
    )

    for label, path in (
        ("industry", industry_path),
        ("audit metadata", audit_path),
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(
                f"Configured {label} file does not exist: {path}"
            )

    panel = build_and_save_panel(local, output, config)
    panel, statuses = enrich_panel_metadata(
        panel,
        industry_path=industry_path,
        audit_metadata_path=audit_path,
        unknown_industry_policy=config.get("sample", {}).get(
            "unknown_industry_policy", "exclude"
        ),
    )
    panel.to_csv(
        output,
        index=False,
        compression="gzip" if str(output).endswith(".gz") else None,
    )

    print(f"Wrote {len(panel):,} enriched panel rows to {output}")
    print(panel.groupby("audit_status").size().to_string())
    print("\nAnalysis eligibility:")
    print(panel.groupby(["exclusion_reason"], dropna=False).size().to_string())
    for name, status in statuses.items():
        print(f"\n{name} metadata status:")
        print(status.to_string(index=False))


if __name__ == "__main__":
    main()
