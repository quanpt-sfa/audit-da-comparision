#!/usr/bin/env python
from __future__ import annotations

import argparse
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


def _validate_financial_source_identity(path: Path) -> None:
    columns = set(
        pd.read_csv(
            path,
            compression="gzip" if str(path).endswith(".gz") else "infer",
            nrows=0,
        ).columns
    )
    if "firm_name_raw" not in columns:
        raise ValueError(
            "The financial-statement source must contain firm_name_raw so reused "
            "VSM/VTS symbols can be resolved before the firm-year pivot."
        )


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
    _validate_financial_source_identity(local)

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
