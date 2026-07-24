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
PANEL_KEYS = ["issuer_ticker", "fiscal_year"]


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


def _apply_population_lock(
    unrestricted_panel: pd.DataFrame,
    eligible_keys_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the pre-existing locked issuer-year population to an enriched panel."""
    if not eligible_keys_path.exists():
        raise FileNotFoundError(
            "The enriched unrestricted panel was created, but the locked population "
            f"key file is missing: {eligible_keys_path}. Recreate the population-lock "
            "artifacts before running Chapter 4."
        )

    keys = pd.read_csv(eligible_keys_path, low_memory=False)
    missing = [column for column in PANEL_KEYS if column not in keys]
    if missing:
        raise ValueError(
            f"Population eligible-key file is missing required columns: {missing}"
        )
    keys = keys[PANEL_KEYS].copy()
    keys["issuer_ticker"] = keys["issuer_ticker"].map(_normalise_ticker)
    keys["fiscal_year"] = pd.to_numeric(keys["fiscal_year"], errors="coerce")
    keys = keys.dropna(subset=PANEL_KEYS).copy()
    keys["fiscal_year"] = keys["fiscal_year"].astype(int)
    keys = keys.drop_duplicates().sort_values(PANEL_KEYS).reset_index(drop=True)

    panel = unrestricted_panel.copy()
    panel["issuer_ticker"] = panel["issuer_ticker"].map(_normalise_ticker)
    panel["fiscal_year"] = pd.to_numeric(panel["fiscal_year"], errors="coerce")
    panel = panel.dropna(subset=PANEL_KEYS).copy()
    panel["fiscal_year"] = panel["fiscal_year"].astype(int)

    duplicate_states = panel.duplicated(PANEL_KEYS + ["audit_status"], keep=False)
    if duplicate_states.any():
        preview = panel.loc[
            duplicate_states, PANEL_KEYS + ["audit_status"]
        ].head(20)
        raise ValueError(
            "Unrestricted panel contains duplicate issuer-year-state rows before "
            f"population locking: {preview.to_dict(orient='records')}"
        )

    available = panel[PANEL_KEYS].drop_duplicates()
    coverage = keys.merge(available, on=PANEL_KEYS, how="left", indicator=True)
    missing_keys = coverage.loc[coverage["_merge"].ne("both"), PANEL_KEYS]
    if not missing_keys.empty:
        raise ValueError(
            "Locked population keys are absent from the rebuilt unrestricted panel. "
            f"Missing examples: {missing_keys.head(20).to_dict(orient='records')}"
        )

    locked = panel.merge(keys, on=PANEL_KEYS, how="inner", validate="many_to_one")
    state_counts = locked.groupby(PANEL_KEYS, observed=True)["audit_status"].nunique()
    invalid = state_counts[state_counts.ne(2)]
    if not invalid.empty:
        raise ValueError(
            "Locked population no longer has exactly two reporting states for every "
            f"issuer-year. Invalid examples: {invalid.head(20).to_dict()}"
        )
    expected_rows = 2 * len(keys)
    if len(locked) != expected_rows:
        raise AssertionError(
            f"Locked panel row count mismatch: expected {expected_rows}, found {len(locked)}"
        )

    locked = locked.sort_values(
        ["fiscal_year", "issuer_ticker", "audit_status"], kind="mergesort"
    ).reset_index(drop=True)
    status = pd.DataFrame(
        [
            {
                "eligible_keys_path": str(eligible_keys_path),
                "eligible_issuer_years": len(keys),
                "unrestricted_rows": len(panel),
                "locked_rows": len(locked),
                "locked_issuers": locked["issuer_ticker"].nunique(),
                "status": "PASS",
            }
        ]
    )
    return locked, status


def _write_panel(panel: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(
        path,
        index=False,
        compression="gzip" if str(path).endswith(".gz") else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and enrich the unrestricted panel, then reapply the locked "
            "issuer-year population"
        )
    )
    parser.add_argument("--config", default="config/signal_gate.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--industry", default=None)
    parser.add_argument("--audit-metadata", default=None)
    parser.add_argument("--eligible-keys", default=None)
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

    locked_output = resolve_path(args.config, config["paths"]["processed_panel"])
    unrestricted_value = config.get("paths", {}).get(
        "unrestricted_panel", "data/processed/accrual_panel_unrestricted.csv.gz"
    )
    unrestricted_output = resolve_path(args.config, unrestricted_value)
    industry_path = _optional_path(
        args.config, config, "industry_input", args.industry
    )
    audit_path = _optional_path(
        args.config, config, "audit_metadata_input", args.audit_metadata
    )
    eligible_keys_path = _optional_path(
        args.config, config, "population_eligible_keys", args.eligible_keys
    )

    for label, path in (
        ("industry", industry_path),
        ("audit metadata", audit_path),
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(
                f"Configured {label} file does not exist: {path}"
            )
    if eligible_keys_path is None:
        raise ValueError(
            "No population eligible-key path is configured. Set "
            "paths.population_eligible_keys or pass --eligible-keys."
        )

    panel = build_and_save_panel(local, unrestricted_output, config)
    panel, statuses = enrich_panel_metadata(
        panel,
        industry_path=industry_path,
        audit_metadata_path=audit_path,
        unknown_industry_policy=config.get("sample", {}).get(
            "unknown_industry_policy", "exclude"
        ),
    )
    _write_panel(panel, unrestricted_output)

    locked, lock_status = _apply_population_lock(panel, eligible_keys_path)
    _write_panel(locked, locked_output)

    print(
        f"Wrote {len(panel):,} enriched unrestricted rows to {unrestricted_output}"
    )
    print(
        f"Wrote {len(locked):,} enriched locked rows to {locked_output}"
    )
    print("\nLocked audit-status counts:")
    print(locked.groupby("audit_status").size().to_string())
    print("\nLocked analysis eligibility:")
    print(locked.groupby("exclusion_reason", dropna=False).size().to_string())
    print("\nPopulation-lock status:")
    print(lock_status.to_string(index=False))
    for name, status in statuses.items():
        print(f"\n{name} metadata status:")
        print(status.to_string(index=False))


if __name__ == "__main__":
    main()
