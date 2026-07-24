#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.cfs_item_map import inventory_and_line_items
from audit_da.results_completion.core import output_hash
from audit_da.supplemental_diagnostics import (
    SupplementalSettings,
    concentration_cases,
    near_zero_cfo_cases,
    near_zero_randomisation,
    supplemental_summary,
)


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _load_or_build_line_items(
    config_path: Path,
    config: dict,
    settings: SupplementalSettings,
    *,
    rebuild: bool,
) -> pd.DataFrame:
    paths = config["paths"]
    cached = resolve(config_path, paths["line_item_input"])
    if cached.exists() and not rebuild:
        print(f"[supplemental] loading mapped line items: {cached}", flush=True)
        return pd.read_csv(cached, low_memory=False)

    raw_path = resolve(config_path, paths["raw_input"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Mapped line-item input missing: {cached}; raw input also missing: {raw_path}"
        )
    mapping_config_path = resolve(config_path, paths["mapping_config"])
    if not mapping_config_path.exists():
        raise FileNotFoundError(f"CFS mapping config missing: {mapping_config_path}")
    mapping_config = yaml.safe_load(mapping_config_path.read_text(encoding="utf-8"))
    mapping = dict(mapping_config["cfs_shifting_validation"])
    mapping["minimum_year"] = settings.minimum_year
    mapping["maximum_year"] = settings.maximum_year
    mapping["audited_label"] = settings.audited_label
    mapping["unaudited_label"] = settings.unaudited_label
    print(f"[supplemental] mapping raw CFS line items: {raw_path}", flush=True)
    tables = inventory_and_line_items(raw_path, mapping)
    frame = tables["cfs_line_item_long"]
    if frame.empty:
        raise ValueError("Raw CFS mapping produced no line-item observations")
    cached.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cached, index=False)
    print(f"[supplemental] wrote mapped line items: {cached}", flush=True)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible supplemental CFS-revision concentration and near-zero "
            "CFO state-swap diagnostics. No external placeholder CSVs are required."
        )
    )
    parser.add_argument(
        "--config",
        default="config/supplemental_diagnostics.yaml",
    )
    parser.add_argument(
        "--rebuild-line-items",
        action="store_true",
        help="Ignore cached cfs_line_item_long and rebuild it from the raw long file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = SupplementalSettings(**config.get("settings", {}))

    panel_path = resolve(config_path, config["paths"]["panel_input"])
    if not panel_path.exists():
        raise FileNotFoundError(f"Processed panel missing: {panel_path}")
    output_dir = resolve(config_path, config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    line_items = _load_or_build_line_items(
        config_path,
        config,
        settings,
        rebuild=args.rebuild_line_items,
    )
    print(f"[supplemental] loading processed panel: {panel_path}", flush=True)
    panel = pd.read_csv(panel_path, low_memory=False)

    print("[supplemental] estimating line-item revision concentration", flush=True)
    concentration = concentration_cases(line_items, settings)
    if concentration.empty:
        raise ValueError("No eligible multi-concept line-item revision cases")

    print("[supplemental] constructing matched near-zero CFO pairs", flush=True)
    near_zero = near_zero_cfo_cases(panel, settings)
    if near_zero.empty:
        raise ValueError("No near-zero CFO pairs matched within absolute-distance bins")

    print("[supplemental] running within-pair state-swap randomisation", flush=True)
    randomisation = near_zero_randomisation(near_zero, settings)
    summary = supplemental_summary(
        concentration,
        near_zero,
        randomisation,
        settings,
    )
    if len(summary) != 2:
        raise ValueError(
            f"Expected two supplemental diagnostic summaries; observed={len(summary)}"
        )

    outputs = {
        "cfs_revision_concentration_cases": concentration,
        "near_zero_cfo_cases": near_zero,
        "near_zero_cfo_permutation_draws": randomisation,
        "supplemental_diagnostics_summary": summary,
    }
    manifest = {
        "config": str(config_path),
        "panel_input": str(panel_path),
        "line_item_rows": int(len(line_items)),
        "settings": settings.__dict__,
        "outputs": {},
    }
    for name, frame in outputs.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest["outputs"][name] = {
            "path": str(path),
            "rows": int(len(frame)),
            "sha256": output_hash(frame),
        }
        print(f"[supplemental] wrote {path} ({len(frame):,} rows)", flush=True)

    manifest_path = output_dir / "supplemental_diagnostics_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[supplemental] complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
