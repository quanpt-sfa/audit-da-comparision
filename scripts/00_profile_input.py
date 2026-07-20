#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.analysis_window import AnalysisWindow
from audit_da.config import load_config, resolve_path
from audit_da.io import materialize_csv_gz, write_json
from audit_da.panel import profile_input


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the TT200-window financial-statement input"
    )
    parser.add_argument("--config", default="config/signal_gate.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    window = AnalysisWindow.from_mapping(
        config.get("analysis_window"),
        fallback={
            "source_start_year": config.get("input", {}).get("minimum_year", 2015),
            "source_end_year": config.get("input", {}).get("maximum_year", 2025),
            "training_start_year": config.get("models", {}).get(
                "training_start_year", 2015
            ),
            "test_start_year": config.get("signal", {}).get(
                "minimum_test_year", 2016
            ),
            "test_end_year": config.get("signal", {}).get(
                "maximum_test_year", 2025
            ),
        },
    )
    input_path = (
        Path(args.input)
        if args.input
        else resolve_path(args.config, config["paths"]["input"])
    )
    local = materialize_csv_gz(input_path, resolve_path(args.config, "data/raw"))
    profile = profile_input(
        local,
        int(config["input"]["chunksize"]),
        minimum_year=window.source_start_year,
        maximum_year=window.source_end_year,
    )
    output = resolve_path(args.config, config["paths"]["profile_output"])
    write_json(profile, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
