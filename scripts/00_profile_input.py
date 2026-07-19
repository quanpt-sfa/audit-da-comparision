#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_da.config import load_config, resolve_path
from audit_da.io import materialize_csv_gz, write_json
from audit_da.panel import profile_input


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the long financial-statement input")
    parser.add_argument("--config", default="config/signal_gate.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    input_path = Path(args.input) if args.input else resolve_path(args.config, config["paths"]["input"])
    local = materialize_csv_gz(input_path, resolve_path(args.config, "data/raw"))
    profile = profile_input(local, int(config["input"]["chunksize"]))
    output = resolve_path(args.config, config["paths"]["profile_output"])
    write_json(profile, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
