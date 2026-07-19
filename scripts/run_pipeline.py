#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete paired DA signal-gate pipeline")
    parser.add_argument("--config", default="config/signal_gate.yaml")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file (.zip, .csv, or .csv.gz). Directories are not supported.",
    )
    args = parser.parse_args()
    python = sys.executable
    root = Path(__file__).resolve().parent.parent
    src = str(root / "src")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])

    run([python, "scripts/00_profile_input.py", "--config", args.config, "--input", args.input], env)
    run([python, "scripts/01_build_panel.py", "--config", args.config, "--input", args.input], env)
    run([python, "scripts/02_run_signal_gate.py", "--config", args.config], env)
    run([python, "scripts/03_run_baselines.py", "--config", args.config], env)


if __name__ == "__main__":
    main()
