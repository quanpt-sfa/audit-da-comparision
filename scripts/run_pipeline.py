#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete paired DA signal-gate pipeline")
    parser.add_argument("--config", default="config/signal_gate.yaml")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    python = sys.executable
    run([python, "scripts/00_profile_input.py", "--config", args.config, "--input", args.input])
    run([python, "scripts/01_build_panel.py", "--config", args.config, "--input", args.input])
    run([python, "scripts/02_run_signal_gate.py", "--config", args.config])
    run([python, "scripts/03_run_baselines.py", "--config", args.config])


if __name__ == "__main__":
    main()
