#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete paired DA signal-gate pipeline")
    parser.add_argument("--config", default="config/signal_gate.yaml")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    input_path = Path(args.input).resolve()

    env = os.environ.copy()
    src_root = str(repo_root / "src")
    env["PYTHONPATH"] = src_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    python = sys.executable
    scripts = repo_root / "scripts"
    run([python, str(scripts / "00_profile_input.py"), "--config", str(config_path), "--input", str(input_path)], repo_root, env)
    run([python, str(scripts / "01_build_panel.py"), "--config", str(config_path), "--input", str(input_path)], repo_root, env)
    run([python, str(scripts / "02_run_signal_gate.py"), "--config", str(config_path)], repo_root, env)
    run([python, str(scripts / "03_run_baselines.py"), "--config", str(config_path)], repo_root, env)


if __name__ == "__main__":
    main()
