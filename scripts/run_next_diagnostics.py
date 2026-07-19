from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the post-baseline transition and falsification diagnostics")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()

    scripts = [
        "05_audit_tails_and_ta.py",
        "06_analyze_sign_transitions.py",
        "07_run_directional_placebos.py",
        "08_analyze_rolling_calibration.py",
        "09_analyze_model_family_discordance.py",
        "10_write_next_diagnostics_report.py",
    ]

    env = os.environ.copy()
    src_root = str(repo_root / "src")
    env["PYTHONPATH"] = src_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    for script in scripts:
        command = [sys.executable, str(scripts_dir / script), "--config", str(config_path)]
        print("Running", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
