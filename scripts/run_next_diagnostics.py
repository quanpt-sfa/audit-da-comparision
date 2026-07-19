from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the post-baseline transition and falsification diagnostics")
    parser.add_argument("--config", default="config/next_diagnostics.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    scripts = [
        "05_audit_tails_and_ta.py",
        "06_analyze_sign_transitions.py",
        "07_run_directional_placebos.py",
        "08_analyze_rolling_calibration.py",
        "09_analyze_model_family_discordance.py",
        "10_write_next_diagnostics_report.py",
    ]
    for script in scripts:
        command = [sys.executable, str(root / script), "--config", args.config]
        print("Running", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
