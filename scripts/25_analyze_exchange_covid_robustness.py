from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run exchange and COVID-period robustness analyses. "
            "This compatibility wrapper delegates to the two independent scripts."
        )
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    parser.add_argument("--bootstrap-repetitions", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent
    env = os.environ.copy()
    src_root = str(repo_root / "src")
    env["PYTHONPATH"] = src_root + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    for script in (
        "25_analyze_exchange_robustness.py",
        "26_analyze_covid_robustness.py",
    ):
        command = [
            sys.executable,
            str(scripts_dir / script),
            "--config",
            args.config,
        ]
        if args.bootstrap_repetitions is not None:
            command += [
                "--bootstrap-repetitions",
                str(args.bootstrap_repetitions),
            ]
        if args.bootstrap_seed is not None:
            command += ["--bootstrap-seed", str(args.bootstrap_seed)]
        print("Running", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=repo_root, env=env)


if __name__ == "__main__":
    main()
