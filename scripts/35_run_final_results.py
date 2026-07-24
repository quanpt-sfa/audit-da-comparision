#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "34_audit_final_results_contract.py"
RESULTS_SCRIPT = REPO_ROOT / "scripts" / "34_complete_final_results.py"


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("final_results_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import final audit script: {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete stale outputs, audit required inputs, run the final Results "
            "pipeline, and audit the completed bundle."
        )
    )
    parser.add_argument("--config", default="config/results_completion.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--simulation-batch-size", type=int, default=None)
    parser.add_argument(
        "--clean",
        action="store_true",
        required=True,
        help="Required: delete final_output_dir before the full rerun.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = resolve(config_path, config["paths"]["final_output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[final-run] removed stale outputs: {output_dir}")

    audit_module = _load_audit_module()
    pre = audit_module.run_audit(config_path, check_outputs=False)
    print(
        "[final-run] input and method contract PASS: "
        f"{pre['contract_sha256']}"
    )

    command = [
        sys.executable,
        str(RESULTS_SCRIPT),
        "--config",
        str(config_path),
        "--overwrite",
    ]
    if args.workers is not None:
        command.extend(["--workers", str(max(1, args.workers))])
    if args.simulation_batch_size is not None:
        command.extend([
            "--simulation-batch-size",
            str(max(1, args.simulation_batch_size)),
        ])

    print("[final-run] executing:", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    final = audit_module.run_audit(config_path, check_outputs=True)
    print(f"[final-run] completed and audited: {final['audit_path']}")


if __name__ == "__main__":
    main()
