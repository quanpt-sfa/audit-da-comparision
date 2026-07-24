#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "32_audit_method_contract.py"
RESULTS_SCRIPT = REPO_ROOT / "scripts" / "31_complete_manuscript_results.py"


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("method_contract_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import method-contract audit: {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the locked method contract and run the corrected Chapter 4 pipeline"
        )
    )
    parser.add_argument("--config", default="config/results_completion.yaml")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--simulation-batch-size", type=int, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--clean",
        action="store_true",
        help="Delete all existing result checkpoints before the corrected full run.",
    )
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after the audit confirms every existing checkpoint contract.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = _resolve(config_path, config["paths"]["output_dir"])

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[method-corrected-run] removed legacy outputs: {output_dir}")

    audit_module = _load_audit_module()
    audit = audit_module.run_audit(
        config_path,
        check_outputs=args.resume,
    )
    print(
        "[method-corrected-run] method contract PASS: "
        f"{audit['contract_sha256']}"
    )

    command = [
        sys.executable,
        str(RESULTS_SCRIPT),
        "--config",
        str(config_path),
    ]
    if args.resume:
        command.append("--resume")
    if args.workers is not None:
        command.extend(["--workers", str(max(1, args.workers))])
    if args.simulation_batch_size is not None:
        command.extend(
            [
                "--simulation-batch-size",
                str(max(1, args.simulation_batch_size)),
            ]
        )

    print("[method-corrected-run] executing:", " ".join(command))
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    final_audit = audit_module.run_audit(config_path, check_outputs=True)
    print(
        "[method-corrected-run] completed and re-audited: "
        f"{final_audit['audit_path']}"
    )


if __name__ == "__main__":
    main()
