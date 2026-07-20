from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve
from audit_da.analysis_window import window_from_section


def read_table(output: Path, name: str) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def add_table(lines: list[str], title: str, table: pd.DataFrame) -> None:
    lines += [f"## {title}", ""]
    lines += [
        table.to_markdown(index=False)
        if not table.empty
        else "No status artifact was produced.",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the shared TT200 time-contract report"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    settings = config["cfs_shifting_validation"]
    window = window_from_section(settings)

    analysis = read_table(output, "cfs_analysis_window_status")
    completion = read_table(output, "cfs_time_contract_status")
    folds = read_table(output, "cfs_expected_cfo_folds")
    common = read_table(output, "cfs_common_sample_status")
    auditor = read_table(output, "cfs_auditor_analysis_window_status")

    effective_start = pd.NA
    effective_end = pd.NA
    if not folds.empty and "fiscal_year" in folds.columns:
        successful = folds.copy()
        if "status" in successful.columns:
            successful = successful[successful["status"].eq("OK")]
        years = pd.to_numeric(successful["fiscal_year"], errors="coerce").dropna()
        if not years.empty:
            effective_start = int(years.min())
            effective_end = int(years.max())

    contract = pd.DataFrame(
        [
            {
                **window.as_dict(),
                "effective_expected_cfo_test_start_year": effective_start,
                "effective_expected_cfo_test_end_year": effective_end,
                "source_target_interpretation": "2015-2025",
                "out_of_sample_interpretation": "2016-2025 or later when training gate binds",
                "common_comparison_interpretation": "issuer-year intersection across prespecified models",
            }
        ]
    )

    lines = [
        "# TT200 Time-Contract Report",
        "",
        "The source and target-construction regime is 2015-2025. Rolling models use only TT200 observations beginning in 2015 and are evaluated out of sample from 2016 onward. The effective first test year may be later when the prespecified minimum-training requirement binds. Cross-model results use the common issuer-year intersection.",
        "",
    ]
    add_table(lines, "Locked contract", contract)
    add_table(lines, "CFS source-window status", analysis)
    add_table(lines, "Artifact-level completion gate", completion)
    add_table(lines, "Common-sample status", common)
    add_table(lines, "Auditor source/test window", auditor)

    report = output / "TT200_TIME_CONTRACT_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
