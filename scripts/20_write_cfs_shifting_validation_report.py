from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve


def maybe_read(output: Path, name: str) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the observed CFS shifting validation report")
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])
    inventory = maybe_read(output, "cfs_item_inventory")
    mapping_review = maybe_read(output, "cfs_item_mapping_review")
    coverage = maybe_read(output, "cfs_line_item_method_coverage")
    folds = maybe_read(output, "cfs_expected_cfo_folds")
    validation = maybe_read(output, "cfs_shifting_proxy_validation")
    yearly = maybe_read(output, "cfs_shifting_proxy_validation_by_year")
    reconciliation = maybe_read(output, "cfs_line_item_reconciliation_summary")
    top = maybe_read(output, "cfs_line_item_top_contributors")

    lines = [
        "# Observed CFS Shifting Validation Report",
        "",
        "## Interpretation boundaries",
        "",
        "- Observed preliminary-to-audited reclassification is a validation outcome, not direct evidence of managerial intent.",
        "- Abnormal-CFO residuals are literature-aligned indirect proxies estimated prospectively from prior-year data.",
        "- A useful shifting proxy should distinguish CFF-downward cases from CFI-upward cases, not merely predict any reporting instability.",
        "- Detailed line-item conclusions remain provisional until mapping-review rows and selected source documents are checked.",
        "",
    ]
    if not inventory.empty:
        status = inventory["mapping_status"].value_counts(dropna=False)
        lines += [
            "## CFS item inventory", "",
            f"- Distinct source items: {len(inventory):,}.",
            f"- Mapped items: {int(status.get('mapped', 0)):,}.",
            f"- Unmapped items: {int(status.get('unmapped', 0)):,}.",
            f"- Ambiguous items: {int(status.get('ambiguous', 0)):,}.", "",
        ]
    if not coverage.empty:
        lines += ["## Selected CFS method coverage", "", coverage.to_markdown(index=False), ""]
    if not folds.empty:
        lines += ["## Rolling expected-CFO folds", "", folds.to_markdown(index=False), ""]
    if not validation.empty:
        lines += ["## Proxy validation against observed outcomes", "", validation.to_markdown(index=False), ""]
    if not yearly.empty:
        lines += ["## Temporal stability", "", yearly.to_markdown(index=False), ""]
    if not reconciliation.empty:
        lines += ["## Detailed line-item reconciliation", "", reconciliation.to_markdown(index=False), ""]
    if not top.empty:
        lines += ["## Largest mapped line-item contributors", "", top.head(100).to_markdown(index=False), ""]
    if not mapping_review.empty:
        lines += [
            "## Mapping review obligation", "",
            f"- Items requiring manual mapping review: {len(mapping_review):,}.",
            "- Do not interpret CFI/CFF subchannels until high-coverage unmapped and ambiguous items are resolved.", "",
        ]
    lines += [
        "## Decision rules", "",
        "1. A shifting proxy is externally validated only if it predicts CFF-downward observed reclassification out of time with meaningful AUC/AP lift.",
        "2. Similar prediction of CFF-downward and CFI-upward outcomes indicates a generic reporting-instability proxy rather than an opportunistic-CFO proxy.",
        "3. Prediction of any candidate but not direction supports reliability screening, not manipulation detection.",
        "4. Line-item reconciliation must explain a material share of aggregate CFI/CFF changes before institutional mechanisms are named.",
        "5. Source-document checks remain mandatory for top contributors, ambiguous mappings, and economically extreme cases.",
    ]
    report = output / "CFS_SHIFTING_VALIDATION_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
