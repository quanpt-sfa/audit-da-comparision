from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _next_diag_common import load_config, resolve


def read_table(output: Path, name: str) -> pd.DataFrame:
    for path in (output / f"{name}.csv", output / f"{name}.csv.gz"):
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def add_table(lines: list[str], title: str, table: pd.DataFrame, empty: str) -> None:
    lines += [f"## {title}", ""]
    lines += [table.to_markdown(index=False) if not table.empty else empty, ""]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write within-exchange and COVID-period robustness report"
    )
    parser.add_argument("--config", default="config/cfs_shifting_validation.yaml")
    args = parser.parse_args()
    config_path, config = load_config(args.config)
    output = resolve(config_path, config["paths"]["output_dir"])

    window = read_table(output, "cfs_regime_robustness_window_status")
    status = read_table(output, "cfs_regime_robustness_status")
    exchange = read_table(output, "cfs_exchange_robustness_metrics")
    exchange_diff = read_table(output, "cfs_exchange_pairwise_differences")
    exchange_boot = read_table(output, "cfs_exchange_cluster_bootstrap")
    exchange_leave = read_table(output, "cfs_exchange_leave_one_out")
    exchange_interaction = read_table(output, "cfs_exchange_interactions")
    covid = read_table(output, "cfs_covid_regime_metrics")
    covid_diff = read_table(output, "cfs_covid_regime_differences")
    covid_sensitivity = read_table(output, "cfs_covid_window_sensitivity")
    covid_boot = read_table(output, "cfs_covid_cluster_bootstrap")
    covid_interaction = read_table(output, "cfs_covid_interactions")

    exchange_focal = (
        exchange_interaction[exchange_interaction["focal_term"].eq(True)].copy()
        if not exchange_interaction.empty and "focal_term" in exchange_interaction
        else pd.DataFrame()
    )
    covid_focal = (
        covid_interaction[covid_interaction["focal_term"].eq(True)].copy()
        if not covid_interaction.empty and "focal_term" in covid_interaction
        else pd.DataFrame()
    )

    lines = [
        "# Exchange and COVID-Period Robustness",
        "",
        "## Interpretation boundaries",
        "",
        "- All analyses use the focal earnings-working-capital score on the common-primary, analysis-core issuer-year sample.",
        "- The TT200 source period is 2015-2025; robustness metrics use out-of-sample test observations from 2016 onward.",
        "- Exchange results test transportability across HOSE, HNX and UPCoM; they do not identify a causal exchange effect.",
        "- COVID results test stability across temporal regimes. They do not identify a causal treatment effect of the pandemic.",
        "- The primary COVID-period definition is 2020-2021; 2020-only and 2020-2022 are prespecified sensitivity windows.",
        "- Pairwise confidence intervals resample issuers, preserving repeated firm-year dependence.",
        "- Full year fixed effects absorb level differences in the COVID interaction models; the focal term is the change in the score slope during the configured period.",
        "",
    ]

    add_table(lines, "Robustness time contract", window, "No robustness-window status was produced.")
    add_table(lines, "Execution gates", status, "No robustness status was produced.")
    add_table(lines, "Criterion validity within exchange", exchange, "No exchange metrics were produced.")
    add_table(lines, "Pairwise exchange differences", exchange_diff, "No exchange differences were produced.")
    add_table(lines, "Issuer-cluster bootstrap by exchange", exchange_boot, "No exchange bootstrap was produced.")
    add_table(lines, "Leave-one-exchange-out sensitivity", exchange_leave, "No leave-one-exchange-out table was produced.")
    add_table(lines, "Score-by-exchange interactions", exchange_focal, "No exchange interaction was estimable.")
    add_table(lines, "Criterion validity by COVID regime", covid, "No COVID-regime metrics were produced.")
    add_table(lines, "COVID and recovery differences from pre-COVID", covid_diff, "No COVID-regime differences were produced.")
    add_table(lines, "Alternative COVID windows", covid_sensitivity, "No alternative-window sensitivity was produced.")
    add_table(lines, "Issuer-cluster bootstrap by COVID regime", covid_boot, "No COVID-regime bootstrap was produced.")
    add_table(lines, "Score-by-COVID-period interactions", covid_focal, "No COVID interaction was estimable.")

    lines += [
        "## Decision rules",
        "",
        "1. A pooled result is considered exchange-transportable when within-exchange signs are consistent, pairwise AUC intervals do not show a material reversal, and leave-one-exchange-out results remain close to pooled estimates.",
        "2. A score-by-exchange interaction is interpreted as slope heterogeneity, not an exchange treatment effect.",
        "3. COVID-period robustness is supported when the score remains discriminative in pre-COVID, shock and recovery regimes and the score-by-shock interaction is not materially adverse.",
        "4. Prevalence changes across COVID regimes are descriptive because enforcement, reporting conditions, sample composition and macroeconomic stress also changed over time.",
        "5. Any directional conclusion must be supported by the primary 2020-2021 window and remain qualitatively stable under at least one alternative window.",
    ]

    report = output / "CFS_EXCHANGE_COVID_ROBUSTNESS_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
