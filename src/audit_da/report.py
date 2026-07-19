from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_signal_report(panel: pd.DataFrame, posterior: pd.DataFrame, config: dict[str, Any]) -> str:
    decisive = float(config["signal"]["decisive_probability"])
    report_cfg = config["report"]
    primary_mask = (posterior["benchmark"] == "version_specific") & (posterior["rho"] == 0.9) & (posterior["delta"] == 0.005)
    if "error_sd_ratio" in posterior:
        primary_mask &= posterior["error_sd_ratio"] == 1.0
    primary = posterior[primary_mask].copy()
    if primary.empty:
        primary = posterior.copy()
    pair_count = primary[["issuer_ticker", "fiscal_year"]].drop_duplicates().shape[0]
    improve = float(primary["prob_improve"].mean())
    deteriorate = float(primary["prob_deteriorate"].mean())
    decisive_rate = float(((primary["prob_improve"] >= decisive) | (primary["prob_deteriorate"] >= decisive)).mean())
    raw_corr = float(primary[["reduction_mean", "raw_ta_shift"]].corr().iloc[0, 1])

    grouping = ["benchmark", "rho"] + (["error_sd_ratio"] if "error_sd_ratio" in posterior else []) + ["delta"]
    direction = posterior.groupby(grouping, observed=True)["reduction_mean"].mean().reset_index()
    direction["positive"] = direction["reduction_mean"] > 0
    direction_agreement = float(direction["positive"].mean())
    go_checks = {
        "minimum_pairs": pair_count >= int(report_cfg["minimum_pairs"]),
        "decisive_rate": decisive_rate >= float(report_cfg["minimum_decisive_rate"]),
        "benchmark_direction_agreement": direction_agreement >= float(report_cfg["minimum_benchmark_direction_agreement"]),
    }
    overall = all(go_checks.values())

    lines = [
        "# Paired DA Signal Gate Report",
        "",
        f"**Decision:** {'PROVISIONAL GO' if overall else 'NO-GO / REVISE'}",
        "",
        "## Primary diagnostics",
        "",
        f"- Complete firm-year pairs evaluated: **{pair_count:,}**",
        f"- Mean posterior probability of improvement: **{improve:.3f}**",
        f"- Mean posterior probability of deterioration: **{deteriorate:.3f}**",
        f"- Decisive posterior rate at {decisive:.2f}: **{decisive_rate:.3f}**",
        f"- Correlation of posterior reduction with raw TA movement: **{raw_corr:.3f}**",
        f"- Direction agreement across benchmark/rho/delta cells: **{direction_agreement:.3f}**",
        "",
        "## Gate checks",
        "",
    ]
    for name, passed in go_checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "This gate establishes whether paired pre/post DA contains a measurable posterior transition signal. It does not identify deterrence, separate detected from waived adjustments, or rank audit firms. Auditor-level estimation should begin only after this gate passes and sensitivity results remain stable.",
        "",
        "## Cell-level sensitivity summary",
        "",
        direction.to_markdown(index=False),
    ])
    return "\n".join(lines) + "\n"


def save_report(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
