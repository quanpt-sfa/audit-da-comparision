from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED = {
    "issuer_ticker", "fiscal_year", "model", "benchmark", "da_pre",
    "da_post", "signed_shift", "reduction", "raw_ta_shift",
}


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values[np.isfinite(values)]


def sign_class(series: pd.Series, tolerance: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.select(
        [values > tolerance, values < -tolerance], [1, -1], default=0
    ), index=series.index)


def distribution_summary(df: pd.DataFrame, tolerance: float, trim: float) -> pd.DataFrame:
    rows = []
    for (model, benchmark), group in df.groupby(["model", "benchmark"], observed=True):
        r = finite(group["reduction"])
        lo, hi = r.quantile([trim, 1 - trim]) if trim else (r.min(), r.max())
        trimmed = r[(r >= lo) & (r <= hi)]
        positive, negative = r[r > tolerance], r[r < -tolerance]
        positive_total = positive.sum()
        top_n = max(1, int(np.ceil(len(positive) * 0.01))) if len(positive) else 0
        rows.append({
            "model": model, "benchmark": benchmark, "count": len(r),
            "firms": group["issuer_ticker"].nunique(),
            "years": group["fiscal_year"].nunique(),
            "mean": r.mean(), "trimmed_mean": trimmed.mean(), "sd": r.std(),
            "p01": r.quantile(.01), "p05": r.quantile(.05),
            "p10": r.quantile(.10), "p25": r.quantile(.25),
            "median": r.median(), "p75": r.quantile(.75),
            "p90": r.quantile(.90), "p95": r.quantile(.95),
            "p99": r.quantile(.99),
            "share_positive": (r > tolerance).mean(),
            "share_negative": (r < -tolerance).mean(),
            "share_exact_zero": (r.abs() <= tolerance).mean(),
            "positive_minus_negative_share": (r > tolerance).mean() - (r < -tolerance).mean(),
            "mean_if_positive": positive.mean(), "mean_if_negative": negative.mean(),
            "top_1pct_share_positive": (
                positive.nlargest(top_n).sum() / positive_total
                if top_n and positive_total > 0 else np.nan
            ),
            "mean_signed_shift": finite(group["signed_shift"]).mean(),
            "mean_raw_ta_shift": finite(group["raw_ta_shift"]).mean(),
        })
    return pd.DataFrame(rows)


def yearly_summary(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["model", "benchmark", "fiscal_year"], observed=True):
        r = finite(group["reduction"])
        rows.append(dict(zip(["model", "benchmark", "fiscal_year"], keys)) | {
            "count": len(r), "mean": r.mean(), "median": r.median(),
            "share_positive": (r > tolerance).mean(),
            "share_negative": (r < -tolerance).mean(),
            "share_exact_zero": (r.abs() <= tolerance).mean(),
        })
    return pd.DataFrame(rows)


def tolerance_grid(df: pd.DataFrame, tolerances: list[float]) -> pd.DataFrame:
    rows = []
    for tol in tolerances:
        for (model, benchmark), group in df.groupby(["model", "benchmark"], observed=True):
            r = finite(group["reduction"])
            rows.append({"model": model, "benchmark": benchmark, "tolerance": tol,
                "count": len(r), "share_positive": (r > tol).mean(),
                "share_negative": (r < -tol).mean(),
                "share_near_zero": (r.abs() <= tol).mean()})
    return pd.DataFrame(rows)


def trim_grid(df: pd.DataFrame, trims: list[float]) -> pd.DataFrame:
    rows = []
    for trim in trims:
        for (model, benchmark), group in df.groupby(["model", "benchmark"], observed=True):
            r = finite(group["reduction"]).sort_values()
            cut = int(np.floor(len(r) * trim))
            kept = r.iloc[cut:len(r)-cut] if cut else r
            rows.append({"model": model, "benchmark": benchmark,
                "trim_fraction_each_tail": trim, "count_retained": len(kept),
                "mean": kept.mean(), "median": kept.median()})
    return pd.DataFrame(rows)


def pair_agreement(df: pd.DataFrame, dimension: str, tolerance: float) -> pd.DataFrame:
    if dimension == "benchmark":
        fixed, varied, index = "model", "benchmark", ["issuer_ticker", "fiscal_year", "model"]
    else:
        fixed, varied, index = "benchmark", "model", ["issuer_ticker", "fiscal_year", "benchmark"]
    rows = []
    for fixed_value, subset in df.groupby(fixed, observed=True):
        pivot = subset.pivot_table(index=index, columns=varied, values="reduction", aggfunc="first")
        for left, right in combinations(sorted(pivot.columns), 2):
            pair = pivot[[left, right]].dropna()
            if pair.empty:
                continue
            rows.append({fixed: fixed_value, f"{varied}_left": left, f"{varied}_right": right,
                "count": len(pair), "pearson_correlation": pair[left].corr(pair[right]),
                "same_sign_class_share": (sign_class(pair[left], tolerance) == sign_class(pair[right], tolerance)).mean(),
                "mean_absolute_difference": (pair[left] - pair[right]).abs().mean()})
    return pd.DataFrame(rows)


def decision(distribution: pd.DataFrame, trims: pd.DataFrame, yearly: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    if not (distribution["mean"] > 0).all():
        reasons.append("At least one model-benchmark mean is non-positive.")
    if not (trims["mean"] > 0).all():
        reasons.append("At least one trimmed mean is non-positive; the signal may depend on tails.")
    yearly_rate = (yearly["mean"] > 0).mean()
    if yearly_rate < .75:
        reasons.append(f"Only {yearly_rate:.1%} of model-benchmark-year cells have positive means.")
    if not (distribution["share_positive"] > distribution["share_negative"]).all():
        reasons.append("At least one cell lacks a positive breadth advantage over deterioration.")
    if not (distribution["mean"] > 0).all() or not (trims["mean"] > 0).all():
        return "NO_GO", reasons
    if reasons:
        return "PASS_DIRECTION_REVIEW_BREADTH", reasons
    return "PASS_BASELINE_GATE", ["Direction, trimming, yearly stability, and breadth checks passed."]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose OLS pre/post DA baseline breadth and robustness")
    parser.add_argument("--config", default="config/ols_diagnostics.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = resolve(config_path, config["paths"]["baseline_input"])
    output = resolve(config_path, config["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(source)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    settings = config["diagnostics"]
    tol = float(settings["zero_tolerance"])
    dist = distribution_summary(df, tol, float(settings["summary_trim_fraction"]))
    yearly = yearly_summary(df, tol)
    tolerances = tolerance_grid(df, list(settings["near_zero_tolerance_grid"]))
    trims = trim_grid(df, list(settings["trim_fraction_grid"]))
    benchmarks = pair_agreement(df, "benchmark", tol)
    models = pair_agreement(df, "model", tol)
    status, reasons = decision(dist, trims, yearly)
    tables = {"ols_distribution_summary.csv": dist, "ols_yearly_summary.csv": yearly,
        "ols_tolerance_grid.csv": tolerances, "ols_trim_sensitivity.csv": trims,
        "ols_benchmark_agreement.csv": benchmarks, "ols_model_agreement.csv": models}
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
        print(f"Wrote {output / name}")
    report = ["# OLS Baseline Diagnostics", "", f"**Decision:** `{status}`", "", "## Reasons", ""]
    report += [f"- {reason}" for reason in reasons]
    report += ["", "## Headline", "",
        f"- All means positive: {(dist['mean'] > 0).all()}",
        f"- All trimmed means positive: {(trims['mean'] > 0).all()}",
        f"- Positive year-cell rate: {(yearly['mean'] > 0).mean():.1%}",
        f"- Median positive share: {dist['share_positive'].median():.1%}",
        f"- Median negative share: {dist['share_negative'].median():.1%}",
        f"- Median exact-zero share: {dist['share_exact_zero'].median():.1%}", "",
        "## Distribution summary", "", dist.to_markdown(index=False), "",
        "## Interpretation boundary", "",
        "Passing establishes only a stable non-Bayesian direction signal; it does not establish Bayesian incremental value, posterior decisiveness, or audit-firm effects."]
    report_path = output / "OLS_BASELINE_DIAGNOSTICS.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"OLS baseline gate: {status}")


if __name__ == "__main__":
    main()
