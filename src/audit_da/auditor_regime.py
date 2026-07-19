from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .diag_common import KEYS


DEFAULT_OUTCOMES = (
    "any_candidate",
    "audited_cfo_decrease",
    "audited_cfo_increase",
    "cff_down_candidate",
    "cfi_up_candidate",
)

SCORE_RULES = {
    "any_candidate": "absolute",
    "audited_cfo_decrease": "positive",
    "audited_cfo_increase": "negative",
    "cff_down_candidate": "positive",
    "cfi_up_candidate": "negative",
}

DEFAULT_AUDITOR_COLUMNS = (
    "auditor_name",
    "audit_firm",
    "auditing_firm",
    "auditing_company",
    "audit_company",
    "auditor_firm_name",
    "auditing_company_name",
    "company_audit_name",
    "ten_cong_ty_kiem_toan",
    "ten_don_vi_kiem_toan",
    "cong_ty_kiem_toan",
)

DEFAULT_TICKER_COLUMNS = (
    "issuer_ticker",
    "ticker",
    "stock_code",
    "symbol",
    "ma_ck",
    "ma_chung_khoan",
)

DEFAULT_YEAR_COLUMNS = (
    "fiscal_year",
    "report_year",
    "year",
    "nam",
)

DEFAULT_AUDIT_STATUS_COLUMNS = (
    "audit_status",
    "report_audit_status",
    "is_audited",
    "audited",
)

DEFAULT_SCOPE_COLUMNS = (
    "scope",
    "report_scope",
    "statement_scope",
)

BIG4_PATTERNS = {
    "DELOITTE": (r"\bdeloitte\b",),
    "PWC": (
        r"\bpwc\b",
        r"\bprice\s*waterhouse\s*coopers\b",
        r"\bpricewaterhousecoopers\b",
    ),
    "EY": (
        r"\bernst\s*(?:and|&)\s*young\b",
        r"\bernst\s+young\b",
        r"\bey\s+vietnam\b",
        r"\bey\b",
    ),
    "KPMG": (r"\bkpmg\b",),
}


def _ascii_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("đ", "d")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_ticker(value: Any) -> str:
    text = str(value).strip().upper() if value is not None and not pd.isna(value) else ""
    text = re.sub(r"\.(?:HO|HN|UPCOM)$", "", text)
    return text


def classify_auditor_name(value: Any) -> dict[str, Any]:
    raw = "" if value is None or pd.isna(value) else str(value).strip()
    normalized = _ascii_text(raw)
    if not normalized:
        return {
            "auditor_name_raw": raw,
            "auditor_name_normalized": "",
            "auditor_brand": "",
            "auditor_group": "UNKNOWN",
            "big4_flag": np.nan,
            "auditor_name_status": "MISSING",
        }
    matches = [
        brand
        for brand, patterns in BIG4_PATTERNS.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    ]
    if len(matches) > 1:
        return {
            "auditor_name_raw": raw,
            "auditor_name_normalized": normalized,
            "auditor_brand": "|".join(sorted(matches)),
            "auditor_group": "AMBIGUOUS",
            "big4_flag": np.nan,
            "auditor_name_status": "MULTIPLE_BIG4_BRANDS",
        }
    if len(matches) == 1:
        return {
            "auditor_name_raw": raw,
            "auditor_name_normalized": normalized,
            "auditor_brand": matches[0],
            "auditor_group": "BIG4",
            "big4_flag": 1.0,
            "auditor_name_status": "MAPPED_BIG4",
        }
    return {
        "auditor_name_raw": raw,
        "auditor_name_normalized": normalized,
        "auditor_brand": normalized,
        "auditor_group": "NON_BIG4",
        "big4_flag": 0.0,
        "auditor_name_status": "MAPPED_NON_BIG4",
    }


def _choose_column(
    columns: Iterable[str],
    configured: str | None,
    candidates: Iterable[str],
    kind: str,
) -> str | None:
    available = list(columns)
    if configured:
        if configured not in available:
            raise ValueError(f"Configured {kind} column is unavailable: {configured}")
        return configured
    lowered = {_ascii_text(column).replace(" ", "_"): column for column in available}
    for candidate in candidates:
        key = _ascii_text(candidate).replace(" ", "_")
        if key in lowered:
            return lowered[key]
    if kind != "auditor-name":
        return None
    excluded = ("status", "opinion", "date", "fee", "partner", "code", "id")
    scored: list[tuple[int, int, str]] = []
    for column in available:
        token = _ascii_text(column).replace(" ", "_")
        if any(term in token for term in excluded):
            continue
        score = 0
        if "auditor" in token:
            score += 5
        if "audit" in token and any(term in token for term in ("firm", "company", "name")):
            score += 4
        if "kiem_toan" in token and any(term in token for term in ("cong_ty", "don_vi", "ten")):
            score += 5
        if score:
            scored.append((score, -len(column), column))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][2]


def inspect_auditor_schema(path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        columns = list(pd.read_parquet(path).columns)
    else:
        columns = list(pd.read_csv(path, nrows=0).columns)
    return {
        "ticker": _choose_column(
            columns,
            settings.get("ticker_column"),
            settings.get("ticker_column_candidates", DEFAULT_TICKER_COLUMNS),
            "ticker",
        ),
        "year": _choose_column(
            columns,
            settings.get("year_column"),
            settings.get("year_column_candidates", DEFAULT_YEAR_COLUMNS),
            "year",
        ),
        "auditor": _choose_column(
            columns,
            settings.get("auditor_name_column"),
            settings.get("auditor_name_column_candidates", DEFAULT_AUDITOR_COLUMNS),
            "auditor-name",
        ),
        "audit_status": _choose_column(
            columns,
            settings.get("audit_status_column"),
            settings.get("audit_status_column_candidates", DEFAULT_AUDIT_STATUS_COLUMNS),
            "audit-status",
        ),
        "scope": _choose_column(
            columns,
            settings.get("scope_column"),
            settings.get("scope_column_candidates", DEFAULT_SCOPE_COLUMNS),
            "scope",
        ),
        "available_columns": columns,
    }


def _read_selected(path: Path, columns: list[str], chunksize: int) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    reader = pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False)
    frames = list(reader)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def load_auditor_firm_year(
    source_paths: list[Path],
    settings: dict[str, Any],
    audited_label: str = "audited",
    required_scope: str | None = "consolidated",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_path: Path | None = None
    schema: dict[str, Any] | None = None
    inspected: list[dict[str, Any]] = []
    for path in source_paths:
        if not path.exists():
            inspected.append({"path": str(path), "status": "MISSING_FILE"})
            continue
        candidate = inspect_auditor_schema(path, settings)
        inspected.append(
            {
                "path": str(path),
                "status": "AUDITOR_COLUMN_FOUND" if candidate["auditor"] else "NO_AUDITOR_COLUMN",
                "auditor_column": candidate["auditor"],
            }
        )
        if candidate["ticker"] and candidate["year"] and candidate["auditor"]:
            selected_path, schema = path, candidate
            break
    if selected_path is None or schema is None:
        detail = "; ".join(f"{row['path']}={row['status']}" for row in inspected)
        if settings.get("required", True):
            raise ValueError(f"No usable auditor-name column was found. {detail}")
        status = pd.DataFrame(inspected)
        return pd.DataFrame(), pd.DataFrame(), status

    usecols = [schema["ticker"], schema["year"], schema["auditor"]]
    for optional in (schema["audit_status"], schema["scope"]):
        if optional and optional not in usecols:
            usecols.append(optional)
    raw = _read_selected(
        selected_path,
        usecols,
        int(settings.get("chunksize", 250_000)),
    )
    raw = raw.rename(
        columns={
            schema["ticker"]: "issuer_ticker",
            schema["year"]: "fiscal_year",
            schema["auditor"]: "auditor_name_raw",
        }
    )
    raw["issuer_ticker"] = raw["issuer_ticker"].map(normalize_ticker)
    raw["fiscal_year"] = pd.to_numeric(raw["fiscal_year"], errors="coerce")
    raw = raw[raw["issuer_ticker"].ne("") & raw["fiscal_year"].notna()].copy()
    raw["fiscal_year"] = raw["fiscal_year"].astype(int)
    if schema["audit_status"]:
        status_values = raw[schema["audit_status"]].map(_ascii_text)
        label = _ascii_text(audited_label)
        raw = raw[status_values.eq(label)].copy()
    if schema["scope"] and required_scope:
        scope_values = raw[schema["scope"]].map(_ascii_text)
        raw = raw[scope_values.eq(_ascii_text(required_scope))].copy()

    classified = pd.DataFrame(
        [classify_auditor_name(value) for value in raw["auditor_name_raw"]],
        index=raw.index,
    )
    raw = pd.concat(
        [raw[["issuer_ticker", "fiscal_year"]], classified], axis=1
    )
    name_map = (
        raw[
            [
                "auditor_name_raw",
                "auditor_name_normalized",
                "auditor_brand",
                "auditor_group",
                "big4_flag",
                "auditor_name_status",
            ]
        ]
        .drop_duplicates()
        .sort_values(["auditor_group", "auditor_brand", "auditor_name_raw"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for (ticker, year), group in raw.groupby(KEYS, observed=True, sort=False):
        valid = group[group["auditor_group"].isin(["BIG4", "NON_BIG4"])]
        brands = sorted(set(valid["auditor_brand"].dropna().astype(str)) - {""})
        groups = sorted(set(valid["auditor_group"].dropna().astype(str)))
        if not brands:
            row = classify_auditor_name("")
            row["auditor_firm_year_status"] = "MISSING_AUDITOR"
        elif len(brands) == 1 and len(groups) == 1:
            first = valid.iloc[0]
            row = {
                "auditor_name_raw": first["auditor_name_raw"],
                "auditor_name_normalized": first["auditor_name_normalized"],
                "auditor_brand": brands[0],
                "auditor_group": groups[0],
                "big4_flag": float(first["big4_flag"]),
                "auditor_name_status": first["auditor_name_status"],
                "auditor_firm_year_status": (
                    "EXACT_ONE_NAME" if len(valid) == 1 else "CONSISTENT_DUPLICATES"
                ),
            }
        else:
            row = {
                "auditor_name_raw": " | ".join(sorted(set(valid["auditor_name_raw"].astype(str)))),
                "auditor_name_normalized": " | ".join(brands),
                "auditor_brand": " | ".join(brands),
                "auditor_group": "AMBIGUOUS",
                "big4_flag": np.nan,
                "auditor_name_status": "MULTIPLE_AUDITORS",
                "auditor_firm_year_status": "AMBIGUOUS_MULTIPLE_AUDITORS",
            }
        row.update({"issuer_ticker": ticker, "fiscal_year": int(year)})
        rows.append(row)
    firm_year = pd.DataFrame(rows)
    if not firm_year.empty:
        firm_year = firm_year[KEYS + [
            "auditor_name_raw",
            "auditor_name_normalized",
            "auditor_brand",
            "auditor_group",
            "big4_flag",
            "auditor_name_status",
            "auditor_firm_year_status",
        ]].sort_values(KEYS).reset_index(drop=True)

    status = pd.DataFrame(
        [
            {
                "status": "EVALUATED",
                "source_path": str(selected_path),
                "ticker_column": schema["ticker"],
                "year_column": schema["year"],
                "auditor_name_column": schema["auditor"],
                "audit_status_column": schema["audit_status"],
                "scope_column": schema["scope"],
                "source_rows_after_filters": len(raw),
                "firm_years": len(firm_year),
                "big4_firm_years": int(firm_year["auditor_group"].eq("BIG4").sum()) if not firm_year.empty else 0,
                "non_big4_firm_years": int(firm_year["auditor_group"].eq("NON_BIG4").sum()) if not firm_year.empty else 0,
                "ambiguous_firm_years": int(firm_year["auditor_group"].eq("AMBIGUOUS").sum()) if not firm_year.empty else 0,
                "unknown_firm_years": int(firm_year["auditor_group"].eq("UNKNOWN").sum()) if not firm_year.empty else 0,
            }
        ]
    )
    return firm_year, name_map, status


def _score(values: pd.Series, rule: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if rule == "absolute":
        return numeric.abs()
    if rule == "negative":
        return -numeric
    return numeric


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    positives = int(y.sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    return float(
        (ranks[y == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    finite = np.isfinite(score)
    y, score = y[finite], score[finite]
    if y.sum() == 0:
        return np.nan
    ranked = y[np.argsort(-score, kind="mergesort")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].mean())


def prepare_auditor_analysis_sample(
    cases: pd.DataFrame,
    auditor_firm_year: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = settings.get("proxy_model", "earnings_working_capital")
    sample_mode = settings.get("sample_mode", "common_primary_models")
    restriction = settings.get("sample_restriction", "analysis_core")
    frame = cases.copy()
    if "proxy_model" in frame:
        frame = frame[frame["proxy_model"].eq(model)]
    if "sample_mode" in frame:
        frame = frame[frame["sample_mode"].eq(sample_mode)]
    if "sample_restriction" in frame:
        frame = frame[frame["sample_restriction"].eq(restriction)]
    frame = frame.drop_duplicates(KEYS).copy()
    frame = frame.merge(
        auditor_firm_year,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    frame["auditor_group"] = frame["auditor_group"].fillna("UNKNOWN")
    frame["auditor_firm_year_status"] = frame["auditor_firm_year_status"].fillna(
        "NO_MATCH"
    )
    coverage = (
        frame.groupby("auditor_group", dropna=False, observed=True)
        .agg(rows=("issuer_ticker", "size"), issuers=("issuer_ticker", "nunique"))
        .reset_index()
    )
    coverage["share"] = coverage["rows"] / len(frame) if len(frame) else np.nan
    coverage.insert(0, "proxy_model", model)
    coverage.insert(1, "sample_mode", sample_mode)
    coverage.insert(2, "sample_restriction", restriction)
    return frame, coverage


def _group_metrics(frame: pd.DataFrame, outcome: str) -> dict[str, float | int]:
    y = pd.to_numeric(frame[outcome], errors="coerce").fillna(0).astype(int)
    score = _score(frame["abnormal_cfo_proxy"], SCORE_RULES[outcome])
    valid = y.notna() & score.notna()
    y, score = y[valid], score[valid]
    n = len(y)
    positives = int(y.sum())
    prevalence = positives / n if n else np.nan
    top = (
        frame.loc[valid]
        .assign(_score=score.to_numpy())
        .groupby("fiscal_year", observed=True)["_score"]
        .rank(pct=True, method="average")
        .ge(0.90)
    )
    top_rate = float(y[top.to_numpy()].mean()) if top.any() else np.nan
    return {
        "rows": n,
        "issuers": int(frame.loc[valid, "issuer_ticker"].nunique()),
        "positives": positives,
        "prevalence": prevalence,
        "auc": _auc(y.to_numpy(), score.to_numpy()),
        "average_precision": _ap(y.to_numpy(), score.to_numpy()),
        "top_decile_rate": top_rate,
        "top_decile_lift": top_rate / prevalence if prevalence and np.isfinite(top_rate) else np.nan,
    }


def stratified_auditor_metrics(
    sample: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = settings.get("outcomes", list(DEFAULT_OUTCOMES))
    groups = settings.get("reported_groups", ["BIG4", "NON_BIG4", "UNKNOWN", "AMBIGUOUS"])
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        if outcome not in sample:
            continue
        for group in groups:
            subset = sample[sample["auditor_group"].eq(group)].copy()
            metrics = _group_metrics(subset, outcome) if not subset.empty else {
                "rows": 0,
                "issuers": 0,
                "positives": 0,
                "prevalence": np.nan,
                "auc": np.nan,
                "average_precision": np.nan,
                "top_decile_rate": np.nan,
                "top_decile_lift": np.nan,
            }
            rows.append(
                {
                    "outcome": outcome,
                    "score_rule": SCORE_RULES[outcome],
                    "auditor_group": group,
                    **metrics,
                }
            )
    table = pd.DataFrame(rows)
    differences: list[dict[str, Any]] = []
    for outcome in outcomes:
        part = table[table["outcome"].eq(outcome)].set_index("auditor_group")
        if not {"BIG4", "NON_BIG4"}.issubset(part.index):
            continue
        differences.append(
            {
                "outcome": outcome,
                "big4_rows": part.loc["BIG4", "rows"],
                "non_big4_rows": part.loc["NON_BIG4", "rows"],
                "delta_prevalence_big4_minus_non_big4": part.loc["BIG4", "prevalence"] - part.loc["NON_BIG4", "prevalence"],
                "delta_auc_big4_minus_non_big4": part.loc["BIG4", "auc"] - part.loc["NON_BIG4", "auc"],
                "delta_ap_big4_minus_non_big4": part.loc["BIG4", "average_precision"] - part.loc["NON_BIG4", "average_precision"],
                "delta_lift_big4_minus_non_big4": part.loc["BIG4", "top_decile_lift"] - part.loc["NON_BIG4", "top_decile_lift"],
            }
        )
    return table, pd.DataFrame(differences)


def cluster_bootstrap_differences(
    sample: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    known = sample[sample["auditor_group"].isin(["BIG4", "NON_BIG4"])].copy()
    issuers = known["issuer_ticker"].dropna().unique()
    if len(issuers) < 2:
        return pd.DataFrame()
    reps = int(settings.get("bootstrap_repetitions", 500))
    seed = int(settings.get("bootstrap_seed", 240719))
    rng = np.random.default_rng(seed)
    outcomes = settings.get("outcomes", list(DEFAULT_OUTCOMES))
    by_issuer = {key: value for key, value in known.groupby("issuer_ticker", sort=False)}
    draws: dict[str, list[dict[str, float]]] = {outcome: [] for outcome in outcomes}
    for _ in range(reps):
        selected = rng.choice(issuers, size=len(issuers), replace=True)
        boot = pd.concat([by_issuer[issuer] for issuer in selected], ignore_index=True)
        for outcome in outcomes:
            if outcome not in boot:
                continue
            group_metrics = {}
            for group in ("BIG4", "NON_BIG4"):
                subset = boot[boot["auditor_group"].eq(group)]
                group_metrics[group] = _group_metrics(subset, outcome)
            if any(group_metrics[group]["rows"] == 0 for group in group_metrics):
                continue
            draws[outcome].append(
                {
                    "delta_prevalence": group_metrics["BIG4"]["prevalence"] - group_metrics["NON_BIG4"]["prevalence"],
                    "delta_auc": group_metrics["BIG4"]["auc"] - group_metrics["NON_BIG4"]["auc"],
                    "delta_ap": group_metrics["BIG4"]["average_precision"] - group_metrics["NON_BIG4"]["average_precision"],
                    "delta_lift": group_metrics["BIG4"]["top_decile_lift"] - group_metrics["NON_BIG4"]["top_decile_lift"],
                }
            )
    _, point_diff = stratified_auditor_metrics(known, settings)
    point_diff = point_diff.set_index("outcome") if not point_diff.empty else pd.DataFrame()
    output: list[dict[str, Any]] = []
    point_columns = {
        "delta_prevalence": "delta_prevalence_big4_minus_non_big4",
        "delta_auc": "delta_auc_big4_minus_non_big4",
        "delta_ap": "delta_ap_big4_minus_non_big4",
        "delta_lift": "delta_lift_big4_minus_non_big4",
    }
    for outcome, records in draws.items():
        if not records:
            continue
        boot = pd.DataFrame(records)
        for metric, point_column in point_columns.items():
            values = pd.to_numeric(boot[metric], errors="coerce").dropna()
            if values.empty:
                continue
            output.append(
                {
                    "outcome": outcome,
                    "metric": metric,
                    "estimate_big4_minus_non_big4": point_diff.loc[outcome, point_column] if outcome in point_diff.index else np.nan,
                    "bootstrap_repetitions_requested": reps,
                    "bootstrap_repetitions_valid": len(values),
                    "ci_lower_2_5pct": float(values.quantile(0.025)),
                    "ci_upper_97_5pct": float(values.quantile(0.975)),
                    "bootstrap_mean": float(values.mean()),
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(output)


def _standardize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    sd = numeric.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / sd


def _design_matrix(frame: pd.DataFrame, outcome: str, settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    work = frame[frame["auditor_group"].isin(["BIG4", "NON_BIG4"])].copy()
    work["score"] = _score(work["abnormal_cfo_proxy"], SCORE_RULES[outcome])
    work["score_z"] = _standardize(work["score"])
    work["big4"] = work["auditor_group"].eq("BIG4").astype(float)
    work["score_x_big4"] = work["score_z"] * work["big4"]
    work["log_lag_assets"] = np.log(pd.to_numeric(work["lag_assets"], errors="coerce").clip(lower=1.0))
    numeric_controls = settings.get("continuous_controls", ["log_lag_assets", "pre_cfo_scaled"])
    x = pd.DataFrame(index=work.index)
    x["intercept"] = 1.0
    x["score_z"] = work["score_z"]
    x["big4"] = work["big4"]
    x["score_x_big4"] = work["score_x_big4"]
    for column in numeric_controls:
        if column in work and column not in {"score_z", "big4", "score_x_big4"}:
            x[column] = _standardize(work[column])
    for column in settings.get("fixed_effects", ["fiscal_year", "raw_exchange", "industry_name"]):
        if column not in work:
            continue
        dummies = pd.get_dummies(work[column].fillna("UNKNOWN").astype(str), prefix=column, drop_first=True, dtype=float)
        x = pd.concat([x, dummies], axis=1)
    y = pd.to_numeric(work[outcome], errors="coerce")
    valid = y.notna() & x.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    return pd.concat([work.loc[valid, ["issuer_ticker"]], x.loc[valid]], axis=1), y.loc[valid].astype(int)


def _fit_logit_clustered(
    x_with_cluster: pd.DataFrame,
    y: pd.Series,
    ridge: float,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    clusters = x_with_cluster.pop("issuer_ticker").astype(str)
    x = x_with_cluster.to_numpy(float)
    target = y.to_numpy(float)
    beta = np.zeros(x.shape[1], dtype=float)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    status = "MAX_ITER"
    for _ in range(max_iter):
        eta = np.clip(x @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.clip(probability * (1.0 - probability), 1e-8, None)
        hessian = x.T @ (x * weight[:, None]) + penalty
        gradient = x.T @ (target - probability) - penalty @ beta
        step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta + step
        if np.max(np.abs(step)) < tolerance:
            beta = beta_new
            status = "CONVERGED"
            break
        beta = beta_new
    eta = np.clip(x @ beta, -30.0, 30.0)
    probability = 1.0 / (1.0 + np.exp(-eta))
    weight = np.clip(probability * (1.0 - probability), 1e-8, None)
    bread = np.linalg.pinv(x.T @ (x * weight[:, None]) + penalty)
    score_rows = x * (target - probability)[:, None]
    score_frame = pd.DataFrame(score_rows)
    score_frame["cluster"] = clusters.to_numpy()
    cluster_scores = score_frame.groupby("cluster", observed=True).sum(numeric_only=True).to_numpy(float)
    meat = cluster_scores.T @ cluster_scores
    n, k = x.shape
    g = len(cluster_scores)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 and n > k else 1.0
    covariance = bread @ meat @ bread * correction
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return beta, se, status


def auditor_interaction_models(sample: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    outcomes = settings.get("outcomes", list(DEFAULT_OUTCOMES))
    for outcome in outcomes:
        if outcome not in sample:
            continue
        x, y = _design_matrix(sample, outcome, settings)
        clusters = int(x["issuer_ticker"].nunique()) if not x.empty else 0
        positives = int(y.sum()) if len(y) else 0
        if len(y) < int(settings.get("minimum_interaction_rows", 300)) or positives < int(settings.get("minimum_interaction_positives", 20)):
            output.append(
                {
                    "outcome": outcome,
                    "term": "score_x_big4",
                    "status": "INSUFFICIENT_SAMPLE",
                    "rows": len(y),
                    "positives": positives,
                    "clusters": clusters,
                }
            )
            continue
        terms = [column for column in x.columns if column != "issuer_ticker"]
        beta, se, status = _fit_logit_clustered(
            x.copy(),
            y,
            float(settings.get("interaction_ridge", 1e-6)),
            int(settings.get("interaction_max_iter", 100)),
            float(settings.get("interaction_tolerance", 1e-8)),
        )
        for term, estimate, standard_error in zip(terms, beta, se):
            z = estimate / standard_error if standard_error > 0 else np.nan
            p_value = math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else np.nan
            output.append(
                {
                    "outcome": outcome,
                    "score_rule": SCORE_RULES[outcome],
                    "term": term,
                    "estimate": float(estimate),
                    "cluster_se": float(standard_error),
                    "z_value": float(z) if np.isfinite(z) else np.nan,
                    "p_value_two_sided": float(p_value) if np.isfinite(p_value) else np.nan,
                    "odds_ratio": float(np.exp(np.clip(estimate, -20, 20))),
                    "rows": len(y),
                    "positives": positives,
                    "clusters": clusters,
                    "status": status,
                    "focal_term": term in {"score_z", "big4", "score_x_big4"},
                }
            )
    return pd.DataFrame(output)


def _smd_continuous(big4: pd.Series, non_big4: pd.Series) -> float:
    a = pd.to_numeric(big4, errors="coerce").dropna()
    b = pd.to_numeric(non_big4, errors="coerce").dropna()
    if a.empty or b.empty:
        return np.nan
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def _smd_binary(p1: float, p0: float) -> float:
    pooled = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2.0)
    return (p1 - p0) / pooled if pooled > 0 else np.nan


def auditor_balance_diagnostics(sample: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    known = sample[sample["auditor_group"].isin(["BIG4", "NON_BIG4"])].copy()
    if known.empty:
        return pd.DataFrame()
    known["log_lag_assets"] = np.log(pd.to_numeric(known["lag_assets"], errors="coerce").clip(lower=1.0))
    rows: list[dict[str, Any]] = []
    for column in settings.get("balance_continuous", ["log_lag_assets", "pre_cfo_scaled", "abnormal_cfo_proxy"]):
        if column not in known:
            continue
        big4 = known.loc[known["auditor_group"].eq("BIG4"), column]
        non = known.loc[known["auditor_group"].eq("NON_BIG4"), column]
        rows.append(
            {
                "variable": column,
                "level": "continuous",
                "big4_mean": pd.to_numeric(big4, errors="coerce").mean(),
                "non_big4_mean": pd.to_numeric(non, errors="coerce").mean(),
                "standardized_mean_difference": _smd_continuous(big4, non),
            }
        )
    for column in settings.get("balance_categorical", ["raw_exchange", "industry_name", "fiscal_year"]):
        if column not in known:
            continue
        levels = sorted(known[column].fillna("UNKNOWN").astype(str).unique())
        for level in levels:
            indicator = known[column].fillna("UNKNOWN").astype(str).eq(level)
            p1 = float(indicator[known["auditor_group"].eq("BIG4")].mean())
            p0 = float(indicator[known["auditor_group"].eq("NON_BIG4")].mean())
            rows.append(
                {
                    "variable": column,
                    "level": level,
                    "big4_mean": p1,
                    "non_big4_mean": p0,
                    "standardized_mean_difference": _smd_binary(p1, p0),
                }
            )
    return pd.DataFrame(rows)


def auditor_switch_diagnostics(auditor_firm_year: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = auditor_firm_year[
        auditor_firm_year["auditor_group"].isin(["BIG4", "NON_BIG4"])
    ].sort_values(["issuer_ticker", "fiscal_year"]).copy()
    if known.empty:
        return pd.DataFrame(), pd.DataFrame()
    known["prior_auditor_group"] = known.groupby("issuer_ticker", observed=True)["auditor_group"].shift(1)
    known["prior_auditor_brand"] = known.groupby("issuer_ticker", observed=True)["auditor_brand"].shift(1)
    known["prior_fiscal_year"] = known.groupby("issuer_ticker", observed=True)["fiscal_year"].shift(1)
    known["consecutive_year"] = known["fiscal_year"].sub(known["prior_fiscal_year"]).eq(1)
    events = known[
        known["consecutive_year"]
        & known["prior_auditor_group"].notna()
        & known["auditor_group"].ne(known["prior_auditor_group"])
    ].copy()
    if events.empty:
        return events, pd.DataFrame([{"switch_type": "NONE", "events": 0, "issuers": 0}])
    events["switch_type"] = events["prior_auditor_group"] + "_TO_" + events["auditor_group"]
    summary = (
        events.groupby("switch_type", observed=True)
        .agg(events=("issuer_ticker", "size"), issuers=("issuer_ticker", "nunique"))
        .reset_index()
    )
    return events, summary


def run_auditor_regime_analysis(
    cases: pd.DataFrame,
    auditor_firm_year: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    sample, coverage = prepare_auditor_analysis_sample(cases, auditor_firm_year, settings)
    metrics, differences = stratified_auditor_metrics(sample, settings)
    bootstrap = cluster_bootstrap_differences(sample, settings)
    interaction = auditor_interaction_models(sample, settings)
    balance = auditor_balance_diagnostics(sample, settings)
    switch_events, switch_summary = auditor_switch_diagnostics(auditor_firm_year)
    known_share = float(sample["auditor_group"].isin(["BIG4", "NON_BIG4"]).mean()) if len(sample) else 0.0
    group_counts = sample["auditor_group"].value_counts()
    min_group = int(settings.get("minimum_group_rows", 100))
    pass_gate = (
        known_share >= float(settings.get("minimum_known_coverage", 0.80))
        and int(group_counts.get("BIG4", 0)) >= min_group
        and int(group_counts.get("NON_BIG4", 0)) >= min_group
        and not interaction.empty
    )
    status = pd.DataFrame(
        [
            {
                "gate": "auditor_regime_heterogeneity",
                "status": "PASS" if pass_gate else "PARTIALLY_EVALUATED",
                "analysis_rows": len(sample),
                "known_auditor_rows": int(sample["auditor_group"].isin(["BIG4", "NON_BIG4"]).sum()),
                "known_auditor_share": known_share,
                "big4_rows": int(group_counts.get("BIG4", 0)),
                "non_big4_rows": int(group_counts.get("NON_BIG4", 0)),
                "unknown_rows": int(group_counts.get("UNKNOWN", 0)),
                "ambiguous_rows": int(group_counts.get("AMBIGUOUS", 0)),
                "interpretation": "Associational heterogeneity; auditor is not used in expected-CFO fitting.",
            }
        ]
    )
    return {
        "cfs_auditor_analysis_sample": sample,
        "cfs_auditor_regime_coverage": coverage,
        "cfs_auditor_regime_metrics": metrics,
        "cfs_auditor_regime_metric_differences": differences,
        "cfs_auditor_regime_bootstrap": bootstrap,
        "cfs_auditor_regime_interaction": interaction,
        "cfs_auditor_regime_balance": balance,
        "cfs_auditor_switch_events": switch_events,
        "cfs_auditor_switch_summary": switch_summary,
        "cfs_auditor_regime_status": status,
    }
