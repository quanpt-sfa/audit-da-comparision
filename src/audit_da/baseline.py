from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .analysis_window import AnalysisWindow


def _paired(
    panel: pd.DataFrame,
    year: int,
    audited: str,
    unaudited: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = panel[panel["fiscal_year"].eq(year)]
    pre = current[current["audit_status"].eq(unaudited)].copy()
    post = current[current["audit_status"].eq(audited)].copy()
    keys = sorted(set(pre["issuer_ticker"]) & set(post["issuer_ticker"]))
    pre = (
        pre[pre["issuer_ticker"].isin(keys)]
        .drop_duplicates("issuer_ticker")
        .set_index("issuer_ticker")
        .loc[keys]
        .reset_index()
    )
    post = (
        post[post["issuer_ticker"].isin(keys)]
        .drop_duplicates("issuer_ticker")
        .set_index("issuer_ticker")
        .loc[keys]
        .reset_index()
    )
    return pre, post


def run_ols_baselines(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    audited = config["input"]["audited_label"]
    unaudited = config["input"]["unaudited_label"]
    signal = config["signal"]
    specs = config["models"]["candidate_models"]
    window = AnalysisWindow.from_mapping(
        config.get("analysis_window"),
        fallback={
            "source_start_year": config.get("input", {}).get("minimum_year", 2015),
            "source_end_year": config.get("input", {}).get("maximum_year", 2025),
            "training_start_year": config.get("models", {}).get(
                "training_start_year", 2015
            ),
            "test_start_year": signal.get("minimum_test_year", 2016),
            "test_end_year": signal.get("maximum_test_year", 2025),
        },
    )
    source = panel.loc[window.source_mask(panel["fiscal_year"])].copy()
    rows: list[pd.DataFrame] = []

    for year in window.test_years():
        train_mask = source["audit_status"].eq(audited) & window.training_mask(
            source["fiscal_year"], year
        )
        train = source.loc[train_mask].copy()
        pre, post = _paired(source, year, audited, unaudited)
        if pre.empty:
            continue

        for model_name, features in specs.items():
            features = list(features)
            needed = ["ta_scaled"] + features
            training = train.replace([np.inf, -np.inf], np.nan).dropna(
                subset=needed
            )

            valid = (
                pre[needed]
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .all(axis=1)
            )
            valid &= (
                post[needed]
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .all(axis=1)
            )
            pre_v = pre.loc[valid].reset_index(drop=True)
            post_v = post.loc[valid].reset_index(drop=True)

            if (
                len(training) < int(config["models"]["minimum_train_rows"])
                or pre_v.empty
            ):
                continue

            scaler = StandardScaler().fit(training[features])
            model = LinearRegression().fit(
                scaler.transform(training[features]),
                training["ta_scaled"],
            )

            train_year = pd.to_numeric(training["fiscal_year"], errors="coerce")
            metadata = {
                "source_start_year_contract": window.source_start_year,
                "source_end_year_contract": window.source_end_year,
                "training_start_year_contract": window.training_start_year,
                "training_min_year": int(train_year.min()),
                "training_max_year": int(train_year.max()),
                "test_start_year_contract": window.test_start_year,
                "test_end_year_contract": window.test_end_year,
                "train_rows": len(training),
            }

            for benchmark in signal["benchmarks"]:
                if benchmark == "version_specific":
                    x_pre = pre_v[features]
                    x_post = post_v[features]
                elif benchmark == "pre_reference":
                    x_pre = pre_v[features]
                    x_post = pre_v[features]
                elif benchmark == "audited_reference":
                    x_pre = post_v[features]
                    x_post = post_v[features]
                else:
                    raise ValueError(f"Unknown benchmark: {benchmark}")

                nda_pre = model.predict(scaler.transform(x_pre))
                nda_post = model.predict(scaler.transform(x_post))
                da_pre = pre_v["ta_scaled"].to_numpy(float) - nda_pre
                da_post = post_v["ta_scaled"].to_numpy(float) - nda_post

                result = pd.DataFrame(
                    {
                        "issuer_ticker": pre_v["issuer_ticker"],
                        "fiscal_year": year,
                        "model": model_name,
                        "benchmark": benchmark,
                        "da_pre": da_pre,
                        "da_post": da_post,
                        "signed_shift": da_post - da_pre,
                        "reduction": np.abs(da_pre) - np.abs(da_post),
                        "raw_ta_shift": (
                            post_v["ta_scaled"].to_numpy(float)
                            - pre_v["ta_scaled"].to_numpy(float)
                        ),
                    }
                )
                for name, value in metadata.items():
                    result[name] = value
                rows.append(result)

    if not rows:
        raise ValueError("No OLS baseline folds were produced")
    output = pd.concat(rows, ignore_index=True)
    if output["training_min_year"].lt(window.training_start_year).any():
        raise AssertionError("OLS training includes observations before the contract")
    if not output["fiscal_year"].between(
        window.test_start_year, window.test_end_year
    ).all():
        raise AssertionError("OLS output contains test years outside the contract")
    return output
