from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def _paired(panel: pd.DataFrame, year: int, audited: str, unaudited: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    current = panel[panel["fiscal_year"] == year]
    pre = current[current["audit_status"] == unaudited].copy()
    post = current[current["audit_status"] == audited].copy()
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
    rows: list[pd.DataFrame] = []

    for year in range(int(signal["minimum_test_year"]), int(signal["maximum_test_year"]) + 1):
        train = panel[
            (panel["audit_status"] == audited)
            & (panel["fiscal_year"] <= year - 1)
        ].copy()
        pre, post = _paired(panel, year, audited, unaudited)
        if pre.empty:
            continue

        for model_name, features in specs.items():
            features = list(features)
            needed = ["ta_scaled"] + features
            training = train.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)

            valid = pre[needed].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            valid &= post[needed].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
            pre_v = pre.loc[valid].reset_index(drop=True)
            post_v = post.loc[valid].reset_index(drop=True)

            if len(training) < int(config["models"]["minimum_train_rows"]) or pre_v.empty:
                continue

            scaler = StandardScaler().fit(training[features])
            model = LinearRegression().fit(
                scaler.transform(training[features]),
                training["ta_scaled"],
            )

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

                rows.append(pd.DataFrame({
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
                }))

    if not rows:
        raise ValueError("No OLS baseline folds were produced")
    return pd.concat(rows, ignore_index=True)
