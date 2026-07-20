from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .cfs_item_map import pair_line_items
from .cfs_proxy_validate import OUTCOMES, _observed_outcomes
from .diag_common import KEYS


DEFAULT_EVENT_OUTCOMES = tuple(OUTCOMES) + (
    "signed_cfo_correction",
    "absolute_cfo_correction",
    "borrowing_proceeds_correction_scaled",
    "debt_repayments_correction_scaled",
)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def prepare_switch_analysis_panel(
    direct_cases: pd.DataFrame,
    auditor_firm_year: pd.DataFrame,
    settings: dict[str, Any],
    cfs_settings: dict[str, Any] | None = None,
    line_item_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the direct-outcome panel used by switch event studies.

    The base-rate outcomes come from the direct CFS target table rather than the
    model-availability intersection. Auditor status is merged only after target
    construction. Optional line-item values add pre-event borrowing intensity and
    borrowing/debt-repayment correction magnitudes.
    """
    cfs_settings = dict(cfs_settings or {})
    frame = _observed_outcomes(direct_cases, cfs_settings)
    frame = frame.drop_duplicates(KEYS).copy()

    delta_cfo = _numeric(frame, "delta_cfo_scaled")
    frame["signed_cfo_correction"] = delta_cfo
    frame["absolute_cfo_correction"] = delta_cfo.abs()

    audit = auditor_firm_year.drop_duplicates(KEYS).copy()
    keep = [
        column
        for column in KEYS
        + [
            "auditor_group",
            "auditor_brand",
            "auditor_name_raw",
            "auditor_firm_year_status",
        ]
        if column in audit.columns
    ]
    frame = frame.merge(audit[keep], on=KEYS, how="left", validate="one_to_one")
    frame["auditor_group"] = frame.get(
        "auditor_group", pd.Series("UNKNOWN", index=frame.index)
    ).fillna("UNKNOWN")

    line_status = {
        "line_item_status": "NOT_PROVIDED",
        "line_item_rows": 0,
        "borrowing_intensity_rows": 0,
    }
    if line_item_panel is not None and not line_item_panel.empty:
        paired = pair_line_items(line_item_panel, cfs_settings)
        if not paired.empty:
            line_status["line_item_status"] = "MERGED"
            line_status["line_item_rows"] = len(paired)
            line_columns = [
                column
                for column in [
                    "cff_borrowing_proceeds_pre",
                    "cff_borrowing_proceeds_post",
                    "delta_cff_borrowing_proceeds",
                    "cff_debt_repayments_pre",
                    "cff_debt_repayments_post",
                    "delta_cff_debt_repayments",
                ]
                if column in paired.columns
            ]
            frame = frame.merge(
                paired[KEYS + line_columns],
                on=KEYS,
                how="left",
                validate="one_to_one",
            )

    scale_column = _first_present(
        frame,
        list(
            settings.get(
                "scale_column_candidates",
                ["lag_assets_common", "lag_assets", "lag_assets_pre"],
            )
        ),
    )
    scale = (
        _numeric(frame, scale_column)
        if scale_column
        else pd.Series(np.nan, index=frame.index)
    )
    scale = scale.where(scale.gt(0))

    borrowing_pre = _numeric(frame, "cff_borrowing_proceeds_pre")
    frame["borrowing_proceeds_pre_scaled"] = borrowing_pre / scale
    frame["borrowing_proceeds_correction_scaled"] = (
        _numeric(frame, "delta_cff_borrowing_proceeds") / scale
    )
    frame["debt_repayments_correction_scaled"] = (
        _numeric(frame, "delta_cff_debt_repayments") / scale
    )
    line_status["borrowing_intensity_rows"] = int(
        frame["borrowing_proceeds_pre_scaled"].notna().sum()
    )
    line_status["scale_column"] = scale_column or "UNAVAILABLE"

    year = pd.to_numeric(frame["fiscal_year"], errors="coerce")
    frame = frame.loc[year.notna()].copy()
    frame["fiscal_year"] = year.loc[frame.index].astype(int)
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    return frame, pd.DataFrame([line_status])


def identify_clean_switch_events(
    auditor_firm_year: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    cfg = settings.get("switch_event_study", settings)
    pre_periods = int(cfg.get("pre_periods", 2))
    post_periods = int(cfg.get("post_periods", 2))
    one_per_direction = bool(cfg.get("one_event_per_direction", True))

    known = (
        auditor_firm_year[
            auditor_firm_year["auditor_group"].isin(["BIG4", "NON_BIG4"])
        ]
        .drop_duplicates(KEYS)
        .sort_values(KEYS)
        .copy()
    )
    if known.empty:
        return pd.DataFrame()

    known["prior_group"] = known.groupby("issuer_ticker", observed=True)[
        "auditor_group"
    ].shift(1)
    known["prior_year"] = known.groupby("issuer_ticker", observed=True)[
        "fiscal_year"
    ].shift(1)
    candidates = known[
        known["prior_group"].notna()
        & known["fiscal_year"].sub(known["prior_year"]).eq(1)
        & known["auditor_group"].ne(known["prior_group"])
    ].copy()
    if candidates.empty:
        return pd.DataFrame()

    histories = {
        str(ticker): group.set_index("fiscal_year")["auditor_group"].to_dict()
        for ticker, group in known.groupby("issuer_ticker", observed=True)
    }
    rows: list[dict[str, Any]] = []
    for record in candidates.itertuples(index=False):
        ticker = str(record.issuer_ticker)
        event_year = int(record.fiscal_year)
        prior_group = str(record.prior_group)
        new_group = str(record.auditor_group)
        direction = (
            "UPGRADE"
            if prior_group == "NON_BIG4" and new_group == "BIG4"
            else "DOWNGRADE"
        )
        history = histories[ticker]
        expected: dict[int, str] = {}
        for offset in range(-pre_periods, 0):
            expected[event_year + offset] = prior_group
        for offset in range(0, post_periods + 1):
            expected[event_year + offset] = new_group
        missing_years = [year for year in expected if year not in history]
        wrong_status = [
            year
            for year, expected_group in expected.items()
            if year in history and history[year] != expected_group
        ]
        clean = not missing_years and not wrong_status
        rows.append(
            {
                "event_id": f"{ticker}:{event_year}:{direction}",
                "issuer_ticker": ticker,
                "event_year": event_year,
                "switch_direction": direction,
                "prior_auditor_group": prior_group,
                "new_auditor_group": new_group,
                "pre_periods_required": pre_periods,
                "post_periods_required": post_periods,
                "window_start_year": event_year - pre_periods,
                "window_end_year": event_year + post_periods,
                "missing_window_years": ",".join(map(str, missing_years)),
                "wrong_status_years": ",".join(map(str, wrong_status)),
                "clean_switch": clean,
                "event_status": (
                    "CLEAN" if clean else "INCOMPLETE_OR_REVERSING_WINDOW"
                ),
            }
        )
    events = pd.DataFrame(rows).sort_values(
        ["switch_direction", "issuer_ticker", "event_year"]
    )
    events["primary_event"] = events["clean_switch"]
    if one_per_direction and not events.empty:
        clean_rank = (
            events.loc[events["clean_switch"]]
            .groupby(["switch_direction", "issuer_ticker"], observed=True)
            .cumcount()
        )
        events.loc[events["clean_switch"], "primary_event"] = clean_rank.eq(
            0
        ).to_numpy()
        later = events["clean_switch"] & ~events["primary_event"]
        events.loc[later, "event_status"] = "LATER_SAME_DIRECTION_EVENT_EXCLUDED"
    return events.reset_index(drop=True)


def _baseline_match_values(
    panel: pd.DataFrame,
    ticker: str,
    year: int,
    columns: list[str],
) -> dict[str, str]:
    row = panel[
        panel["issuer_ticker"].astype(str).eq(ticker)
        & panel["fiscal_year"].eq(year)
    ]
    if row.empty:
        return {}
    first = row.iloc[0]
    return {
        column: str(first[column])
        for column in columns
        if column in row.columns and pd.notna(first[column])
    }


def build_stacked_switch_sample(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = settings.get("switch_event_study", settings)
    pre_periods = int(cfg.get("pre_periods", 2))
    post_periods = int(cfg.get("post_periods", 2))
    exact_match_columns = list(
        cfg.get("exact_match_columns", ["raw_exchange", "industry_name"])
    )
    minimum_controls = int(cfg.get("minimum_controls_per_event", 5))

    frame = panel.copy()
    frame["issuer_ticker"] = frame["issuer_ticker"].astype(str)
    primary = events[events.get("primary_event", False).astype(bool)].copy()
    if primary.empty:
        return pd.DataFrame(), pd.DataFrame()

    by_issuer = {
        str(ticker): group.set_index("fiscal_year", drop=False).sort_index()
        for ticker, group in frame.groupby("issuer_ticker", observed=True)
    }
    stacks: list[pd.DataFrame] = []
    support: list[dict[str, Any]] = []
    for event in primary.itertuples(index=False):
        ticker = str(event.issuer_ticker)
        event_year = int(event.event_year)
        years = list(
            range(event_year - pre_periods, event_year + post_periods + 1)
        )
        baseline_year = event_year - 1
        baseline_group = str(event.prior_auditor_group)
        treated_history = by_issuer.get(ticker)
        if treated_history is None or not set(years).issubset(
            treated_history.index
        ):
            support.append(
                {
                    "event_id": event.event_id,
                    "switch_direction": event.switch_direction,
                    "event_year": event_year,
                    "status": "TREATED_OUTCOME_WINDOW_INCOMPLETE",
                    "controls": 0,
                }
            )
            continue

        match_values = _baseline_match_values(
            frame, ticker, baseline_year, exact_match_columns
        )
        controls: list[str] = []
        for candidate, history in by_issuer.items():
            if candidate == ticker or not set(years).issubset(history.index):
                continue
            window = history.loc[years]
            if not window["auditor_group"].eq(baseline_group).all():
                continue
            baseline = history.loc[baseline_year]
            matched = True
            for column, value in match_values.items():
                if column not in history.columns or str(baseline[column]) != value:
                    matched = False
                    break
            if matched:
                controls.append(candidate)

        if len(controls) < minimum_controls:
            support.append(
                {
                    "event_id": event.event_id,
                    "switch_direction": event.switch_direction,
                    "event_year": event_year,
                    "status": "INSUFFICIENT_STABLE_CONTROLS",
                    "controls": len(controls),
                    "match_values": "|".join(
                        f"{k}={v}" for k, v in match_values.items()
                    ),
                }
            )
            continue

        selected = [ticker] + controls
        stack = frame[
            frame["issuer_ticker"].isin(selected)
            & frame["fiscal_year"].isin(years)
        ].copy()
        stack["event_id"] = event.event_id
        stack["cohort_year"] = event_year
        stack["switch_direction"] = event.switch_direction
        stack["event_time"] = stack["fiscal_year"] - event_year
        stack["treated"] = stack["issuer_ticker"].eq(ticker).astype(int)
        stack["treated_issuer"] = ticker
        stack["baseline_auditor_group"] = baseline_group
        stack["new_auditor_group"] = event.new_auditor_group
        stack["available_controls"] = len(controls)
        stacks.append(stack)
        support.append(
            {
                "event_id": event.event_id,
                "switch_direction": event.switch_direction,
                "event_year": event_year,
                "status": "STACKED",
                "controls": len(controls),
                "match_values": "|".join(
                    f"{k}={v}" for k, v in match_values.items()
                ),
            }
        )

    if not stacks:
        return pd.DataFrame(), pd.DataFrame(support)
    stacked = pd.concat(stacks, ignore_index=True)
    duplicated = stacked.groupby(
        ["switch_direction", "issuer_ticker", "fiscal_year"], observed=True
    )["event_id"].transform("nunique")
    stacked["stack_weight"] = np.where(
        stacked["treated"].eq(1), 1.0, 1.0 / duplicated.clip(lower=1)
    )

    pre_offsets = list(cfg.get("borrowing_intensity_event_times", [-2, -1]))
    borrowing = (
        stacked[stacked["event_time"].isin(pre_offsets)]
        .groupby(["event_id", "issuer_ticker"], observed=True)[
            "borrowing_proceeds_pre_scaled"
        ]
        .mean()
        .rename("pre_event_borrowing_intensity")
        .reset_index()
    )
    stacked = stacked.merge(
        borrowing,
        on=["event_id", "issuer_ticker"],
        how="left",
        validate="many_to_one",
    )
    treated_intensity = (
        stacked[stacked["treated"].eq(1)][
            ["event_id", "switch_direction", "pre_event_borrowing_intensity"]
        ]
        .drop_duplicates("event_id")
        .copy()
    )
    treated_intensity["treated_borrowing_median"] = treated_intensity.groupby(
        "switch_direction", observed=True
    )["pre_event_borrowing_intensity"].transform("median")
    treated_intensity["treated_borrowing_group"] = np.where(
        treated_intensity["pre_event_borrowing_intensity"].notna()
        & treated_intensity["pre_event_borrowing_intensity"].ge(
            treated_intensity["treated_borrowing_median"]
        ),
        "HIGH",
        np.where(
            treated_intensity["pre_event_borrowing_intensity"].notna(),
            "LOW",
            "UNAVAILABLE",
        ),
    )
    stacked = stacked.merge(
        treated_intensity[
            ["event_id", "treated_borrowing_median", "treated_borrowing_group"]
        ],
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    return stacked, pd.DataFrame(support)


def _group_codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values.astype(str), sort=False)[0].astype(int)


def _weighted_demean_once(
    matrix: np.ndarray,
    codes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    output = matrix.copy()
    groups = int(codes.max()) + 1 if len(codes) else 0
    denominator = np.bincount(codes, weights=weights, minlength=groups)
    for column in range(matrix.shape[1]):
        numerator = np.bincount(
            codes, weights=weights * matrix[:, column], minlength=groups
        )
        means = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0,
        )
        output[:, column] -= means[codes]
    return output


def _two_way_residualize(
    matrix: np.ndarray,
    first_fe: np.ndarray,
    second_fe: np.ndarray,
    weights: np.ndarray,
    tolerance: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    residual = matrix.astype(float, copy=True)
    for _ in range(max_iter):
        prior = residual.copy()
        residual = _weighted_demean_once(residual, first_fe, weights)
        residual = _weighted_demean_once(residual, second_fe, weights)
        if np.max(np.abs(residual - prior)) < tolerance:
            break
    return residual


def _cluster_weighted_ols(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    xtwx = x.T @ (x * weights[:, None])
    bread = np.linalg.pinv(xtwx)
    beta = bread @ (x.T @ (weights * y))
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    unique = np.unique(clusters)
    for cluster in unique:
        index = np.flatnonzero(clusters == cluster)
        score = x[index].T @ (weights[index] * residual[index])
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    n, k, g = len(y), x.shape[1], len(unique)
    if g > 1 and n > k:
        covariance *= (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    status = "OK" if np.isfinite(beta).all() else "NONFINITE_ESTIMATE"
    return beta, se, covariance, status


def fit_stacked_event_study(
    stacked: pd.DataFrame,
    settings: dict[str, Any],
    borrowing_group: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = settings.get("switch_event_study", settings)
    outcomes = list(cfg.get("outcomes", DEFAULT_EVENT_OUTCOMES))
    reference = int(cfg.get("reference_event_time", -1))
    minimum_events = int(cfg.get("minimum_events", 20))
    event_times = list(
        range(
            -int(cfg.get("pre_periods", 2)),
            int(cfg.get("post_periods", 2)) + 1,
        )
    )
    event_times = [value for value in event_times if value != reference]

    frame = stacked.copy()
    if borrowing_group is not None:
        frame = frame[frame["treated_borrowing_group"].eq(borrowing_group)].copy()
    estimates: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for direction in ["UPGRADE", "DOWNGRADE"]:
        direction_frame = frame[
            frame["switch_direction"].eq(direction)
        ].copy()
        event_count = int(direction_frame["event_id"].nunique())
        for outcome in outcomes:
            if outcome not in direction_frame.columns:
                continue
            x_frame = pd.DataFrame(index=direction_frame.index)
            term_names: list[str] = []
            for event_time in event_times:
                name = f"treated_x_event_{event_time:+d}"
                x_frame[name] = (
                    direction_frame["treated"].eq(1)
                    & direction_frame["event_time"].eq(event_time)
                ).astype(float)
                term_names.append(name)
            y = pd.to_numeric(direction_frame[outcome], errors="coerce")
            valid = (
                y.notna()
                & x_frame.notna().all(axis=1)
                & pd.to_numeric(
                    direction_frame["stack_weight"], errors="coerce"
                ).gt(0)
            )
            work = direction_frame.loc[valid].copy()
            x_valid = x_frame.loc[valid].copy()
            y_valid = y.loc[valid].astype(float)
            if event_count < minimum_events or work.empty:
                for event_time, term in zip(event_times, term_names):
                    estimates.append(
                        {
                            "switch_direction": direction,
                            "borrowing_group": borrowing_group or "ALL",
                            "outcome": outcome,
                            "event_time": event_time,
                            "term": term,
                            "estimate": np.nan,
                            "cluster_se": np.nan,
                            "p_value_two_sided": np.nan,
                            "events": event_count,
                            "rows": len(work),
                            "status": "INSUFFICIENT_EVENTS",
                        }
                    )
                continue

            matrix = np.column_stack(
                [y_valid.to_numpy(), x_valid.to_numpy(float)]
            )
            weights = pd.to_numeric(
                work["stack_weight"], errors="coerce"
            ).to_numpy(float)
            issuer_stack = _group_codes(
                work["event_id"].astype(str)
                + "|"
                + work["issuer_ticker"].astype(str)
            )
            stack_year = _group_codes(
                work["event_id"].astype(str)
                + "|"
                + work["fiscal_year"].astype(str)
            )
            residualized = _two_way_residualize(
                matrix, issuer_stack, stack_year, weights
            )
            y_residual = residualized[:, 0]
            x_residual = residualized[:, 1:]
            variance = np.average(x_residual**2, axis=0, weights=weights)
            keep = variance > 1e-14
            if not keep.any():
                continue
            kept_terms = [term for term, flag in zip(term_names, keep) if flag]
            beta, se, covariance, status = _cluster_weighted_ols(
                y_residual,
                x_residual[:, keep],
                weights,
                work["issuer_ticker"].astype(str).to_numpy(),
            )
            coefficient_map = {
                term: index for index, term in enumerate(kept_terms)
            }
            for event_time, term in zip(event_times, term_names):
                if term not in coefficient_map:
                    estimates.append(
                        {
                            "switch_direction": direction,
                            "borrowing_group": borrowing_group or "ALL",
                            "outcome": outcome,
                            "event_time": event_time,
                            "term": term,
                            "estimate": np.nan,
                            "cluster_se": np.nan,
                            "p_value_two_sided": np.nan,
                            "events": event_count,
                            "rows": len(work),
                            "status": "COLLINEAR_OR_NO_SUPPORT",
                        }
                    )
                    continue
                index = coefficient_map[term]
                z = beta[index] / se[index] if se[index] > 0 else np.nan
                estimates.append(
                    {
                        "switch_direction": direction,
                        "borrowing_group": borrowing_group or "ALL",
                        "outcome": outcome,
                        "event_time": event_time,
                        "term": term,
                        "estimate": float(beta[index]),
                        "cluster_se": float(se[index]),
                        "ci_lower_95": float(beta[index] - 1.96 * se[index]),
                        "ci_upper_95": float(beta[index] + 1.96 * se[index]),
                        "z_value": float(z) if np.isfinite(z) else np.nan,
                        "p_value_two_sided": float(2 * norm.sf(abs(z)))
                        if np.isfinite(z)
                        else np.nan,
                        "events": event_count,
                        "issuers": int(work["issuer_ticker"].nunique()),
                        "rows": len(work),
                        "status": status,
                    }
                )

            lead_terms = [
                f"treated_x_event_{value:+d}"
                for value in event_times
                if value < reference
                and f"treated_x_event_{value:+d}" in coefficient_map
            ]
            if lead_terms:
                indices = [coefficient_map[term] for term in lead_terms]
                lead_beta = beta[indices]
                lead_cov = covariance[np.ix_(indices, indices)]
                statistic = float(
                    lead_beta.T @ np.linalg.pinv(lead_cov) @ lead_beta
                )
                tests.append(
                    {
                        "switch_direction": direction,
                        "borrowing_group": borrowing_group or "ALL",
                        "outcome": outcome,
                        "test": "JOINT_PRETREND",
                        "lead_terms": "|".join(lead_terms),
                        "chi_square": statistic,
                        "df": len(indices),
                        "p_value": float(chi2.sf(statistic, len(indices))),
                        "events": event_count,
                        "status": status,
                    }
                )
    return pd.DataFrame(estimates), pd.DataFrame(tests)


def run_switch_event_study(
    direct_cases: pd.DataFrame,
    auditor_firm_year: pd.DataFrame,
    settings: dict[str, Any],
    cfs_settings: dict[str, Any] | None = None,
    line_item_panel: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    panel, line_status = prepare_switch_analysis_panel(
        direct_cases,
        auditor_firm_year,
        settings,
        cfs_settings=cfs_settings,
        line_item_panel=line_item_panel,
    )
    events = identify_clean_switch_events(auditor_firm_year, settings)
    stacked, support = build_stacked_switch_sample(panel, events, settings)
    estimates, pretrend = fit_stacked_event_study(stacked, settings)

    borrowing_estimates: list[pd.DataFrame] = []
    borrowing_tests: list[pd.DataFrame] = []
    if not stacked.empty and stacked["treated_borrowing_group"].isin(
        ["HIGH", "LOW"]
    ).any():
        for group in ("HIGH", "LOW"):
            group_estimates, group_tests = fit_stacked_event_study(
                stacked, settings, borrowing_group=group
            )
            borrowing_estimates.append(group_estimates)
            borrowing_tests.append(group_tests)
    borrowing_status = pd.DataFrame(
        [
            {
                "status": "EVALUATED"
                if borrowing_estimates
                else "NOT_EVALUATED",
                "treated_events_with_borrowing_intensity": int(
                    stacked.loc[
                        stacked["treated"].eq(1)
                        & stacked["treated_borrowing_group"].isin(
                            ["HIGH", "LOW"]
                        ),
                        "event_id",
                    ].nunique()
                )
                if not stacked.empty
                else 0,
                "interpretation": (
                    "Subgroups are defined by treated firms' mean preliminary "
                    "borrowing proceeds in the configured pre-event years."
                ),
            }
        ]
    )
    event_count = (
        int(events.get("primary_event", pd.Series(dtype=bool)).sum())
        if not events.empty
        else 0
    )
    stacked_events = (
        int(stacked["event_id"].nunique()) if not stacked.empty else 0
    )
    successful = (
        not estimates.empty
        and "status" in estimates.columns
        and estimates["status"].eq("OK").any()
    )
    status = pd.DataFrame(
        [
            {
                "gate": "auditor_switch_event_study",
                "status": "PASS"
                if stacked_events > 0 and successful
                else "PARTIALLY_EVALUATED",
                "candidate_switch_events": len(events),
                "primary_clean_events": event_count,
                "stacked_events": stacked_events,
                "analysis_rows": len(stacked),
                "interpretation": (
                    "Within-firm dynamic association around clean auditor-tier "
                    "transitions; not a causal auditor effect."
                ),
            }
        ]
    )
    return {
        "cfs_auditor_switch_direct_panel": panel,
        "cfs_auditor_switch_line_item_status": line_status,
        "cfs_auditor_switch_event_diagnostics": events,
        "cfs_auditor_switch_stack_support": support,
        "cfs_auditor_switch_stacked_sample": stacked,
        "cfs_auditor_switch_event_study": estimates,
        "cfs_auditor_switch_pretrend_tests": pretrend,
        "cfs_auditor_switch_borrowing_heterogeneity": pd.concat(
            borrowing_estimates, ignore_index=True
        )
        if borrowing_estimates
        else pd.DataFrame(),
        "cfs_auditor_switch_borrowing_pretrend": pd.concat(
            borrowing_tests, ignore_index=True
        )
        if borrowing_tests
        else pd.DataFrame(),
        "cfs_auditor_switch_borrowing_status": borrowing_status,
        "cfs_auditor_switch_event_study_status": status,
    }
