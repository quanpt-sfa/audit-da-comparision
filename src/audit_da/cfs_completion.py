from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cfs_line_reconcile import line_item_reconciliation
from .diag_common import KEYS


def _key_mask(frame: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    if frame.empty or keys.empty:
        return pd.Series(False, index=frame.index)
    key_index = pd.MultiIndex.from_frame(frame[KEYS])
    allowed = pd.MultiIndex.from_frame(keys[KEYS].drop_duplicates())
    return pd.Series(key_index.isin(allowed), index=frame.index)


def restrict_estimation_panel(
    panel: pd.DataFrame,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the prespecified population restrictions before model fitting.

    This is deliberately separate from outcome-sample filtering. The same
    eligible panel is used to construct prior firm history, estimate each
    rolling fold, and generate test-year predictions.
    """
    cfg = settings.get("estimation_sample", {})
    frame = panel.copy()
    mask = pd.Series(True, index=frame.index)
    reasons: list[str] = []

    if bool(cfg.get("require_nonfinancial", True)):
        if "financial_flag" not in frame.columns:
            raise ValueError(
                "estimation_sample.require_nonfinancial=true but financial_flag is unavailable"
            )
        known = frame["financial_flag"].notna()
        nonfinancial = known & frame["financial_flag"].eq(False)
        mask &= nonfinancial
        reasons.append("known non-financial ICB classification")

    if bool(cfg.get("require_listed", True)):
        listed_values = {
            str(value).upper()
            for value in cfg.get("listed_exchanges", ["HOSE", "HNX", "UPCOM"])
        }
        exchange = frame.get("raw_exchange", pd.Series("", index=frame.index))
        mask &= exchange.astype(str).str.upper().isin(listed_values)
        reasons.append("listed exchange")

    if bool(cfg.get("require_valid_ticker", True)):
        ticker = frame["issuer_ticker"].astype(str).str.upper()
        mask &= ticker.str.fullmatch(r"[A-Z][A-Z0-9]{1,7}")
        reasons.append("valid ticker")

    if bool(cfg.get("require_positive_lag_assets", True)):
        lag_assets = pd.to_numeric(frame.get("lag_assets"), errors="coerce")
        mask &= lag_assets.gt(0)
        reasons.append("positive lagged assets")

    eligible = frame.loc[mask].copy()
    raw_keys = frame[KEYS].drop_duplicates().shape[0]
    eligible_keys = eligible[KEYS].drop_duplicates().shape[0]
    status = pd.DataFrame(
        [
            {
                "status": "EVALUATED",
                "restriction": "rolling_expected_cfo_estimation_sample",
                "input_rows": len(frame),
                "eligible_rows": len(eligible),
                "input_firm_years": raw_keys,
                "eligible_firm_years": eligible_keys,
                "rows_removed": len(frame) - len(eligible),
                "firm_years_removed": raw_keys - eligible_keys,
                "eligible_share": len(eligible) / len(frame) if len(frame) else np.nan,
                "rules": "; ".join(reasons),
            }
        ]
    )
    return eligible, status


def restrict_to_estimation_keys(
    table: pd.DataFrame,
    estimation_panel: pd.DataFrame,
) -> pd.DataFrame:
    keys = estimation_panel[KEYS].drop_duplicates()
    return table.loc[_key_mask(table, keys)].copy()


def history_incremental_comparison(
    validation: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    cfg = settings.get("history_nested_comparison", {})
    base_model = cfg.get("base_model", "earnings_working_capital")
    nested_model = cfg.get(
        "nested_model", "earnings_working_capital_history"
    )
    sample_mode = cfg.get("sample_mode", "common_all_models")
    sample_restriction = cfg.get("sample_restriction", "analysis_core")
    subset = validation[
        validation["sample_mode"].eq(sample_mode)
        & validation["sample_restriction"].eq(sample_restriction)
        & validation["proxy_model"].isin([base_model, nested_model])
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    metrics = [
        "rows",
        "positives",
        "prevalence",
        "auc",
        "average_precision",
        "top_decile_lift",
    ]
    base = subset[subset["proxy_model"].eq(base_model)][["outcome"] + metrics]
    nested = subset[subset["proxy_model"].eq(nested_model)][["outcome"] + metrics]
    base = base.rename(columns={column: f"base_{column}" for column in metrics})
    nested = nested.rename(columns={column: f"nested_{column}" for column in metrics})
    comparison = base.merge(nested, on="outcome", how="inner", validate="one_to_one")
    comparison.insert(0, "base_model", base_model)
    comparison.insert(1, "nested_model", nested_model)
    comparison.insert(2, "sample_mode", sample_mode)
    comparison.insert(3, "sample_restriction", sample_restriction)
    comparison["delta_auc_nested_minus_base"] = (
        comparison["nested_auc"] - comparison["base_auc"]
    )
    comparison["delta_ap_nested_minus_base"] = (
        comparison["nested_average_precision"]
        - comparison["base_average_precision"]
    )
    comparison["delta_lift_nested_minus_base"] = (
        comparison["nested_top_decile_lift"]
        - comparison["base_top_decile_lift"]
    )
    return comparison


def _rename_reconciliation_outputs(
    tables: dict[str, pd.DataFrame], suffix: str
) -> dict[str, pd.DataFrame]:
    return {f"{name}_{suffix}": value for name, value in tables.items()}


def core_reconciliation_outputs(
    line_item_panel: pd.DataFrame,
    observed_cases: pd.DataFrame,
    panel: pd.DataFrame,
    primary_cases: pd.DataFrame,
    all_model_cases: pd.DataFrame,
    settings: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for suffix, sample in [
        ("common_primary_core", primary_cases),
        ("common_all_core", all_model_cases),
    ]:
        if sample.empty:
            output.update(
                _rename_reconciliation_outputs(
                    line_item_reconciliation(
                        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), settings
                    ),
                    suffix,
                )
            )
            continue
        keys = sample[KEYS].drop_duplicates()
        line_subset = line_item_panel.loc[_key_mask(line_item_panel, keys)].copy()
        observed_subset = observed_cases.loc[_key_mask(observed_cases, keys)].copy()
        panel_subset = panel.loc[_key_mask(panel, keys)].copy()
        tables = line_item_reconciliation(
            line_subset, observed_subset, panel_subset, settings
        )
        output.update(_rename_reconciliation_outputs(tables, suffix))
    return output


def build_pdf_verification_manifest(
    reconciliation_cases: pd.DataFrame,
    settings: dict[str, Any],
) -> pd.DataFrame:
    if reconciliation_cases.empty:
        return pd.DataFrame()
    cfg = settings.get("pdf_verification", {})
    candidate_label = settings.get(
        "candidate_label",
        "identity_consistent_offsetting_reclassification_candidate",
    )
    cases = reconciliation_cases[
        reconciliation_cases["cfs_resolution"].eq(candidate_label)
    ].copy()
    if cases.empty:
        return pd.DataFrame()
    cases["absolute_aggregate_change"] = pd.to_numeric(
        cases["aggregate_section_change_scaled"], errors="coerce"
    ).abs()
    cases["absolute_reconciliation_residual"] = pd.to_numeric(
        cases["reconciliation_residual_scaled"], errors="coerce"
    ).abs()
    share = pd.to_numeric(cases["mapped_share_of_aggregate"], errors="coerce")
    cases["mapped_share_gap"] = (share - 1.0).abs()

    selected: list[pd.DataFrame] = []

    def add_bucket(
        condition: pd.Series,
        n: int,
        priority: int,
        reason: str,
        sort_columns: list[str],
    ) -> None:
        bucket = cases.loc[condition].copy()
        if bucket.empty or n <= 0:
            return
        bucket = bucket.sort_values(sort_columns, ascending=False).head(n)
        bucket["verification_priority"] = priority
        bucket["selection_reason"] = reason
        selected.append(bucket)

    quotas = cfg.get("quotas", {})
    add_bucket(
        cases["section"].eq("financing")
        & cases["offset_channel_pattern"].eq("cff_dominant")
        & cases["cfo_adjustment_direction"].eq("audited_cfo_decrease")
        & cases["dominant_line_item"].eq("cff_borrowing_proceeds"),
        int(quotas.get("cff_down_borrowing", 12)),
        2,
        "CFF-down borrowing-proceeds mechanism",
        ["absolute_aggregate_change"],
    )
    add_bucket(
        cases["section"].eq("investing")
        & cases["offset_channel_pattern"].eq("cfi_dominant")
        & cases["cfo_adjustment_direction"].eq("audited_cfo_increase")
        & cases["dominant_line_item"].eq("cfi_ppe_purchase"),
        int(quotas.get("cfi_up_ppe", 10)),
        3,
        "CFI-up PPE-purchase mechanism",
        ["absolute_aggregate_change"],
    )
    add_bucket(
        cases["section"].eq("investing")
        & cases["offset_channel_pattern"].eq("cfi_dominant")
        & cases["cfo_adjustment_direction"].eq("audited_cfo_increase")
        & cases["dominant_line_item"].eq("cfi_loans_advanced"),
        int(quotas.get("cfi_up_loans", 10)),
        3,
        "CFI-up loans-advanced mechanism",
        ["absolute_aggregate_change"],
    )
    add_bucket(
        pd.Series(True, index=cases.index),
        int(quotas.get("reconciliation_outliers", 10)),
        1,
        "Largest reconciliation exception",
        ["absolute_reconciliation_residual", "mapped_share_gap"],
    )

    force_rows: list[pd.DataFrame] = []
    for forced in cfg.get("force_cases", []):
        ticker = str(forced.get("issuer_ticker", "")).upper()
        year = int(forced.get("fiscal_year"))
        hit = cases[
            cases["issuer_ticker"].astype(str).str.upper().eq(ticker)
            & pd.to_numeric(cases["fiscal_year"], errors="coerce").eq(year)
        ].copy()
        if hit.empty:
            force_rows.append(
                pd.DataFrame(
                    [
                        {
                            "issuer_ticker": ticker,
                            "fiscal_year": year,
                            "verification_priority": 0,
                            "selection_reason": "Forced case not present in common-primary core sample",
                            "selection_status": "NOT_IN_SAMPLE",
                        }
                    ]
                )
            )
        else:
            hit["verification_priority"] = 0
            hit["selection_reason"] = forced.get(
                "reason", "Prespecified forced verification case"
            )
            hit["selection_status"] = "SELECTED"
            force_rows.append(hit)

    frames = force_rows + selected
    if not frames:
        return pd.DataFrame()
    manifest = pd.concat(frames, ignore_index=True, sort=False)
    if "selection_status" not in manifest:
        manifest["selection_status"] = "SELECTED"
    else:
        manifest["selection_status"] = manifest["selection_status"].fillna(
            "SELECTED"
        )
    manifest = manifest.sort_values(
        ["verification_priority", "absolute_aggregate_change"],
        ascending=[True, False],
        na_position="last",
    )
    dedup_keys = ["issuer_ticker", "fiscal_year", "section"]
    present_keys = [column for column in dedup_keys if column in manifest]
    manifest = manifest.drop_duplicates(present_keys, keep="first")
    manifest["document_checked"] = False
    manifest["verification_result"] = "PENDING"
    manifest["source_document"] = ""
    manifest["reviewer_notes"] = ""
    preferred = [
        "verification_priority",
        "issuer_ticker",
        "fiscal_year",
        "section",
        "selection_reason",
        "selection_status",
        "offset_channel_pattern",
        "cfo_adjustment_direction",
        "dominant_line_item",
        "aggregate_section_change_scaled",
        "dominant_line_item_change_scaled",
        "mapped_share_of_aggregate",
        "reconciliation_residual_scaled",
        "document_checked",
        "verification_result",
        "source_document",
        "reviewer_notes",
    ]
    columns = [column for column in preferred if column in manifest.columns]
    return manifest[columns].reset_index(drop=True)


def completion_gate_status(
    estimation_status: pd.DataFrame,
    history_comparison: pd.DataFrame,
    primary_reconciliation: pd.DataFrame,
    verification_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "gate": "nonfinancial_estimation_sample",
            "status": "PASS" if not estimation_status.empty else "FAILED",
            "evidence_rows": len(estimation_status),
        },
        {
            "gate": "nested_history_incremental_test",
            "status": "PASS" if not history_comparison.empty else "FAILED",
            "evidence_rows": len(history_comparison),
        },
        {
            "gate": "common_primary_core_reconciliation",
            "status": "PASS" if not primary_reconciliation.empty else "FAILED",
            "evidence_rows": len(primary_reconciliation),
        },
        {
            "gate": "pdf_verification_manifest",
            "status": "PASS" if not verification_manifest.empty else "FAILED",
            "evidence_rows": len(verification_manifest),
        },
        {
            "gate": "scale_scope_screening",
            "status": "WAIVED_BY_DESIGN",
            "evidence_rows": 0,
        },
    ]
    return pd.DataFrame(rows)
