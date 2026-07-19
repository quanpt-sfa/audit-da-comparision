from .cfs_item_map import classify_cfs_item, compile_item_rules, inventory_and_line_items
from .cfs_proxy_validate import (
    rolling_expected_cfo_proxies,
    validate_proxy_predictions as validate_proxy_predictions_legacy,
)
from .cfs_proxy_validate_samples import validate_proxy_predictions_dual_common
from .cfs_line_reconcile import line_item_reconciliation


def _apply_scale_scope_waiver(tables, settings):
    restrictions = settings.get("sample_restrictions", {})
    if restrictions.get("require_scale_scope_screening", True):
        return tables

    status = tables.get("cfs_proxy_sample_restriction_status")
    if status is None or status.empty:
        return tables

    status = status.copy()
    reason = restrictions.get(
        "scale_scope_waiver_reason",
        "Scale/scope screening waived by the locked data-source design.",
    )
    scale_row = status["sample_restriction"].eq("exclude_scale_scope_flags")
    status.loc[scale_row, "status"] = "WAIVED_BY_DESIGN"
    status.loc[scale_row, "reason"] = reason

    nonfinancial = status[
        status["sample_restriction"].eq("nonfinancial_only")
    ]
    financial_ready = (
        not nonfinancial.empty
        and nonfinancial["status"].iloc[0] == "EVALUATED"
    )
    core_row = status["sample_restriction"].eq("analysis_core")
    if financial_ready:
        status.loc[core_row, "status"] = "EVALUATED"
        status.loc[core_row, "reason"] = (
            "All required sample restrictions evaluated; scale/scope screening "
            "waived because preliminary and audited records share the same "
            "source, monetary unit, reporting scope, and period convention."
        )
    else:
        status.loc[core_row, "status"] = "PARTIALLY_EVALUATED"
        status.loc[core_row, "reason"] = (
            "Non-financial classification unavailable; scale/scope screening "
            "waived by design."
        )
    tables["cfs_proxy_sample_restriction_status"] = status
    return tables


def validate_proxy_predictions(predictions, observed_cases, settings):
    """Dispatch to the validation engine implied by the configuration schema.

    Production configurations define ``common_primary_models`` and/or
    ``common_all_models`` and therefore use the strict dual-common-sample
    engine. Older callers and focused tests that define only
    ``common_sample_models`` retain the legacy ``common_models`` output and
    its historical missing-metadata behavior.
    """
    uses_dual_common = (
        "common_primary_models" in settings
        or "common_all_models" in settings
    )
    if uses_dual_common:
        tables = validate_proxy_predictions_dual_common(
            predictions, observed_cases, settings
        )
        return _apply_scale_scope_waiver(tables, settings)
    return validate_proxy_predictions_legacy(
        predictions, observed_cases, settings
    )


def run_cfs_shifting_validation(panel, observed_cases, line_item_panel, settings):
    predictions, folds = rolling_expected_cfo_proxies(panel, settings)
    output = {
        "cfs_expected_cfo_predictions": predictions,
        "cfs_expected_cfo_folds": folds,
    }
    output.update(
        validate_proxy_predictions(predictions, observed_cases, settings)
    )
    output.update(
        line_item_reconciliation(
            line_item_panel, observed_cases, panel, settings
        )
    )
    return output


__all__ = [
    "classify_cfs_item",
    "compile_item_rules",
    "inventory_and_line_items",
    "rolling_expected_cfo_proxies",
    "validate_proxy_predictions",
    "line_item_reconciliation",
    "run_cfs_shifting_validation",
]
