from .cfs_item_map import classify_cfs_item, compile_item_rules, inventory_and_line_items
from .cfs_proxy_validate import rolling_expected_cfo_proxies
from .cfs_proxy_validate_samples import validate_proxy_predictions_dual_common
from .cfs_line_reconcile import line_item_reconciliation


def validate_proxy_predictions(predictions, observed_cases, settings):
    return validate_proxy_predictions_dual_common(predictions, observed_cases, settings)


def run_cfs_shifting_validation(panel, observed_cases, line_item_panel, settings):
    predictions, folds = rolling_expected_cfo_proxies(panel, settings)
    output = {"cfs_expected_cfo_predictions": predictions, "cfs_expected_cfo_folds": folds}
    output.update(validate_proxy_predictions(predictions, observed_cases, settings))
    output.update(line_item_reconciliation(line_item_panel, observed_cases, panel, settings))
    return output


__all__ = [
    "classify_cfs_item", "compile_item_rules", "inventory_and_line_items",
    "rolling_expected_cfo_proxies", "validate_proxy_predictions",
    "line_item_reconciliation", "run_cfs_shifting_validation",
]
