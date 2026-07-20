from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .auditor_source import load_auditor_firm_year_resilient


def load_auditor_firm_year_safe(
    source_paths: list[Path],
    settings: dict[str, Any],
    audited_label: str = "audited",
    required_scope: str | None = "consolidated",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load auditor data without allowing zero-row schema filters to crash.

    Dedicated annual audit tables are already audit-specific. Some of them
    nevertheless expose generic `audit_status` or `scope` fields whose values do
    not match the CFS labels. The legacy wide loader could therefore filter all
    rows and then fail while constructing the name map. This wrapper retries the
    same sources without those two filters, then degrades to NOT_EVALUATED.
    """
    first_error = ""
    try:
        return load_auditor_firm_year_resilient(
            source_paths,
            settings,
            audited_label=audited_label,
            required_scope=required_scope,
        )
    except (KeyError, ValueError) as exc:
        first_error = f"{type(exc).__name__}: {exc}"

    retry = dict(settings)
    retry["audit_status_column_candidates"] = []
    retry["scope_column_candidates"] = []
    retry.pop("audit_status_column", None)
    retry.pop("scope_column", None)
    try:
        firm_year, name_map, status = load_auditor_firm_year_resilient(
            source_paths,
            retry,
            audited_label=audited_label,
            required_scope=None,
        )
        if not firm_year.empty:
            status = status.copy()
            status["filter_strategy"] = "RETRY_WITHOUT_AUDIT_STATUS_OR_SCOPE"
            status["initial_error"] = first_error
        return firm_year, name_map, status
    except Exception as exc:  # pragma: no cover - defensive terminal guard
        rows = [
            {
                "overall_status": "NOT_EVALUATED",
                "status": "LOAD_ERROR",
                "path": str(path),
                "initial_error": first_error,
                "retry_error": f"{type(exc).__name__}: {exc}",
            }
            for path in source_paths
        ]
        if not rows:
            rows = [
                {
                    "overall_status": "NOT_EVALUATED",
                    "status": "NO_SOURCE_PATHS",
                    "initial_error": first_error,
                    "retry_error": f"{type(exc).__name__}: {exc}",
                }
            ]
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(rows)
