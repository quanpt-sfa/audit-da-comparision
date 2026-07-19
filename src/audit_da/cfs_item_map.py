from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import unicodedata

import numpy as np
import pandas as pd

from .diag_common import KEYS
from .io import read_long_chunks

RAW_KEYS = ["issuer_ticker", "raw_exchange", "fiscal_year", "audit_status", "scope"]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _normalise_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def compile_item_rules(settings: dict[str, Any]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for rule in settings.get("line_item_rules", []):
        include = [re.compile(p, re.IGNORECASE) for p in rule.get("include", [])]
        exclude = [re.compile(p, re.IGNORECASE) for p in rule.get("exclude", [])]
        if include:
            rules.append({"concept": rule["concept"], "section": rule["section"], "include": include, "exclude": exclude})
    return rules


def classify_cfs_item(source_item_id: Any, item_name_raw: Any, statement_family: Any, rules: list[dict[str, Any]]) -> tuple[str | None, str | None, int]:
    text = " | ".join(_normalise_text(v) for v in (source_item_id, item_name_raw, statement_family))
    matched = [r for r in rules if any(p.search(text) for p in r["include"]) and not any(p.search(text) for p in r["exclude"])]
    if len(matched) == 1:
        return str(matched[0]["concept"]), str(matched[0]["section"]), 1
    return None, None, len(matched)


def inventory_and_line_items(raw_path: str | Path, settings: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rules = compile_item_rules(settings)
    audited = settings.get("audited_label", "audited")
    unaudited = settings.get("unaudited_label", "unaudited")
    usecols = RAW_KEYS + [
        "statement_family", "source_item_id", "item_name_raw", "value_numeric",
        "identity_match_status", "retrospective_eligible", "prospective_flag",
    ]
    inventories: list[pd.DataFrame] = []
    mapped_parts: list[pd.DataFrame] = []
    for chunk in read_long_chunks(raw_path, int(settings.get("chunksize", 250_000)), usecols):
        year = pd.to_numeric(chunk["fiscal_year"], errors="coerce")
        mask = (
            chunk["statement_family"].fillna("").astype(str).str.contains(settings.get("statement_family_regex", r"cash[_ ]?flow"), case=False, regex=True)
            & chunk["scope"].eq(settings.get("required_scope", "consolidated"))
            & chunk["identity_match_status"].isin(settings.get("allowed_identity_status", ["exact", "casefold_match"]))
            & chunk["retrospective_eligible"].astype(str).eq("1")
            & chunk["prospective_flag"].astype(str).eq("0")
            & year.between(int(settings.get("minimum_year", 2018)), int(settings.get("maximum_year", 2025)))
            & chunk["audit_status"].isin([audited, unaudited])
        )
        part = chunk.loc[mask].copy()
        if part.empty:
            continue
        part["fiscal_year"] = year.loc[mask].astype(int)
        part["value_numeric"] = pd.to_numeric(part["value_numeric"], errors="coerce")
        classified = [classify_cfs_item(*row, rules) for row in part[["source_item_id", "item_name_raw", "statement_family"]].itertuples(index=False, name=None)]
        part["concept"] = [x[0] for x in classified]
        part["concept_section"] = [x[1] for x in classified]
        part["rule_match_count"] = [x[2] for x in classified]
        inventories.append(
            part.groupby(["source_item_id", "item_name_raw", "statement_family"], dropna=False, observed=True)
            .agg(
                rows=("issuer_ticker", "size"), nonmissing_rows=("value_numeric", "count"),
                audited_rows=("audit_status", lambda x: int(x.eq(audited).sum())),
                unaudited_rows=("audit_status", lambda x: int(x.eq(unaudited).sum())),
                minimum_year=("fiscal_year", "min"), maximum_year=("fiscal_year", "max"),
                mapped_concept=("concept", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
                mapped_section=("concept_section", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
                maximum_rule_matches=("rule_match_count", "max"),
            ).reset_index()
        )
        mapped = part[part["concept"].notna() & part["value_numeric"].notna() & part["rule_match_count"].eq(1)]
        if not mapped.empty:
            mapped_parts.append(mapped[RAW_KEYS + ["statement_family", "concept", "concept_section", "value_numeric"]])

    inventory = pd.DataFrame()
    if inventories:
        inventory = (
            pd.concat(inventories, ignore_index=True)
            .groupby(["source_item_id", "item_name_raw", "statement_family"], dropna=False, observed=True)
            .agg(
                rows=("rows", "sum"), nonmissing_rows=("nonmissing_rows", "sum"),
                audited_rows=("audited_rows", "sum"), unaudited_rows=("unaudited_rows", "sum"),
                minimum_year=("minimum_year", "min"), maximum_year=("maximum_year", "max"),
                mapped_concept=("mapped_concept", lambda x: "|".join(sorted(set(filter(None, x))))),
                mapped_section=("mapped_section", lambda x: "|".join(sorted(set(filter(None, x))))),
                maximum_rule_matches=("maximum_rule_matches", "max"),
            ).reset_index()
        )
        inventory["mapping_status"] = np.select(
            [inventory["maximum_rule_matches"].gt(1), inventory["mapped_concept"].ne("")],
            ["ambiguous", "mapped"], default="unmapped",
        )
    if not mapped_parts:
        return {
            "cfs_item_inventory": inventory,
            "cfs_item_mapping_review": inventory.copy(),
            "cfs_line_item_long": pd.DataFrame(),
            "cfs_line_item_panel": pd.DataFrame(),
            "cfs_line_item_method_coverage": pd.DataFrame(),
        }

    long = pd.concat(mapped_parts, ignore_index=True)
    long = long.groupby(RAW_KEYS + ["statement_family", "concept", "concept_section"], as_index=False, observed=True)["value_numeric"].sum(min_count=1)
    method = long.groupby(RAW_KEYS + ["statement_family"], observed=True)["concept"].nunique().rename("mapped_concepts").reset_index()
    preference = {name: rank for rank, name in enumerate(settings.get("statement_family_preference", ["cash_flow_indirect", "cash_flow_direct"]))}
    method["preference_rank"] = method["statement_family"].map(preference).fillna(len(preference))
    chosen = (
        method.sort_values(RAW_KEYS + ["mapped_concepts", "preference_rank"], ascending=[True, True, True, True, True, False, True])
        .drop_duplicates(RAW_KEYS)[RAW_KEYS + ["statement_family", "mapped_concepts"]]
        .rename(columns={"statement_family": "selected_statement_family"})
    )
    selected = long.merge(chosen, on=RAW_KEYS, how="inner", validate="many_to_one")
    selected = selected[selected["statement_family"].eq(selected["selected_statement_family"])]
    wide = selected.pivot_table(index=RAW_KEYS + ["selected_statement_family"], columns="concept", values="value_numeric", aggfunc="sum").reset_index()
    wide.columns.name = None
    coverage = chosen.groupby(["audit_status", "selected_statement_family"], observed=True).agg(rows=("issuer_ticker", "size"), median_mapped_concepts=("mapped_concepts", "median")).reset_index()
    coverage["share_within_audit_status"] = coverage["rows"] / coverage.groupby("audit_status")["rows"].transform("sum")
    return {
        "cfs_item_inventory": inventory,
        "cfs_item_mapping_review": inventory[inventory["mapping_status"].ne("mapped")].copy(),
        "cfs_line_item_long": selected,
        "cfs_line_item_panel": wide,
        "cfs_line_item_method_coverage": coverage,
    }


def pair_line_items(line_item_panel: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    if line_item_panel.empty:
        return pd.DataFrame()
    pre = line_item_panel[line_item_panel["audit_status"].eq(settings.get("unaudited_label", "unaudited"))].drop_duplicates(KEYS)
    post = line_item_panel[line_item_panel["audit_status"].eq(settings.get("audited_label", "audited"))].drop_duplicates(KEYS)
    concepts = sorted((set(pre.columns) | set(post.columns)) - set(RAW_KEYS + ["selected_statement_family"]))
    pre_cols, post_cols = [c for c in concepts if c in pre], [c for c in concepts if c in post]
    paired = pre[KEYS + pre_cols].rename(columns={c: f"{c}_pre" for c in pre_cols}).merge(
        post[KEYS + post_cols].rename(columns={c: f"{c}_post" for c in post_cols}),
        on=KEYS, how="inner", validate="one_to_one",
    )
    for concept in concepts:
        paired[f"delta_{concept}"] = _numeric(paired, f"{concept}_post") - _numeric(paired, f"{concept}_pre")
    return paired
