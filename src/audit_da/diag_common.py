from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["issuer_ticker", "fiscal_year"]


def trimmed_mean(values: np.ndarray, trim: float) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values): return float("nan")
    if trim <= 0: return float(values.mean())
    cut = int(np.floor(len(values) * trim))
    if 2 * cut >= len(values): return float("nan")
    return float(np.sort(values)[cut:len(values)-cut].mean())


def paired_panel(panel: pd.DataFrame, audited="audited", unaudited="unaudited") -> pd.DataFrame:
    missing = set(KEYS + ["audit_status"]) - set(panel.columns)
    if missing: raise ValueError(f"Panel missing columns: {sorted(missing)}")
    pre = panel[panel.audit_status.eq(unaudited)].drop_duplicates(KEYS).copy()
    post = panel[panel.audit_status.eq(audited)].drop_duplicates(KEYS).copy()
    shared = sorted((set(pre.columns) & set(post.columns)) - set(KEYS + ["audit_status"]))
    pre = pre[KEYS + shared].rename(columns={c: f"{c}_pre" for c in shared})
    post = post[KEYS + shared].rename(columns={c: f"{c}_post" for c in shared})
    return pre.merge(post, on=KEYS, how="inner", validate="one_to_one")


def write_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        gz = name.endswith("_cases") or len(frame) > 100_000
        path = output / f"{name}.csv{'.gz' if gz else ''}"
        frame.to_csv(path, index=False, compression="gzip" if gz else None)
        print(f"Wrote {path}")
