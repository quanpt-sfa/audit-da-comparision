from __future__ import annotations

import gzip
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd


def materialize_csv_gz(input_path: str | Path, work_dir: str | Path) -> Path:
    """Return a local .csv.gz path from .csv.gz, .zip containing .csv.gz, or .csv."""
    input_path = Path(input_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if input_path.suffix == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one file in ZIP, found {members}")
            member = members[0]
            target = work_dir / Path(member).name
            if not target.exists() or target.stat().st_size != archive.getinfo(member).file_size:
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            return target

    if input_path.name.endswith(".csv.gz"):
        return input_path

    if input_path.suffix == ".csv":
        target = work_dir / f"{input_path.name}.gz"
        with input_path.open("rb") as source, gzip.open(target, "wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        return target

    raise ValueError(f"Unsupported input format: {input_path}")


def read_long_chunks(path: str | Path, chunksize: int, usecols: list[str] | None = None) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        compression="gzip" if str(path).endswith(".gz") else "infer",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )


def write_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
