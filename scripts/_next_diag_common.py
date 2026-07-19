from __future__ import annotations
from pathlib import Path
import yaml


def load_config(path: str | Path):
    path = Path(path).resolve()
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(config_path: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else (config_path.parent.parent / value).resolve()
