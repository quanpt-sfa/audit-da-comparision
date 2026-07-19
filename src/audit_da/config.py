from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    value = Path(value)
    if value.is_absolute():
        return value
    return (config_path.parent.parent / value).resolve()
