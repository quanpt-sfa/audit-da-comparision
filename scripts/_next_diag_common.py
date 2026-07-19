from __future__ import annotations

import sys
from pathlib import Path

# Support direct execution from a src-layout repository without requiring
# `pip install -e .` or a manually configured PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import yaml


def load_config(path: str | Path):
    path = Path(path).resolve()
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(config_path: Path, value: str | Path) -> Path:
    value = Path(value)
    return value if value.is_absolute() else (config_path.parent.parent / value).resolve()
