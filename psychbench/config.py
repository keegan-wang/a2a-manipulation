"""Load and minimally validate YAML experiment configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_KEYS = ("experiment", "agents", "environment")


def load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg
