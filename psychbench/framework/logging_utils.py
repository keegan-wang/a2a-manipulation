"""JSONL logger (one record per line) and summary JSON writer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> "JsonlLogger":
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlLogger used outside of context manager")
        self._fh.write(json.dumps(record, default=_json_default) + "\n")
        self._fh.flush()


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, default=_json_default))


def _json_default(o: Any) -> Any:
    from dataclasses import asdict, is_dataclass
    from enum import Enum

    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Enum):
        return o.value
    raise TypeError(f"Not JSON serializable: {type(o).__name__}")
