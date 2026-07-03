"""Generic cross-product config enumerator for IV sweeps."""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SweepCell:
    config: dict[str, Any]
    cell_id: str
    sweep_values: dict[str, Any] = field(default_factory=dict)


def expand_sweep(
    config: dict[str, Any],
    sweep_fields: list[str],
) -> list[SweepCell]:
    """Enumerate the Cartesian product of list-valued sweep fields.

    Only fields listed in ``sweep_fields`` are expanded; any other list-valued
    fields in ``config`` pass through untouched. A missing path raises
    ``KeyError`` with the offending path.
    """
    per_field_values: list[list[Any]] = []
    for path in sweep_fields:
        value = _get_path(config, path)
        if isinstance(value, list):
            per_field_values.append(list(value))
        else:
            per_field_values.append([value])

    cells: list[SweepCell] = []
    for combo in itertools.product(*per_field_values):
        cell_config = copy.deepcopy(config)
        sweep_values: dict[str, Any] = {}
        id_parts: list[str] = []
        for path, value in zip(sweep_fields, combo):
            _set_path(cell_config, path, value)
            sweep_values[path] = value
            basename = path.rsplit(".", 1)[-1]
            id_parts.append(f"{basename}={_format_value(value)}")
        cell_id = "__".join(id_parts)
        cells.append(
            SweepCell(
                config=cell_config,
                cell_id=cell_id,
                sweep_values=sweep_values,
            )
        )
    return cells


def _get_path(cfg: dict[str, Any], path: str) -> Any:
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _set_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
