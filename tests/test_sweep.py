"""Unit tests for framework.sweep — cross-product config enumeration."""
from __future__ import annotations

import pytest

from psychbench.framework.sweep import SweepCell, expand_sweep


def _base():
    return {
        "experiment": {"name": "x", "trials": 12},
        "agents": {"n_confederates": 5},
        "documents": {"document_type": "wikipedia"},
        "environment": {"vis": "public"},
        "irrelevant_list": [2, 3, 5],
    }


def test_scalar_only_config_yields_one_cell():
    cells = expand_sweep(_base(), sweep_fields=["agents.n_confederates"])
    assert len(cells) == 1
    assert isinstance(cells[0], SweepCell)
    assert cells[0].config["agents"]["n_confederates"] == 5
    assert cells[0].sweep_values == {"agents.n_confederates": 5}


def test_single_list_field_expands():
    cfg = _base()
    cfg["agents"]["n_confederates"] = [1, 3, 5]
    cells = expand_sweep(cfg, sweep_fields=["agents.n_confederates"])
    assert [c.config["agents"]["n_confederates"] for c in cells] == [1, 3, 5]
    assert [c.sweep_values["agents.n_confederates"] for c in cells] == [1, 3, 5]


def test_two_list_fields_cartesian_product():
    cfg = _base()
    cfg["agents"]["n_confederates"] = [1, 3]
    cfg["documents"]["document_type"] = ["wiki", "forum"]
    cells = expand_sweep(
        cfg,
        sweep_fields=[
            "agents.n_confederates",
            "documents.document_type",
        ],
    )
    assert len(cells) == 4
    pairs = [
        (c.sweep_values["agents.n_confederates"],
         c.sweep_values["documents.document_type"])
        for c in cells
    ]
    assert pairs == [(1, "wiki"), (1, "forum"), (3, "wiki"), (3, "forum")]


def test_non_sweep_list_passes_through_untouched():
    cfg = _base()
    cfg["agents"]["n_confederates"] = [1, 3]
    cells = expand_sweep(cfg, sweep_fields=["agents.n_confederates"])
    for cell in cells:
        assert cell.config["irrelevant_list"] == [2, 3, 5]


def test_missing_sweep_path_raises_keyerror_with_path():
    with pytest.raises(KeyError, match="agents.n_gadgets"):
        expand_sweep(_base(), sweep_fields=["agents.n_gadgets"])


def test_cell_id_is_deterministic():
    cfg = _base()
    cfg["agents"]["n_confederates"] = [1, 3]
    cfg["documents"]["document_type"] = ["wiki", "forum"]
    cells = expand_sweep(
        cfg,
        sweep_fields=[
            "agents.n_confederates",
            "documents.document_type",
        ],
    )
    assert cells[0].cell_id == "n_confederates=1__document_type=wiki"
    assert cells[1].cell_id == "n_confederates=1__document_type=forum"
    assert cells[2].cell_id == "n_confederates=3__document_type=wiki"
    assert cells[3].cell_id == "n_confederates=3__document_type=forum"


def test_cell_config_is_deep_copy():
    cfg = _base()
    cfg["agents"]["n_confederates"] = [1, 3]
    cells = expand_sweep(cfg, sweep_fields=["agents.n_confederates"])
    cells[0].config["agents"]["n_confederates"] = 999
    assert cells[1].config["agents"]["n_confederates"] == 3


def test_boolean_values_expand():
    cfg = _base()
    cfg["agents"]["dissenter"] = [False, True]
    cells = expand_sweep(cfg, sweep_fields=["agents.dissenter"])
    assert [c.sweep_values["agents.dissenter"] for c in cells] == [False, True]
    assert cells[0].cell_id == "dissenter=false"
    assert cells[1].cell_id == "dissenter=true"
