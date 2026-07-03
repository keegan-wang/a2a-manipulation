"""Repeated sampling + the model-panel leaderboard (hermetic, echo backends)."""
from __future__ import annotations

import json
from pathlib import Path

from psychbench.config import load_config
from psychbench.experiments.conformity.challenge import (
    ConformityChallengeExperiment,
)
from psychbench.experiments.conformity.panel import (
    estimate_panel_calls, format_leaderboard, run_panel,
)

SMOKE = "config/experiments/conformity_story_panel_smoke.yaml"


def test_n_repeats_pools_samples(tmp_path):
    cfg = load_config(SMOKE)
    cfg["experiment"]["n_repeats"] = 3
    summary = ConformityChallengeExperiment(cfg).run(output_dir=tmp_path)
    assert summary["n_repeats"] == 3
    rows = [
        json.loads(line)
        for line in (Path(summary["run_dir"]) / "challenge_neutral.jsonl")
        .read_text().splitlines()
    ]
    assert len(rows) == 4 * 3          # trials x repeats, pooled
    assert all("repeat" in r for r in rows)
    # the summary carries a standard error once samples exist
    assert "induced_hallucination_se" in summary["headline"]["arms"]["neutral"]


def test_panel_runs_and_builds_leaderboard(tmp_path):
    cfg = load_config(SMOKE)
    lb = run_panel(cfg, output_dir=tmp_path)
    assert lb["n_models"] == 2
    assert {r["model"] for r in lb["rows"]} == {"echo-a", "echo-b"}
    for r in lb["rows"]:
        assert "induced_neutral" in r and "induced_reflect" in r
        assert "baseline_accuracy_alone" in r
    assert (Path(lb["panel_dir"]) / "leaderboard.json").exists()
    assert "echo-a" in format_leaderboard(lb)


def test_estimate_panel_calls():
    cfg = load_config(SMOKE)
    # 2 models x 2 arms x 4 trials x (2 rounds + 1 commit) x 1 repeat
    assert estimate_panel_calls(cfg) == 2 * 2 * 4 * 3 * 1
