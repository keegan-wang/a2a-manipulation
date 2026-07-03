"""End-to-end (hermetic, echo backend) for the conformity clone."""
from __future__ import annotations

import json
from pathlib import Path

from psychbench.config import load_config
from psychbench.experiments.conformity.experiment import ConformityExperiment


def _cfg():
    return load_config("config/experiments/conformity_knowledge_smoke.yaml")


def test_run_produces_headline_and_both_conditions(tmp_path):
    exp = ConformityExperiment(_cfg())
    summary = exp.run(output_dir=tmp_path)

    assert summary["experiment"] == "conformity"
    assert summary["task_type"] == "knowledge"
    headline = summary["headline"]
    for key in (
        "baseline_accuracy_alone", "group_conformity_rate",
        "induced_hallucination_rate", "group_abstention_rate",
    ):
        assert key in headline
        assert 0.0 <= headline[key] <= 1.0

    rep = summary["repeats"][0]
    assert rep["alone"]["condition"] == "alone"
    assert rep["group"]["condition"] == "group"
    assert rep["alone"]["n_critical"] == rep["group"]["n_critical"]
    assert rep["induced"]["n_items"] == rep["group"]["n_critical"]

    run_dir = Path(summary["run_dir"])
    assert (run_dir / "conformity_alone.jsonl").exists()
    assert (run_dir / "conformity_group.jsonl").exists()
    written = json.loads((run_dir / "summary.json").read_text())
    assert written["headline"] == headline


def test_group_naive_sees_peers_alone_does_not(tmp_path):
    exp = ConformityExperiment(_cfg())
    summary = exp.run(output_dir=tmp_path)
    run_dir = Path(summary["run_dir"])

    def naive_prompt(path):
        rec = json.loads(path.read_text().splitlines()[0])
        naive = next(r for r in rec["responses"] if r["agent_id"] == "naive")
        return naive["prompt"]

    assert "Participant 1:" in naive_prompt(run_dir / "conformity_group.jsonl")
    assert "Participant 1:" not in naive_prompt(
        run_dir / "conformity_alone.jsonl"
    )


def test_n_repeats_runs_multiple_seeds(tmp_path):
    cfg = _cfg()
    cfg["experiment"]["n_repeats"] = 2
    summary = ConformityExperiment(cfg).run(output_dir=tmp_path)
    assert summary["n_repeats"] == 2
    assert len(summary["repeats"]) == 2
    assert summary["repeats"][0]["seed"] != summary["repeats"][1]["seed"]


def test_concurrent_run_matches_serial(tmp_path):
    cfg = _cfg()
    serial = ConformityExperiment(cfg).run(
        output_dir=tmp_path / "serial", max_concurrency=1,
    )
    parallel = ConformityExperiment(cfg).run(
        output_dir=tmp_path / "par", max_concurrency=8,
    )
    # Deterministic echo backend ⇒ concurrency must not change results.
    assert parallel["headline"] == serial["headline"]
    assert parallel["max_concurrency"] == 8


def test_resume_skips_completed_sessions(tmp_path):
    cfg = _cfg()
    first = ConformityExperiment(cfg).run(
        output_dir=tmp_path, run_id="fixed", resume=True,
    )
    run_dir = Path(first["run_dir"])
    group_log = run_dir / "conformity_group.jsonl"
    mtime_before = group_log.stat().st_mtime_ns

    # Re-run with the same run_id + resume: completed sessions are loaded, not
    # re-executed, so the group log file is not rewritten.
    second = ConformityExperiment(cfg).run(
        output_dir=tmp_path, run_id="fixed", resume=True,
    )
    assert second["headline"] == first["headline"]
    assert group_log.stat().st_mtime_ns == mtime_before
