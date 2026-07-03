"""Faithful Asch (1956) replication: fixed stimulus set, procedure, pipeline.

Echo can't judge line lengths (it returns the last A/B/C it sees), so the
end-to-end tests use small *perceptual* fake backends that actually compare the
numbers — a conformist that copies the announced majority and a stubborn one
that trusts its own eyes — to exercise the alone/group + induced-hallucination
pipeline deterministically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from psychbench.config import load_config
from psychbench.framework.types import Stimulus, TrialContext
from psychbench.agents.scripted_agent import ScriptedAgent
from psychbench.experiments.conformity import experiment as exp_mod
from psychbench.experiments.conformity.experiment import ConformityExperiment
from psychbench.experiments.conformity.prompts import build_conformity_prompt
from psychbench.experiments.conformity.asch_classic import (
    ASCH_1956_TRIALS, ASCH_GROUP_SIZE, N_CRITICAL, N_TRIALS,
    build_asch_classic_stimuli, validate_trials,
)


# ----------------------------------------------------------------- stimulus set
def test_table_invariants_hold():
    validate_trials()  # raises if the fixed table drifts
    assert len(ASCH_1956_TRIALS) == N_TRIALS == 18
    crit = [t for t in ASCH_1956_TRIALS if t["majority"] is not None]
    assert len(crit) == N_CRITICAL == 12
    # first two trials neutral (Asch establishes normalcy before pressure)
    assert ASCH_1956_TRIALS[0]["majority"] is None
    assert ASCH_1956_TRIALS[1]["majority"] is None


def test_correct_and_majority_labels_vary():
    # The B-confound fix: neither the correct line nor the conformity target is
    # pinned to one letter.
    correct = {t["correct"] for t in ASCH_1956_TRIALS}
    majority = {t["majority"] for t in ASCH_1956_TRIALS if t["majority"]}
    assert correct == {"A", "B", "C"}
    assert majority == {"A", "B", "C"}


def test_builder_marks_critical_and_carries_target():
    stims = build_asch_classic_stimuli()
    assert len(stims) == 18
    assert sum(s.is_critical for s in stims) == 12
    for s, t in zip(stims, ASCH_1956_TRIALS):
        assert s.metadata["task_type"] == "lines_classic"
        # the matching comparison line is exactly the correct label
        assert s.options[s.correct_label] == s.reference_cm
        if s.is_critical:
            assert s.wrong_label == t["majority"] != s.correct_label
        else:
            assert s.wrong_label is None


# ----------------------------------------------------------------- prompt
def _ctx(stim: Stimulus, priors=None) -> TrialContext:
    return TrialContext(
        stimulus=stim, agent_position=ASCH_GROUP_SIZE, agent_id="naive",
        prior_responses=priors or [],
    )


def test_classic_prompt_is_two_card_vision_test():
    stim = build_asch_classic_stimuli()[2]  # a critical trial
    p = build_conformity_prompt(_ctx(stim))
    assert "visual perception test" in p
    assert "standard line" in p.lower()
    assert "Line A:" in p and "Line B:" in p and "Line C:" in p
    assert "aloud" in p  # public/oral framing
    # alone (no priors): the group's answers are absent
    assert "Participant" not in p


def test_classic_prompt_announces_majority_in_order():
    stim = build_asch_classic_stimuli()[2]
    priors = [(f"confederate_{i+1}", stim.wrong_label) for i in range(7)]
    p = build_conformity_prompt(_ctx(stim, priors))
    assert p.count("Participant ") == 7
    assert "announced their answers aloud, in order" in p
    assert f"Participant 1 says: {stim.wrong_label}" in p


# --------------------------------------------------------------- confederates
def test_confederates_unanimous_wrong_on_critical_correct_on_neutral():
    stims = build_asch_classic_stimuli()
    conf = ScriptedAgent("c1", position=0, behavior="stimulus_wrong_on_critical")
    for s in stims:
        ans = conf.respond(_ctx(s)).parsed_answer
        if s.is_critical:
            assert ans == s.wrong_label != s.correct_label
        else:
            assert ans == s.correct_label


# --------------------------------------------------------------- perceptual fakes
def _match_line(prompt: str) -> str:
    """Pick the comparison line whose length equals the standard."""
    std = float(re.search(r"Standard line:\s*([\d.]+)\s*cm", prompt).group(1))
    for lab, val in re.findall(r"Line ([ABC]):\s*([\d.]+)\s*cm", prompt):
        if abs(float(val) - std) < 1e-6:
            return lab
    return "A"


class _ConformistPerceptual:
    """Trusts its eyes alone; copies the announced majority when present."""
    def __init__(self, model="fake", **_):
        self.model = model

    def generate(self, prompt: str, stateful: bool = False) -> str:
        said = re.findall(r"Participant \d+ says:\s*([ABC])", prompt)
        if said:
            return max(set(said), key=said.count)  # modal peer answer
        return _match_line(prompt)

    def reset(self):
        pass


class _StubbornPerceptual(_ConformistPerceptual):
    def generate(self, prompt: str, stateful: bool = False) -> str:
        return _match_line(prompt)  # ignores peers entirely


@pytest.fixture
def smoke_cfg():
    return load_config("config/experiments/asch_classic_smoke.yaml")


def test_end_to_end_conformist_induces_hallucination(tmp_path, monkeypatch, smoke_cfg):
    monkeypatch.setattr(exp_mod, "get_backend",
                        lambda *a, **k: _ConformistPerceptual())
    summary = ConformityExperiment(smoke_cfg).run(output_dir=tmp_path)
    h = summary["headline"]
    # knows every line alone, but copies the unanimous-wrong majority -> 100%
    assert h["baseline_accuracy_alone"] == 1.0
    assert h["group_conformity_rate"] == 1.0
    assert h["induced_hallucination_rate"] == 1.0
    assert summary["config"]["agents"]["confederates"]["count"] == ASCH_GROUP_SIZE


def test_end_to_end_stubborn_resists(tmp_path, monkeypatch, smoke_cfg):
    monkeypatch.setattr(exp_mod, "get_backend",
                        lambda *a, **k: _StubbornPerceptual())
    summary = ConformityExperiment(smoke_cfg).run(output_dir=tmp_path)
    h = summary["headline"]
    assert h["baseline_accuracy_alone"] == 1.0
    assert h["group_conformity_rate"] == 0.0
    assert h["induced_hallucination_rate"] == 0.0


def test_default_group_size_is_seven(smoke_cfg):
    # even if a config omits count, lines_classic uses Asch's majority of 7
    smoke_cfg["agents"]["confederates"].pop("count", None)
    agents = ConformityExperiment(smoke_cfg).build_agents()
    confederates = [a for a in agents if a.agent_id.startswith("confederate")]
    assert len(confederates) == ASCH_GROUP_SIZE


def test_smoke_run_writes_both_conditions(tmp_path, smoke_cfg):
    # echo plumbing run: artifacts only, but proves files + structure exist
    summary = ConformityExperiment(smoke_cfg).run(output_dir=tmp_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "conformity_alone.jsonl").exists()
    assert (run_dir / "conformity_group.jsonl").exists()
    group = json.loads((run_dir / "conformity_group.summary.json").read_text())
    assert group["n_critical"] == 12
