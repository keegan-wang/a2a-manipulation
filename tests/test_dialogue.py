"""Live agent-to-agent persuasion: prompts, one item, and the cross-model matrix.

Echo can't argue or read passages, so we use a fake agent that plays all three
roles by prompt: commits the passage answer, argues for the wrong option as a
persuader, and (by default) caves to the group on rebuttal.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from psychbench.config import load_config
from psychbench.experiments.conformity.corpus import load_story_corpus
from psychbench.experiments.conformity.stimuli import build_story_stimuli
from psychbench.experiments.conformity import dialogue as dlg_mod
from psychbench.experiments.conformity.dialogue import (
    ConformityDialogueExperiment, build_persuader_prompt,
    build_subject_commit_prompt, build_subject_rebuttal_prompt,
    format_matrix, run_dialogue_item, run_dialogue_matrix,
)

_CORPUS = "psychbench/experiments/conformity/corpus/story_facts.yaml"


def _stim():
    return build_story_stimuli(load_story_corpus(_CORPUS), seed=0)[0]


def _passage_answer(prompt: str) -> str:
    passage = prompt.split("Question:")[0]
    for lab, text in re.findall(r"^([AB]): (.+)$", prompt, re.MULTILINE):
        if text.strip() and text.strip() in passage:
            return lab
    return "A"


# ---------------------------------------------------------------- prompts
def test_prompts_have_their_roles():
    s = _stim()
    commit = build_subject_commit_prompt(s)
    assert "Read the following passage" in commit and "Answer:" in commit
    pers = build_persuader_prompt(s, "I think it's A.", 0)
    assert s.options[s.wrong_label] in pers          # argues the wrong option
    assert "group discussion" in pers
    reb = build_subject_rebuttal_prompt(s, ["arg one", "arg two", "arg three"])
    assert reb.count("Participant ") == 3
    assert s.wrong_label in reb


def test_authority_persuader_uses_authority_tactics():
    s = _stim()
    peer = build_persuader_prompt(s, "I think A.", 0, "peer")
    auth = build_persuader_prompt(s, "I think A.", 0, "authority")
    assert "fellow participant" in peer
    assert "fellow participant" not in auth
    assert ("authority" in auth.lower()) or ("expert" in auth.lower())
    assert "cite" in auth.lower()                       # invokes citations
    assert s.options[s.wrong_label] in auth             # still argues the wrong answer


# ---------------------------------------------------------------- fake agent
class _FakeAgent:
    def __init__(self, model="fake", caves=True, **_):
        self.model = model
        self.caves = caves
        self.committed = None

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if "Read the following passage" in prompt:           # subject commit
            self.committed = _passage_answer(prompt)
            return f"Reading it, Answer: {self.committed}"
        if "group discussion" in prompt:                      # persuader role
            return "It is clearly the other option; your reasoning is mistaken."
        m = re.search(r"answer is ([AB])", prompt)            # subject rebuttal
        if self.caves and m:
            return f"You're right. Answer: {m.group(1)}"
        return f"I'll stick with it. Answer: {self.committed}"

    def reset(self):
        self.committed = None


def test_one_item_capitulates(monkeypatch):
    s = _stim()
    item = run_dialogue_item(
        s, make_subject=lambda: _FakeAgent(caves=True),
        make_persuader=lambda: _FakeAgent(), n_persuaders=3, n_rounds=2,
    )
    assert item["knew_alone"] is True                # committed the passage answer
    assert item["induced_hallucination"] is True     # caved to the group
    # transcript: commit + 2 rounds, each round with 3 live arguments
    assert len(item["transcript"]) == 3
    assert len(item["transcript"][1]["persuader_args"]) == 3


def test_one_item_emits_live_events_in_order():
    s = _stim()
    events: list[dict[str, Any]] = []

    item = run_dialogue_item(
        s,
        make_subject=lambda: _FakeAgent(caves=True),
        make_persuader=lambda: _FakeAgent(),
        n_persuaders=2,
        n_rounds=1,
        emit_event=events.append,
    )

    assert item["induced_hallucination"] is True
    assert [e["type"] for e in events] == [
        "run_start",
        "subject_commit",
        "round_start",
        "persuader_argument",
        "persuader_argument",
        "subject_response",
        "run_complete",
    ]
    assert events[0]["speaker_role"] == "system"
    assert events[1]["speaker_id"] == "subject"
    assert events[1]["answer"] == item["answers"][0]
    assert events[3]["speaker_id"] == "persuader_1"
    assert events[4]["speaker_id"] == "persuader_2"
    assert events[-1]["summary"]["induced_hallucination"] is True


def test_one_item_holds_firm():
    s = _stim()
    item = run_dialogue_item(
        s, make_subject=lambda: _FakeAgent(caves=False),
        make_persuader=lambda: _FakeAgent(), n_persuaders=3, n_rounds=2,
    )
    assert item["knew_alone"] is True
    assert item["induced_hallucination"] is False


# ---------------------------------------------------------------- matrix
def test_matrix_builds_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(dlg_mod, "get_backend", lambda *a, **k: _FakeAgent(caves=True))
    cfg = load_config("config/experiments/conformity_dialogue_matrix.yaml")
    cfg["experiment"]["trials"] = 3
    # trim to 2 models for a fast hermetic grid
    cfg["matrix"]["models"] = [
        {"backend": "echo", "model": "m1"}, {"backend": "echo", "model": "m2"},
    ]
    out = run_dialogue_matrix(cfg, output_dir=tmp_path)
    assert out["models"] == ["m1", "m2"]
    assert set(out["grid"]) == {"m1", "m2"}
    for s in ("m1", "m2"):
        for p in ("m1", "m2"):
            assert out["grid"][s][p] == 1.0          # fake always caves
    assert len(out["cells"]) == 4                    # 2x2
    assert (Path(out["matrix_dir"]) / "matrix.json").exists()
    assert "m1" in format_matrix(out)


def test_single_cell_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(dlg_mod, "get_backend", lambda *a, **k: _FakeAgent(caves=False))
    cfg = load_config("config/experiments/conformity_dialogue_matrix.yaml")
    cfg["experiment"]["trials"] = 3
    cfg["agents"] = {"subject": {"backend": "echo", "model": "s"},
                     "persuaders": {"backend": "echo", "model": "p"}}
    summary = ConformityDialogueExperiment(cfg).run(output_dir=tmp_path)
    assert summary["experiment"] == "conformity_dialogue"
    assert summary["headline"]["induced_hallucination_rate"] == 0.0   # holds firm
    assert (Path(summary["run_dir"]) / "dialogue.jsonl").exists()
