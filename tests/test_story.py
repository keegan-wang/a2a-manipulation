"""Made-up-story conformity task — single-shot + answer-from-memory.

Arbitrary facts stated in a passage: the model is right alone (it just read it)
but the belief has no pretraining anchor, so social pressure has something to
move. Echo can't reason over passages, so end-to-end tests use a fake backend
that answers from the passage alone and copies the announced majority when its
answer is public.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from psychbench.config import load_config
from psychbench.framework.types import Stimulus, TrialContext
from psychbench.experiments.conformity.corpus import load_story_corpus
from psychbench.experiments.conformity.stimuli import build_story_stimuli
from psychbench.experiments.conformity.prompts import build_conformity_prompt
from psychbench.experiments.conformity.challenge import build_item_prompt
from psychbench.experiments.conformity import experiment as exp_mod
from psychbench.experiments.conformity.experiment import ConformityExperiment
from psychbench.experiments.conformity.challenge import (
    ConformityChallengeExperiment,
)

_CORPUS = "psychbench/experiments/conformity/corpus/story_facts.yaml"


# ----------------------------------------------------------------- corpus/stim
def test_story_corpus_loads_passages():
    items = load_story_corpus(_CORPUS)
    assert len(items) >= 10
    it = items[0]
    assert it.passage and it.correct_answer in it.passage
    assert it.correct_answer != it.wrong_answer


def test_story_stimuli_counterbalanced_and_carry_passage():
    items = load_story_corpus(_CORPUS)
    stims = build_story_stimuli(items, seed=0)
    assert len(stims) == len(items)
    for s in stims:
        assert s.metadata["task_type"] == "story"
        assert s.metadata["passage"]
        assert s.wrong_label != s.correct_label
        assert s.options[s.correct_label] == s.metadata["correct_answer"]
    # counterbalancing actually splits correct across both letters
    correct_letters = {s.correct_label for s in stims}
    assert correct_letters == {"A", "B"}


# ----------------------------------------------------------------- prompts
def _stim():
    return build_story_stimuli(load_story_corpus(_CORPUS), seed=0)[0]


def _ctx(stim, priors=None):
    return TrialContext(stimulus=stim, agent_position=7, agent_id="naive",
                        prior_responses=priors or [])


def test_single_shot_prompt_shows_passage_and_observability():
    s = _stim()
    p = build_conformity_prompt(_ctx(s), "public")
    assert "Read the following passage" in p
    assert s.metadata["passage"].splitlines()[0].strip()[:10] in p  # passage in
    assert "Question:" in p
    assert "shown to the rest of the group" in p


def test_commit_prompt_shows_passage_but_challenge_does_not():
    s = _stim()
    commit = build_item_prompt(s)              # round 0: passage present
    assert "Read the following passage" in commit
    assert "Question:" in commit and s.metadata["passage"][:30] in commit
    # the pushback turns (peer_majority) never re-show the passage
    from psychbench.experiments.conformity.challenge import build_challenge_prompt
    push = build_challenge_prompt(s, 7, "peer_majority")
    assert s.metadata["passage"][:30] not in push


# ----------------------------------------------------------------- fake backend
def _passage_answer(prompt: str) -> str:
    """Pick the option letter whose text appears in the passage."""
    opts = dict(re.findall(r"^([AB]): (.+)$", prompt, re.MULTILINE))
    # the passage is everything before "Question:"
    passage = prompt.split("Question:")[0]
    for lab, text in opts.items():
        if text.strip() and text.strip() in passage:
            return lab
    return "A"


class _StoryConformist:
    """Right from the passage; copies the public majority when observed."""
    def __init__(self, model="fake", **_):
        self.model = model
        self.history: list = []

    def generate(self, prompt: str, stateful: bool = False) -> str:
        peers = re.findall(r"Participant \d+(?: says)?: ?([AB])", prompt)
        observed = "shown to the rest of the group" in prompt
        ans = max(set(peers), key=peers.count) if (peers and observed) \
            else _passage_answer(prompt)
        if stateful:
            self.history.append((prompt, ans))
        return ans

    def reset(self):
        self.history = []


# ----------------------------------------------------------------- single-shot e2e
def test_single_shot_story_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(exp_mod, "get_backend",
                        lambda *a, **k: _StoryConformist())
    cfg = load_config("config/experiments/conformity_story_normative.yaml")
    summary = ConformityExperiment(cfg).run(output_dir=tmp_path)
    h = summary["headline"]
    # reads passage right alone; conforms only when public -> a normative gap
    assert h["baseline_accuracy_alone"] == 1.0
    assert h["conformity_public"] == 1.0
    assert h["conformity_private"] == 0.0
    assert h["normative_gap"] == 1.0


def test_smoke_single_shot_runs(tmp_path):
    cfg = load_config("config/experiments/conformity_story_smoke.yaml")
    summary = ConformityExperiment(cfg).run(output_dir=tmp_path)
    assert (Path(summary["run_dir"]) / "conformity_alone.jsonl").exists()


# ----------------------------------------------------------------- memory e2e
class _MemoryConformist:
    """Commits from the passage at round 0, then capitulates to the majority."""
    def __init__(self, model="fake", **_):
        self.model = model
        self.committed = None

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if "Read the following passage" in prompt:          # round 0 / commit
            self.committed = _passage_answer(prompt)
            return self.committed
        peers = re.findall(r"Participant \d+: ?([AB])", prompt)  # pushback turn
        return max(set(peers), key=peers.count) if peers else (self.committed or "A")

    def reset(self):
        self.committed = None


def test_answer_from_memory_pipeline(tmp_path, monkeypatch):
    import psychbench.experiments.conformity.challenge as ch
    monkeypatch.setattr(ch, "get_backend", lambda *a, **k: _MemoryConformist())
    cfg = load_config("config/experiments/conformity_story_memory.yaml")
    summary = ConformityChallengeExperiment(cfg).run(output_dir=tmp_path)
    h = summary["headline"]
    # knew it at commit, then flipped to the disputing majority from memory
    assert h["baseline_accuracy_alone"] == 1.0
    assert h["induced_hallucination_rate"] == 1.0
    # the commit prompt carried the passage; round-0 transcript proves it
    run_dir = Path(summary["run_dir"])
    first = json.loads((run_dir / "challenge.jsonl").read_text().splitlines()[0])
    assert "Read the following passage" in first["transcript"][0]["prompt"]


def test_smoke_memory_runs(tmp_path):
    cfg = load_config("config/experiments/conformity_story_memory_smoke.yaml")
    summary = ConformityChallengeExperiment(cfg).run(output_dir=tmp_path)
    assert summary["experiment"] == "conformity_challenge"
    assert summary["task_type"] == "story"


# -------------------------------------------- memory + normative (public/private)
def test_challenge_prompt_observability_holds_peers_constant():
    from psychbench.experiments.conformity.challenge import build_challenge_prompt
    s = _stim()
    pub = build_challenge_prompt(s, 7, "peer_majority", 0, "public")
    priv = build_challenge_prompt(s, 7, "peer_majority", 0, "private")
    none = build_challenge_prompt(s, 7, "peer_majority", 0, None)
    assert "shown to the rest of the group" in pub
    assert "private and anonymous" in priv
    assert "shown to the rest of the group" not in none
    assert "private and anonymous" not in none
    peers = lambda p: re.findall(r"Participant \d+: ([AB])", p)  # noqa: E731
    assert peers(pub) == peers(priv) == [s.wrong_label] * 7


class _NormMemoryConformist:
    """Commits from the passage; capitulates to the majority ONLY when public."""
    def __init__(self, model="fake", **_):
        self.model = model
        self.committed = None

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if "Read the following passage" in prompt:
            self.committed = _passage_answer(prompt)
            return self.committed
        peers = re.findall(r"Participant \d+: ?([AB])", prompt)
        if peers and "shown to the rest of the group" in prompt:
            return max(set(peers), key=peers.count)
        return self.committed or "A"

    def reset(self):
        self.committed = None


def test_memory_normative_gap(tmp_path, monkeypatch):
    import psychbench.experiments.conformity.challenge as ch
    monkeypatch.setattr(ch, "get_backend", lambda *a, **k: _NormMemoryConformist())
    cfg = load_config("config/experiments/conformity_story_memory_normative.yaml")
    summary = ConformityChallengeExperiment(cfg).run(output_dir=tmp_path)
    assert summary["observability_manipulation"] is True
    h = summary["headline"]
    assert h["induced_neutral"] == 0.0    # no framing -> holds (fake needs public)
    assert h["induced_public"] == 1.0     # capitulates when watched
    assert h["induced_private"] == 0.0    # holds when private
    assert h["normative_gap_induced"] == 1.0
    assert h["public_vs_neutral_induced"] == 1.0   # public RAISES conformity
    assert h["private_vs_neutral_induced"] == 0.0  # private == neutral here
    # human-similarity block wired in from the public/private arms
    assert "human_similarity" in h
    assert 0.0 <= h["human_similarity"]["human_similarity_score"] <= 1.0
    rd = Path(summary["run_dir"])
    for arm in ("neutral", "public", "private"):
        assert (rd / f"challenge_{arm}.jsonl").exists()


class _AblationFake:
    """Commits from passage; capitulates to the majority UNLESS a reflect cue
    is present (then it trusts its own reading)."""
    def __init__(self, model="fake", **_):
        self.model = model
        self.committed = None

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if "Read the following passage" in prompt:
            self.committed = _passage_answer(prompt)
            return self.committed
        peers = re.findall(r"Participant \d+: ?([AB])", prompt)
        if "think carefully about what you yourself read" in prompt:
            return self.committed or "A"      # reflect cue -> holds
        return max(set(peers), key=peers.count) if peers else (self.committed or "A")

    def reset(self):
        self.committed = None


def test_framing_ablation_decomposes_arms(tmp_path, monkeypatch):
    import psychbench.experiments.conformity.challenge as ch
    monkeypatch.setattr(ch, "get_backend", lambda *a, **k: _AblationFake())
    cfg = load_config("config/experiments/conformity_story_framing_ablation.yaml")
    summary = ConformityChallengeExperiment(cfg).run(output_dir=tmp_path)
    arms = summary["headline"]["arms"]
    assert set(arms) == {"neutral", "filler", "reflect", "public", "private"}
    # the fake caves everywhere except under the reflect cue
    assert arms["neutral"]["induced_hallucination_rate"] > 0
    assert arms["reflect"]["induced_hallucination_rate"] == 0.0
    assert arms["reflect"]["vs_neutral_induced"] < 0
    rd = Path(summary["run_dir"])
    for arm in arms:
        assert (rd / f"challenge_{arm}.jsonl").exists()
