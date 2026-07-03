"""Normative-pressure manipulation: Asch's public/private dissociation for LLMs.

The lever changes ONLY whether the subject's own answer is observed by the
group; the peer information is held identical. A model that conforms more when
observed shows a normative-pressure analog (normative_gap > 0); one that
conforms the same either way is doing pure informational deference (gap ~0).

Echo can't compare line lengths, so the end-to-end test uses a fake "norm-
sensitive" backend that conforms only when it is told its answer is public.
"""
from __future__ import annotations

import re

import pytest

from psychbench.config import load_config
from psychbench.framework.types import Stimulus, TrialContext
from psychbench.experiments.conformity import experiment as exp_mod
from psychbench.experiments.conformity.experiment import ConformityExperiment
from psychbench.experiments.conformity.prompts import build_conformity_prompt
from psychbench.experiments.conformity.asch_classic import build_asch_classic_stimuli


# ----------------------------------------------------------------- prompt lever
def _ctx(stim, priors):
    return TrialContext(
        stimulus=stim, agent_position=7, agent_id="naive",
        prior_responses=priors,
    )


def test_observability_changes_only_accountability_not_peers():
    stim = build_asch_classic_stimuli()[2]  # critical
    priors = [(f"c{i}", stim.wrong_label) for i in range(7)]
    public = build_conformity_prompt(_ctx(stim, priors), "public")
    private = build_conformity_prompt(_ctx(stim, priors), "private")
    none = build_conformity_prompt(_ctx(stim, priors), None)

    # the accountability sentence is the only thing that flips
    assert "shown to the rest of the group" in public
    assert "shown to the rest of the group" not in private
    assert "private and anonymous" in private
    assert "private and anonymous" not in public
    # no observability framing at all when unspecified (back-compat default)
    assert "shown to the rest of the group" not in none
    assert "private and anonymous" not in none

    # peer information is IDENTICAL across the arms (the whole point)
    def peers(p):
        return re.findall(r"Participant \d+ says: ([ABC])", p)
    assert peers(public) == peers(private) == [stim.wrong_label] * 7


def test_knowledge_prompt_also_supports_observability():
    stim = Stimulus(
        trial_index=0, is_critical=True, reference_cm=0.0,
        options={"A": "Paris", "B": "Lyon"}, correct_label="A", wrong_label="B",
        question="Capital of France?", metadata={"task_type": "knowledge"},
    )
    p = build_conformity_prompt(_ctx(stim, []), "public")
    assert "shown to the rest of the group" in p


# ----------------------------------------------------------------- end-to-end
def _match_line(prompt: str) -> str:
    std = float(re.search(r"Standard line:\s*([\d.]+)\s*cm", prompt).group(1))
    for lab, val in re.findall(r"Line ([ABC]):\s*([\d.]+)\s*cm", prompt):
        if abs(float(val) - std) < 1e-6:
            return lab
    return "A"


class _NormSensitivePerceptual:
    """Conforms to the majority ONLY when its answer is public; else trusts eyes."""
    def __init__(self, model="fake", **_):
        self.model = model

    def generate(self, prompt: str, stateful: bool = False) -> str:
        peers = re.findall(r"Participant \d+ says:\s*([ABC])", prompt)
        observed = "shown to the rest of the group" in prompt
        if peers and observed:
            return max(set(peers), key=peers.count)
        return _match_line(prompt)

    def reset(self):
        pass


@pytest.fixture
def cfg():
    return load_config("config/experiments/asch_classic_normative_smoke.yaml")


def test_normative_gap_is_measured(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr(exp_mod, "get_backend",
                        lambda *a, **k: _NormSensitivePerceptual())
    summary = ConformityExperiment(cfg).run(output_dir=tmp_path)
    assert summary["observability_manipulation"] is True
    h = summary["headline"]
    # knows every line alone; conforms when public, resists when private
    assert h["baseline_accuracy_alone"] == 1.0
    assert h["conformity_public"] == 1.0
    assert h["conformity_private"] == 0.0
    assert h["normative_gap"] == 1.0
    assert h["informational_conformity"] == 0.0


def test_three_conditions_written(tmp_path, monkeypatch, cfg):
    from pathlib import Path
    monkeypatch.setattr(exp_mod, "get_backend",
                        lambda *a, **k: _NormSensitivePerceptual())
    summary = ConformityExperiment(cfg).run(output_dir=tmp_path)
    run_dir = Path(summary["run_dir"])
    for label in ("alone", "group_public", "group_private"):
        assert (run_dir / f"conformity_{label}.jsonl").exists()
