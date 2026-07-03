"""Framework-level unit tests: types, agent base, registry, env, trial, session."""
from __future__ import annotations

import json as _json
from pathlib import Path

import pytest

from psychbench.framework.agent import BaseAgent
from psychbench.framework.environment import Environment
from psychbench.framework.experiment import (
    BaseExperiment, EXPERIMENT_REGISTRY, get_experiment_class,
    register_experiment,
)
from psychbench.framework.logging_utils import JsonlLogger, write_summary
from psychbench.framework.session import Session
from psychbench.framework.trial import run_trial
from psychbench.framework.types import (
    AgentResponse, ResponseVisibility, Stimulus, TrialContext, TrialResult,
)


# ---------- types ---------- #

def test_stimulus_has_correct_answer_and_options():
    s = Stimulus(
        trial_index=0, is_critical=True, reference_cm=10.0,
        options={"A": 10.0, "B": 12.0, "C": 7.0}, correct_label="A",
        metadata={},
    )
    assert s.correct_label == "A"
    assert s.options["A"] == 10.0
    assert s.is_critical is True


def test_trial_context_carries_prior_responses():
    s = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(
        stimulus=s, agent_position=5, agent_id="naive",
        prior_responses=[("confed_1", "A"), ("confed_2", "A")],
    )
    assert len(ctx.prior_responses) == 2
    assert ctx.agent_position == 5


def test_agent_response_fields():
    r = AgentResponse(
        agent_id="naive", raw_text="I think A", parsed_answer="A",
        prompt="...", metadata={"backend": "openai"},
    )
    assert r.parsed_answer == "A"


def test_response_visibility_enum():
    assert ResponseVisibility("public") == ResponseVisibility.PUBLIC
    assert ResponseVisibility("private") == ResponseVisibility.PRIVATE
    assert ResponseVisibility("partial") == ResponseVisibility.PARTIAL


# ---------- BaseAgent ---------- #

def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent(agent_id="x", position=0)  # type: ignore[abstract]


def test_base_agent_subclass_implements_respond():
    class DummyAgent(BaseAgent):
        def respond(self, context):
            return AgentResponse(
                agent_id=self.agent_id, raw_text="A", parsed_answer="A",
                prompt="", metadata={},
            )

    a = DummyAgent(agent_id="d", position=0)
    s = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(stimulus=s, agent_position=0, agent_id="d")
    r = a.respond(ctx)
    assert r.parsed_answer == "A"
    a.reset()


# ---------- experiment registry ---------- #

def test_register_and_retrieve_experiment():
    @register_experiment("dummy_exp")
    class DummyExp(BaseExperiment):
        def build_stimuli(self):
            return []

        def build_agents(self):
            return []

    assert "dummy_exp" in EXPERIMENT_REGISTRY
    assert get_experiment_class("dummy_exp") is DummyExp


def test_get_experiment_class_raises_on_unknown():
    with pytest.raises(KeyError):
        get_experiment_class("definitely_not_registered")


# ---------- environment visibility ---------- #

def test_environment_public_visibility_gives_all_prior_responses():
    env = Environment(visibility=ResponseVisibility.PUBLIC)
    stim = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    env.begin_trial(stim)
    env.record_response("agent_1", 0, "A")
    env.record_response("agent_2", 1, "A")
    visible = env.visible_prior_responses(agent_id="agent_3", position=2)
    assert visible == [("agent_1", "A"), ("agent_2", "A")]


def test_environment_private_visibility_hides_responses():
    env = Environment(visibility=ResponseVisibility.PRIVATE)
    stim = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    env.begin_trial(stim)
    env.record_response("agent_1", 0, "A")
    assert env.visible_prior_responses("agent_2", 1) == []


def test_environment_partial_visibility_uses_per_agent_map():
    env = Environment(
        visibility=ResponseVisibility.PARTIAL,
        per_agent_visibility={"naive": "public", "confed_1": "private"},
    )
    stim = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    env.begin_trial(stim)
    env.record_response("confed_1", 0, "A")
    env.record_response("confed_2", 1, "A")
    assert env.visible_prior_responses("naive", 2) == [
        ("confed_1", "A"), ("confed_2", "A"),
    ]
    assert env.visible_prior_responses("confed_1", 0) == []


# ---------- trial runner ---------- #

class _FixedAgent(BaseAgent):
    def __init__(self, agent_id, position, answer):
        super().__init__(agent_id, position)
        self._answer = answer
        self.seen: list[tuple[str, str]] | None = None

    def respond(self, context):
        self.seen = list(context.prior_responses)
        return AgentResponse(
            agent_id=self.agent_id, raw_text=self._answer,
            parsed_answer=self._answer,
            prompt=f"pos={self.position}", metadata={},
        )


def test_run_trial_sequential_visibility_public():
    env = Environment(visibility=ResponseVisibility.PUBLIC)
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    a1 = _FixedAgent("a1", 0, "B")
    a2 = _FixedAgent("a2", 1, "B")
    a3 = _FixedAgent("a3", 2, "A")
    result = run_trial(stim, [a1, a2, a3], env)
    assert [r.parsed_answer for r in result.responses] == ["B", "B", "A"]
    assert a3.seen == [("a1", "B"), ("a2", "B")]
    assert a1.seen == []


def test_run_trial_private_visibility_hides_all():
    env = Environment(visibility=ResponseVisibility.PRIVATE)
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    a1 = _FixedAgent("a1", 0, "B")
    a2 = _FixedAgent("a2", 1, "A")
    run_trial(stim, [a1, a2], env)
    assert a2.seen == []


# ---------- logging ---------- #

def test_jsonl_logger_writes_one_record_per_line(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    with JsonlLogger(p) as log:
        log.write({"a": 1})
        log.write({"b": 2})
    lines = p.read_text().strip().splitlines()
    assert [_json.loads(line) for line in lines] == [{"a": 1}, {"b": 2}]


def test_write_summary_produces_valid_json(tmp_path: Path):
    p = tmp_path / "summary.json"
    write_summary(p, {"conformity_rate": 0.33, "n_critical": 12})
    loaded = _json.loads(p.read_text())
    assert loaded["conformity_rate"] == 0.33


# ---------- session ---------- #

class _ScriptedForSession(BaseAgent):
    def __init__(self, agent_id, position, critical_answer, normal_answer="A"):
        super().__init__(agent_id, position)
        self._crit = critical_answer
        self._norm = normal_answer

    def respond(self, context):
        ans = self._crit if context.stimulus.is_critical else self._norm
        return AgentResponse(
            agent_id=self.agent_id, raw_text=ans, parsed_answer=ans,
            prompt=f"pos={self.position}", metadata={},
        )


def test_session_runs_all_trials_and_logs_jsonl(tmp_path: Path):
    stims = [
        Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {}),
        Stimulus(1, True, 10.0, {"A": 10.0, "B": 13.0, "C": 7.0}, "A", {}),
    ]
    agents = [
        _ScriptedForSession("c1", 0, "B"),
        _ScriptedForSession("naive", 1, "A"),
    ]
    env = Environment(visibility=ResponseVisibility.PUBLIC)
    log_path = tmp_path / "run.jsonl"
    summary_path = tmp_path / "summary.json"
    session = Session(
        stimuli=stims, agents=agents, environment=env,
        log_path=log_path, summary_path=summary_path,
        config_snapshot={"test": True},
    )
    results = session.run()
    assert len(results) == 2
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    rec = _json.loads(lines[1])
    assert rec["trial_index"] == 1
    assert rec["is_critical"] is True
    assert any(r["agent_id"] == "naive" for r in rec["responses"])
