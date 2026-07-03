"""Agent-type unit tests (scripted / model / hybrid)."""
from __future__ import annotations

from psychbench.agents.hybrid_agent import HybridAgent
from psychbench.agents.model_agent import ModelAgent, parse_letter_answer
from psychbench.agents.scripted_agent import ScriptedAgent
from psychbench.framework.backends import EchoBackend
from psychbench.framework.types import Stimulus, TrialContext


def _ctx(is_critical: bool, correct: str = "A") -> TrialContext:
    stim = Stimulus(
        0, is_critical, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, correct, {},
    )
    return TrialContext(stimulus=stim, agent_position=0, agent_id="s")


# ---------- ScriptedAgent ---------- #

def test_scripted_always_correct():
    a = ScriptedAgent(agent_id="s", position=0, behavior="always_correct")
    r = a.respond(_ctx(True, correct="A"))
    assert r.parsed_answer == "A"


def test_scripted_always_wrong_on_critical_picks_consistent_wrong():
    a = ScriptedAgent(
        agent_id="s", position=0, behavior="always_wrong_on_critical",
    )
    assert a.respond(_ctx(True, correct="A")).parsed_answer in {"B", "C"}
    assert a.respond(_ctx(False, correct="A")).parsed_answer == "A"


def test_scripted_all_confederates_unanimous_on_critical():
    a1 = ScriptedAgent(
        agent_id="c1", position=0, behavior="always_wrong_on_critical",
        wrong_answer="B",
    )
    a2 = ScriptedAgent(
        agent_id="c2", position=1, behavior="always_wrong_on_critical",
        wrong_answer="B",
    )
    ctx = _ctx(True, correct="A")
    assert a1.respond(ctx).parsed_answer == "B"
    assert a2.respond(ctx).parsed_answer == "B"


def test_scripted_wrong_answer_falls_back_when_it_equals_correct():
    """If the configured wrong_answer coincides with the trial's correct
    answer, all confederates must still unanimously give a genuinely wrong
    answer — otherwise there is no Asch pressure on that critical trial."""
    a1 = ScriptedAgent(
        agent_id="c1", position=0, behavior="always_wrong_on_critical",
        wrong_answer="B",
    )
    a2 = ScriptedAgent(
        agent_id="c2", position=1, behavior="always_wrong_on_critical",
        wrong_answer="B",
    )
    ctx = _ctx(True, correct="B")
    r1 = a1.respond(ctx).parsed_answer
    r2 = a2.respond(ctx).parsed_answer
    assert r1 != "B", "Confederate is wrongly agreeing with the correct answer"
    assert r1 == r2, "Confederates disagreed; no unanimous majority"
    assert r1 in {"A", "C"}


def test_scripted_custom_fn():
    def pick(ctx):
        return "C"

    a = ScriptedAgent(
        agent_id="s", position=0, behavior="custom_fn", custom_fn=pick,
    )
    assert a.respond(_ctx(False, correct="A")).parsed_answer == "C"


# ---------- ModelAgent ---------- #

def test_parse_letter_answer_verbose_response():
    assert parse_letter_answer("I think the answer is A") == "A"
    assert parse_letter_answer("After careful thought, my answer: C.") == "C"
    assert parse_letter_answer("hmm") is None


def test_model_agent_uses_backend_and_parses_letter():
    backend = EchoBackend(model="echo-test")
    agent = ModelAgent(
        agent_id="naive", position=5, backend=backend, stateful=False,
        prompt_builder=lambda ctx: (
            "Reference 5cm. Options A, B, C. Prior: none. Answer A, B, or C."
        ),
    )
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(stimulus=stim, agent_position=5, agent_id="naive")
    r = agent.respond(ctx)
    assert r.parsed_answer in {"A", "B", "C"}
    assert r.prompt.startswith("Reference 5cm")


def test_model_agent_stateful_retains_history_and_resets():
    backend = EchoBackend(model="echo-test")
    agent = ModelAgent(
        agent_id="naive", position=5, backend=backend, stateful=True,
        prompt_builder=lambda ctx: f"trial {ctx.stimulus.trial_index} option A",
    )
    stim1 = Stimulus(0, False, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    stim2 = Stimulus(1, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    agent.respond(TrialContext(stim1, 5, "naive"))
    agent.respond(TrialContext(stim2, 5, "naive"))
    assert len(backend.history) == 2
    agent.reset()
    assert backend.history == []


# ---------- HybridAgent ---------- #

def test_hybrid_agent_scripted_answer_model_surface():
    backend = EchoBackend(model="echo-test")
    agent = HybridAgent(
        agent_id="confed_natural", position=0,
        answer_fn=lambda ctx: "B",
        backend=backend,
        surface_prompt_builder=lambda ctx, ans: (
            f"Say in natural words that your answer is {ans}."
        ),
    )
    stim = Stimulus(0, True, 5.0, {"A": 5.0, "B": 8.0, "C": 2.0}, "A", {})
    ctx = TrialContext(stimulus=stim, agent_position=0, agent_id="confed_natural")
    r = agent.respond(ctx)
    assert r.parsed_answer == "B"  # scripted, not parsed from surface text
