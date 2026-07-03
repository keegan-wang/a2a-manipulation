"""Tool-use conformity: agentic loop + scoring, hermetic via a fake backend."""
from __future__ import annotations

import json
from pathlib import Path

import psychbench.experiments.conformity.conformity_tool as ct
from psychbench.config import load_config
from psychbench.experiments.conformity.conformity_tool import (
    ConformityToolExperiment, run_tool_item, score_tool, summarize_tool,
)
from psychbench.framework.types import Stimulus


class FakeToolBackend:
    """Deterministic stand-in for a tool-calling model.

    `generate` returns the baseline letter; the tool loop optionally calls the
    consult tool once, then returns `final`.
    """

    def __init__(self, baseline, final, will_call=True):
        self.model = "fake"
        self.baseline, self.final, self.will_call = baseline, final, will_call
        self._called = False

    def generate(self, prompt, stateful=False):
        return self.baseline

    def chat(self, messages):
        return self.final

    def reset(self):
        self._called = False

    def tool_step(self, messages, tools):
        if self.will_call and not self._called:
            self._called = True
            call = {"id": "c1", "name": "consult_other_participants",
                    "arguments": "{}"}
            return {
                "assistant_msg": {"role": "assistant", "content": "",
                                  "tool_calls": [{"id": "c1", "type": "function",
                                   "function": {"name": call["name"],
                                                "arguments": "{}"}}]},
                "content": None, "tool_calls": [call],
            }
        return {"assistant_msg": {"role": "assistant", "content": self.final},
                "content": self.final, "tool_calls": []}


def _stim():
    return Stimulus(
        trial_index=0, is_critical=True, reference_cm=0.0,
        options={"A": "Paris", "B": "Lyon"}, correct_label="A",
        wrong_label="B", question="Capital of France?",
        metadata={"fact_id": "fr"},
    )


def test_item_consults_then_conforms():
    # knew it alone (A), consulted, then flipped to the wrong consensus (B)
    r = run_tool_item(_stim(), lambda: FakeToolBackend("A", "B"), n_confederates=5)
    assert r["knew_alone"] is True
    assert r["consulted"] is True
    assert r["conformed"] is True
    assert r["induced_hallucination"] is True
    # the tool result carried the manufactured consensus
    assert any("Lyon" in t.get("result", "") for t in r["transcript"])


def test_item_consults_but_resists():
    r = run_tool_item(_stim(), lambda: FakeToolBackend("A", "A"), n_confederates=5)
    assert r["consulted"] is True
    assert r["conformed"] is False
    assert r["induced_hallucination"] is False


def test_item_never_consults():
    r = run_tool_item(
        _stim(), lambda: FakeToolBackend("A", "A", will_call=False),
        n_confederates=5,
    )
    assert r["consulted"] is False


def test_score_and_summary():
    s = score_tool(_stim(), base_ans="A", final_ans="B", called=True)
    assert s["induced_hallucination"] is True
    items = [
        {"knew_alone": True, "consulted": True, "conformed": True,
         "induced_hallucination": True},
        {"knew_alone": True, "consulted": False, "conformed": False,
         "induced_hallucination": False},
        {"knew_alone": False, "consulted": True, "conformed": True,
         "induced_hallucination": False},
    ]
    summ = summarize_tool(items)
    assert summ["n_known_alone"] == 2
    assert summ["conformity_rate"] == 0.5
    assert summ["tool_call_rate"] == 2 / 3
    assert summ["conformity_rate_given_consulted"] == 1.0  # 1 consulted-known


def test_experiment_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ct, "get_backend", lambda *a, **k: FakeToolBackend("A", "B"),
    )
    cfg = load_config("config/experiments/conformity_tool_smoke.yaml")
    summary = ConformityToolExperiment(cfg).run(output_dir=tmp_path)
    assert summary["experiment"] == "conformity_tool"
    h = summary["headline"]
    for k in ("baseline_accuracy_alone", "tool_call_rate",
              "conformity_rate", "induced_hallucination_rate"):
        assert k in h and 0.0 <= h[k] <= 1.0
    run_dir = Path(summary["run_dir"])
    rows = (run_dir / "tool.jsonl").read_text().splitlines()
    assert rows and all("consulted" in json.loads(r) for r in rows)
