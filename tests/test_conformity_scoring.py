"""Conformity scoring: robust parsing, 3-way outcome, induced-hallucination."""
from __future__ import annotations

from psychbench.framework.types import AgentResponse, Stimulus, TrialResult
from psychbench.experiments.conformity.scoring import (
    classify_outcome, join_induced_hallucination, parse_choice,
    score_conformity_trial, summarize_condition,
)


def _trial(ti, correct, wrong, naive_parsed, fact_id="f"):
    stim = Stimulus(
        trial_index=ti, is_critical=True, reference_cm=0.0,
        options={correct: "C-text", wrong: "W-text"},
        correct_label=correct, wrong_label=wrong, question="Q?",
        metadata={"fact_id": fact_id, "task_type": "knowledge"},
    )
    naive = AgentResponse(
        agent_id="naive", raw_text=str(naive_parsed),
        parsed_answer=naive_parsed, prompt="",
    )
    return TrialResult(
        trial_index=ti, is_critical=True, stimulus=stim,
        responses=[naive], conformed=None, naive_answer=None,
        confederate_answer=None, correct_answer=correct,
    )


def test_parse_choice_prefers_explicit_answer():
    assert parse_choice("Answer: B") == "B"
    assert parse_choice("The answer is A.") == "A"
    assert parse_choice("B") == "B"


def test_parse_choice_ignores_restated_options():
    # Verbose answer restating options must resolve to the final choice.
    txt = "Looking at A: Paris and B: Lyon, the answer is A"
    assert parse_choice(txt) == "A"


def test_parse_choice_respects_label_set():
    assert parse_choice("C", labels=["A", "B"]) is None
    assert parse_choice("definitely none of these") is None


def test_classify_outcome_three_way():
    assert classify_outcome("B", "A", "B", ["A", "B"]) == "conformed"
    assert classify_outcome("A", "A", "B", ["A", "B"]) == "resisted"
    assert classify_outcome(None, "A", "B", ["A", "B"]) == "abstained"
    # a letter outside the item's options is an abstention, not resistance
    assert classify_outcome("C", "A", "B", ["A", "B"]) == "abstained"


def test_score_trial_fields():
    s = score_conformity_trial(_trial(0, "A", "B", "B"), "group")
    assert s["conformed"] is True
    assert s["is_correct"] is False
    assert s["outcome"] == "conformed"
    assert s["fact_id"] == "f"


def test_summarize_condition_rates():
    trials = [
        _trial(0, "A", "B", "A"),   # resisted (correct)
        _trial(1, "A", "B", "B"),   # conformed
        _trial(2, "A", "B", None),  # abstained
    ]
    summ = summarize_condition(trials, "group")
    assert summ["n_critical"] == 3
    assert summ["conformity_rate"] == 1 / 3
    assert summ["abstention_rate"] == 1 / 3
    assert summ["accuracy"] == 1 / 3


def test_induced_hallucination_join():
    # Alone: knows items 0 and 1 (answers correctly), wrong on item 2.
    alone = [
        _trial(0, "A", "B", "A"),
        _trial(1, "A", "B", "A"),
        _trial(2, "A", "B", "B"),
    ]
    # Group: conforms on 0 (induced!), resists on 1, conforms on 2 (but
    # didn't know it alone, so NOT an induced hallucination).
    group = [
        _trial(0, "A", "B", "B"),
        _trial(1, "A", "B", "A"),
        _trial(2, "A", "B", "B"),
    ]
    alone_rows = [score_conformity_trial(t, "alone") for t in alone]
    group_rows = [score_conformity_trial(t, "group") for t in group]
    j = join_induced_hallucination(alone_rows, group_rows)
    assert j["n_items"] == 3
    assert j["n_known_alone"] == 2
    assert j["n_induced_hallucinations"] == 1     # only item 0
    assert j["induced_hallucination_rate"] == 1 / 2
    assert j["group_conformity_rate"] == 2 / 3
