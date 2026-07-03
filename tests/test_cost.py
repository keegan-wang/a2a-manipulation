"""Cost estimator: call counting + echo zero-cost + graceful unknown model."""
from __future__ import annotations

from psychbench.framework.cost import estimate_cost, estimate_naive_calls


def _knowledge_cfg(backend="echo", model="echo-test", n_repeats=1, trials=None):
    exp = {
        "type": "conformity", "task_type": "knowledge",
        "seed": 1, "n_repeats": n_repeats,
    }
    if trials is not None:
        exp["trials"] = trials
    return {
        "experiment": exp,
        "corpus": {
            "path": "psychbench/experiments/conformity/corpus/"
                    "knowledge_facts.yaml",
        },
        "agents": {"naive": {"backend": backend, "model": model}},
        "environment": {"response_visibility": "public"},
    }


def test_naive_call_count_two_conditions_times_repeats():
    cfg = _knowledge_cfg(trials=10, n_repeats=3)
    # 10 items x 2 conditions x 3 repeats
    assert estimate_naive_calls(cfg) == 10 * 2 * 3


def test_naive_call_count_uses_corpus_when_uncapped():
    cfg = _knowledge_cfg()  # no trials cap -> full corpus
    n = estimate_naive_calls(cfg)
    assert n >= 20 * 2  # >=20 facts x 2 conditions x 1 repeat


def test_lines_call_count():
    cfg = {
        "experiment": {"type": "conformity", "task_type": "lines",
                       "trials": 18, "n_repeats": 2},
        "agents": {"naive": {"backend": "echo", "model": "echo-test"}},
        "environment": {"response_visibility": "public"},
    }
    assert estimate_naive_calls(cfg) == 18 * 2 * 2


def test_causal_call_count_uses_factorial_cells_and_two_subject_turns():
    cfg = {
        "experiment": {
            "type": "conformity_causal",
            "task_type": "story",
            "trials": 180,
        },
        "corpus": {
            "path": "psychbench/experiments/conformity/corpus/"
                    "relational_facts_powered.yaml",
        },
        "causal": {"condition_set": "factorial", "n_rounds": 1},
        "agents": {"subject": {"backend": "echo", "model": "echo-test"}},
        "environment": {"response_visibility": "public"},
    }
    assert estimate_naive_calls(cfg) == 180 * 9 * 2


def test_causal_adaptivity_estimate_counts_three_conditions_and_live_persuader():
    cfg = {
        "experiment": {
            "type": "conformity_causal",
            "task_type": "story",
            "trials": 180,
        },
        "corpus": {
            "path": "psychbench/experiments/conformity/corpus/"
                    "relational_facts_powered.yaml",
        },
        "causal": {"condition_set": "adaptivity", "n_rounds": 1},
        "agents": {"subject": {"backend": "echo", "model": "echo-test"}},
        "environment": {"response_visibility": "public"},
    }
    out = estimate_cost(cfg)
    assert out["n_subject_calls"] == 180 * 3 * 2
    assert out["n_persuader_calls"] == 180


def test_causal_verification_estimate_counts_two_conditions():
    cfg = {
        "experiment": {
            "type": "conformity_causal",
            "task_type": "story",
            "trials": 180,
        },
        "corpus": {
            "path": "psychbench/experiments/conformity/corpus/"
                    "relational_facts_powered.yaml",
        },
        "causal": {"condition_set": "verification", "n_rounds": 1},
        "agents": {"subject": {"backend": "echo", "model": "echo-test"}},
        "environment": {"response_visibility": "public"},
    }
    out = estimate_cost(cfg)
    assert out["n_subject_calls"] == 180 * 2 * 2
    assert out["n_persuader_calls"] == 0


def test_echo_is_free():
    out = estimate_cost(_knowledge_cfg(trials=10))
    assert out["est_usd"] == 0.0
    assert out["n_naive_calls"] == 20


def test_unknown_model_does_not_crash():
    out = estimate_cost(
        _knowledge_cfg(backend="openai", model="totally-made-up-model-xyz")
    )
    # either a float (if litellm somehow prices it) or None with a note
    assert out["est_usd"] is None or isinstance(out["est_usd"], float)
