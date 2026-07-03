"""Rough pre-run cost estimate for the naive (subject) model calls.

Confederates in the conformity clone are free scripted agents, so the dollar
cost is dominated by the naive subject's calls: ``n_trials × 2 conditions ×
n_repeats``. Token counts are coarse assumptions; pricing comes from litellm's
model table. The point is a sanity gate for the cheap pilot, not accounting.
"""
from __future__ import annotations

from typing import Any

from .backends import litellm_model_string

# Coarse per-call token assumptions for an MC letter-answer task with a few
# visible peer answers. Override via estimate_cost(...) for other shapes.
_ASSUMED_PROMPT_TOKENS = 220
_ASSUMED_COMPLETION_TOKENS = 8


def estimate_naive_calls(config: dict[str, Any]) -> int:
    exp = config.get("experiment", {})
    if exp.get("type") == "conformity_causal":
        return estimate_causal_subject_calls(config)
    n_repeats = max(1, int(exp.get("n_repeats", 1)))
    task = exp.get("task_type", "knowledge")
    if task == "knowledge":
        n_trials = _knowledge_item_count(config)
    else:
        n_trials = int(exp.get("trials", 18))
    # alone + group conditions both call the naive once per trial.
    return n_trials * 2 * n_repeats


def estimate_causal_subject_calls(config: dict[str, Any]) -> int:
    n_trials = _story_item_count(config)
    n_conditions = _causal_condition_count(config)
    # Initial commitment + final answer for each condition.
    return n_trials * n_conditions * 2


def estimate_causal_persuader_calls(config: dict[str, Any]) -> int:
    causal = config.get("causal", {})
    condition_set = causal.get("condition_set", "factorial")
    n_trials = _story_item_count(config)
    n_rounds = int(causal.get("n_rounds", 1))
    if condition_set == "adaptivity":
        return n_trials * n_rounds
    if causal.get("pressure_mode", "static") != "live":
        return 0
    # Factorial has 8 pressure cells plus the no-pressure control.
    pressure_conditions = max(0, _causal_condition_count(config) - 1)
    return n_trials * pressure_conditions * n_rounds


def _causal_condition_count(config: dict[str, Any]) -> int:
    condition_set = config.get("causal", {}).get("condition_set", "factorial")
    if condition_set == "factorial":
        return 9
    if condition_set == "adaptivity":
        return 3
    if condition_set == "verification":
        return 2
    return int(config.get("causal", {}).get("n_conditions", 9))


def _knowledge_item_count(config: dict[str, Any]) -> int:
    exp = config.get("experiment", {})
    cap = exp.get("trials")
    try:
        from psychbench.experiments.conformity.corpus import (
            load_knowledge_corpus,
        )
        n = len(load_knowledge_corpus(config["corpus"]["path"]))
    except Exception:
        n = int(cap) if cap else 0
    return min(n, int(cap)) if cap else n


def _story_item_count(config: dict[str, Any]) -> int:
    exp = config.get("experiment", {})
    cap = exp.get("trials")
    try:
        from psychbench.experiments.conformity.corpus import load_story_corpus
        n = len(load_story_corpus(config["corpus"]["path"]))
    except Exception:
        n = int(cap) if cap else 0
    return min(n, int(cap)) if cap else n


def estimate_cost(
    config: dict[str, Any],
    prompt_tokens: int = _ASSUMED_PROMPT_TOKENS,
    completion_tokens: int = _ASSUMED_COMPLETION_TOKENS,
) -> dict[str, Any]:
    exp = config.get("experiment", {})
    agent_key = "subject" if exp.get("type") == "conformity_causal" else "naive"
    naive = config.get("agents", {}).get(agent_key, {})
    kind = naive.get("backend", "echo")
    model = naive.get("model", "")
    n_calls = estimate_naive_calls(config)
    out: dict[str, Any] = {
        "n_naive_calls": n_calls,
        "n_subject_calls": n_calls,
        "backend": kind,
        "model": model,
        "assumed_prompt_tokens": prompt_tokens,
        "assumed_completion_tokens": completion_tokens,
    }
    if exp.get("type") == "conformity_causal":
        out["n_persuader_calls"] = estimate_causal_persuader_calls(config)
    if kind == "echo":
        out["est_usd"] = 0.0
        return out
    try:
        import litellm  # type: ignore
        ms = litellm_model_string(kind, model)
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        per_call = float(prompt_cost) + float(completion_cost)
        out["est_usd_per_call"] = round(per_call, 6)
        out["est_usd"] = round(per_call * n_calls, 4)
    except Exception as e:  # unknown model / litellm missing
        out["est_usd"] = None
        out["note"] = f"no pricing available for '{model}': {e}"
    return out
