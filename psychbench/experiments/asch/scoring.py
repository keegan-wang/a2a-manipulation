"""Binary conformity scoring for Asch Phase 1."""
from __future__ import annotations

from typing import Any

from psychbench.framework.types import AgentResponse, TrialResult


def _find(
    responses: list[AgentResponse], agent_id: str
) -> AgentResponse | None:
    for r in responses:
        if r.agent_id == agent_id:
            return r
    return None


def score_asch_trial(
    result: TrialResult, naive_id: str, confederate_ids: list[str],
) -> dict[str, Any]:
    naive = _find(result.responses, naive_id)
    naive_ans = naive.parsed_answer if naive is not None else None
    confed_answers = [
        r.parsed_answer for r in result.responses
        if r.agent_id in confederate_ids and r.parsed_answer is not None
    ]
    confed_ans = confed_answers[0] if confed_answers else None
    if not result.is_critical:
        return {
            "conformed": None,
            "naive_answer": naive_ans,
            "confederate_answer": confed_ans,
        }
    conformed = (
        naive_ans is not None
        and confed_ans is not None
        and naive_ans == confed_ans
        and naive_ans != result.correct_answer
    )
    return {
        "conformed": bool(conformed),
        "naive_answer": naive_ans,
        "confederate_answer": confed_ans,
    }


def summarize_asch_session(trials: list[TrialResult]) -> dict[str, Any]:
    critical = [t for t in trials if t.is_critical]
    n_critical = len(critical)
    n_conformed = sum(1 for t in critical if t.conformed is True)
    conformity_rate = n_conformed / n_critical if n_critical else 0.0
    per_trial = [
        {
            "trial_index": t.trial_index,
            "is_critical": t.is_critical,
            "naive_answer": t.naive_answer,
            "confederate_answer": t.confederate_answer,
            "correct_answer": t.correct_answer,
            "conformed": t.conformed,
        }
        for t in trials
    ]
    return {
        "n_trials": len(trials),
        "n_critical": n_critical,
        "n_conformed": n_conformed,
        "conformity_rate": conformity_rate,
        "ever_conformed": n_conformed > 0,
        "per_trial": per_trial,
    }
