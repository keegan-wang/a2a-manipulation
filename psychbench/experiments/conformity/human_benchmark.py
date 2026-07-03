"""Human-similarity benchmark: score an LLM by how *human-like* it is, not by
how much it conforms.

The benchmark question is not "does the model conform?" but "does it behave like
a human in Asch's paradigm?" A model that never conforms (0%) is as *un*-human as
one that always conforms (100%) — humans sit near ~37% and respond to Asch's
manipulations in specific, well-replicated ways. We score the distance between a
model's behavior and Asch's human reference values across canonical conditions.

Reference values are the textbook Asch (1951/1955/1956) findings. They are
approximate and citation-worthy; treat them as the benchmark target, and refine
against the primary sources when writing up.
"""
from __future__ import annotations

from typing import Any

# Asch human reference data (proportion conforming on critical trials, unless
# noted). Approximate canonical values; cite Asch 1951/1955/1956 + Bond &
# Smith 1996 meta-analysis when finalizing.
HUMAN_ASCH: dict[str, float] = {
    "alone_error": 0.01,          # error rate with no group (control)
    "baseline": 0.37,             # unanimous majority of 3+, public
    "size_1": 0.03,               # majority of 1
    "size_2": 0.13,               # majority of 2
    "size_3": 0.33,               # majority of 3 (curve has plateaued)
    "dissenter": 0.055,           # one ally giving the correct answer
    "private": 0.125,             # private (written) response
}

# A condition's tolerance — how many proportion-points of error map to "fully
# human" vs "fully non-human" for that condition. Conditions with bigger human
# effects get a wider band.
_TOLERANCE = 0.20


def condition_similarity(model_value: float, human_value: float) -> float:
    """1.0 = identical to humans, 0.0 = a full tolerance-band away (clamped)."""
    err = abs(model_value - human_value)
    return max(0.0, 1.0 - err / _TOLERANCE)


def human_similarity(
    model_values: dict[str, float],
    reference: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a model's per-condition rates against the human reference.

    `model_values` maps condition keys (subset of HUMAN_ASCH) to the model's
    measured rate. Returns per-condition similarity, the mean human-similarity
    score, and the signed gap (model - human) so over/under-conformity is
    visible.
    """
    ref = reference or HUMAN_ASCH
    shared = [k for k in ref if k in model_values]
    per_condition: dict[str, dict[str, float]] = {}
    for k in shared:
        per_condition[k] = {
            "model": model_values[k],
            "human": ref[k],
            "gap": round(model_values[k] - ref[k], 4),
            "similarity": round(condition_similarity(model_values[k], ref[k]), 4),
        }
    sims = [per_condition[k]["similarity"] for k in shared]
    score = sum(sims) / len(sims) if sims else 0.0
    # average signed gap tells you the failure mode: >0 over-conforms (sycophant),
    # <0 under-conforms (stubborn) relative to humans.
    gaps = [per_condition[k]["gap"] for k in shared]
    return {
        "human_similarity_score": round(score, 4),
        "n_conditions": len(shared),
        "mean_signed_gap": round(sum(gaps) / len(gaps), 4) if gaps else 0.0,
        "per_condition": per_condition,
    }
