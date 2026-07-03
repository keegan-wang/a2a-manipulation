"""Human-similarity scoring: human-like models score high, extremes score low."""
from __future__ import annotations

from psychbench.experiments.conformity.human_benchmark import (
    HUMAN_ASCH, condition_similarity, human_similarity,
)


def test_condition_similarity_bounds():
    assert condition_similarity(0.37, 0.37) == 1.0      # identical
    assert condition_similarity(0.99, 0.0) == 0.0       # far -> clamped to 0
    assert 0.0 < condition_similarity(0.27, 0.37) < 1.0  # partial


def test_human_like_model_scores_high():
    # a model matching the human reference exactly
    model = dict(HUMAN_ASCH)
    out = human_similarity(model)
    assert out["human_similarity_score"] == 1.0
    assert abs(out["mean_signed_gap"]) < 1e-9


def test_super_conformist_scores_low_and_positive_gap():
    # always conforms everywhere -> far from human, over-conforms
    model = {k: 1.0 for k in HUMAN_ASCH}
    out = human_similarity(model)
    assert out["human_similarity_score"] < 0.2
    assert out["mean_signed_gap"] > 0      # over-conforms vs humans


def test_stubborn_model_scores_low_and_negative_gap():
    # never conforms -> also un-human, under-conforms
    model = {k: 0.0 for k in HUMAN_ASCH}
    out = human_similarity(model)
    assert out["human_similarity_score"] < 0.6
    assert out["mean_signed_gap"] < 0      # under-conforms vs humans


def test_only_shared_conditions_scored():
    out = human_similarity({"baseline": 0.37})
    assert out["n_conditions"] == 1
    assert "baseline" in out["per_condition"]
