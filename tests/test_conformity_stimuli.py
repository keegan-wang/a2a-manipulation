"""Conformity stimulus generation: counterbalancing + wrong-target wiring."""
from __future__ import annotations

import collections

from psychbench.experiments.conformity.corpus import Fact, load_knowledge_corpus
from psychbench.experiments.conformity.stimuli import (
    build_knowledge_stimuli, build_line_stimuli,
)

_FACTS = [
    Fact("a", "Q a?", "Aye", "Nay"),
    Fact("b", "Q b?", "Bee", "Sea"),
    Fact("c", "Q c?", "Cee", "Dee"),
]


def test_knowledge_stimuli_basic_shape():
    stims = build_knowledge_stimuli(_FACTS, seed=1)
    assert len(stims) == 3
    for s in stims:
        assert s.is_critical
        assert s.question is not None
        assert set(s.options) == {"A", "B"}
        assert s.wrong_label != s.correct_label
        assert s.wrong_label in s.options
        # the wrong option text is the fact's distractor
        assert s.options[s.wrong_label] == s.metadata["wrong_answer"]


def test_knowledge_correct_label_is_counterbalanced():
    # Across seeds the correct answer should land on both A and B.
    seen = collections.Counter()
    for seed in range(60):
        for s in build_knowledge_stimuli(_FACTS, seed=seed):
            seen[s.correct_label] += 1
    assert seen["A"] > 0 and seen["B"] > 0


def test_knowledge_n_trials_cap():
    stims = build_knowledge_stimuli(_FACTS, seed=1, n_trials=2)
    assert len(stims) == 2


def test_line_stimuli_have_counterbalanced_wrong_target():
    crit = [2, 3, 5]
    targets = collections.Counter()
    for seed in range(80):
        stims = build_line_stimuli(6, crit, seed=seed)
        for s in stims:
            if s.is_critical:
                assert s.wrong_label is not None
                assert s.wrong_label != s.correct_label
                assert s.wrong_label in s.options
                targets[s.wrong_label] += 1
            else:
                assert s.wrong_label is None
    # the target is not pinned to a single letter (the Phase-1 confound)
    assert len(targets) >= 2


def test_real_corpus_loads_and_is_consistent():
    corpus = load_knowledge_corpus(
        "psychbench/experiments/conformity/corpus/knowledge_facts.yaml"
    )
    assert len(corpus) >= 20
    for f in corpus.facts:
        assert f.correct_answer and f.wrong_answer
        assert f.correct_answer != f.wrong_answer
