"""Invariants for the made-up-story corpus — the methodology depends on these.

For the task to measure conformity (not capability), each item must be: right
alone (correct answer stated in the passage) and a genuine substitution under
pressure (plausible wrong answer that is NOT in the passage). A drift here would
silently turn "induced hallucination" into "the model never knew it".
"""
from __future__ import annotations

from psychbench.experiments.conformity.corpus import load_story_corpus

CORPUS = "psychbench/experiments/conformity/corpus/story_facts.yaml"


def test_corpus_is_large_enough_for_power():
    items = load_story_corpus(CORPUS)
    assert len(items) >= 40


def test_every_correct_answer_is_in_its_passage():
    for it in load_story_corpus(CORPUS):
        assert it.correct_answer.lower() in it.passage.lower(), (
            f"{it.id}: correct answer {it.correct_answer!r} not stated in passage"
        )


def test_no_wrong_answer_leaks_into_its_passage():
    for it in load_story_corpus(CORPUS):
        assert it.wrong_answer.lower() not in it.passage.lower(), (
            f"{it.id}: wrong answer {it.wrong_answer!r} appears in the passage"
        )


def test_distractors_are_distinct_and_nonempty():
    for it in load_story_corpus(CORPUS):
        assert it.correct_answer and it.wrong_answer
        assert it.correct_answer != it.wrong_answer
        assert it.question.strip().endswith("?")
