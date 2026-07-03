"""Stimulus generation for the conformity clone — knowledge and line tasks.

Both task types carry a per-item, counterbalanced ``wrong_label`` (the
conformity target the confederates assert on critical trials). Counterbalancing
the target across option letters fixes the Phase-1 confound where the target
was always "B" (and never "C"), which let a model's letter bias masquerade as
conformity.
"""
from __future__ import annotations

import random

from psychbench.experiments.asch.stimuli import generate_asch_stimuli
from psychbench.framework.types import Stimulus

from .corpus import Fact, StoryItem


def build_knowledge_stimuli(
    facts: list[Fact],
    seed: int = 0,
    n_trials: int | None = None,
) -> list[Stimulus]:
    """One two-option (correct vs. plausible-wrong) MC item per fact.

    The correct answer is placed at a seed-counterbalanced letter so neither
    "correct" nor the conformity target is tied to a fixed position.
    """
    rng = random.Random(seed)
    chosen = facts if n_trials is None else facts[:n_trials]
    stims: list[Stimulus] = []
    for i, f in enumerate(chosen):
        if rng.random() < 0.5:
            correct_label, wrong_label = "A", "B"
        else:
            correct_label, wrong_label = "B", "A"
        options = {
            correct_label: f.correct_answer,
            wrong_label: f.wrong_answer,
        }
        stims.append(Stimulus(
            trial_index=i,
            is_critical=True,      # every knowledge item is a pressure trial
            reference_cm=0.0,      # unused for knowledge
            options=options,
            correct_label=correct_label,
            wrong_label=wrong_label,
            question=f.question,
            metadata={
                "fact_id": f.id,
                "correct_answer": f.correct_answer,
                "wrong_answer": f.wrong_answer,
                "task_type": "knowledge",
                "seed": seed,
            },
        ))
    return stims


def build_story_stimuli(
    items: list[StoryItem],
    seed: int = 0,
    n_trials: int | None = None,
) -> list[Stimulus]:
    """One two-option MC item per made-up-story fact, passage carried along.

    Same counterbalancing as the knowledge task (correct answer at a
    seed-chosen letter), but each stimulus also carries its ``passage`` in
    metadata so the prompt can show the story the model is judging.
    """
    rng = random.Random(seed)
    chosen = items if n_trials is None else items[:n_trials]
    stims: list[Stimulus] = []
    for i, it in enumerate(chosen):
        if rng.random() < 0.5:
            correct_label, wrong_label = "A", "B"
        else:
            correct_label, wrong_label = "B", "A"
        options = {
            correct_label: it.correct_answer,
            wrong_label: it.wrong_answer,
        }
        stims.append(Stimulus(
            trial_index=i,
            is_critical=True,      # every story item is a pressure trial
            reference_cm=0.0,      # unused for story
            options=options,
            correct_label=correct_label,
            wrong_label=wrong_label,
            question=it.question,
            metadata={
                "fact_id": it.id,
                "passage": it.passage,
                "correct_answer": it.correct_answer,
                "wrong_answer": it.wrong_answer,
                "task_type": "story",
                "seed": seed,
            },
        ))
    return stims


def build_line_stimuli(
    n_trials: int,
    critical_indices: list[int],
    seed: int = 0,
) -> list[Stimulus]:
    """Line-length items (the literal Asch task) with a counterbalanced target.

    Reuses the Phase-1 length generator, then assigns each critical trial a
    ``wrong_label`` drawn uniformly from that trial's distractor letters.
    """
    base = generate_asch_stimuli(n_trials, critical_indices, seed=seed)
    rng = random.Random(seed + 7919)
    for s in base:
        s.metadata = {**s.metadata, "task_type": "lines"}
        if s.is_critical:
            distractors = sorted(
                label for label in s.options if label != s.correct_label
            )
            s.wrong_label = rng.choice(distractors)
    return base
