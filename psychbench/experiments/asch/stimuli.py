"""Generate line-length comparison stimuli for Asch Phase 1."""
from __future__ import annotations

import random

from psychbench.framework.types import Stimulus


def generate_asch_stimuli(
    n_trials: int,
    critical_indices: list[int],
    seed: int = 0,
    reference_min: float = 5.0,
    reference_max: float = 15.0,
    wrong_gap_min: float = 2.0,
) -> list[Stimulus]:
    """Produce ``n_trials`` Asch-style line-length stimuli.

    Reference line length is uniform in [reference_min, reference_max] rounded
    to the nearest 0.5cm. Exactly one option equals the reference; the other
    two differ by at least ``wrong_gap_min`` cm.
    """
    rng = random.Random(seed)
    stimuli: list[Stimulus] = []
    critical_set = set(critical_indices)
    labels = ["A", "B", "C"]
    for i in range(n_trials):
        reference = round(rng.uniform(reference_min, reference_max) * 2) / 2
        correct_label = rng.choice(labels)
        wrong_labels = [label for label in labels if label != correct_label]
        wrong_lengths = _pick_unambiguous_wrong_lengths(
            rng, reference, reference_min, reference_max, wrong_gap_min,
        )
        options = {correct_label: reference}
        for label, length in zip(wrong_labels, wrong_lengths):
            options[label] = length
        stimuli.append(Stimulus(
            trial_index=i,
            is_critical=i in critical_set,
            reference_cm=reference,
            options=options,
            correct_label=correct_label,
            metadata={"seed": seed},
        ))
    return stimuli


def _pick_unambiguous_wrong_lengths(
    rng: random.Random,
    reference: float,
    ref_min: float,
    ref_max: float,
    gap_min: float,
) -> list[float]:
    low = max(1.0, ref_min - 4.0)
    high = ref_max + 4.0
    picks: list[float] = []
    attempts = 0
    while len(picks) < 2 and attempts < 500:
        attempts += 1
        candidate = round(rng.uniform(low, high) * 2) / 2
        if abs(candidate - reference) < gap_min:
            continue
        if any(abs(candidate - p) < gap_min for p in picks):
            continue
        picks.append(candidate)
    if len(picks) < 2:
        picks = [reference + 3.0, reference - 3.0]
        picks = [p if p > 0 else reference + 4.0 for p in picks]
    return picks
