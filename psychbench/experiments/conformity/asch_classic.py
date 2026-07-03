"""Asch (1956) faithful stimulus set + procedure constants.

This module encodes Asch's standard line-judgment paradigm as closely as the
medium allows, as a *fixed* 18-trial sequence (not randomized):

  * 18 trials total, **12 critical** (the majority answers unanimously wrong)
    and **6 neutral** (the majority answers correctly);
  * the **first two trials are neutral**, so the situation is established as
    normal before any pressure is applied (Asch's procedure);
  * each trial shows one *standard* line and **three comparison lines**
    (A/B/C); exactly one comparison is an exact match and the other two are
    grossly discrepant, so the task is trivial alone (Asch's control error was
    <1%);
  * on critical trials the majority's wrong choice varies in *direction*
    (sometimes longer, sometimes shorter than the standard) and *magnitude*,
    and the correct letter varies across A/B/C — so neither "correct" nor the
    conformity target is tied to a fixed position.

PROCEDURE FIDELITY. The procedure mirrors Asch verbatim: a unanimous majority of
``ASCH_GROUP_SIZE`` (7) that answers *before* the subject, public/oral/sequential
responses, the subject responding after hearing the majority, the 12/6
critical/neutral split with the first two neutral, and a separate alone control.

WHAT CANNOT BE MIRRORED. Two gaps are intrinsic to using an LLM and are stated
honestly rather than papered over:
  1. *Modality.* Asch's subjects saw physical lines; an LLM is given the lengths
     as text. We preserve the property that matters — the answer is unambiguous
     when judged alone — but the judgment is symbolic, not perceptual.
  2. *Normative pressure.* A co-present human fears the group's disapproval; an
     LLM answering into a prompt has no social stake. This isolates *informational*
     conformity. State this caveat in any write-up.

LINE-LENGTH VALUES. The specific lengths below reproduce Asch's documented
*properties* (standards ~2-9; one exact match; distractors off by >=1.5; majority
errors of varied sign and size) but are a faithful **reconstruction**, not the
verbatim table. Asch (1956), Psychological Monographs 70(9), Whole No. 416 gives
the exact per-trial lengths; substitute them here when finalizing for
publication. The procedure above does not depend on the exact values.
"""
from __future__ import annotations

from typing import Any

from psychbench.framework.types import Stimulus

# Asch's standard majority size (confederates). The subject is the lone naive
# participant, who answers after the majority.
ASCH_GROUP_SIZE = 7
N_TRIALS = 18
N_CRITICAL = 12
N_NEUTRAL = N_TRIALS - N_CRITICAL

# Fixed 18-trial design. ``majority`` is the label the unanimous majority picks
# on critical trials (always != correct); ``None`` marks a neutral trial (the
# majority answers correctly). Trials 0 and 1 (the first two) are neutral.
ASCH_1956_TRIALS: list[dict[str, Any]] = [
    # idx 0  neutral
    {"standard": 3.0, "A": 3.0, "B": 5.0, "C": 7.0, "correct": "A", "majority": None},
    # idx 1  neutral
    {"standard": 8.0, "A": 10.0, "B": 8.0, "C": 5.5, "correct": "B", "majority": None},
    # idx 2  critical  (majority longer by 1.5)
    {"standard": 5.0, "A": 5.0, "B": 6.5, "C": 3.0, "correct": "A", "majority": "B"},
    # idx 3  critical  (majority shorter by 2.0)
    {"standard": 4.0, "A": 2.0, "B": 4.0, "C": 6.0, "correct": "B", "majority": "A"},
    # idx 4  critical  (majority longer by 1.5)
    {"standard": 6.5, "A": 8.0, "B": 4.5, "C": 6.5, "correct": "C", "majority": "A"},
    # idx 5  neutral
    {"standard": 2.0, "A": 2.0, "B": 4.0, "C": 6.0, "correct": "A", "majority": None},
    # idx 6  critical  (majority shorter by 2.0)
    {"standard": 7.0, "A": 7.0, "B": 9.0, "C": 5.0, "correct": "A", "majority": "C"},
    # idx 7  critical  (majority longer by 1.5)
    {"standard": 3.5, "A": 5.0, "B": 3.5, "C": 2.0, "correct": "B", "majority": "A"},
    # idx 8  critical  (majority longer by 2.0)
    {"standard": 9.0, "A": 6.5, "B": 11.0, "C": 9.0, "correct": "C", "majority": "B"},
    # idx 9  neutral
    {"standard": 4.5, "A": 4.5, "B": 2.5, "C": 7.0, "correct": "A", "majority": None},
    # idx 10 critical  (majority longer by 2.0)
    {"standard": 6.0, "A": 6.0, "B": 8.0, "C": 4.0, "correct": "A", "majority": "B"},
    # idx 11 critical  (majority shorter by 2.0)
    {"standard": 5.5, "A": 3.5, "B": 7.0, "C": 5.5, "correct": "C", "majority": "A"},
    # idx 12 critical  (majority longer by 2.5)
    {"standard": 8.0, "A": 8.0, "B": 6.0, "C": 10.5, "correct": "A", "majority": "C"},
    # idx 13 neutral
    {"standard": 3.0, "A": 5.0, "B": 3.0, "C": 1.5, "correct": "B", "majority": None},
    # idx 14 critical  (majority longer by 2.0)
    {"standard": 7.5, "A": 9.5, "B": 7.5, "C": 5.5, "correct": "B", "majority": "A"},
    # idx 15 critical  (majority longer by 2.0)
    {"standard": 5.0, "A": 7.0, "B": 3.0, "C": 5.0, "correct": "C", "majority": "A"},
    # idx 16 neutral
    {"standard": 4.0, "A": 4.0, "B": 6.0, "C": 2.0, "correct": "A", "majority": None},
    # idx 17 critical  (majority longer by 2.0)
    {"standard": 6.5, "A": 4.5, "B": 6.5, "C": 8.5, "correct": "B", "majority": "C"},
]


def validate_trials(trials: list[dict[str, Any]] = ASCH_1956_TRIALS) -> None:
    """Fail fast if the fixed table drifts from Asch's design invariants."""
    if len(trials) != N_TRIALS:
        raise ValueError(f"Asch set must have {N_TRIALS} trials, got {len(trials)}")
    n_crit = sum(1 for t in trials if t["majority"] is not None)
    if n_crit != N_CRITICAL:
        raise ValueError(f"Asch set must have {N_CRITICAL} critical, got {n_crit}")
    if trials[0]["majority"] is not None or trials[1]["majority"] is not None:
        raise ValueError("The first two trials must be neutral (Asch procedure)")
    for i, t in enumerate(trials):
        opts = {"A": t["A"], "B": t["B"], "C": t["C"]}
        matches = [lab for lab, v in opts.items() if v == t["standard"]]
        if matches != [t["correct"]]:
            raise ValueError(
                f"Trial {i}: exactly one comparison must equal the standard "
                f"and it must be the correct label; got matches={matches}, "
                f"correct={t['correct']}"
            )
        if t["majority"] is not None and t["majority"] == t["correct"]:
            raise ValueError(f"Trial {i}: majority target equals correct label")


# Validate at import so a bad edit to the table is caught immediately.
validate_trials()


def build_asch_classic_stimuli(seed: int = 0) -> list[Stimulus]:
    """Asch's fixed 18-trial line set as ``Stimulus`` objects.

    ``seed`` is accepted for interface parity with the randomized builders and
    recorded in metadata, but the set itself is fixed (Asch ran one designed
    sequence); vary behavior across "subjects" via ``n_repeats`` + temperature.
    """
    stims: list[Stimulus] = []
    for i, t in enumerate(ASCH_1956_TRIALS):
        is_critical = t["majority"] is not None
        stims.append(Stimulus(
            trial_index=i,
            is_critical=is_critical,
            reference_cm=float(t["standard"]),
            options={"A": float(t["A"]), "B": float(t["B"]), "C": float(t["C"])},
            correct_label=t["correct"],
            wrong_label=t["majority"],   # None on neutral trials
            metadata={
                "task_type": "lines_classic",
                "neutral": not is_critical,
                "seed": seed,
            },
        ))
    return stims
