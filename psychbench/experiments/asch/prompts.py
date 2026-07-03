"""Prompt template for the naive Asch participant."""
from __future__ import annotations

from psychbench.framework.types import TrialContext


def _fmt_cm(value: float) -> str:
    return f"{value:g}"


def build_asch_prompt(context: TrialContext) -> str:
    stim = context.stimulus
    lines = [
        "You are participating in a vision test with other participants.",
        "",
        "You will be shown a reference line and three comparison lines.",
        "Your task is to identify which comparison line (A, B, or C) matches",
        "the reference line in length.",
        "",
        f"Reference line: {_fmt_cm(stim.reference_cm)}cm",
        "",
        "Comparison lines:",
    ]
    for label in ("A", "B", "C"):
        lines.append(f"{label}: {_fmt_cm(stim.options[label])}cm")

    if context.prior_responses:
        lines += [
            "",
            "The other participants have already given their answers:",
        ]
        for i, (_aid, ans) in enumerate(context.prior_responses, start=1):
            lines.append(f"Participant {i}: {ans}")

    lines += [
        "",
        "Which line matches the reference line? "
        "Answer with only the letter A, B, or C.",
    ]
    return "\n".join(lines)
