"""Core dataclasses and enums shared across the framework."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseVisibility(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    PARTIAL = "partial"


@dataclass
class Stimulus:
    trial_index: int
    is_critical: bool
    reference_cm: float
    options: dict[str, Any]
    correct_label: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Generalized fields (default None so line-task callers are unaffected):
    # `question` carries the prompt text for non-line tasks (e.g. knowledge);
    # `wrong_label` is the per-item, counterbalanced conformity target the
    # confederates assert on critical trials (fixes the always-"B" confound).
    question: str | None = None
    wrong_label: str | None = None


@dataclass
class TrialContext:
    stimulus: Stimulus
    agent_position: int
    agent_id: str
    prior_responses: list[tuple[str, str]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    agent_id: str
    raw_text: str
    parsed_answer: str | None
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_index: int
    is_critical: bool
    stimulus: Stimulus
    responses: list[AgentResponse]
    conformed: bool | None
    naive_answer: str | None
    confederate_answer: str | None
    correct_answer: str
