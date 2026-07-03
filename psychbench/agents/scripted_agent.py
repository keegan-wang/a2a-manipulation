"""Non-LLM rule-based agent used for confederates."""
from __future__ import annotations

from typing import Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.types import AgentResponse, TrialContext


Behavior = str
# "always_correct" | "always_wrong_on_critical"
# | "stimulus_wrong_on_critical" | "custom_fn"


class ScriptedAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        behavior: Behavior,
        wrong_answer: str | None = None,
        custom_fn: Callable[[TrialContext], str] | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            position=position,
            metadata={"type": "scripted", "behavior": behavior},
        )
        self.behavior = behavior
        self.wrong_answer = wrong_answer
        self.custom_fn = custom_fn
        if behavior == "custom_fn" and custom_fn is None:
            raise ValueError("behavior='custom_fn' requires custom_fn callable")

    def respond(self, context: TrialContext) -> AgentResponse:
        answer = self._pick_answer(context)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=answer,
            parsed_answer=answer,
            prompt="",
            metadata={"scripted": True, "behavior": self.behavior},
        )

    def _pick_answer(self, context: TrialContext) -> str:
        if self.behavior == "always_correct":
            return context.stimulus.correct_label
        if self.behavior == "always_wrong_on_critical":
            if not context.stimulus.is_critical:
                return context.stimulus.correct_label
            correct = context.stimulus.correct_label
            # Prefer the configured wrong answer when it's actually wrong,
            # but fall back to the first other option (deterministic across
            # confederates) on trials where the configured letter happens to
            # be correct. Without this, confederates silently "agree" with the
            # ground truth on a fraction of critical trials and the naive is
            # never under Asch pressure there.
            if (
                self.wrong_answer is not None
                and self.wrong_answer != correct
                and self.wrong_answer in context.stimulus.options
            ):
                return self.wrong_answer
            for label in sorted(context.stimulus.options):
                if label != correct:
                    return label
            raise RuntimeError("No wrong options available")
        if self.behavior == "stimulus_wrong_on_critical":
            # Read the per-item, counterbalanced wrong target off the stimulus
            # so the conformity target varies across A/B/C instead of being a
            # fixed letter. Correct on non-critical trials.
            if not context.stimulus.is_critical:
                return context.stimulus.correct_label
            target = context.stimulus.wrong_label
            if target is None or target == context.stimulus.correct_label:
                raise ValueError(
                    "stimulus_wrong_on_critical needs a wrong_label that "
                    "differs from correct_label on critical trials"
                )
            return target
        if self.behavior == "custom_fn":
            assert self.custom_fn is not None
            return self.custom_fn(context)
        raise ValueError(f"Unknown behavior: {self.behavior}")
