"""Scripted answer, model-generated surface text (natural-sounding confederate)."""
from __future__ import annotations

from typing import Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import ModelBackend
from psychbench.framework.types import AgentResponse, TrialContext


AnswerFn = Callable[[TrialContext], str]
SurfacePromptBuilder = Callable[[TrialContext, str], str]


class HybridAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        answer_fn: AnswerFn,
        backend: ModelBackend,
        surface_prompt_builder: SurfacePromptBuilder,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            position=position,
            metadata={"type": "hybrid", "model": backend.model},
        )
        self.answer_fn = answer_fn
        self.backend = backend
        self.surface_prompt_builder = surface_prompt_builder

    def respond(self, context: TrialContext) -> AgentResponse:
        answer = self.answer_fn(context)
        prompt = self.surface_prompt_builder(context, answer)
        raw = self.backend.generate(prompt, stateful=False)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=raw,
            parsed_answer=answer,
            prompt=prompt,
            metadata={"model": self.backend.model, "hybrid": True},
        )

    def reset(self) -> None:
        self.backend.reset()
