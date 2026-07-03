"""LLM-backed agent with optional cross-trial stateful history."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import ModelBackend
from psychbench.framework.types import AgentResponse, TrialContext


if TYPE_CHECKING:  # pragma: no cover
    from psychbench.interpretability.collector import ActivationCollector


PromptBuilder = Callable[[TrialContext], str]


def parse_letter_answer(text: str) -> str | None:
    """Extract A/B/C from model output.

    Prefer a final standalone letter (handles verbose responses ending in
    the letter), fall back to the first standalone letter anywhere.
    """
    stripped = text.strip()
    m_final = re.search(r"\b([ABC])\b\W*$", stripped)
    if m_final:
        return m_final.group(1)
    m_any = re.search(r"\b([ABC])\b", stripped)
    if m_any:
        return m_any.group(1)
    return None


class ModelAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str,
        position: int,
        backend: ModelBackend,
        stateful: bool,
        prompt_builder: PromptBuilder,
        activation_collector: "ActivationCollector | None" = None,
        answer_parser: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            position=position,
            metadata={
                "type": "model",
                "model": backend.model,
                "stateful": stateful,
            },
        )
        self.backend = backend
        self.stateful = stateful
        self.prompt_builder = prompt_builder
        self.activation_collector = activation_collector
        self.answer_parser = answer_parser or parse_letter_answer

    def respond(self, context: TrialContext) -> AgentResponse:
        prompt = self.prompt_builder(context)
        if self.activation_collector is not None:
            raw, record = self.activation_collector.collect(
                self.backend.hooked_model, prompt, token_labels=None,
            )
            metadata = {
                "model": self.backend.model,
                "stateful": self.stateful,
                "interpretability_record": record,
            }
        else:
            raw = self.backend.generate(prompt, stateful=self.stateful)
            metadata = {
                "model": self.backend.model,
                "stateful": self.stateful,
            }
        parsed = self.answer_parser(raw)
        return AgentResponse(
            agent_id=self.agent_id,
            raw_text=raw,
            parsed_answer=parsed,
            prompt=prompt,
            metadata=metadata,
        )

    def reset(self) -> None:
        self.backend.reset()
