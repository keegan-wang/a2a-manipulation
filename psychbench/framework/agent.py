"""Abstract base class for all agent types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import AgentResponse, TrialContext


class BaseAgent(ABC):
    """Shared interface for scripted, model-backed, and hybrid agents."""

    def __init__(self, agent_id: str, position: int,
                 metadata: dict[str, Any] | None = None) -> None:
        self.agent_id = agent_id
        self.position = position
        self.metadata = metadata or {}

    @abstractmethod
    def respond(self, context: TrialContext) -> AgentResponse:
        """Produce a response for the given trial context."""

    def reset(self) -> None:
        """Clear any per-session state. Default no-op."""
        return None
