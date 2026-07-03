"""Abstract base for experiments + lightweight string-keyed registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .agent import BaseAgent
from .types import AgentResponse, Stimulus


EXPERIMENT_REGISTRY: dict[str, type["BaseExperiment"]] = {}


def register_experiment(
    name: str,
) -> Callable[[type["BaseExperiment"]], type["BaseExperiment"]]:
    def decorator(cls: type["BaseExperiment"]) -> type["BaseExperiment"]:
        EXPERIMENT_REGISTRY[name] = cls
        cls.experiment_name = name
        return cls
    return decorator


def get_experiment_class(name: str) -> type["BaseExperiment"]:
    if name not in EXPERIMENT_REGISTRY:
        raise KeyError(
            f"Unknown experiment '{name}'. "
            f"Registered: {list(EXPERIMENT_REGISTRY)}"
        )
    return EXPERIMENT_REGISTRY[name]


class BaseExperiment(ABC):
    """Subclasses define how to build stimuli, agents, and score trials."""

    experiment_name: str = "base"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def build_stimuli(self) -> list[Stimulus]:
        """Return the full ordered list of stimuli for the session."""

    @abstractmethod
    def build_agents(self) -> list[BaseAgent]:
        """Return the ordered list of agents (response order = list order)."""

    def score_trial(
        self, stimulus: Stimulus, responses: list[AgentResponse]
    ) -> dict[str, Any]:
        """Override to add experiment-specific scoring. Default: empty dict."""
        return {}
