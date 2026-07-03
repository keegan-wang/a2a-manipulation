"""Controls what each agent sees of other agents' responses each trial."""
from __future__ import annotations

from typing import Any

from .types import ResponseVisibility, Stimulus


class Environment:
    """Sequential-response environment with configurable visibility.

    Each trial, agents respond in position order. `visible_prior_responses`
    tells each agent what it is allowed to see when its turn comes up.
    """

    def __init__(
        self,
        visibility: ResponseVisibility = ResponseVisibility.PUBLIC,
        per_agent_visibility: dict[str, str] | None = None,
    ) -> None:
        self.visibility = visibility
        self.per_agent_visibility = per_agent_visibility or {}
        self._current_stimulus: Stimulus | None = None
        self._trial_responses: list[tuple[str, int, str]] = []

    def clone(self) -> "Environment":
        """A fresh environment with the same visibility config, no trial state.

        Used to give each trial its own environment when trials run
        concurrently (the per-trial response buffer is not thread-safe).
        """
        return Environment(
            visibility=self.visibility,
            per_agent_visibility=dict(self.per_agent_visibility),
        )

    def begin_trial(self, stimulus: Stimulus) -> None:
        self._current_stimulus = stimulus
        self._trial_responses = []

    def record_response(
        self, agent_id: str, position: int, answer: str
    ) -> None:
        self._trial_responses.append((agent_id, position, answer))

    def visible_prior_responses(
        self, agent_id: str, position: int
    ) -> list[tuple[str, str]]:
        mode = self._resolve_mode(agent_id)
        if mode == ResponseVisibility.PRIVATE:
            return []
        return [
            (aid, ans)
            for aid, pos, ans in self._trial_responses
            if pos < position
        ]

    def _resolve_mode(self, agent_id: str) -> ResponseVisibility:
        if self.visibility != ResponseVisibility.PARTIAL:
            return self.visibility
        per = self.per_agent_visibility.get(agent_id, "private")
        return ResponseVisibility(per)

    def snapshot(self) -> dict[str, Any]:
        return {
            "visibility": self.visibility.value,
            "per_agent_visibility": dict(self.per_agent_visibility),
            "current_trial_responses": list(self._trial_responses),
        }
