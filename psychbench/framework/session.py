"""Orchestrate a full session: run all trials, log contexts, write summary."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .agent import BaseAgent
from .environment import Environment
from .logging_utils import JsonlLogger, write_summary
from .progress import emit
from .trial import run_trial
from .types import AgentResponse, Stimulus, TrialResult


def _concurrency_is_safe(agents: list[BaseAgent]) -> bool:
    """Trials may run concurrently only if no agent carries per-call state.

    Stateful conversation history and the interpretability activation collector
    both mutate shared model state, so those sessions must run serially.
    """
    for a in agents:
        if getattr(a, "stateful", False):
            return False
        if getattr(a, "activation_collector", None) is not None:
            return False
    return True


class Session:
    def __init__(
        self,
        stimuli: list[Stimulus],
        agents: list[BaseAgent],
        environment: Environment,
        log_path: str | Path,
        summary_path: str | Path,
        config_snapshot: dict[str, Any],
        score_trial: Callable[[TrialResult], dict[str, Any]] | None = None,
        summarize: Callable[[list[TrialResult]], dict[str, Any]] | None = None,
        session_label: str = "session",
        max_concurrency: int = 1,
    ) -> None:
        self.stimuli = stimuli
        self.agents = agents
        self.environment = environment
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path)
        self.config_snapshot = config_snapshot
        self.score_trial = score_trial
        self.summarize = summarize
        self.session_label = session_label
        self.max_concurrency = max(1, int(max_concurrency))

    def _process_trial(
        self, stim: Stimulus, env: Environment,
    ) -> tuple[TrialResult, dict[str, Any]]:
        result = run_trial(
            stim, self.agents, env, session_label=self.session_label,
        )
        if self.score_trial is not None:
            scores = self.score_trial(result)
            for k, v in scores.items():
                if hasattr(result, k):
                    setattr(result, k, v)
            return result, scores
        return result, {}

    def run(self) -> list[TrialResult]:
        for agent in self.agents:
            agent.reset()
        concurrent = self.max_concurrency > 1 and _concurrency_is_safe(
            self.agents
        )
        emit(
            "session_start",
            {
                "session_label": self.session_label,
                "n_trials": len(self.stimuli),
                "max_concurrency": self.max_concurrency if concurrent else 1,
            },
        )
        t0 = time.time()

        # Each trial gets its own environment so the per-trial response buffer
        # is never shared across threads. Stateless agents are shared safely.
        if concurrent:
            processed: list[tuple[TrialResult, dict[str, Any]] | None] = (
                [None] * len(self.stimuli)
            )
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
                futures = {
                    ex.submit(
                        self._process_trial, stim, self.environment.clone(),
                    ): i
                    for i, stim in enumerate(self.stimuli)
                }
                for fut in futures:
                    i = futures[fut]
                    processed[i] = fut.result()
            ordered = [p for p in processed if p is not None]
        else:
            ordered = [
                self._process_trial(stim, self.environment)
                for stim in self.stimuli
            ]

        results: list[TrialResult] = []
        with JsonlLogger(self.log_path) as log:
            for result, result_extra in ordered:
                emit(
                    "trial_end",
                    {
                        "trial_index": result.trial_index,
                        "scoring": result_extra,
                        "session_label": self.session_label,
                    },
                )
                log.write(self._serialize_trial(result, result_extra))
                results.append(result)
        summary: dict[str, Any] = {
            "session_label": self.session_label,
            "timestamp": time.time(),
            "n_trials": len(results),
            "config": self.config_snapshot,
        }
        if self.summarize is not None:
            summary.update(self.summarize(results))
        write_summary(self.summary_path, summary)
        emit(
            "session_end",
            {
                "session_label": self.session_label,
                "n_trials": len(results),
                "elapsed_s": time.time() - t0,
            },
        )
        return results

    def _serialize_trial(
        self, result: TrialResult, extra: dict[str, Any]
    ) -> dict[str, Any]:
        responses_payload: list[dict[str, Any]] = []
        interp_block: dict[str, Any] | None = None
        for r in result.responses:
            metadata = dict(r.metadata) if r.metadata else {}
            record = metadata.pop("interpretability_record", None)
            stripped = AgentResponse(
                agent_id=r.agent_id,
                raw_text=r.raw_text,
                parsed_answer=r.parsed_answer,
                prompt=r.prompt,
                metadata=metadata,
            )
            responses_payload.append(asdict(stripped))
            if record is not None and interp_block is None:
                rel_path = (
                    f"activations/{self.session_label}/"
                    f"trial_{result.trial_index:03d}.npz"
                )
                try:
                    from psychbench.interpretability.storage import (
                        save_activation_record,
                    )
                    save_activation_record(
                        record, self.log_path.parent / rel_path,
                    )
                    interp_block = {
                        "activations_path": rel_path,
                        "layers": list(record.layers),
                        "n_prompt_tokens": int(record.n_prompt_tokens),
                        "token_positions": dict(record.token_positions),
                    }
                except OSError:
                    interp_block = None
        payload: dict[str, Any] = {
            "trial_index": result.trial_index,
            "is_critical": result.is_critical,
            "stimulus": asdict(result.stimulus),
            "correct_answer": result.correct_answer,
            "responses": responses_payload,
            "scoring": extra,
            "environment": self.environment.snapshot(),
            "session_label": self.session_label,
            "timestamp": time.time(),
        }
        if interp_block is not None:
            payload["interpretability"] = interp_block
        return payload
