"""ConformityExperiment — faithful Asch clone for LLMs (knowledge + lines).

Each repeat runs the same items in two conditions:
- **alone** (private visibility): the subject answers with no peers visible —
  its independent belief and the capability baseline.
- **group** (public visibility): confederates answer first with a unanimous,
  counterbalanced wrong target; the subject answers last.

Joining the two per item yields the headline measure — the
**induced-hallucination rate** (right alone, conformed under the group) — plus
the classic conformity/abstention rates and the alone-vs-group delta.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from psychbench.agents.model_agent import ModelAgent
from psychbench.agents.scripted_agent import ScriptedAgent
from psychbench.framework.agent import BaseAgent
from psychbench.framework.backends import get_backend
from psychbench.framework.environment import Environment
from psychbench.framework.experiment import (
    BaseExperiment, register_experiment,
)
from psychbench.framework.progress import emit
from psychbench.framework.session import Session
from psychbench.framework.types import ResponseVisibility, Stimulus

from .asch_classic import ASCH_GROUP_SIZE, build_asch_classic_stimuli
from .corpus import load_knowledge_corpus, load_story_corpus
from .prompts import build_conformity_prompt
from .scoring import (
    join_induced_hallucination, parse_choice, score_conformity_trial,
    summarize_condition,
)
from .stimuli import (
    build_knowledge_stimuli, build_line_stimuli, build_story_stimuli,
)

_DEFAULT_CRITICAL_INDICES = [2, 3, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17]


@register_experiment("conformity")
class ConformityExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._task_type = config["experiment"].get("task_type", "knowledge")
        self._corpus = None
        self._story_items = None
        if self._task_type == "knowledge":
            self._corpus = load_knowledge_corpus(config["corpus"]["path"])
        elif self._task_type == "story":
            self._story_items = load_story_corpus(config["corpus"]["path"])

    def build_stimuli(self, seed_override: int | None = None) -> list[Stimulus]:
        exp = self.config["experiment"]
        seed = seed_override if seed_override is not None else exp.get("seed", 0)
        if self._task_type == "knowledge":
            return build_knowledge_stimuli(
                self._corpus.facts, seed=seed,
                n_trials=exp.get("trials"),
            )
        if self._task_type == "story":
            return build_story_stimuli(
                self._story_items, seed=seed, n_trials=exp.get("trials"),
            )
        if self._task_type == "lines_classic":
            # Asch's fixed designed 18-trial set (faithful replication).
            return build_asch_classic_stimuli(seed=seed)
        return build_line_stimuli(
            n_trials=int(exp.get("trials", 18)),
            critical_indices=list(
                exp.get("critical_trial_indices", _DEFAULT_CRITICAL_INDICES)
            ),
            seed=seed,
        )

    def build_agents(
        self, observability: str | None = None,
    ) -> list[BaseAgent]:
        agents_cfg = self.config["agents"]
        confed_cfg = agents_cfg["confederates"]
        naive_cfg = agents_cfg["naive"]
        # Asch's standard majority is 7; the randomized line/knowledge tasks
        # default to a smaller 5 unless the config overrides it.
        default_count = (
            ASCH_GROUP_SIZE if self._task_type == "lines_classic" else 5
        )
        n_conf = int(confed_cfg.get("count", default_count))
        dissenter = bool(confed_cfg.get("dissenter", False))

        confederates: list[BaseAgent] = []
        for i in range(n_conf):
            is_dissenter = dissenter and i == 0
            confederates.append(ScriptedAgent(
                agent_id=f"confederate_{i+1}",
                position=i,
                behavior=(
                    "always_correct" if is_dissenter
                    else "stimulus_wrong_on_critical"
                ),
            ))
            confederates[-1].metadata["dissenter"] = is_dissenter

        naive_backend = get_backend(
            naive_cfg.get("backend", "echo"),
            naive_cfg.get("model", "echo-test"),
            temperature=float(naive_cfg.get("temperature", 0.0)),
            max_tokens=int(naive_cfg.get("max_tokens", 64)),
        )
        naive = ModelAgent(
            agent_id="naive",
            position=n_conf,
            backend=naive_backend,
            stateful=bool(naive_cfg.get("stateful", False)),
            prompt_builder=lambda ctx: build_conformity_prompt(ctx, observability),
            answer_parser=lambda raw: parse_choice(raw),
        )
        return [*confederates, naive]

    def _condition_summary(
        self, label: str, visibility: str, stimuli: list[Stimulus],
        run_dir: Path, max_concurrency: int, resume: bool,
        observability: str | None = None,
    ) -> dict[str, Any]:
        """Run one condition session (or load it if resuming).

        Returns the condition summary dict (with a ``per_trial`` row list). On
        resume, a previously-completed session's summary is loaded from disk and
        the model calls are skipped entirely.
        """
        log_path = run_dir / f"conformity_{label}.jsonl"
        summary_path = run_dir / f"conformity_{label}.summary.json"
        if resume and summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text())
                if "per_trial" in loaded:
                    emit("session_skipped",
                         {"session_label": label, "reason": "resume"})
                    return loaded
            except (ValueError, OSError):
                pass  # corrupt/partial — re-run below
        agents = self.build_agents(observability)
        env = Environment(visibility=ResponseVisibility(visibility))
        session = Session(
            stimuli=stimuli, agents=agents, environment=env,
            log_path=log_path, summary_path=summary_path,
            config_snapshot=self.config,
            score_trial=lambda t, _lbl=label: score_conformity_trial(t, _lbl),
            summarize=lambda trials, _lbl=label: summarize_condition(
                trials, _lbl
            ),
            session_label=label,
            max_concurrency=max_concurrency,
        )
        session.run()
        return json.loads(summary_path.read_text())

    def run(
        self, output_dir: str | Path, run_id: str | None = None,
        resume: bool | None = None, max_concurrency: int | None = None,
    ) -> dict[str, Any]:
        timestamp = int(time.time())
        exp = self.config["experiment"]
        run_cfg = self.config.get("run", {})
        run_id = (
            run_id or exp.get("run_id") or f"conformity_{timestamp}"
        )
        resume = (
            resume if resume is not None
            else bool(run_cfg.get("resume", False))
        )
        max_concurrency = int(
            max_concurrency if max_concurrency is not None
            else run_cfg.get("max_concurrency", 1)
        )
        run_dir = Path(output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        base_seed = int(exp.get("seed", 0))
        n_repeats = max(1, int(exp.get("n_repeats", 1)))
        # Normative-pressure manipulation (Asch public/private): when on, the
        # group condition splits into public vs private, peer info held constant.
        normative = bool(exp.get("observability_manipulation", False))

        repeats: list[dict[str, Any]] = []
        for r in range(n_repeats):
            stimuli = self.build_stimuli(seed_override=base_seed + r)
            suffix = f"_r{r}" if n_repeats > 1 else ""
            alone = self._condition_summary(
                f"alone{suffix}", "private", stimuli, run_dir,
                max_concurrency, resume,
            )
            if normative:
                # Same unanimous-wrong majority shown in both; only the
                # subject's own-answer observability differs.
                group_public = self._condition_summary(
                    f"group_public{suffix}", "public", stimuli, run_dir,
                    max_concurrency, resume, observability="public",
                )
                group_private = self._condition_summary(
                    f"group_private{suffix}", "public", stimuli, run_dir,
                    max_concurrency, resume, observability="private",
                )
                repeats.append({
                    "repeat": r,
                    "seed": base_seed + r,
                    "alone": alone,
                    "group_public": group_public,
                    "group_private": group_private,
                    "induced": join_induced_hallucination(
                        alone["per_trial"], group_public["per_trial"],
                    ),
                })
            else:
                group = self._condition_summary(
                    f"group{suffix}", "public", stimuli, run_dir,
                    max_concurrency, resume,
                )
                repeats.append({
                    "repeat": r,
                    "seed": base_seed + r,
                    "alone": alone,
                    "group": group,
                    "induced": join_induced_hallucination(
                        alone["per_trial"], group["per_trial"],
                    ),
                })

        headline = (
            _aggregate_repeats_normative(repeats) if normative
            else _aggregate_repeats(repeats)
        )
        summary = {
            "experiment": "conformity",
            "task_type": self._task_type,
            "run_id": run_id,
            "timestamp": timestamp,
            "n_repeats": n_repeats,
            "max_concurrency": max_concurrency,
            "observability_manipulation": normative,
            "headline": headline,
            "repeats": repeats,
            "config": self.config,
        }
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        summary["summary_path"] = str(summary_path)
        summary["run_dir"] = str(run_dir)
        return summary


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the headline rates across repeats (each repeat = a fresh seed)."""
    alone_acc = [r["alone"]["accuracy"] for r in repeats]
    group_conf = [r["group"]["conformity_rate"] for r in repeats]
    group_abst = [r["group"]["abstention_rate"] for r in repeats]
    induced = [r["induced"]["induced_hallucination_rate"] for r in repeats]
    n_known = sum(r["induced"]["n_known_alone"] for r in repeats)
    n_induced = sum(r["induced"]["n_induced_hallucinations"] for r in repeats)
    return {
        "baseline_accuracy_alone": _mean(alone_acc),
        "group_conformity_rate": _mean(group_conf),
        "group_abstention_rate": _mean(group_abst),
        # Pooled across repeats: induced hallucinations / items known alone.
        "induced_hallucination_rate": (
            n_induced / n_known if n_known else 0.0
        ),
        "pooled_n_known_alone": n_known,
        "pooled_n_induced_hallucinations": n_induced,
        # Delta = how much group pressure lowered correctness vs. alone.
        "conformity_delta_vs_alone": _mean(group_conf),
    }


def _aggregate_repeats_normative(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool the public/private observability conditions across repeats.

    The headline is the **normative gap**: extra conformity when the subject's
    answer is observed vs. private, with the peer information held identical.
    Gap ~0 => the model shows no normative-pressure analog (it only does
    informational deference); gap > 0 => an observable normative-like effect.
    """
    alone_acc = [r["alone"]["accuracy"] for r in repeats]
    pub = [r["group_public"]["conformity_rate"] for r in repeats]
    priv = [r["group_private"]["conformity_rate"] for r in repeats]
    n_known = sum(r["induced"]["n_known_alone"] for r in repeats)
    n_induced = sum(r["induced"]["n_induced_hallucinations"] for r in repeats)
    pub_m, priv_m = _mean(pub), _mean(priv)
    return {
        "baseline_accuracy_alone": _mean(alone_acc),
        "conformity_public": pub_m,
        "conformity_private": priv_m,
        # informational floor = conformity that survives going private;
        # normative gap = the public-only surplus on top of it.
        "informational_conformity": priv_m,
        "normative_gap": round(pub_m - priv_m, 4),
        # induced hallucination measured under the public (full-pressure) arm.
        "induced_hallucination_rate": (
            n_induced / n_known if n_known else 0.0
        ),
        "pooled_n_known_alone": n_known,
        "pooled_n_induced_hallucinations": n_induced,
    }
