"""Asch (1951) line-length conformity experiment."""
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
from psychbench.framework.experiment import BaseExperiment, register_experiment
from psychbench.framework.session import Session
from psychbench.framework.types import ResponseVisibility, Stimulus, TrialResult

from .prompts import build_asch_prompt
from .scoring import score_asch_trial, summarize_asch_session
from .stimuli import generate_asch_stimuli


@register_experiment("asch")
class AschExperiment(BaseExperiment):
    def build_stimuli(self, seed_override: int | None = None) -> list[Stimulus]:
        exp_cfg = self.config["experiment"]
        seed = seed_override if seed_override is not None else exp_cfg.get("seed", 0)
        return generate_asch_stimuli(
            n_trials=exp_cfg["trials"],
            critical_indices=list(exp_cfg["critical_trial_indices"]),
            seed=seed,
        )

    def build_agents(self, *, for_control: bool = False) -> list[BaseAgent]:
        agents_cfg = self.config["agents"]
        confed_cfg = agents_cfg["confederates"]
        naive_cfg = agents_cfg["naive"]
        n_conf = int(confed_cfg.get("count", 5))
        dissenter = bool(confed_cfg.get("dissenter", False))
        behavior = confed_cfg.get("behavior", "always_wrong_on_critical")
        wrong_letter = confed_cfg.get("wrong_answer", "B")

        confederates: list[BaseAgent] = []
        for i in range(n_conf):
            if dissenter and i == 0:
                agent_behavior = "always_correct"
                wrong = None
            else:
                agent_behavior = behavior
                wrong = (
                    wrong_letter
                    if behavior == "always_wrong_on_critical"
                    else None
                )
            confederates.append(ScriptedAgent(
                agent_id=f"confederate_{i+1}",
                position=i,
                behavior=agent_behavior,
                wrong_answer=wrong,
            ))

        from psychbench.interpretability.config import (
            resolve_interpretability,
        )
        interp_cfg = resolve_interpretability(self.config)
        if interp_cfg is not None and not for_control:
            from psychbench.interpretability.backend import (
                TransformerLensBackend,
            )
            from psychbench.interpretability.collector import (
                ActivationCollector,
            )
            naive_backend = TransformerLensBackend(
                model=interp_cfg.model, device=interp_cfg.device,
            )
            collector = ActivationCollector(
                layers=interp_cfg.layers,
                max_new_tokens=interp_cfg.max_new_tokens,
            )
        else:
            naive_backend = get_backend(
                naive_cfg.get("backend", "echo"),
                naive_cfg.get("model", "echo-test"),
            )
            collector = None

        position_cfg = naive_cfg.get("position", "last")
        naive_position = (
            n_conf if position_cfg == "last" else int(position_cfg)
        )
        naive = ModelAgent(
            agent_id="naive",
            position=naive_position,
            backend=naive_backend,
            stateful=bool(naive_cfg.get("stateful", False)),
            prompt_builder=build_asch_prompt,
            activation_collector=collector,
        )

        return [*confederates, naive]

    def _environment(self, *, for_control: bool) -> Environment:
        if for_control:
            vis = self.config.get("control", {}).get(
                "response_visibility", "private"
            )
        else:
            vis = self.config["environment"]["response_visibility"]
        return Environment(visibility=ResponseVisibility(vis))

    def run(self, output_dir: str | Path) -> dict[str, Any]:
        timestamp = int(time.time())
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exp_cfg = self.config["experiment"]
        base_seed = int(exp_cfg.get("seed", 0))
        n_repeats = max(1, int(exp_cfg.get("n_repeats", 1)))

        summary: dict[str, Any] = {}

        for label, for_control in self._conditions():
            per_repeat: list[dict[str, Any]] = []
            for r in range(n_repeats):
                agents = self.build_agents(for_control=for_control)
                stimuli = self.build_stimuli(seed_override=base_seed + r)
                env = self._environment(for_control=for_control)

                suffix = f"_r{r}" if n_repeats > 1 else ""
                log_path = out_dir / f"asch_{label}_{timestamp}{suffix}.jsonl"
                summary_path = (
                    out_dir / f"asch_{label}_{timestamp}{suffix}.summary.json"
                )

                confederate_ids = [
                    a.agent_id for a in agents
                    if a.agent_id.startswith("confederate_")
                ]

                def _score(
                    trial: TrialResult,
                    _confederate_ids: list[str] = confederate_ids,
                ) -> dict[str, Any]:
                    return score_asch_trial(
                        trial, "naive", _confederate_ids
                    )

                session = Session(
                    stimuli=stimuli, agents=agents, environment=env,
                    log_path=log_path, summary_path=summary_path,
                    config_snapshot=self.config,
                    score_trial=_score,
                    summarize=summarize_asch_session,
                    session_label=(
                        label if n_repeats == 1 else f"{label}_r{r}"
                    ),
                )
                results = session.run()
                per_repeat.append({
                    "repeat": r,
                    "seed": base_seed + r,
                    "log_path": str(log_path),
                    "summary_path": str(summary_path),
                    "n_trials": len(results),
                })

            cond_summary: dict[str, Any] = {
                "n_repeats": n_repeats,
                "repeats": per_repeat,
            }
            if n_repeats == 1:
                cond_summary["log_path"] = per_repeat[0]["log_path"]
                cond_summary["summary_path"] = per_repeat[0]["summary_path"]
                cond_summary["n_trials"] = per_repeat[0]["n_trials"]
            else:
                aggregate = self._aggregate_repeats(per_repeat)
                agg_path = (
                    out_dir / f"asch_{label}_{timestamp}.aggregate.json"
                )
                agg_path.write_text(json.dumps(aggregate, indent=2))
                cond_summary["aggregate"] = aggregate
                cond_summary["aggregate_path"] = str(agg_path)

            summary[label] = cond_summary

        if len(summary) == 2:
            summary["comparison"] = self._load_comparison(summary)
        return summary

    @staticmethod
    def _aggregate_repeats(
        per_repeat: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Aggregate conformity rates across repeats: mean / std / individual."""
        rates: list[float] = []
        n_conformed_total = 0
        n_critical_total = 0
        ever_conformed = False
        for r in per_repeat:
            s = json.loads(Path(r["summary_path"]).read_text())
            rate = float(s.get("conformity_rate") or 0.0)
            rates.append(rate)
            n_conformed_total += int(s.get("n_conformed") or 0)
            n_critical_total += int(s.get("n_critical") or 0)
            ever_conformed = ever_conformed or bool(s.get("ever_conformed"))

        n = len(rates)
        mean = sum(rates) / n if n else 0.0
        var = (
            sum((x - mean) ** 2 for x in rates) / (n - 1) if n > 1 else 0.0
        )
        std = var ** 0.5
        sem = std / (n ** 0.5) if n else 0.0
        return {
            "n_repeats": n,
            "per_repeat_rate": rates,
            "mean_conformity_rate": mean,
            "std_conformity_rate": std,
            "sem_conformity_rate": sem,
            "pooled_n_conformed": n_conformed_total,
            "pooled_n_critical": n_critical_total,
            "pooled_conformity_rate": (
                n_conformed_total / n_critical_total
                if n_critical_total else 0.0
            ),
            "ever_conformed_any": ever_conformed,
        }

    def _conditions(self) -> list[tuple[str, bool]]:
        run_ctrl = bool(
            self.config.get("control", {}).get("run_control", False)
        )
        conds = [("experimental", False)]
        if run_ctrl:
            conds.append(("control", True))
        return conds

    def _load_comparison(self, summary: dict[str, Any]) -> dict[str, Any]:
        def _rate(cond: dict[str, Any]) -> tuple[float, float | None]:
            """Return (point_estimate, std_across_repeats_or_None)."""
            agg = cond.get("aggregate")
            if agg is not None:
                return (
                    float(agg["mean_conformity_rate"]),
                    float(agg["std_conformity_rate"]),
                )
            s = json.loads(Path(cond["summary_path"]).read_text())
            return float(s.get("conformity_rate") or 0.0), None

        exp_rate, exp_std = _rate(summary["experimental"])
        ctrl_rate, ctrl_std = _rate(summary["control"])
        block: dict[str, Any] = {
            "experimental_conformity_rate": exp_rate,
            "control_conformity_rate": ctrl_rate,
            "delta": exp_rate - ctrl_rate,
        }
        if exp_std is not None and ctrl_std is not None:
            block["experimental_std"] = exp_std
            block["control_std"] = ctrl_std
            block["n_repeats"] = summary["experimental"].get("n_repeats", 1)
        return block
