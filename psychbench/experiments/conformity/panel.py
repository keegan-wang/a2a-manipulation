"""Run the conformity challenge across a panel of models -> a leaderboard.

A panel config is an ordinary `conformity_challenge` config plus a top-level
``panel.models`` list of ``{backend, model[, label, temperature]}`` specs. Each
model is run with the same task/corpus/arms; results are collected into one
leaderboard.json (and a printed table). Provider keys must be in the environment
(OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ...).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from .challenge import ConformityChallengeExperiment
from .human_benchmark import human_similarity


def estimate_panel_calls(config: dict[str, Any]) -> int:
    """Rough naive-call count: models x arms x items x (rounds+1) x repeats."""
    panel = config.get("panel", {})
    n_models = len(panel.get("models", []))
    exp = config.get("experiment", {})
    ch = config.get("challenge", {})
    n_arms = len(ch.get("framing_arms") or [None]) or 1
    if config.get("experiment", {}).get("observability_manipulation"):
        n_arms = max(n_arms, 3)
    n_items = int(exp.get("trials") or 40)
    n_rounds = int(ch.get("n_rounds", 3))
    n_repeats = max(1, int(exp.get("n_repeats", 1)))
    return n_models * n_arms * n_items * (n_rounds + 1) * n_repeats


def _model_label(spec: dict[str, Any]) -> str:
    if spec.get("label"):
        return spec["label"]
    base = spec["model"]
    eff = spec.get("reasoning_effort")
    return f"{base} ({eff})" if eff else base


def run_panel(
    config: dict[str, Any], output_dir: str | Path,
    timestamp: int | None = None,
) -> dict[str, Any]:
    panel = config.get("panel", {})
    models = panel.get("models")
    if not models:
        raise ValueError(
            "panel config needs panel.models: a list of {backend, model}"
        )
    ts = timestamp if timestamp is not None else int(time.time())
    name = config["experiment"].get("name", "panel")
    panel_id = config["experiment"].get("run_id") or f"{name}_{ts}"
    panel_dir = Path(output_dir) / panel_id
    panel_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in models:
        cfg = copy.deepcopy(config)
        cfg.pop("panel", None)
        naive = cfg.setdefault("agents", {}).setdefault("naive", {})
        naive["backend"] = spec["backend"]
        naive["model"] = spec["model"]
        for k in ("temperature", "max_tokens", "reasoning_effort"):
            if k in spec:
                naive[k] = spec[k]
        naive.setdefault("max_tokens", 16)
        naive.setdefault("stateful", True)
        label = _model_label(spec)
        cfg["experiment"]["run_id"] = (
            f"{panel_id}__{label.replace('/', '_').replace(':', '_')}"
        )
        summary = ConformityChallengeExperiment(cfg).run(output_dir=panel_dir)
        rows.append(_leaderboard_row(label, spec, summary))

    leaderboard = {
        "panel_id": panel_id, "timestamp": ts, "n_models": len(rows),
        "rows": rows, "config": config,
    }
    (panel_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    leaderboard["panel_dir"] = str(panel_dir)
    return leaderboard


def _leaderboard_row(
    label: str, spec: dict[str, Any], summary: dict[str, Any],
) -> dict[str, Any]:
    h = summary["headline"]
    row: dict[str, Any] = {
        "model": label, "backend": spec["backend"],
        "run_dir": summary.get("run_dir"),
        "n_failed": summary.get("n_failed", 0),
        "baseline_accuracy_alone": h.get("baseline_accuracy_alone"),
    }
    if "arms" in h:                                   # framing-ablation shape
        arms = h["arms"]
        for arm, a in arms.items():
            row[f"induced_{arm}"] = a["induced_hallucination_rate"]
        if "public" in arms and "private" in arms:
            row["human_similarity"] = human_similarity({
                "baseline": arms["public"]["induced_hallucination_rate"],
                "private": arms["private"]["induced_hallucination_rate"],
            })["human_similarity_score"]
    elif "induced_public" in h:                       # normative 3-arm shape
        row["induced_neutral"] = h.get("induced_neutral")
        row["induced_public"] = h.get("induced_public")
        row["induced_private"] = h.get("induced_private")
        row["normative_gap_induced"] = h.get("normative_gap_induced")
        row["human_similarity"] = (
            h.get("human_similarity", {}).get("human_similarity_score")
        )
    else:                                             # plain single-pass shape
        row["induced_hallucination_rate"] = h.get("induced_hallucination_rate")
    return row


def format_leaderboard(leaderboard: dict[str, Any]) -> str:
    rows = leaderboard["rows"]
    if not rows:
        return "(no models)"
    cols = [k for k in rows[0] if k not in ("model", "backend", "run_dir")]
    out = [f"{'model':<26} " + " ".join(f"{c:>14}" for c in cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(
                f"{v:>14.3f}" if isinstance(v, (int, float)) else f"{str(v):>14}"
            )
        out.append(f"{r['model']:<26} " + " ".join(cells))
    return "\n".join(out)
