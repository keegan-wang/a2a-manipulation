"""Pub-sub hook for live progress events from sessions and trials.

Sessions/trials call `emit(event_name, payload)` at key points; listeners
registered via `subscribe(fn)` receive them. If nobody is subscribed, `emit`
is essentially free.

The built-in `stderr_printer` formats events as one-line-per-event to stderr
so you can follow an experiment live in a terminal or a Jupyter cell.

Events emitted by the framework:

    session_start   {session_label, n_trials}
    session_end     {session_label, elapsed_s, n_trials}
    trial_start     {trial_index, is_critical, correct_answer, session_label}
    agent_response  {trial_index, agent_id, position, parsed_answer,
                     raw_text, agent_type, session_label}
    trial_end       {trial_index, scoring, session_label}
"""
from __future__ import annotations

import sys
import time
from typing import Any, Callable

Listener = Callable[[str, dict[str, Any]], None]

_listeners: list[Listener] = []


def subscribe(listener: Listener) -> None:
    _listeners.append(listener)


def unsubscribe(listener: Listener) -> None:
    try:
        _listeners.remove(listener)
    except ValueError:
        pass


def clear() -> None:
    _listeners.clear()


def emit(event: str, payload: dict[str, Any]) -> None:
    if not _listeners:
        return
    for lst in list(_listeners):
        try:
            lst(event, payload)
        except Exception:
            # Listeners must never break the run.
            pass


def _truncate(s: str, n: int = 80) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def stderr_printer(event: str, payload: dict[str, Any]) -> None:
    """Human-readable live view written to stderr (unbuffered)."""
    ts = time.strftime("%H:%M:%S")
    out = sys.stderr
    label = payload.get("session_label", "")

    if event == "session_start":
        print(
            f"\n[{ts}] ▶ SESSION start  {label!r}  "
            f"n_trials={payload['n_trials']}",
            file=out, flush=True,
        )
    elif event == "session_end":
        print(
            f"[{ts}] ■ SESSION end    {label!r}  "
            f"elapsed={payload.get('elapsed_s', 0):.1f}s\n",
            file=out, flush=True,
        )
    elif event == "trial_start":
        crit = "CRITICAL" if payload.get("is_critical") else "filler  "
        correct = payload.get("correct_answer")
        correct_str = _truncate(str(correct), 40) if correct is not None else "?"
        print(
            f"[{ts}]   trial {payload['trial_index']:>3} {crit}  "
            f"correct={correct_str!r}",
            file=out, flush=True,
        )
    elif event == "agent_response":
        aid = payload.get("agent_id", "?")
        parsed = payload.get("parsed_answer")
        raw = payload.get("raw_text", "") or ""
        if parsed is not None and len(str(parsed)) <= 3:
            shown = f"→ {parsed}"
        else:
            shown = f"→ {_truncate(raw, 70)!r}"
        print(f"                    {aid:>18}  {shown}", file=out, flush=True)
    elif event == "trial_end":
        s = payload.get("scoring") or {}
        if "conformed" in s:
            if s.get("conformed"):
                marker = "✗ CONFORMED"
            else:
                marker = "✓ independent"
            naive = s.get("naive_answer")
            major = s.get("confederate_answer")
            print(
                f"                         {marker}  "
                f"(naive={naive!r}, majority={major!r})",
                file=out, flush=True,
            )
        elif "full_conformity" in s:
            full = s.get("full_conformity")
            heur = s.get("partial_conformity_heuristic")
            judge = s.get("partial_conformity_judge")
            unan = s.get("unanimity")
            naive = _truncate(str(s.get("naive_answer", "")), 60)
            print(
                f"                         full={full} heur={heur} "
                f"judge={judge} unan={unan}  naive={naive!r}",
                file=out, flush=True,
            )
