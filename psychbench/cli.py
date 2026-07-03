"""Command-line interface: `python -m psychbench run|matrix|panel|estimate`."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import experiments so their @register_experiment decorators run.
from psychbench.experiments import asch  # noqa: F401
from psychbench.experiments import conformity  # noqa: F401
from psychbench.config import load_config
from psychbench.framework.cost import estimate_cost
from psychbench.framework.experiment import get_experiment_class
from psychbench.framework.sweep import expand_sweep


COST_GATE = 5000


def _cost_gate_total(cfg: dict) -> int:
    sweep_fields = list(cfg.get("sweep", {}).get("fields", []))
    if not sweep_fields:
        n_cells = 1
    else:
        cells = expand_sweep(cfg, sweep_fields=sweep_fields)
        n_cells = len(cells)
    n_repeats = int(cfg.get("experiment", {}).get("n_repeats", 1))
    n_trials = int(cfg.get("experiment", {}).get("trials", 1))
    return n_cells * n_repeats * n_trials


def _cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "verbose", False):
        from psychbench.framework.progress import stderr_printer, subscribe
        subscribe(stderr_printer)

    cfg = load_config(args.config)
    exp_type = cfg["experiment"]["type"]

    if cfg.get("sweep", {}).get("fields"):
        total = _cost_gate_total(cfg)
        if total > COST_GATE and not args.i_know:
            print(
                f"Sweep would run {total} trials (>{COST_GATE}). "
                f"Re-run with --i-know to proceed.",
                file=sys.stderr,
            )
            return 2

    out_dir = Path(
        args.output_dir
        or cfg.get("logging", {}).get("output_dir", "results")
    )
    exp_cls = get_experiment_class(exp_type)
    exp = exp_cls(cfg)
    summary = exp.run(output_dir=out_dir)
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_panel(args: argparse.Namespace) -> int:
    if getattr(args, "verbose", False):
        from psychbench.framework.progress import stderr_printer, subscribe
        subscribe(stderr_printer)
    from psychbench.experiments.conformity.panel import (
        estimate_panel_calls, format_leaderboard, run_panel,
    )
    cfg = load_config(args.config)
    calls = estimate_panel_calls(cfg)
    if calls > COST_GATE and not args.i_know:
        print(
            f"Panel would make ~{calls} model calls (>{COST_GATE}). "
            f"Re-run with --i-know to proceed.",
            file=sys.stderr,
        )
        return 2
    out_dir = Path(
        args.output_dir or cfg.get("logging", {}).get("output_dir", "results")
    )
    leaderboard = run_panel(cfg, output_dir=out_dir)
    print(format_leaderboard(leaderboard))
    print(f"\nleaderboard -> {leaderboard['panel_dir']}/leaderboard.json")
    return 0


def _cmd_matrix(args: argparse.Namespace) -> int:
    if getattr(args, "verbose", False):
        from psychbench.framework.progress import stderr_printer, subscribe
        subscribe(stderr_printer)
    from psychbench.experiments.conformity.dialogue import (
        format_matrix, run_dialogue_matrix,
    )
    cfg = load_config(args.config)
    out_dir = Path(
        args.output_dir or cfg.get("logging", {}).get("output_dir", "results")
    )
    matrix = run_dialogue_matrix(cfg, output_dir=out_dir)
    print(format_matrix(matrix))
    print(f"\nmatrix -> {matrix['matrix_dir']}/matrix.json")
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(json.dumps(estimate_cost(cfg), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psychbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run an experiment from a YAML config")
    pr.add_argument("--config", required=True)
    pr.add_argument("--output-dir", default=None)
    pr.add_argument(
        "--i-know", action="store_true",
        help="Bypass cost gate for large sweeps",
    )
    pr.add_argument(
        "--verbose", "-v", action="store_true",
        help="Stream live trial/agent events to stderr as the run proceeds",
    )
    pr.set_defaults(func=_cmd_run)

    pe = sub.add_parser(
        "estimate", help="Estimate the naive-model dollar cost of a config"
    )
    pe.add_argument("--config", required=True)
    pe.set_defaults(func=_cmd_estimate)

    pp = sub.add_parser(
        "panel", help="Run the conformity challenge across a panel of models"
    )
    pp.add_argument("--config", required=True)
    pp.add_argument("--output-dir", default=None)
    pp.add_argument(
        "--i-know", action="store_true", help="Bypass the panel call-count gate"
    )
    pp.add_argument("--verbose", "-v", action="store_true")
    pp.set_defaults(func=_cmd_panel)

    pm = sub.add_parser(
        "matrix", help="Run the cross-model agent-persuasion conformity matrix"
    )
    pm.add_argument("--config", required=True)
    pm.add_argument("--output-dir", default=None)
    pm.add_argument("--verbose", "-v", action="store_true")
    pm.set_defaults(func=_cmd_matrix)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
