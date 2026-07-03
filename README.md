# Agent-to-Agent Manipulation Through Delegated Evidence

Code and data for the paper **"Agent-to-Agent Manipulation Through Delegated Evidence"** (Keegan Wang, 2026).

Can one model agent convince another to abandon a correct answer? We introduce a controlled evaluation of agent-to-agent (A2A) manipulation: a subject model answers a fictional-fact question correctly from a source passage, then an attacker model tries to induce a *harmful revision* — a correct initial answer flipped to the attacker-favored wrong one. This repository contains the full experiment framework, run configs, raw per-trial transcripts, and analysis scripts behind every number and figure in the paper.

## Findings

1. **Directed influence is target-defined.** In a 14-model pairwise influence map, target resistance explains 88% of pairwise variance in harmful revision; attacker strength explains 2%.
2. **Concrete evidence access is protective.** A visible source passage — or the target's own retained rationale — protects capable models against static false claims (e.g. GPT-4o-mini: 25/180 → 180/180 harmful revisions when the retained rationale is removed under a hidden source).
3. **Live adaptation defeats rationale-only defenses.** A live persuader that reads the subject's reasoning raises harmful revision from 12% to 99% (GPT-4o-mini) and 2% to 56% (Claude Haiku 4.5); a neutral live challenger causes almost none.
4. **Delegated evidence breaks most models, including frontier models.** Claude Opus 4.6 and Sonnet 4.6 reject a fabricated source in every trial when it arrives as a peer message, but accept the same fabrication as a delegated sub-agent tool result (167/180 and 106/180 harmful revisions).
5. **Grounded verification stops manipulation and propagation.** A verifier eliminates the attack only when it restores source evidence; asserting the correct answer is not enough. Grounding also stops a corrupted worker's report from propagating to a downstream aggregator (aggregator error 59/60 → 0/60).

## Repository layout

```
psychbench/            experiment framework (runner, backends, agents)
  experiments/
    conformity/        all paper experiments + the fictional-fact corpus
config/experiments/    YAML configs for every paper run (+ cheap *_smoke.yaml)
scripts/               analysis, statistics, figure and table generation
results/               raw per-trial transcripts + summaries for every paper run
results/reports/       aggregated statistics consumed by figures/tables
tests/                 unit tests (no API calls)
figures/, tables/      regenerated outputs (see below)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                # offline; verifies the install
```

## Reproducing the paper from the shipped data

All raw run data is included, so every figure, table, and statistic regenerates offline:

```bash
python scripts/network_analysis.py    # variance decomposition -> results/reports/network_analysis.json
python scripts/causal_stats.py        # appendix factorial stats -> results/reports/causal_stats.json
python scripts/a2a_stats.py           # paired McNemar tests quoted in the text (prints)
python scripts/make_a2a_figures.py    # Figs: source x rationale, live A2A, channel, verifier+cascade, decomposition
python scripts/make_paper_figures.py  # Figs: influence map + appendix figures
python scripts/make_tables.py         # appendix tables -> tables/tables.tex
python scripts/make_causal_tables.py  # appendix factorial tables -> tables/tables_causal.tex
```

## Experiment ↔ config ↔ data map

| Paper section | Experiment | Configs / entry point | Run data |
|---|---|---|---|
| §4.1 influence map | 14×14 live dialogue matrix | `python -m psychbench matrix --config config/experiments/conformity_dialogue_matrix_unified_aws.yaml` | `results/dialogue_matrix_unified_aws/` |
| §4.2 evidence access | source × rationale factorial | `python -m psychbench run --config config/experiments/conformity_causal_srt_<model>.yaml` | `results/causal_srt_*/` |
| §4.3 live adaptation | static vs live vs neutral | `conformity_causal_livea2a_{gpt4omini,haiku45}.yaml` | `results/livea2a_*/` |
| §4.4 delegated evidence | peer vs sub-agent channel | `python scripts/probe_subagent_channel.py` | `results/subagent_channel_powered/` |
| §4.5 verification | grounded vs ungrounded verifier | `conformity_causal_verifier_*.yaml` | `results/causal_verifier_*/` |
| §4.6 propagation | attacker→worker→aggregator cascade | `python scripts/run_cascade.py` | `results/reports/cascade_main.json` |
| App. B factorial | 14-model static factorial | `conformity_causal_powered_*.yaml` (see `scripts/run_unified_causal.sh`) | `results/causal_powered_*/` |
| App. adaptivity/verification controls | replay + forced-source | `conformity_causal_adaptivity_*.yaml`, `conformity_causal_verification_*.yaml` | `results/causal_adaptivity_*/`, `results/causal_verification_*/` |

## Re-running experiments from scratch

Runs call provider APIs and cost real money (a powered 14-model condition set is thousands of trials). Copy `.env.example` to `.env` and fill in the providers you need, then start with a smoke config:

```bash
python -m psychbench run --config config/experiments/conformity_causal_srt_smoke.yaml --verbose
```

Full-scale runs are orchestrated by the shell helpers in `scripts/` (`run_unified_causal.sh`, `run_source_rationale.sh`, `run_unified_matrix.sh`). Runs are resume-safe: re-running a config with the same `run_id` skips completed trials.

Bedrock runs require an explicit `AWS_PROFILE`; an optional spend guard (`PSYCHBENCH_DENIED_AWS_PROFILE_PREFIXES`, `PSYCHBENCH_DENIED_AWS_ACCOUNT_IDS`) refuses runs against accounts you list.

## Data format

Each run directory contains:

- `causal.jsonl` (or `dialogue.jsonl`) — one JSON object per trial: item id, condition name, full prompt/response transcript, parsed initial and final answers, and the `harmful_revision` outcome.
- `summary.json` — aggregated per-condition counts (`headline.cells.<condition> = {x, n}`) plus the resolved run config.

The task corpus (180 fictional-fact items, two-option questions, counterbalanced wrong answers) lives in `psychbench/experiments/conformity/corpus/`.

## Citation

```bibtex
@article{wang2026a2a,
  title  = {Agent-to-Agent Manipulation Through Delegated Evidence},
  author = {Wang, Keegan},
  year   = {2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
