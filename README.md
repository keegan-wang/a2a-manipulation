# Agentic systems are only as safe as their weakest link

Code and data for [**“Agentic systems are only as safe as their weakest link”**](https://marginsofintelligence.com/research-logs/agent-to-agent-manipulation/), by Anantika Mannby and Keegan Wang (2026).

Can one model agent convince another to abandon a correct answer? This paper introduces a controlled evaluation of agent-to-agent (A2A) manipulation. A subject model first answers a fictional-fact question correctly from a source passage; an attacker then tries to induce a *harmful revision*: a correct initial answer changed to the specific wrong answer favored by the attacker.

The experiments progress from static messages to live adaptive dialogue, fabricated evidence returned by a delegated sub-agent, and an attacker–worker–aggregator pipeline. They test which models can be moved, how access to evidence changes their resistance, whether the delivery channel affects what they trust, how corruption propagates, and whether verification stops it.

This repository contains the experiment framework, configurations, raw per-trial transcripts, aggregated results, and analysis scripts behind the paper.

## Findings

1. **Directed influence is subject-governed.** In a 14-model pairwise influence map, subject-model resistance explains 88.3% of pairwise variance in harmful revision, pair-specific structure 9.3%, and attacker strength 2.4%.
2. **Evidence access controls manipulation.** With the source passage visible, capable models do not revise under static false claims. With the source hidden, removing the subject's retained rationale raises harmful revision from 13.9% to 100% for GPT-4o-mini and from 0.6% to 99.4% for Claude Haiku 4.5.
3. **Live adaptation defeats rationale-only defenses.** A live persuader that reads the subject's reasoning raises harmful revision from 12.2% to 98.9% for GPT-4o-mini and from 2.2% to 55.6% for Claude Haiku 4.5. A neutral live challenger causes almost none.
4. **Delegated evidence creates a model-dependent channel failure.** Claude Opus 4.6 and Sonnet 4.6 reject a fabricated source on every trial when it arrives as a peer message, but accept the same fabrication as a delegated sub-agent tool result on 92.8% and 58.9% of trials. GPT-5.4 shows no comparable asymmetry.
5. **Corruption propagates through the pipeline.** An attacked worker causes a downstream aggregator that resisted direct attack to answer incorrectly on 98.3% of trials (59/60), even though the aggregator never sees the attack.
6. **Grounded verification stops both the attack and the cascade.** A verifier that restores the source passage reduces harmful revision to 0% for every tested subject and reduces pipeline error from 98.3% to 0%. A verifier that merely asserts the correct answer is inconsistent.

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

All raw run data is included, so the analyses, figures, and tables can be regenerated offline:

```bash
python scripts/network_analysis.py    # variance decomposition -> results/reports/network_analysis.json
python scripts/causal_stats.py        # factorial statistics -> results/reports/causal_stats.json
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
| §4.5 propagation | attacker→worker→aggregator cascade | `python scripts/run_cascade.py` | `results/reports/cascade_main.json` |
| §4.6 verification | grounded vs ungrounded verifier | `conformity_causal_verifier_*.yaml` | `results/causal_verifier_*/` |

Additional static-factorial, adaptivity, and forced-source verification runs are available under `config/experiments/` with their corresponding data under `results/`.

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
@misc{mannbywang2026weakestlink,
  title  = {Agentic systems are only as safe as their weakest link},
  author = {Mannby, Anantika and Wang, Keegan},
  year   = {2026},
  url    = {https://marginsofintelligence.com/research-logs/agent-to-agent-manipulation/},
}
```

## License

MIT — see [LICENSE](LICENSE).
