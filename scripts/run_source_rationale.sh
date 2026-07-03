#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."
set -a; . ./.env 2>/dev/null; set +a
export PYTHONPATH="$(pwd)"; PY=.venv/bin/python
LOG=results/reports/source_rationale_run.log; mkdir -p results/reports; : > "$LOG"
run_one(){ echo "[$(date +%H:%M:%S)] START $1" >>"$LOG"; $PY -m psychbench run --config config/experiments/conformity_causal_srt_$1.yaml --i-know >>"$LOG" 2>&1; echo "[$(date +%H:%M:%S)] DONE $1 ($?)" >>"$LOG"; }
bedrock(){ for m in bedrock_haiku45 bedrock_deepseek bedrock_mistral bedrock_llama31_8b; do run_one "$m"; done; }
openai(){ run_one openai_gpt4omini; }
bedrock & openai & wait
echo "[$(date +%H:%M:%S)] ALL SOURCE-RATIONALE DONE" >>"$LOG"
