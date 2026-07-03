#!/usr/bin/env bash
# Run the causal factorial for the 7 new unified-panel models.
# Bedrock and OpenAI streams run in parallel (independent services); models
# within a stream run sequentially. All runs are resume-safe (stable run_id).
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env 2>/dev/null; set +a
export PYTHONPATH="$(pwd)"
PY=.venv/bin/python
LOG=results/reports/unified_causal_run.log
mkdir -p results/reports
: > "$LOG"

run_one () {
  local cfg="config/experiments/conformity_causal_powered_$1.yaml"
  echo "[$(printf '%(%H:%M:%S)T')] START $1" >> "$LOG"
  $PY -m psychbench run --config "$cfg" --i-know >> "$LOG" 2>&1
  echo "[$(printf '%(%H:%M:%S)T')] DONE  $1 (exit $?)" >> "$LOG"
}

bedrock_stream () { for m in bedrock_llama31_8b bedrock_llama33_70b bedrock_deepseek bedrock_mistral; do run_one "$m"; done; }
openai_stream  () { for m in openai_gpt35turbo openai_gpt5nano openai_gpt54mini; do run_one "$m"; done; }

bedrock_stream &
B=$!
openai_stream &
O=$!
wait $B $O
echo "[$(printf '%(%H:%M:%S)T')] ALL CAUSAL RUNS COMPLETE" >> "$LOG"
