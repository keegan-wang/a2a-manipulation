#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env 2>/dev/null; set +a
export PYTHONPATH="$(pwd)"
PY=.venv/bin/python
LOG=results/reports/openrouter_extra_run.log
mkdir -p results/reports
: > "$LOG"

causal_stream() {
  for m in openrouter_gemini25flash openrouter_qwen3_235b; do
    echo "[$(date +%H:%M:%S)] START causal $m" >> "$LOG"
    $PY -m psychbench run --config config/experiments/conformity_causal_powered_$m.yaml --i-know >> "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] DONE  causal $m (exit $?)" >> "$LOG"
  done
}
matrix_stream() {
  echo "[$(date +%H:%M:%S)] START matrix (resume; +gemini/qwen3 cells)" >> "$LOG"
  $PY -m psychbench matrix --config config/experiments/conformity_dialogue_matrix_unified_aws.yaml --output-dir results/ >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  matrix (exit $?)" >> "$LOG"
}
causal_stream & C=$!
matrix_stream & M=$!
wait $C $M
echo "[$(date +%H:%M:%S)] ALL OPENROUTER-EXTRA RUNS COMPLETE" >> "$LOG"
