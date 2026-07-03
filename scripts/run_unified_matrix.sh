#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
set -a; . ./.env 2>/dev/null; set +a
export PYTHONPATH="$(pwd)"
LOG=results/reports/unified_matrix_run.log
mkdir -p results/reports
echo "[$(date +%H:%M:%S)] START unified matrix (12 models, n=20, 144 cells)" > "$LOG"
.venv/bin/python -m psychbench matrix \
  --config config/experiments/conformity_dialogue_matrix_unified_aws.yaml \
  --output-dir results/ >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] DONE unified matrix (exit $?)" >> "$LOG"
