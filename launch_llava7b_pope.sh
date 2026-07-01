#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUT="${1:-$HOME/results/llava7b_pope_foresight_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

PYTHON_BIN="${PYTHON_BIN:-python}"

nohup env PYTHONUNBUFFERED=1 PYTHONPATH=. \
  "$PYTHON_BIN" -u run_llava7b_pope.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --dataset-name lmms-lab/POPE \
  --split test \
  --category all \
  --sampling first \
  --k-min 32 \
  --k-max 128 \
  --k-text 64 \
  --rho 0.90 \
  --load-in-4bit \
  --output-dir "$OUT" \
  > "$OUT/run.log" 2>&1 < /dev/null &

PID=$!
echo "$PID" > "$OUT/pid.txt"
echo "PID:$PID"
echo "OUT:$OUT"
echo "LOG:$OUT/run.log"
