#!/usr/bin/env bash
# Light 5% ablations AFTER main v5 finishes (avoids fighting MPS).
# Recipes: min_w 0.10 / 0.15, ent_w on/off. Seed 42. Shorter patience for speed.
set -euo pipefail

PY="${PY:-/Users/liuqiwu/Downloads/deepyeast_msmm_bundle/.venv_pytorch/bin/python}"
REPO="${REPO:-/Users/liuqiwu/Downloads/deepyeast-mase}"
DATA="${DATA:-/Users/liuqiwu/Downloads/deepyeast_5pct}"
DEVICE="${DEVICE:-mps}"
LOG="$DATA/checkpoints/local_jobs/light_ablation_master.log"
mkdir -p "$(dirname "$LOG")"

wait_for_v5() {
  local pid_file="$DATA/checkpoints/local_jobs/v5_mps.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[$(date)] waiting for v5 pid=$pid ..." | tee -a "$LOG"
      while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    fi
  fi
  # also wait if results.json not yet written
  local r="$DATA/checkpoints/mase_lite_5pct_v5/results.json"
  echo "[$(date)] waiting for $r ..." | tee -a "$LOG"
  while [[ ! -f "$r" ]]; do sleep 60; done
}

run_one() {
  local name="$1"; shift
  mkdir -p "$DATA/checkpoints/$name"
  echo "[$(date)] START $name $*" | tee -a "$LOG"
  caffeinate -dims "$PY" "$REPO/pytorch/train_mase_lite.py" \
    --data-dir "$DATA" \
    --device "$DEVICE" \
    --epochs 40 \
    --patience 10 \
    --batch-size 64 \
    --top-k 40 \
    --soft-alpha 0.5 \
    --mask-sparsity-weight 0.05 \
    --label-smoothing 0.1 \
    --seed 42 \
    --num-workers 0 \
    --no-uq \
    --checkpoint-name "$name" \
    "$@" \
    > "$DATA/checkpoints/$name/train.log" 2>&1 || true
  echo "[$(date)] DONE $name" | tee -a "$LOG"
}

wait_for_v5
mkdir -p \
  "$DATA/checkpoints/abl_5pct_min10_ent0" \
  "$DATA/checkpoints/abl_5pct_min15_ent0" \
  "$DATA/checkpoints/abl_5pct_min15_ent01"

# learnable fused-CE (not freeze, not legacy): closest to v5 family
run_one abl_5pct_min10_ent0 \
  --min-ensemble-weight 0.10 --ensemble-entropy-weight 0.0 \
  --ensemble-lr 0.0005

run_one abl_5pct_min15_ent0 \
  --min-ensemble-weight 0.15 --ensemble-entropy-weight 0.0 \
  --ensemble-lr 0.0005

run_one abl_5pct_min15_ent01 \
  --min-ensemble-weight 0.15 --ensemble-entropy-weight 0.01 \
  --ensemble-lr 0.0005

echo "[$(date)] all light ablations finished" | tee -a "$LOG"
