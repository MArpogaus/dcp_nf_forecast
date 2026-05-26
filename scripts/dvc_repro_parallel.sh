#!/usr/bin/env bash
# Run evaluate stages in parallel across both GPUs.
# Usage:
#   ./scripts/dvc_repro_parallel.sh              # full repro (checks deps)
#   ./scripts/dvc_repro_parallel.sh --force-evaluation  # force re-eval only
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

source .venv/bin/activate
mkdir -p logs

RESAMPLE=false
[ "${1:-}" = "--force-evaluation" ] && RESAMPLE=true

STAGES=()
while IFS= read -r line; do
  STAGES+=("$(echo "$line" | awk '{print $1}')")
done < <(dvc stage list | grep '^evaluate@')

echo "Found ${#STAGES[@]} evaluate stages"

half=$(( (${#STAGES[@]} + 1) / 2 ))
gpu0=("${STAGES[@]:0:$half}")
gpu1=("${STAGES[@]:$half}")

run_batch() {
  local gpu=$1; shift
  local flags=()
  $RESAMPLE && flags=(--single-item --force)
  for stage in "$@"; do
    local log="logs/gpu${gpu}_${stage}.log"
    local attempt=0 max_attempts=10
    while true; do
      echo "[GPU-$gpu] $stage (attempt $((attempt+1))) ..."
      CUDA_VISIBLE_DEVICES="$gpu" dvc repro "${flags[@]}" "$stage" > "$log" 2>&1; rc=$?
      if [ $rc -eq 0 ]; then
        echo "[GPU-$gpu] done: $stage"
        break
      elif grep -q "Unable to acquire lock" "$log" 2>/dev/null; then
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
          echo "[GPU-$gpu] FAILED (lock timeout): $stage"
          break
        fi
        local delay=$(( attempt * 5 + RANDOM % 5 ))
        echo "[GPU-$gpu] lock contention, retry in ${delay}s ..."
        sleep "$delay"
      else
        echo "[GPU-$gpu] FAILED (exit=$rc): $stage"
        break
      fi
    done
  done
}

run_batch 0 "${gpu0[@]}" & pid0=$!
run_batch 1 "${gpu1[@]}" & pid1=$!
echo "GPU0 PID=$pid0  GPU1 PID=$pid1"
echo "Monitor: tail -f logs/gpu*"

wait $pid0 $pid1
echo "All done."
