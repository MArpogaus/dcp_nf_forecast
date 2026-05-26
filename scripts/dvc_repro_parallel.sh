#!/usr/bin/env bash
# Run evaluate stages in parallel across both GPUs.
# Usage:
#   ./scripts/dvc_repro_parallel.sh              # full repro (checks deps)
#   ./scripts/dvc_repro_parallel.sh --force-evaluation  # force re-eval only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

source .venv/bin/activate
mkdir -p logs

RESAMPLE=false
[ "${1:-}" = "--force-evaluation" ] && RESAMPLE=true

# Gather all evaluate stages
STAGES=()
while IFS= read -r line; do
  STAGES+=("$(echo "$line" | awk '{print $1}')")
done < <(dvc stage list | grep '^evaluate@')

echo "Found ${#STAGES[@]} evaluate stages"

# Split across GPUs
half=$(( (${#STAGES[@]} + 1) / 2 ))
gpu0=("${STAGES[@]:0:$half}")
gpu1=("${STAGES[@]:$half}")

run_batch() {
  local gpu=$1; shift
  local flags=()
  $RESAMPLE && flags=(--single-item --force)
  for stage in "$@"; do
    echo "[GPU-$gpu] $stage ..."
    CUDA_VISIBLE_DEVICES="$gpu" dvc repro "${flags[@]}" "$stage" \
      > "logs/gpu${gpu}_${stage}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then echo "[GPU-$gpu] done: $stage"
    else echo "[GPU-$gpu] FAILED (exit=$rc): $stage"
    fi
  done
}

run_batch 0 "${gpu0[@]}" & pid0=$!
run_batch 1 "${gpu1[@]}" & pid1=$!
echo "GPU0 PID=$pid0  GPU1 PID=$pid1"
echo "Monitor: tail -f logs/gpu*"

wait $pid0 $pid1
echo "All done."
