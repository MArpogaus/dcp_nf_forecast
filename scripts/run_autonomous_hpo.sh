#!/usr/bin/env bash
set -e

MODEL="$1"
TARGET="$2"
ITER="$3"
EPOCHS="$4"
PATIENCE="$5"
LR="$6"
HIDDEN="$7"
NBINS="$8"

CONFIG="/app/params/models/${TARGET}/${MODEL}.yaml"
LOGFILE="/app/logs/hpo_iter${ITER}_${TARGET}-${MODEL}.log"
STATUS_FILE="/tmp/hpo_${TARGET}-${MODEL}_status_iter${ITER}"

log() { echo "[$(date)] $*" | tee -a "$LOGFILE"; }

log "=== HPO Iter ${ITER}: ${MODEL} (${TARGET}) ==="
log "Config: epochs=${EPOCHS} patience=${PATIENCE} lr=${LR} h=${HIDDEN} nbins=${NBINS}"

# Update config via python yaml
python3 -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
c['fit_kwargs']['max_epochs'] = $EPOCHS
c['fit_kwargs']['epochs'] = $EPOCHS
c['fit_kwargs']['early_stopping_patience'] = $PATIENCE
c['fit_kwargs']['learning_rate'] = '$LR'
if 'hidden_units' in c.get('model_kwargs', {}):
    c['model_kwargs']['hidden_units'] = $HIDDEN
if 'parameters_fn_kwargs' in c.get('model_kwargs', {}):
    c['model_kwargs']['parameters_fn_kwargs']['hidden_units'] = $HIDDEN
with open('$CONFIG', 'w') as f:
    yaml.dump(c, f, default_flow_style=False)
print('Config updated')
"

# Launch training
export PATH="/app/.venv/bin:$PATH"
cd /app
dvc repro "train@dataset0-${MODEL}" 2>&1 | tee -a "$LOGFILE"

# Extract results
METRICS_FILE="/app/results/${TARGET}/${MODEL}/metrics.yaml"
if [ -f "$METRICS_FILE" ]; then
    MIN_VAL_LOSS=$(grep "min_val_loss:" "$METRICS_FILE" | awk '{print $2}')
    BEST_EPOCH=$(grep "best_epoch:" "$METRICS_FILE" | awk '{print $2}')
    log "COMPLETED: min_val_loss=${MIN_VAL_LOSS} best_epoch=${BEST_EPOCH}"
    echo "${MIN_VAL_LOSS}" > "${STATUS_FILE}_result"
else
    log "ERROR: No metrics file found"
    echo "ERROR" > "${STATUS_FILE}_result"
fi

echo "completed" > "$STATUS_FILE"
log "=== Iter ${ITER} done ==="
