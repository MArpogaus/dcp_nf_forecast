#!/usr/bin/env bash
# HPO Autonomous Monitor — survives handover/compaction
# Usage: source this file or run: nohup bash /app/logs/hpo_monitor_loop.sh &
# Logs to: /app/logs/hpo_monitor_loop.log

MONITOR_LOG="/app/logs/hpo_monitor_loop.log"
HPO_LOG="/app/hpo_study.md"
DLA_RESULTS="/app/results/dla"
ITER_LOG="/app/logs/hpo_iter1_dla-spline_nf_lognormal.log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" | tee -a "$MONITOR_LOG"
}

get_val_loss() {
    grep -oP "val_loss: [-\d.]+" "$ITER_LOG" 2>/dev/null | tail -1 | cut -d' ' -f2
}

get_epoch() {
    grep -oP "Epoch \d+/\d+" "$ITER_LOG" 2>/dev/null | tail -1
}

is_training_alive() {
    ps aux | grep -q "[t]rain.py.*spline_nf_lognormal"
}

check_metrics() {
    local mf="$DLA_RESULTS/spline_nf_lognormal/metrics.yaml"
    if [ -f "$mf" ]; then
        cat "$mf"
        return 0
    fi
    return 1
}

log "=== HPO Autonomous Monitor Started ==="
log "Monitoring: spline_nf_lognormal (lr=0.0003, epochs=400)"
log "PID to watch: $(ps aux | grep '[t]rain.py.*spline_nf_lognormal' | awk '{print $2}' | head -1)"

# Wait for training to complete
LAST_VAL=""
STALLED=0
while is_training_alive; do
    VAL=$(get_val_loss)
    EP=$(get_epoch)
    if [ -n "$VAL" ] && [ "$VAL" != "$LAST_VAL" ]; then
        LAST_VAL=$VAL
        STALLED=0
        log "$EP | val_loss: $VAL"
    elif [ -n "$VAL" ]; then
        STALLED=$((STALLED + 1))
        if [ $STALLED -gt 20 ]; then  # >10 min no change
            log "WARNING: val_loss stalled at $VAL for $STALLED checks"
        fi
    fi
    sleep 30
done

log "Training process exited."
sleep 15  # wait for final writes

# Check results
log "=== Checking final metrics ==="
METRICS=$(check_metrics)
if [ -n "$METRICS" ]; then
    MIN_VAL=$(echo "$METRICS" | grep "min_val_loss:" | awk '{print $2}')
    BEST_EP=$(echo "$METRICS" | grep "best_epoch:" | awk '{print $2}')
    FINAL_VAL=$(echo "$METRICS" | grep "final_val_loss:" | awk '{print $2}')
    log "min_val_loss=$MIN_VAL (best_epoch=$BEST_EP) final_val_loss=$FINAL_VAL"
    
    PREV_BEST=-159.39
    if (( $(echo "$MIN_VAL < $PREV_BEST" | bc -l) )); then
        log "✅ IMPROVED: $PREV_BEST → $MIN_VAL"
    else
        log "❌ Did not improve: $PREV_BEST vs $MIN_VAL"
    fi
    
    # Commit
    log "Committing results..."
    cd /app
    git add -A
    git commit -m "hpo(dla): spline_nf_lognormal iter2 lr=0.0005→0.0003: $PREV_BEST→${MIN_VAL}"
    log "Committed. Ready for next iteration."
else
    log "⚠️ No metrics.yaml found — training may have failed."
    log "Config: lr=0.0003 epochs=400"
    log "Check $ITER_LOG for errors."
fi

log "=== Monitor cycle complete ==="
echo "Done. Check $MONITOR_LOG for details."
