#!/usr/bin/env python3
"""Autonomous HPO master loop for DLA models.

Runs sequentially through all models. One param change per iteration.
Updates hpo_study.md after each completion.
"""
import subprocess
import time
import sys
import os
import re
import yaml

MODELS = [
    "spline_nf_lognormal",
    "spline_nf",
    "spline_nf_scale",
    "bernstein_nf",
    "bernstein_nf_lognormal",
    "bernstein_nf_scale",
    "bernstein_nf_scale_lognormal",
    "spline_nf_scale_lognormal",
]

TARGET = "dla"
HPO_LOG = "/app/hpo_study.md"

LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    return line


def read_config(model):
    path = f"/app/params/models/{TARGET}/{model}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def write_config(model, cfg):
    path = f"/app/params/models/{TARGET}/{model}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def get_metrics(model):
    path = f"/app/results/{TARGET}/{model}/metrics.yaml"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    m = re.search(r"min_val_loss:\s*([-\d.]+)", content)
    best = re.search(r"best_epoch:\s*(\d+)", content)
    final = re.search(r"final_val_loss:\s*([-\d.]+)", content)
    return {
        "min_val_loss": float(m.group(1)) if m else None,
        "best_epoch": int(best.group(1)) if best else None,
        "final_val_loss": float(final.group(1)) if final else None,
    }


def run_training(model, logfile):
    cmd = (
        f"export PATH=/app/.venv/bin:$PATH && "
        f"cd /app && "
        f"dvc repro train@dataset0-{model} 2>&1"
    )
    with open(logfile, "w") as lf:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
            executable="/bin/bash"
        )
    return proc


def train_model(model, cfg_override):
    cfg = read_config(model)
    for k, v in cfg_override.items():
        if k == "lr":
            cfg["fit_kwargs"]["learning_rate"] = v
        elif k == "epochs":
            cfg["fit_kwargs"]["epochs"] = v
            cfg["fit_kwargs"]["max_epochs"] = v
        elif k == "patience":
            cfg["fit_kwargs"]["early_stopping_patience"] = v
        elif k == "hidden_units":
            if "parameters_fn_kwargs" in cfg.get("model_kwargs", {}):
                cfg["model_kwargs"]["parameters_fn_kwargs"]["hidden_units"] = v
        elif k == "nbins":
            if "parameters_constraint_fn_kwargs" in cfg.get("model_kwargs", {}):
                cfg["model_kwargs"]["parameters_constraint_fn_kwargs"]["nbins"] = v
        elif k == "bijector_kwargs":
            cfg["model_kwargs"]["bijector_kwargs"].update(v)
    write_config(model, cfg)

    iter_num = get_next_iter_num(model)
    logfile = f"{LOG_DIR}/hpo_iter{iter_num}_{TARGET}-{model}.log"
    log(f"=== Launching {model} iter{iter_num}: {cfg_override} ===")
    log(f"Log: {logfile}")

    proc = run_training(model, logfile)

    # Monitor
    last_val = None
    stalled = 0
    while proc.poll() is None:
        metrics = get_metrics(model)
        if metrics and metrics["min_val_loss"] is not None:
            val = metrics["min_val_loss"]
            if val != last_val:
                log(f"  {model} iter{iter_num}: best_val_loss={val}")
                last_val = val
                stalled = 0
            else:
                stalled += 1
        time.sleep(30)

    proc.wait()
    log(f"Training process exited (code {proc.returncode})")
    time.sleep(5)

    metrics = get_metrics(model)
    if metrics:
        log(f"  FINAL {model} iter{iter_num}: min_val_loss={metrics['min_val_loss']} "
            f"best_epoch={metrics['best_epoch']}")
    return metrics


def get_next_iter_num(model):
    """Determine next iteration number by scanning hpo_study.md."""
    if not os.path.exists(HPO_LOG):
        return 1
    with open(HPO_LOG) as f:
        content = f.read()
    # Find all HPO log entries for this model
    pattern = rf"\| \d+ \| .*? \| .*?{model}.*? \|.*?\|"
    matches = re.findall(r"\| (\d+) \|.*?" + re.escape(model) + r".*?\|", content)
    if not matches:
        return 1
    return max(int(m) for m in matches) + 1


def update_hpo_log(model, iter_num, cfg_override, old_val, new_val):
    line = (f"| {iter_num} | {time.strftime('%Y-%m-%d')} | "
            f"{model}: {cfg_override} | {old_val} | {new_val} | "
            f"{'✅' if new_val and old_val and new_val < old_val else '❌'} | "
            f"autonomous |\n")
    # Append to the HPO table in hpo_study.md
    with open(HPO_LOG, "a") as f:
        f.write(line)
    log(f"Updated HPO log: {line.strip()}")


def save_status(msg):
    with open(f"{LOG_DIR}/hpo_status.txt", "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")


def main():
    log("=== HPO Master Loop Started ===")
    save_status("Starting HPO master loop")

    # Current best metrics by model (from full pipeline)
    best_metrics = {}
    for model in MODELS:
        m = get_metrics(model)
        if m:
            best_metrics[model] = m["min_val_loss"]
            log(f"  Current best {model}: {m['min_val_loss']}")

    LNB_BASELINE = -163.31  # lognormal_baseline best

    # =====================================================================
    # Model-specific HPO plans
    # Each entry: (model, [list of config overrides to try in order])
    # =====================================================================
    hpo_plans = {
        # spline_nf_lognormal: already did iter1 (400ep, -159.39), now try LR
        "spline_nf_lognormal": [
            {"lr": 0.0003, "epochs": 400, "patience": 20},          # iter2: lower LR
            {"lr": 0.0001, "epochs": 400, "patience": 20},          # iter3: even lower
            {"lr": "0.0005_cyclic", "epochs": 400, "patience": 20}, # iter4: cyclic schedule
            {"hidden_units": [256, 256], "epochs": 400, "patience": 20},  # iter5: more capacity
        ],
        # spline_nf: only had 200ep default, try more
        "spline_nf": [
            {"epochs": 400, "patience": 20},
        ],
        # spline_nf_scale: already had 5 HPO iters, try more epochs
        "spline_nf_scale": [
            {"epochs": 400, "patience": 20},
        ],
        # bernstein models: try more epochs first
        "bernstein_nf": [
            {"epochs": 400, "patience": 20},
        ],
        "bernstein_nf_lognormal": [
            {"epochs": 400, "patience": 20},
        ],
        "bernstein_nf_scale": [
            {"epochs": 400, "patience": 20},
        ],
        "bernstein_nf_scale_lognormal": [
            {"epochs": 400, "patience": 20},
        ],
        "spline_nf_scale_lognormal": [
            {"epochs": 400, "patience": 20},
        ],
    }

    for model in MODELS:
        if model not in hpo_plans:
            log(f"No HPO plan for {model}, skipping")
            continue

        plans = hpo_plans[model]
        current_best = best_metrics.get(model)

        for i, cfg_override in enumerate(plans):
            old_val = current_best
            metrics = train_model(model, cfg_override)

            if metrics and metrics["min_val_loss"] is not None:
                new_val = metrics["min_val_loss"]
                iter_num = get_next_iter_num(model) - 1  # already incremented
                update_hpo_log(model, iter_num, cfg_override, old_val, new_val)

                if new_val < (current_best or 0):
                    log(f"✅ {model}: IMPROVED {old_val} → {new_val}")
                    current_best = new_val
                    best_metrics[model] = new_val
                else:
                    log(f"❌ {model}: did not improve ({old_val} vs {new_val})")

                # Check if we beat the baseline
                if new_val < LNB_BASELINE:
                    log(f"🎉 {model}: BEAT LOGNORMAL BASELINE ({new_val} < {LNB_BASELINE})")
                    # Move to next model
                    break
            else:
                log(f"⚠️ {model} iter{i+1}: No metrics found, skipping remaining plans")
                break

        save_status(f"Completed HPO for {model} (best: {current_best})")

    log("=== All DLA models optimized ===")
    save_status("All DLA models complete")

    # Summary
    log("\n=== FINAL DLA RANKING ===")
    sorted_models = sorted(best_metrics.items(), key=lambda x: x[1] or 0, reverse=True)
    for model, val in sorted_models:
        gap = val - LNB_BASELINE if val else 0
        log(f"  {model}: {val} (gap: {gap:+.2f})")


if __name__ == "__main__":
    main()
