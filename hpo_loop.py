#!/usr/bin/env python3
"""HPO loop for spline_nf_scale on DLA — fully autonomous."""
import subprocess
import time
import json
import urllib.request
import sys
from pathlib import Path

TARGET = "dla-spline-nf-scale"
MODEL = "spline_nf_scale"
EXPERIMENT_NAME = "dcp_nf_forecast-dla"
MLFLOW_URI = "http://server:5000"
LOG_DIR = Path("/app/logs")
HPO_FILE = Path("/app/hpo_study.md")
CONFIG_FILE = Path("/app/params/models/dla/spline_nf_scale.yaml")

SEARCH_SPACE = {
    "learning_rate": [0.0005, 0.0001, 0.003, 0.005],
    "nbins": [8, 16, 24],
    "hidden_units": [[128, 64], [256, 128], [256, 256]],
}

# Current index in search path
param_order = [
    ("learning_rate", 0),  # start at index 0 (0.0005)
]


def get_experiment_id(name):
    url = f"{MLFLOW_URI}/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())["experiment"]["experiment_id"]
    except Exception:
        return None


def get_last_min_val_loss(exp_id, model_name):
    """Get min_val_loss from the most recent FINISHED training run for this model."""
    data = json.dumps({
        "experiment_ids": [exp_id],
        "max_results": 10,
        "order_by": ["attribute.start_time DESC"],
    }).encode()
    req = urllib.request.Request(
        f"{MLFLOW_URI}/api/2.0/mlflow/runs/search",
        data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            runs = json.loads(r.read()).get("runs", [])
    except Exception:
        return None

    for run in runs:
        name = run["info"]["run_name"]
        status = run["info"]["status"]
        if model_name in name and "_training" in name and status == "FINISHED":
            metrics = {m["key"]: m["value"] for m in run["data"].get("metrics", [])}
            return metrics.get("min_val_loss")
    return None


def update_study_log(iter_num, change, old_val, new_val, delta, commit, status):
    row = f"| {iter_num} | 2026-05-22 | {change} | {old_val} | {new_val} | {delta} | {commit} | {status} |\n"
    with open(HPO_FILE, "a") as f:
        f.write(row)


def launch_training():
    """Launch dvc repro for this model."""
    env = {"CUDA_VISIBLE_DEVICES": "0", "TF_CPP_MIN_LOG_LEVEL": "3"}
    cmd = ["dvc", "repro", "-f", f"train@dataset0-{MODEL}"]
    log_file = open(LOG_DIR / f"hpo_iter_{MODEL}.log", "w")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env,
        preexec_fn=lambda: None,  # detach from process group
    )
    return proc


def main():
    exp_id = get_experiment_id(EXPERIMENT_NAME)
    if not exp_id:
        print("ERROR: Cannot find experiment")
        sys.exit(1)

    print(f"=== HPO Loop: {TARGET} ===")
    print(f"Experiment: {EXPERIMENT_NAME} ({exp_id})")

    base_val = get_last_min_val_loss(exp_id, MODEL)
    print(f"Baseline min_val_loss: {base_val}")

    if base_val is None:
        print("ERROR: No baseline found")
        sys.exit(1)

    best_val = base_val
    iter_num = 1

    while True:
        # Get next param to tweak
        param_name, param_idx = param_order[0]
        param_values = SEARCH_SPACE[param_name]
        if param_idx >= len(param_values):
            print(f"Exhausted {param_name}, moving on")
            param_order.pop(0)
            if not param_order:
                print("Search space exhausted. Stopping.")
                break
            continue

        new_value = param_values[param_idx]
        # Read current config
        config_text = CONFIG_FILE.read_text()

        # Determine old value (simplified)
        old_value = param_idx  # placeholder, we'll read from config

        change_desc = f"{param_name}: current->{new_value}"
        print(f"\nIter {iter_num}: {change_desc}")

        # Update study log
        update_study_log(iter_num, change_desc, f"{best_val:.4f}", "—", "—", "—", "launched")

        # Launch
        proc = launch_training()
        print(f"  Launched PID {proc.pid}")

        # Monitor
        log_file = LOG_DIR / f"hpo_iter_{MODEL}.log"
        last_size = 0
        stall_count = 0

        while proc.poll() is None:
            time.sleep(30)
            if log_file.exists():
                current_size = log_file.stat().st_size
                if current_size == last_size:
                    stall_count += 1
                    if stall_count >= 10:  # 5 min stalled
                        print("  STALL detected, killing process")
                        proc.kill()
                        break
                else:
                    stall_count = 0
                last_size = current_size

        exit_code = proc.wait()
        print(f"  Process exited with code {exit_code}")

        # Read new metrics
        new_val = get_last_min_val_loss(exp_id, MODEL)
        print(f"  New min_val_loss: {new_val}")

        if new_val is None:
            print("  No metrics found — treating as failure")
            update_study_log(iter_num, change_desc, f"{best_val:.4f}", "N/A", "N/A", "—", "failed")
            param_order[0] = (param_name, param_idx + 1)  # try next value
            iter_num += 1
            continue

        # Compare
        delta = best_val - new_val  # negative = better
        improvement = -delta  # positive = improvement

        if new_val < best_val and improvement > 0.5:
            print(f"  IMPROVED! {best_val:.4f} → {new_val:.4f} (Δ={delta:.4f})")
            best_val = new_val
            # Commit
            commit_msg = f"feat(hpo): {TARGET} iter{iter_num} — {change_desc} (val: {best_val:.4f})"
            result = subprocess.run(
                ["git", "add", str(CONFIG_FILE), str(HPO_FILE)],
                capture_output=True, text=True,
            )
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True, text=True,
            )
            commit_hash = result.stdout.strip().split()[-1] if result.returncode == 0 else "no-commit"
            update_study_log(iter_num, change_desc, f"{new_val:.4f}", f"{delta:.4f}", commit_hash, "committed")
        elif new_val >= best_val:
            print(f"  WORSENED / NO IMPROVEMENT: {best_val:.4f} → {new_val:.4f}")
            # Revert config
            subprocess.run(["git", "checkout", "--", str(CONFIG_FILE)], capture_output=True)
            update_study_log(iter_num, change_desc, f"{best_val:.4f}", f"+{delta:.4f}", "—", "reverted")
            param_order[0] = (param_name, param_idx + 1)  # try next value

        # Check stopping criteria
        if best_val <= -170:
            print(f"\n=== TARGET REACHED: {best_val:.4f} ≤ -170 ===")
            break

        iter_num += 1
        if iter_num > 20:
            print("\n=== MAX ITERATIONS REACHED ===")
            break

        # Check plateau
        time.sleep(5)

    print(f"\n=== HPO COMPLETE ===")
    print(f"Best min_val_loss: {best_val:.4f}")
    subprocess.run(["git", "add", str(HPO_FILE)], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"feat(hpo): {TARGET} — complete (val_loss: {best_val:.4f})"], capture_output=True)


if __name__ == "__main__":
    main()
