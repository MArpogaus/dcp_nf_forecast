#!/usr/bin/env python3
"""Monitor DLA training, log results to hpo_study.md, and commit."""
import subprocess
import time
import json
import urllib.request
import re
from pathlib import Path

MLFLOW_URI = "http://server:5000"
EXPERIMENT_NAME = "dcp_nf_forecast-dla"
HPO_FILE = Path("/app/hpo_study.md")
LOG_FILE = Path("/app/logs/dvc_train_dla.log")
MODELS = [
    "normal_baseline", "lognormal_baseline", "bernstein_nf",
    "bernstein_nf_lognormal", "bernstein_nf_scale", "bernstein_nf_scale_lognormal",
    "spline_nf", "spline_nf_lognormal", "spline_nf_scale", "spline_nf_scale_lognormal",
]


def get_experiment_id(name):
    url = f"{MLFLOW_URI}/api/2.0/mlflow/experiments/get-by-name?experiment_name={name}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())["experiment"]["experiment_id"]
    except Exception as e:
        print(f"[monitor] Failed to get experiment: {e}")
        return None


def list_runs(experiment_id):
    url = f"{MLFLOW_URI}/api/2.0/mlflow/runs/search"
    data = json.dumps({
        "experiment_ids": [experiment_id],
        "max_results": 50,
        "order_by": ["attribute.start_time DESC"],
    }).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()).get("runs", [])
    except Exception as e:
        print(f"[monitor] Failed to search runs: {e}")
        return []


def get_best_val_loss(run):
    metrics = run.get("data", {}).get("metrics", [])
    val_losses = [m["value"] for m in metrics if m["key"] == "val_loss"]
    return min(val_losses) if val_losses else None


def parse_log_progress():
    """Parse log for current training progress."""
    if not LOG_FILE.exists():
        return "no log yet"
    with open(LOG_FILE) as f:
        lines = f.readlines()
    if not lines:
        return "empty log"
    tail = "".join(lines[-30:])
    # Extract model name and epoch
    for line in reversed(lines):
        if "Running stage" in line:
            m = re.search(r"train@dataset0-(\S+)", line)
            if m:
                return f"current model: {m.group(1)}"
    for line in reversed(lines):
        if "Epoch" in line and "/" in line:
            return line.strip()[:120]
    return tail.strip()[:200]


def update_hpo(run_summary, status):
    """Append a results section to hpo_study.md."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(HPO_FILE, "a") as f:
        f.write(f"\n## DLA Full Run — {timestamp} — Status: {status}\n\n")
        f.write("| Model | Status | Min Val Loss |\n")
        f.write("|-------|--------|-------------|\n")
        for model, s, loss in run_summary:
            loss_str = f"{loss:.4f}" if loss is not None else "-"
            f.write(f"| {model} | {s} | {loss_str} |\n")
        f.write("\n")


def commit_hpo(msg="docs: update hpo_study.md with DLA full run results"):
    """Commit hpo_study.md changes."""
    subprocess.run(["git", "add", str(HPO_FILE)], check=False)
    subprocess.run(["git", "commit", "-m", msg], check=False)


def is_dvc_running():
    result = subprocess.run(["pgrep", "-f", "dvc repro"], capture_output=True, text=True)
    return result.returncode == 0


def is_process_running(pid):
    try:
        subprocess.run(["kill", "-0", str(pid)], capture_output=True)
        return True
    except:
        return False


def main():
    print("[monitor] Starting DLA training monitor...")
    experiment_id = get_experiment_id(EXPERIMENT_NAME)
    if experiment_id:
        print(f"[monitor] Experiment ID: {experiment_id}")
    else:
        print("[monitor] Will retry experiment lookup")

    last_status = {}
    idle_cycles = 0

    while True:
        time.sleep(120)  # Poll every 2 minutes

        dvc_running = is_dvc_running()
        progress = parse_log_progress()
        print(f"[monitor] DVC running: {dvc_running} | {progress}")

        if not experiment_id:
            experiment_id = get_experiment_id(EXPERIMENT_NAME)
            if not experiment_id:
                if not dvc_running:
                    print("[monitor] DVC not running, experiment not found — done")
                    break
                continue

        runs = list_runs(experiment_id)
        run_summary = []
        all_done = True

        for model in MODELS:
            model_runs = [r for r in runs if model in r["info"].get("run_name", "")]
            if not model_runs:
                run_summary.append((model, "PENDING", None))
                all_done = False
                continue
            # Get the latest run
            latest = model_runs[0]
            status = latest["info"]["status"]
            best_loss = get_best_val_loss(latest)
            run_summary.append((model, status, best_loss))

            prev = last_status.get(model)
            if prev != (status, best_loss):
                print(f"[monitor] {model}: {status} loss={best_loss}")
                last_status[model] = (status, best_loss)

            if status in ("RUNNING", "PENDING", "SCHEDULED"):
                all_done = False

        # Update hpo_study.md every cycle
        overall_status = "COMPLETE" if all_done else ("RUNNING" if dvc_running else "IDLE")
        update_hpo(run_summary, overall_status)
        commit_hpo()

        if not dvc_running and all_done:
            print("[monitor] All models done, DVC stopped — exiting")
            break

        if not dvc_running:
            idle_cycles += 1
            if idle_cycles >= 3:  # 6 minutes idle
                print("[monitor] DVC idle for 3 cycles — exiting")
                break
        else:
            idle_cycles = 0

    print("[monitor] Final update...")
    # Final update
    experiment_id = experiment_id or get_experiment_id(EXPERIMENT_NAME)
    if experiment_id:
        runs = list_runs(experiment_id)
        run_summary = []
        for model in MODELS:
            model_runs = [r for r in runs if model in r["info"].get("run_name", "")]
            if not model_runs:
                run_summary.append((model, "PENDING", None))
                continue
            latest = model_runs[0]
            run_summary.append((latest["info"]["run_name"], latest["info"]["status"], get_best_val_loss(latest)))
        update_hpo(run_summary, "FINAL")
        commit_hpo(msg="docs: hpo_study.md final DLA full run results")


if __name__ == "__main__":
    main()
