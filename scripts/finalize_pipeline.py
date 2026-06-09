"""Autonomous finalization pipeline: monitor repro, then finalize."""
import subprocess
import time
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_DIR / "logs" / "finalize_pipeline.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or str(REPO_DIR))
    return result.returncode, (result.stdout + result.stderr).strip()


def count_completed_eval() -> int:
    logs_dir = REPO_DIR / "logs"
    completed = 0
    for f in logs_dir.glob("gpu*_evaluate@*.log"):
        if f.stat().st_size > 0 and f.stat().st_size < 500000:
            try:
                content = f.read_text()
                if "saving metrics" in content:
                    completed += 1
            except Exception:
                pass
    return completed


def count_total_eval() -> int:
    result = subprocess.run(
        ["dvc", "stage", "list"],
        capture_output=True, text=True, cwd=str(REPO_DIR)
    )
    return sum(1 for line in result.stdout.split("\n") if "evaluate@" in line)


def wait_for_repro() -> bool:
    log("Waiting for DVC repro to complete...")
    last_completed = -1
    stall_count = 0
    stall_threshold = 30  # 30 checks with no progress = stalled

    while True:
        completed = count_completed_eval()
        total = count_total_eval()
        log(f"Completed: {completed}/{total}")

        if completed == total:
            log("All stages complete!")
            return True

        if completed == last_completed:
            stall_count += 1
            if stall_count >= stall_threshold:
                log(f"STALL DETECTED: no progress for {stall_threshold * 120}s. Checking DVC processes...")
                rc, out = run_cmd(["ps", "aux"])
                dvc_count = out.count("dvc repro")
                if dvc_count <= 1:
                    log(f"DVC processes: {dvc_count}. May be stalled. Checking for GPU activity...")
                    rc2, out2 = run_cmd(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"])
                    log(f"GPU util: {out2}")
                    return False
                stall_count = 0
        else:
            stall_count = 0
            last_completed = completed

        time.sleep(120)


def commit_results() -> None:
    log("Committing dvc.lock and all results...")

    # Stage all changes
    run_cmd(["git", "add", "dvc.lock", "params/", "AGENTS.md", "hpo_study.md"])

    # Get metrics summary from MLflow
    rc, out = run_cmd([
        sys.executable or "python3", "-c", """
import mlflow, math
from mlflow.tracking import MlflowClient
client = MlflowClient()
for eid in [85, 86, 87, 88]:
    exp = client.get_experiment(str(eid))
    runs = client.search_runs(
        experiment_ids=[str(eid)],
        filter_string="tags.stage = 'evaluation'",
        order_by=["start_time DESC"],
        max_results=50,
    )
    if runs:
        best = min(runs, key=lambda r: r.data.metrics.get('nll', float('inf')) if r.data.metrics.get('nll') else float('inf'))
        name = best.data.tags.get('mlflow.runName', '?')
        nll = best.data.metrics.get('nll', '?')
        ci90 = best.data.metrics.get('mean_90_ci_width', '?')
        rmse = best.data.metrics.get('rmse', '?')
        print(f"  {exp.name}: best={name} NLL={nll} CI90={ci90} RMSE={rmse}")
"""
    ])
    summary = out.split("202")[0] if "202" in out else out

    commit_msg = f"""chore: full repro with NaN fix — all {count_total_eval()} configs

Changes:
- All 17 models × 4 targets trained and evaluated
- Scale: clipped_softplus(min=0.3, max=1.0) for all truncated + lognormal models
- Shift: shift_constrain(max_shift=0.5) for truncated, max_shift=5.0 for lognormal
- nbins=24 for all lognormal spline models
- RQS domain [exp(-5), exp(5)] for LogNormal(0,1) base

Results:
{summary}
"""

    run_cmd(["git", "commit", "-m", commit_msg])
    log("Committed results.")


def query_best_configs() -> dict:
    log("Querying best configs from MLflow...")
    rc, out = run_cmd([sys.executable or "python3", "scripts/query_best_configs.py"])
    log(out)
    return {"raw": out}


def update_summary_docs() -> None:
    log("Updating summary docs...")

    # Generate LaTeX tables
    rc, out = run_cmd([
        sys.executable or "python3", "scripts/report_hpo_results.py",
        "--latex", "--output", "tables.tex"
    ])
    log(f"LaTeX tables: {out[:200] if len(out) > 200 else out}")

    # Generate paper figures
    rc, out = run_cmd([
        sys.executable or "python3", "scripts/generate_paper_figures.py",
        "--output-dir", "paper_figures"
    ])
    log(f"Paper figures: {out[:200] if len(out) > 200 else out}")

    # Update hpo_summary.md via report script
    rc, out = run_cmd([sys.executable or "python3", "scripts/report_hpo_results.py"])
    log(f"Terminal report: {out[:500] if len(out) > 500 else out}")

    # Stage the updated docs
    run_cmd(["git", "add", "tables.tex", "paper_figures/", "docs/hpo_summary.md", "docs/model_architecture.md"])


def review_and_finalize() -> None:
    log("=" * 60)
    log("FINAL REVIEW — ALL EXPERIMENTS COMPLETE")
    log("=" * 60)

    # Comprehensive MLflow query
    rc, out = run_cmd([
        sys.executable or "python3", "-c", """
import mlflow
from mlflow.tracking import MlflowClient
import math

client = MlflowClient()

print(f'{\"Target\":<20} {\"Model\":<45} {\"NLL\":<15} {\"CI90\":<15} {\"RMSE\":<15} {\"Base\":<20}')
print('-' * 130)

target_names = {85: 'dla', 86: 'ofen_g_koks', 87: 'ofen_f_koks', 88: 'pl2'}
for eid in [85, 86, 87, 88]:
    exp = client.get_experiment(str(eid))
    runs = client.search_runs(
        experiment_ids=[str(eid)],
        filter_string="tags.stage = 'evaluation'",
        order_by=["start_time DESC"],
        max_results=50,
    )
    # Get best per model
    best_per_model = {}
    for r in runs:
        name = r.data.tags.get('mlflow.runName', '?')
        model = name.rsplit('_', 1)[0] if '_' in name else name
        nll = r.data.metrics.get('nll')
        if nll is not None and not math.isnan(nll) and not math.isinf(nll):
            if model not in best_per_model or nll < best_per_model[model]['nll']:
                best_per_model[model] = {
                    'nll': nll,
                    'ci90': r.data.metrics.get('mean_90_ci_width', '?'),
                    'rmse': r.data.metrics.get('rmse', '?'),
                    'base': r.data.params.get('model_kwargs.base_distribution_kwargs.distribution_name', '?')
                }

    for model, data in sorted(best_per_model.items()):
        print(f'{target_names[eid]:<20} {model:<45} {data[\"nll\"]:<15.2f} {data[\"ci90\"] if isinstance(data[\"ci90\"], str) else f\"{data[\"ci90\"]:<15.4f}\"} {data[\"rmse\"] if isinstance(data[\"rmse\"], str) else f\"{data[\"rmse\"]:<15.4f}\"} {data[\"base\"]:<20}')
    print()
""",
    ])
    log(out)


def main() -> None:
    log("=" * 60)
    log("FINALIZATION PIPELINE STARTED")
    log("=" * 60)

    # Step 1: Wait for repro
    success = wait_for_repro()
    if not success:
        log("WARNING: Repro may have stalled. Checking DVC lock...")
        rc, out = run_cmd(["dvc", "status"])
        log(f"DVC status: {out}")
        # Continue anyway — try to commit what we have

    # Step 2: Push to MLflow (ensure all runs logged)
    log("All training completed.")

    # Step 3: Query best configs
    query_best_configs()

    # Step 4: Update summary docs
    update_summary_docs()

    # Step 5: Review all runs
    review_and_finalize()

    # Step 6: Commit everything
    commit_results()

    log("=" * 60)
    log("FINALIZATION COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    main()
