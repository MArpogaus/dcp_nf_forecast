#!/usr/bin/env python3
"""Query MLflow for best training config per (target, model) and print summary."""
import mlflow
from mlflow.tracking import MlflowClient
import sys

TARGETS = {
    85: "dla",
    86: "ofen_g_koks",
    87: "ofen_f_koks",
    88: "pl2",
}

client = MlflowClient()

def get_best_training_runs():
    rows = []
    for eid, target in TARGETS.items():
        runs = client.search_runs(
            experiment_ids=[str(eid)],
            filter_string="tags.stage = 'training'",
            order_by=["metrics.min_val_loss ASC"],
        )
        seen = set()
        for r in runs:
            tags = r.data.tags
            run_name = tags.get("mlflow.runName", "")
            model = run_name.replace(f"_{target}", "").replace("_training", "")
            if model in seen:
                continue
            if "baseline" in model:
                continue  # skip baselines
            seen.add(model)
            metrics = r.data.metrics
            rows.append({
                "target": target,
                "model": model,
                "min_val_loss": metrics.get("min_val_loss"),
                "final_val_loss": metrics.get("val_loss"),
                "best_epoch": int(metrics.get("best_epoch", 0)),
                "run_id": r.info.run_id[:12],
            })
    return rows

def print_table(rows):
    header = f"{'Target':<15} {'Model':<40} {'Min Val Loss':<14} {'Final Val Loss':<14} {'Best Ep':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        mvl = f"{r['min_val_loss']:.2f}" if r['min_val_loss'] else "N/A"
        fvl = f"{r['final_val_loss']:.2f}" if r['final_val_loss'] else "N/A"
        print(f"{r['target']:<15} {r['model']:<40} {mvl:<14} {fvl:<14} {r['best_epoch']:<8}")

    print()
    # Per-target winners
    print("=" * 60)
    print("BEST MODEL PER TARGET (by min_val_loss)")
    print("=" * 60)
    by_target = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)
    for target, models in by_target.items():
        best = min(models, key=lambda x: x["min_val_loss"] or float("inf"))
        print(f"  {target:<15}: {best['model']:<40} {best['min_val_loss']:.2f}")

if __name__ == "__main__":
    rows = get_best_training_runs()
    if not rows:
        print("No training runs found. Run DVC repro first.")
        sys.exit(1)
    print_table(rows)
