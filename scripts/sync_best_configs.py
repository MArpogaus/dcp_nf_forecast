"""Fetch best model configs from MLflow and compare with local YAML files.

Usage:
    python scripts/sync_best_configs.py            # dry-run: report mismatches only
    python scripts/sync_best_configs.py --apply    # apply fixes
    python scripts/sync_best_configs.py --check    # exit 1 if any mismatches
"""

import argparse
import sys
from pathlib import Path

import mlflow
import yaml


TARGETS = {
    85: "dla",
    86: "ofen_g_koks",
    87: "ofen_f_koks",
    88: "pl2",
}

PARAMS_DIR = Path("params/models")

# MLflow param keys we care about
LR_KEYS = [
    "fit_kwargs.learning_rate",
    "fit_kwargs.learning_rate.scheduler_name",
    "fit_kwargs.learning_rate.scheduler_kwargs.initial_learning_rate",
    "fit_kwargs.learning_rate.scheduler_kwargs.decay_steps",
]
HP_KEYS = [
    ("fit_kwargs.epochs", "epochs"),
    ("fit_kwargs.early_stopping_patience", "early_stopping_patience"),
    ("model_kwargs.parameters_fn_kwargs.hidden_units.0", "hidden_units.0"),
    ("model_kwargs.parameters_fn_kwargs.hidden_units.1", "hidden_units.1"),
]

# Targets that have NaN issues with Scale+TruncatedNormal
NAN_NON_DLA_TARGETS = {"ofen_g_koks", "ofen_f_koks", "pl2"}

# Models that use Scale + TruncatedNormal and produce NaN on non-DLA
NAN_SCALE_TRUNCATED_MODELS = {
    "spline_nf_scale_truncated",
    "spline_nf_scale_shift_truncated",
    "bernstein_nf_scale_truncated",
    "bernstein_nf_scale_shift_truncated",
}

ACTIVE_MODELS = [
    "normal_baseline",
    "truncated_baseline",
    "spline_nf",
    "spline_nf_truncated",
    "spline_nf_scale",
    "spline_nf_scale_truncated",
    "spline_nf_scale_shift",
    "spline_nf_scale_shift_truncated",
    "bernstein_nf_scale",
    "bernstein_nf_scale_truncated",
    "bernstein_nf_scale_shift",
    "bernstein_nf_scale_shift_truncated",
]


def get_best_run(exp_id, model_name, target_name):
    """Get the best evaluation run for a model in an experiment."""
    client = mlflow.tracking.MlflowClient()
    # Match exact model name in run name (eval_<model>_<target>_evaluation)
    target_suffix = target_name.replace("-", "_")
    pattern = f"eval_{model_name}_{target_suffix}_evaluation"
    runs = client.search_runs(
        str(exp_id),
        filter_string=f"tags.stage='evaluation'",
        max_results=200,
    )
    # Filter client-side for exact model name
    target_suffix = target_name.replace("-", "_")
    runs = [
        r for r in runs
        if r.data.tags.get("mlflow.runName", "") == f"eval_{model_name}_{target_suffix}_evaluation"
    ]
    # Pick the run with lowest NLL (excluding inf/NaN)
    best = None
    best_nll = float("inf")
    for r in runs:
        nll = r.data.metrics.get("nll")
        if nll is not None and nll < best_nll and nll < 1e100:
            best_nll = nll
            best = r
    return best
    best = None
    best_nll = float("inf")
    for r in runs:
        nll = r.data.metrics.get("nll")
        if nll is not None and nll < best_nll and nll < 1e100:
            best_nll = nll
            best = r
    return best


def extract_params(run):
    """Extract model hyperparameters from an MLflow run."""
    params = run.data.params
    result = {}

    # Learning rate
    lr_scheduler = params.get("fit_kwargs.learning_rate.scheduler_name")
    if lr_scheduler:
        result["lr_type"] = "CosineDecay"
        result["initial_lr"] = float(
            params.get("fit_kwargs.learning_rate.scheduler_kwargs.initial_learning_rate", 0.001)
        )
    else:
        lr_val = params.get("fit_kwargs.learning_rate")
        if lr_val:
            result["lr_type"] = "constant"
            result["initial_lr"] = float(lr_val)
        else:
            result["lr_type"] = "unknown"
            result["initial_lr"] = None

    # Hidden units
    h0 = params.get("model_kwargs.parameters_fn_kwargs.hidden_units.0")
    h1 = params.get("model_kwargs.parameters_fn_kwargs.hidden_units.1")
    if h0:
        result["hidden_units"] = [int(h0), int(h1)] if h1 else [int(h0)]

    # Epochs and patience
    epochs = params.get("fit_kwargs.epochs")
    if epochs:
        result["epochs"] = int(epochs)
    patience = params.get("fit_kwargs.early_stopping_patience")
    if patience:
        result["patience"] = int(patience)

    # Base distribution (extract from model config name if possible)
    run_name = run.data.tags.get("mlflow.runName", "")
    result["base"] = "truncated_normal" if "truncated" in run_name.lower() else "normal"

    return result


def read_local_config(target, model):
    """Read local YAML config file for a target+model."""
    path = PARAMS_DIR / target / f"{model}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def compare_config(target, model, mlflow_params, local_config):
    """Compare MLflow best config with local config. Return list of mismatches."""
    mismatches = []

    if local_config is None:
        return [f"  LOCAL FILE MISSING: params/models/{target}/{model}.yaml"]

    fit_kwargs = local_config.get("fit_kwargs", {})
    model_kwargs = local_config.get("model_kwargs", {})

    # Compare LR
    local_lr = fit_kwargs.get("learning_rate")
    if isinstance(local_lr, dict):
        # CosineDecay
        local_lr_scheduler = local_lr.get("scheduler_name")
        local_initial_lr = local_lr.get("scheduler_kwargs", {}).get("initial_learning_rate")
        mlflow_lr_type = mlflow_params.get("lr_type")
        if mlflow_lr_type == "CosineDecay":
            mlflow_lr_initial = mlflow_params["initial_lr"]
            if local_initial_lr is not None and abs(local_initial_lr - mlflow_lr_initial) > 1e-6:
                mismatches.append(
                    f"  LR: local CosineDecay({local_initial_lr}) vs MLflow CosineDecay({mlflow_lr_initial})"
                )
        else:
            mismatches.append(
                f"  LR: local CosineDecay({local_initial_lr}) vs MLflow constant({mlflow_params.get('initial_lr')})"
            )
    elif isinstance(local_lr, (int, float)):
        if mlflow_params["lr_type"] == "constant":
            if abs(local_lr - mlflow_params["initial_lr"]) > 1e-6:
                mismatches.append(
                    f"  LR: local {local_lr} vs MLflow {mlflow_params['initial_lr']}"
                )
        else:
            mismatches.append(
                f"  LR: local constant({local_lr}) vs MLflow CosineDecay({mlflow_params.get('initial_lr')})"
            )

    # Compare epochs
    local_epochs = fit_kwargs.get("epochs")
    mlflow_epochs = mlflow_params.get("epochs")
    if local_epochs and mlflow_epochs and local_epochs != mlflow_epochs:
        mismatches.append(f"  epochs: local {local_epochs} vs MLflow {mlflow_epochs}")

    # Compare patience
    local_patience = fit_kwargs.get("early_stopping_patience")
    mlflow_patience = mlflow_params.get("patience")
    if local_patience and mlflow_patience and local_patience != mlflow_patience:
        mismatches.append(f"  patience: local {local_patience} vs MLflow {mlflow_patience}")

    # Compare hidden units
    local_hidden = model_kwargs.get("parameters_fn_kwargs", {}).get("hidden_units")
    mlflow_hidden = mlflow_params.get("hidden_units")
    if local_hidden and mlflow_hidden and local_hidden != mlflow_hidden:
        mismatches.append(f"  hidden_units: local {local_hidden} vs MLflow {mlflow_hidden}")

    return mismatches


def apply_fix(target, model, mlflow_params, local_config):
    """Apply MLflow best config to local YAML file."""
    path = PARAMS_DIR / target / f"{model}.yaml"
    config = yaml.safe_load(path.read_text()) if path.exists() else {}

    fit_kwargs = config.setdefault("fit_kwargs", {})
    model_kwargs = config.setdefault("model_kwargs", {})

    # Fix LR
    if mlflow_params["lr_type"] == "CosineDecay":
        fit_kwargs["learning_rate"] = {
            "scheduler_name": "CosineDecay",
            "scheduler_kwargs": {
                "initial_learning_rate": mlflow_params["initial_lr"],
                "decay_steps": 22000,
            },
        }
    else:
        fit_kwargs["learning_rate"] = mlflow_params["initial_lr"]

    # Fix epochs, patience
    if "epochs" in mlflow_params:
        fit_kwargs["epochs"] = mlflow_params["epochs"]
    if "patience" in mlflow_params:
        fit_kwargs["early_stopping_patience"] = mlflow_params["patience"]

    # Fix hidden units
    if "hidden_units" in mlflow_params:
        model_kwargs.setdefault("parameters_fn_kwargs", {})["hidden_units"] = mlflow_params["hidden_units"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(config, default_flow_style=None, sort_keys=False))
    return f"  ✓ Applied fix to params/models/{target}/{model}.yaml"


def main():
    parser = argparse.ArgumentParser(description="Sync best MLflow configs to local YAML")
    parser.add_argument("--apply", action="store_true", help="Apply fixes to local YAML files")
    parser.add_argument("--check", action="store_true", help="Exit 1 if any mismatches found")
    args = parser.parse_args()

    total_mismatches = 0
    total_fixes = 0
    skipped_nan = 0

    for exp_id, target in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"  Target: {target}")
        print(f"{'='*60}")

        for model in ACTIVE_MODELS:
            # Skip models that will produce NaN on non-DLA targets
            if target in NAN_NON_DLA_TARGETS and model in NAN_SCALE_TRUNCATED_MODELS:
                skipped_nan += 1
                continue

            best_run = get_best_run(exp_id, model, target)
            if best_run is None:
                print(f"  {model:40s}  NO MLFLOW RUN")
                total_mismatches += 1
                continue

            mlflow_params = extract_params(best_run)
            local_config = read_local_config(target, model)

            mismatches = compare_config(target, model, mlflow_params, local_config)

            nll = best_run.data.metrics.get("nll", "?")
            if isinstance(nll, float) and nll > 0 and nll < 1e100:
                nll_str = f"{nll:.4f}"
            else:
                nll_str = str(nll)

            if mismatches:
                total_mismatches += len(mismatches)
                print(f"  {model:40s}  MISMATCH ({nll_str})")
                for m in mismatches:
                    print(m)
                if args.apply:
                    result = apply_fix(target, model, mlflow_params, local_config)
                    print(result)
                    total_fixes += 1
            else:
                print(f"  {model:40s}  OK ({nll_str})")

    print(f"\n{'='*60}")
    print(f"  Summary: {total_mismatches} mismatches, {total_fixes} fixes applied, {skipped_nan} NaN models skipped")
    print(f"{'='*60}")

    if args.check and total_mismatches > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
