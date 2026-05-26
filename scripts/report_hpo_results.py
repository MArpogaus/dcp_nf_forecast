"""Report HPO results from MLflow: parameter variations and winning configs.

Produces both terminal output and publication-ready LaTeX tables via
pandas.DataFrame.to_latex().

Usage:
    python scripts/report_hpo_results.py                  # terminal tables
    python scripts/report_hpo_results.py --latex           # LaTeX tables
    python scripts/report_hpo_results.py --latex --output tables.tex
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

TARGET_NAMES = {"85": "dla", "86": "ofen_g_koks", "87": "ofen_f_koks", "88": "pl2"}
TARGET_LABELS = {
    "dla": r"\textsc{dla}",
    "ofen_g_koks": r"\textsc{ofen\_g\_koks}",
    "ofen_f_koks": r"\textsc{ofen\_f\_koks}",
    "pl2": r"\textsc{pl2}",
}
EXPERIMENT_IDS = list(TARGET_NAMES)

TOP_PARAM_KEYS = [
    "model_kwargs.learning_rate",
    "fit_kwargs.learning_rate",
    "fit_kwargs.learning_rate.scheduler_name",
    "fit_kwargs.learning_rate.scheduler_kwargs.initial_learning_rate",
    "fit_kwargs.epochs",
    "fit_kwargs.early_stopping_patience",
    "fit_kwargs.max_epochs",
    "model_kwargs.num_parameters",
    "model_kwargs.num_layers",
    "model_kwargs.bijector",
    "model_kwargs.bijector_kwargs.nbins",
    "model_kwargs.parameters_constraint_fn_kwargs.nbins",
    "model_kwargs.bijector_kwargs.interval_width",
    "model_kwargs.parameters_constraint_fn_kwargs.interval_width",
    "model_kwargs.bijector_kwargs.range_min",
    "model_kwargs.bijector_kwargs.range_max",
    "model_kwargs.parameters_fn_kwargs.hidden_units.0",
    "model_kwargs.parameters_fn_kwargs.hidden_units.1",
    "model_kwargs.parameters_fn_kwargs.dropout",
    "model_kwargs.parameters_fn_kwargs.batch_norm",
    "model_kwargs.base_distribution_kwargs.distribution_name",
    "model_kwargs.base_distribution_kwargs.low",
    "model_kwargs.base_distribution_kwargs.high",
    "model_kwargs.parameters_constraint_fn_kwargs.low",
    "model_kwargs.parameters_constraint_fn_kwargs.high",
    "model_kwargs.parameters_constraint_fn_kwargs.interval_width",
    "model_kwargs.order",
]

MODEL_SHORT = {
    "normal_baseline": "$\\mathcal{N}$-base",
    "truncated_baseline": "Trunc-$\\mathcal{N}$-base",
    "lognormal_baseline": "LogNormal-base",
    "spline_nf": "Spline",
    "spline_nf_truncated": "Spline (trunc)",
    "spline_nf_lognormal": "Spline (log)",
    "spline_nf_scale": "Spline+Scale",
    "spline_nf_scale_truncated": "Spline+Scale (trunc)",
    "spline_nf_scale_lognormal": "Spline+Scale (log)",
    "spline_nf_scale_shift": "Spline+Scale+Shift",
    "spline_nf_scale_shift_truncated": "Spline+Scale+Shift (trunc)",
    "bernstein_nf": "Bernstein",
    "bernstein_nf_truncated": "Bernstein (trunc)",
    "bernstein_nf_lognormal": "Bernstein (log)",
    "bernstein_nf_scale": "Bernstein+Scale",
    "bernstein_nf_scale_truncated": "Bernstein+Scale (trunc)",
    "bernstein_nf_scale_lognormal": "Bernstein+Scale (log)",
    "bernstein_nf_scale_shift": "Bernstein+Scale+Shift",
    "bernstein_nf_scale_shift_truncated": "Bernstein+Scale+Shift (trunc)",
}


def fmt_val(v: str | None) -> str:
    if v is None:
        return "\u2014"
    return v


def fmt_metric(val: float | None) -> str:
    if val is None or (isinstance(val, float) and val != val):
        return "NaN"
    if isinstance(val, float) and abs(val) > 1e10:
        return "INF"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def fmt_lr(params: dict) -> str:
    lr_val = params.get("fit_kwargs.learning_rate")
    sched_name = params.get("fit_kwargs.learning_rate.scheduler_name")
    init_lr = params.get("fit_kwargs.learning_rate.scheduler_kwargs.initial_learning_rate")
    if sched_name and init_lr:
        return f"{init_lr} ({sched_name})"
    if lr_val:
        return str(lr_val)
    return "\u2014"


def fmt_lr_latex(params: dict) -> str:
    lr_val = params.get("fit_kwargs.learning_rate")
    sched_name = params.get("fit_kwargs.learning_rate.scheduler_name")
    init_lr = params.get("fit_kwargs.learning_rate.scheduler_kwargs.initial_learning_rate")
    if sched_name and init_lr:
        return f"{init_lr} (cosine)"
    if lr_val:
        return str(lr_val)
    return "--"


def fmt_base_latex(base: str) -> str:
    mapping = {
        "normal": "$\\mathcal{N}(0,1)$",
        "truncated_normal": "Trunc-$\\mathcal{N}(0,5)$",
        "lognormal": "LogNormal",
        "\u2014": "--",
    }
    return mapping.get(base, base)


def get_parent_run_id(eval_run) -> str | None:
    return eval_run.data.tags.get("mlflow.parentRunId")


def get_model_name(run_name: str) -> str:
    if run_name.startswith("eval_"):
        return "_".join(run_name.split("_")[1:-2])
    return "_".join(run_name.split("_")[:-2])


def fmt_hidden(params: dict) -> str:
    hidden = []
    for i in range(3):
        hk = f"model_kwargs.parameters_fn_kwargs.hidden_units.{i}"
        if hk in params:
            hidden.append(params[hk])
    return f"[{','.join(hidden)}]" if hidden else "\u2014"


def get_nbins(params: dict) -> str:
    return params.get(
        "model_kwargs.bijector_kwargs.nbins",
        params.get("model_kwargs.parameters_constraint_fn_kwargs.nbins", "\u2014"),
    )


def get_interval(params: dict) -> str:
    return params.get(
        "model_kwargs.bijector_kwargs.interval_width",
        params.get("model_kwargs.parameters_constraint_fn_kwargs.interval_width", "\u2014"),
    )


def get_epochs(params: dict) -> str:
    return params.get("fit_kwargs.max_epochs", params.get("fit_kwargs.epochs", "\u2014"))


def escape_underscore(s: str) -> str:
    return s.replace("_", "\\_")


def collect_runs(args) -> dict:
    """Query MLflow and return structured data per target."""
    client = MlflowClient()
    target_data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    best_per_target_model: dict[str, dict[str, dict]] = defaultdict(dict)

    for exp_id in args.experiment_ids:
        target = TARGET_NAMES.get(exp_id, exp_id)
        exp = client.get_experiment(exp_id)
        if exp is None:
            print(f"Experiment {exp_id} not found, skipping.", file=sys.stderr)
            continue

        all_runs = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["start_time ASC"],
            max_results=2000,
        )

        eval_runs = [r for r in all_runs if "evaluation" in r.data.tags.get("mlflow.runName", "")]
        parent_cache: dict[str, dict] = {}

        for er in eval_runs:
            run_name = er.data.tags.get("mlflow.runName", "")
            model = get_model_name(run_name)
            metrics = dict(er.data.metrics)
            nll = metrics.get("nll")
            if nll is not None and (nll != nll or nll > 1e10 or nll < args.min_nll):
                continue

            parent_id = get_parent_run_id(er)
            if parent_id and parent_id not in parent_cache:
                try:
                    parent_cache[parent_id] = dict(client.get_run(parent_id).data.params)
                except Exception:
                    parent_cache[parent_id] = {}
            params = parent_cache.get(parent_id, {})

            metric_data = {k: metrics.get(k) for k in ("nll", "rmse", "mae", "mean_90_ci_width")}

            param_data = {k: params.get(k, "\u2014") for k in TOP_PARAM_KEYS if k in params}
            param_data["_lr"] = fmt_lr(params)
            param_data["_hidden"] = fmt_hidden(params)

            entry = {
                "run_name": run_name,
                "model": model,
                "nll": nll,
                **metric_data,
                **param_data,
                "start_time": er.info.start_time,
                "run_id": er.info.run_id,
            }
            target_data[target][model].append(entry)

            prev = best_per_target_model[target].get(model)
            if prev is None or (nll is not None and (prev["nll"] is None or nll < prev["nll"])):
                best_per_target_model[target][model] = entry

    return target_data, best_per_target_model


def print_terminal(target_data, best_per_target_model):
    """Print console-friendly tables."""
    EQ = "="
    DASH = "\u2500"
    CHECK = "\u2713"
    BALLOT = "\u2717"
    STAR = "\u2605"
    NDASH = "\u2014"

    for target in sorted(target_data):
        print(f"\n{EQ*90}")
        print(f"  Target: {target.upper()}")
        print(f"{EQ*90}")
        models = sorted(target_data[target].keys())

        for model in sorted(models, key=lambda m: (
            best_per_target_model[target][m]["nll"]
            if best_per_target_model[target].get(m, {}).get("nll") is not None
            and best_per_target_model[target][m]["nll"] == best_per_target_model[target][m]["nll"]
            else float("inf")
        )):
            entries = target_data[target][model]
            best_entry = best_per_target_model[target].get(model)
            best_nll = best_entry["nll"] if best_entry else None
            ok = best_nll is not None and best_nll == best_nll
            marker = CHECK if ok else BALLOT
            print(f"\n  {marker} {model}")
            print(f"  {DASH*80}")
            if len(entries) == 1:
                e = entries[0]
                print(
                    f"     NLL={fmt_metric(e.get('nll'))}  "
                    f"RMSE={fmt_metric(e.get('rmse'))}  "
                    f"MAE={fmt_metric(e.get('mae'))}  "
                    f"CI90={fmt_metric(e.get('mean_90_ci_width'))}"
                )
                nbins = get_nbins(e)
                print(f"     LR={e.get('_lr', NDASH)}  "
                      f"hidden={e.get('_hidden', NDASH)}  "
                      f"epochs={fmt_val(get_epochs(e))}  "
                      f"patience={fmt_val(e.get('fit_kwargs.early_stopping_patience'))}  "
                      f"nbins={fmt_val(nbins)}  "
                      f"nparams={fmt_val(e.get('model_kwargs.num_parameters'))}  "
                      f"base={fmt_val(e.get('model_kwargs.base_distribution_kwargs.distribution_name'))}")
            else:
                print(f"     {len(entries)} configs evaluated. Best NLL={fmt_metric(best_nll)}")
                header = (
                    f"  {'#':>3}  {'NLL':>10}  {'RMSE':>8}  {'MAE':>8}  "
                    f"{'CI90':>8}  {'LR/schedule':>24}  {'hidden':>14}  "
                    f"{'epochs':>6}  {'pat':>4}  {'nbins':>5}  {'nparams':>4}  {'base'}"
                )
                print(header)
                print(f"  {DASH * len(header)}")
                sorted_entries = sorted(
                    entries,
                    key=lambda x: (x["nll"] if x["nll"] is not None and x["nll"] == x["nll"] else float("inf")),
                )
                for i, e in enumerate(sorted_entries):
                    nbins = get_nbins(e)
                    is_best = best_entry and e["run_id"] == best_entry["run_id"]
                    rm = STAR if is_best else " "
                    print(
                        f"  {rm}{i+1:>2}  "
                        f"{fmt_metric(e.get('nll')):>10}  "
                        f"{fmt_metric(e.get('rmse')):>8}  "
                        f"{fmt_metric(e.get('mae')):>8}  "
                        f"{fmt_metric(e.get('mean_90_ci_width')):>8}  "
                        f"{e.get('_lr', NDASH):>24}  "
                        f"{e.get('_hidden', NDASH):>14}  "
                        f"{fmt_val(get_epochs(e)):>6}  "
                        f"{fmt_val(e.get('fit_kwargs.early_stopping_patience')):>4}  "
                        f"{fmt_val(nbins):>5}  "
                        f"{fmt_val(e.get('model_kwargs.num_parameters')):>4}  "
                        f"{fmt_val(e.get('model_kwargs.base_distribution_kwargs.distribution_name'))}"
                    )

    # Summary tables
    print(f"\n\n{EQ*90}")
    print(f"  FULL RESULTS TABLE {NDASH} ALL MODELS {chr(215)} ALL TARGETS")
    print(f"{EQ*90}")
    h = (f"{'Target':<18}  {'Model':<36}  {'NLL':>10}  {'RMSE':>8}  "
         f"{'MAE':>8}  {'CI90':>8}  {'LR':>18}  {'hidden':>14}  "
         f"{'ep':>4}  {'nb':>4}  {'base'}")
    print(h)
    print(f"{DASH * len(h)}")
    for target in sorted(target_data):
        models_sorted = sorted(
            target_data[target].keys(),
            key=lambda m: (best_per_target_model[target][m]["nll"]
                          if best_per_target_model[target].get(m, {}).get("nll") is not None
                          and best_per_target_model[target][m]["nll"] == best_per_target_model[target][m]["nll"]
                          else float("inf")),
        )
        for model in models_sorted:
            e = best_per_target_model[target].get(model)
            if not e:
                continue
            nbins = get_nbins(e)
            print(
                f"{target:<18}  {model:<36}  "
                f"{fmt_metric(e.get('nll')):>10}  "
                f"{fmt_metric(e.get('rmse')):>8}  "
                f"{fmt_metric(e.get('mae')):>8}  "
                f"{fmt_metric(e.get('mean_90_ci_width')):>8}  "
                f"{e.get('_lr', NDASH):>18}  "
                f"{e.get('_hidden', NDASH):>14}  "
                f"{fmt_val(get_epochs(e)):>4}  "
                f"{fmt_val(nbins):>4}  "
                f"{fmt_val(e.get('model_kwargs.base_distribution_kwargs.distribution_name'))}"
            )


def build_best_df(target_data, best_per_target_model) -> pd.DataFrame:
    """Build DataFrame of best config per (target, model)."""
    rows = []
    for target in sorted(target_data):
        models_sorted = sorted(
            target_data[target].keys(),
            key=lambda m: (best_per_target_model[target][m]["nll"]
                          if best_per_target_model[target].get(m, {}).get("nll") is not None
                          and best_per_target_model[target][m]["nll"] == best_per_target_model[target][m]["nll"]
                          else float("inf")),
        )
        for model in models_sorted:
            e = best_per_target_model[target].get(model)
            if not e:
                continue
            base = e.get("model_kwargs.base_distribution_kwargs.distribution_name", "\u2014")
            rows.append({
                "Target": TARGET_LABELS.get(target, target),
                "Model": MODEL_SHORT.get(model, model),
                "NLL": e.get("nll"),
                "RMSE": e.get("rmse"),
                "MAE": e.get("mae"),
                "CI90": e.get("mean_90_ci_width"),
                "LR": e.get("_lr", "\u2014"),
                "Hidden": e.get("_hidden", "\u2014"),
                "Epochs": get_epochs(e),
                "Nbins": get_nbins(e),
                "NParams": e.get("model_kwargs.num_parameters", "\u2014"),
                "Base": base,
            })
    return pd.DataFrame(rows)


def build_winner_df(target_data, best_per_target_model) -> pd.DataFrame:
    """Build DataFrame of best model per target."""
    rows = []
    for target in sorted(target_data):
        best_entry = None
        best_nll = float("inf")
        for m in target_data[target]:
            e = best_per_target_model[target].get(m)
            if e and e.get("nll") is not None and e["nll"] < best_nll:
                best_nll = e["nll"]
                best_entry = e
        if best_entry:
            base = best_entry.get("model_kwargs.base_distribution_kwargs.distribution_name", "\u2014")
            rows.append({
                "Target": TARGET_LABELS.get(target, target),
                "Model": MODEL_SHORT.get(best_entry["model"], best_entry["model"]),
                "NLL": best_entry.get("nll"),
                "RMSE": best_entry.get("rmse"),
                "MAE": best_entry.get("mae"),
                "CI90": best_entry.get("mean_90_ci_width"),
                "LR": best_entry.get("_lr", "\u2014"),
                "Hidden": best_entry.get("_hidden", "\u2014"),
                "Epochs": get_epochs(best_entry),
                "Nbins": get_nbins(best_entry),
                "NParams": best_entry.get("model_kwargs.num_parameters", "\u2014"),
                "Base": base,
            })
    return pd.DataFrame(rows)


def print_latex(target_data, best_per_target_model):
    """Print publication-ready LaTeX tables."""

    df_best = build_best_df(target_data, best_per_target_model)
    if df_best.empty:
        return

    df_best["NLL_s"] = df_best["NLL"].apply(
        lambda x: f"{x:.2f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
    )
    df_best["RMSE_s"] = df_best["RMSE"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
    )
    df_best["MAE_s"] = df_best["MAE"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
    )
    df_best["CI90_s"] = df_best["CI90"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
    )

    latex = (
        df_best[
            ["Target", "Model", "NLL_s", "RMSE_s", "MAE_s", "CI90_s",
             "LR", "Hidden", "Epochs", "Nbins", "NParams", "Base"]
        ]
        .rename(columns={
            "NLL_s": "NLL", "RMSE_s": "RMSE", "MAE_s": "MAE", "CI90_s": "CI90",
            "NParams": "\\#Params", "Base": "Base dist.",
        })
        .to_latex(
            index=False,
            escape=False,
            column_format="l" + "l" + "S[table-format=-4.2]" * 2 + "S[table-format=-1.4]" * 2
                          + "c" * 6,
            position="htbp",
            caption="Best test metrics per model and target from MLflow. "
                    "Models ranked by NLL within each target.",
            label="tab:hpo_best_per_model",
        )
    )
    print("% " + "=" * 72)
    print("% BEST MODEL PER (TARGET, MODEL) — FULL TABLE")
    print("% " + "=" * 72)
    print(latex)

    df_winner = build_winner_df(target_data, best_per_target_model)
    if not df_winner.empty:
        df_winner["NLL_s"] = df_winner["NLL"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
        )
        df_winner["RMSE_s"] = df_winner["RMSE"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
        )
        df_winner["MAE_s"] = df_winner["MAE"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
        )
        df_winner["CI90_s"] = df_winner["CI90"].apply(
            lambda x: f"{x:.4f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
        )
        base_latex = df_winner["Base"].apply(fmt_base_latex)

        lw = (
            df_winner[
                ["Target", "Model", "NLL_s", "RMSE_s", "MAE_s", "CI90_s",
                 "LR", "Hidden", "Epochs", "Nbins", "NParams"]
            ]
            .rename(columns={
                "NLL_s": "NLL", "RMSE_s": "RMSE", "MAE_s": "MAE", "CI90_s": "CI90",
                "NParams": "\\#Params",
            })
            .to_latex(
                index=False,
                escape=False,
                column_format="l" + "l" + "S[table-format=-4.2]" * 2 + "S[table-format=-1.4]" * 2
                              + "c" * 6,
                position="htbp",
                caption="Overall best model per target by NLL.",
                label="tab:hpo_winners",
            )
        )
        print("\n% " + "=" * 72)
        print("% WINNER PER TARGET")
        print("% " + "=" * 72)
        print(lw)

    # Per-target parameter variation tables
    print("\n% " + "=" * 72)
    print("% PARAMETER VARIATIONS (models with 2+ eval runs)")
    print("% " + "=" * 72)
    for target in sorted(target_data):
        for model in sorted(target_data[target].keys()):
            entries = target_data[target][model]
            if len(entries) <= 1:
                continue
            rows = []
            for e in entries:
                rows.append({
                    "Iter": len(rows) + 1,
                    "NLL": e.get("nll"),
                    "RMSE": e.get("rmse"),
                    "MAE": e.get("mae"),
                    "CI90": e.get("mean_90_ci_width"),
                    "LR": e.get("_lr", "\u2014"),
                    "Hidden": e.get("_hidden", "\u2014"),
                    "Epochs": get_epochs(e),
                    "Patience": e.get("fit_kwargs.early_stopping_patience", "\u2014"),
                    "Nbins": get_nbins(e),
                    "Interval": get_interval(e),
                    "NParams": e.get("model_kwargs.num_parameters", "\u2014"),
                    "Base": e.get("model_kwargs.base_distribution_kwargs.distribution_name", "\u2014"),
                })
            df_v = pd.DataFrame(rows)
            df_v["NLL_s"] = df_v["NLL"].apply(
                lambda x: f"{x:.2f}" if isinstance(x, float) and x == x and abs(x) < 1e10 else "NaN"
            )
            mname_esc = escape_underscore(model)
            target_esc = escape_underscore(target)
            cap = f"HPO parameter variations for \\texttt{{{mname_esc}}} on \\texttt{{{target_esc}}}."
            try:
                lv = (
                    df_v[["Iter", "NLL_s", "LR", "Hidden", "Epochs", "Patience",
                          "Nbins", "Interval", "NParams", "Base"]]
                    .rename(columns={
                        "NLL_s": "NLL", "Patience": "Pat.", "Interval": "Int.",
                        "NParams": "\\#Params",
                    })
                    .to_latex(
                        index=False,
                        escape=False,
                        column_format="c" + "S[table-format=-4.2]" + "c" * 8,
                        position="htbp",
                        caption=cap,
                        label=f"tab:hpo_var_{target}_{model}",
                    )
                )
                print(lv)
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report HPO results from MLflow"
    )
    parser.add_argument(
        "--experiment-ids", nargs="+", default=EXPERIMENT_IDS,
        help="MLflow experiment IDs to query",
    )
    parser.add_argument(
        "--min-nll", type=float, default=-1e6,
        help="Minimum NLL threshold to filter out failed runs",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Output LaTeX tables instead of terminal formatting",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write LaTeX output to file (implies --latex)",
    )
    args = parser.parse_args()

    if args.output:
        args.latex = True

    target_data, best_per_target_model = collect_runs(args)

    if args.latex:
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        print_latex(target_data, best_per_target_model)
        sys.stdout = old_stdout
        latex_str = buf.getvalue()
        if args.output:
            with open(args.output, "w") as f:
                f.write(latex_str)
            print(f"LaTeX tables written to {args.output}", file=sys.stderr)
        else:
            print(latex_str)
    else:
        print_terminal(target_data, best_per_target_model)


if __name__ == "__main__":
    main()
