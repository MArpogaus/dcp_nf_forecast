#!/usr/bin/env python3
"""Generate publication-ready figures and LaTeX tables from evaluation results.

Usage:
    source /app/.venv/bin/activate
    python scripts/generate_paper_figures.py --output-dir paper_figures

Outputs:
    paper_figures/
        tables/
            model_comparison.tex       — per-target model comparison
            cross_target_winners.tex   — best model per target
        figures/
            example_forecast.pdf       — best-model forecast with CI bands
            pit_histograms.pdf         — 2x2 PIT grid, one per target
            nll_comparison.pdf         — 4x3 bar chart by model family
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dcp_nf_forecast.utils import setup_plotting_style


TARGETS = {
    "dla": "DLA",
    "ofen_g_koks": "Ofen G Koks",
    "ofen_f_koks": "Ofen F Koks",
    "pl2": "PL2",
}

RESULTS_DIR = Path("results")

COL_WIDTH = 3.417
TEXT_WIDTH = 7.0

MODEL_FAMILIES = {
    "Baselines": ["normal_baseline", "truncated_baseline", "lognormal_baseline"],
    "Spline RQS": [m for m in [
        "spline_nf", "spline_nf_truncated", "spline_nf_lognormal",
        "spline_nf_scale", "spline_nf_scale_truncated", "spline_nf_scale_lognormal",
        "spline_nf_scale_shift", "spline_nf_scale_shift_truncated", "spline_nf_scale_shift_lognormal",
    ]],
    "Bernstein": [m for m in [
        "bernstein_nf", "bernstein_nf_truncated", "bernstein_nf_lognormal",
        "bernstein_nf_scale", "bernstein_nf_scale_truncated", "bernstein_nf_scale_lognormal",
        "bernstein_nf_scale_shift", "bernstein_nf_scale_shift_truncated", "bernstein_nf_scale_shift_lognormal",
    ]],
}

FAMILY_ORDER = ["Baselines", "Spline RQS", "Bernstein"]

FAMILY_COLORS = {
    "Baselines": "#7f7f7f",
    "Spline RQS": "#1f77b4",
    "Bernstein": "#2ca02c",
}


BASELINE_LABELS = {
    "normal_baseline": "Normal",
    "truncated_baseline": "TruncNorm",
    "lognormal_baseline": "LogNormal",
}


def _short_model_name(model: str) -> str:
    """Short readable label for a model name.

    Format: ``{S|B}[+Sc][+Sh] | {N|Tr|LN}``
    """
    if model in BASELINE_LABELS:
        return BASELINE_LABELS[model]
    if model.startswith("spline_nf"):
        prefix = "S"
        rest = model[len("spline_nf"):].strip("_")
    elif model.startswith("bernstein_nf"):
        prefix = "B"
        rest = model[len("bernstein_nf"):].strip("_")
    else:
        return model
    parts = []
    if "scale_shift" in rest:
        parts.append("Sc")
        parts.append("Sh")
    elif "scale" in rest:
        parts.append("Sc")
    if "truncated" in rest:
        base = "Tr"
    elif "lognormal" in rest:
        base = "LN"
    else:
        base = "N"
    if parts:
        return f"{prefix}+{'+'.join(parts)} | {base}"
    return f"{prefix} | {base}"


def load_samples(eval_dir: Path) -> np.ndarray | None:
    """Load samples from single samples.feather, return (n_samp, n_eval, pred_h) array."""
    samp_file = eval_dir / "samples.feather"
    if not samp_file.exists():
        return None
    df = pd.read_feather(samp_file)
    n_eval = df["forecast_origin"].nunique()
    n_samples = df["sample_number"].nunique()
    step_cols = [c for c in df.columns if c.startswith("step_")]
    flat = df[step_cols].values
    samples = flat.reshape(n_eval, n_samples, -1).transpose(1, 0, 2)
    return samples


def load_all_metrics() -> pd.DataFrame:
    rows = []
    for target, target_label in TARGETS.items():
        model_dir = RESULTS_DIR / target
        if not model_dir.exists():
            continue
        for model in sorted(model_dir.iterdir()):
            metrics_file = model / "evaluation" / "metrics.yaml"
            if not metrics_file.exists():
                continue
            with open(metrics_file) as f:
                metrics = yaml.safe_load(f)
            rows.append({
                "target": target,
                "target_label": target_label,
                "model": model.name,
                "nll": metrics.get("nll"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mean_90_ci_width": metrics.get("mean_90_ci_width"),
            })
    return pd.DataFrame(rows)


def get_best_model(df: pd.DataFrame, target: str) -> pd.Series:
    tdf = df[df["target"] == target].dropna(subset=["nll"])
    return tdf.loc[tdf["nll"].idxmin()]


def generate_latex_tables(df: pd.DataFrame, output_dir: Path) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("nll", "NLL $\\downarrow$", "{:.2f}"),
        ("rmse", "RMSE $\\downarrow$", "{:.4f}"),
        ("mae", "MAE $\\downarrow$", "{:.4f}"),
        ("mean_90_ci_width", "CI90 $\\downarrow$", "{:.3f}"),
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Test-set metrics per target and model. Best per metric bolded.}",
        r"\label{tab:model_comparison}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        r"Target / Model & " + " & ".join(m[1] for m in metrics) + r" \\",
        r"\midrule",
    ]
    for target in sorted(df["target"].unique()):
        tdf = df[df["target"] == target].sort_values("nll")
        lines.append(fr"\multicolumn{{{len(metrics) + 1}}}{{l}}{{\textbf{{{tdf.iloc[0]['target_label']}}}}} \\")
        for _, row in tdf.iterrows():
            model_short = row["model"].replace("_", r"\_")
            vals = []
            for metric, _, fmt in metrics:
                val = row[metric]
                if val is None:
                    vals.append("---")
                    continue
                best = tdf[metric].min()
                val_str = fmt.format(val)
                if abs(val - best) < 1e-6:
                    val_str = f"\\textbf{{{val_str}}}"
                vals.append(val_str)
            lines.append("  " + model_short + " & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (table_dir / "model_comparison.tex").write_text("\n".join(lines) + "\n")
    print(f"  Wrote {table_dir / 'model_comparison.tex'}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Best model per target by NLL.}",
        r"\label{tab:cross_target_winners}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Target & Best Model & NLL & RMSE & CI90 \\",
        r"\midrule",
    ]
    for target in sorted(df["target"].unique()):
        best = get_best_model(df, target)
        model_label = best['model'].replace('_', '$\\_$')
        lines.append(
            f"  {best['target_label']} & {model_label} & "
            f"{best['nll']:.2f} & {best['rmse']:.4f} & {best['mean_90_ci_width']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (table_dir / "cross_target_winners.tex").write_text("\n".join(lines) + "\n")
    print(f"  Wrote {table_dir / 'cross_target_winners.tex'}")


def plot_example_forecast(best_row: pd.Series, output_dir: Path) -> None:
    """Time series: actual, median, 50/80/90% CI bands, history/forecast split."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    target = best_row["target"]
    model = best_row["model"]
    eval_dir = RESULTS_DIR / target / model / "evaluation"

    samples = load_samples(eval_dir)
    if samples is None:
        print(f"  WARNING: No samples.feather for {target}/{model}")
        return

    processed_dir = Path("data/processed") / target
    y_test_path = processed_dir / "y_test.feather"
    if not y_test_path.exists():
        print(f"  WARNING: No y_test at {y_test_path}")
        return
    y_true = pd.read_feather(y_test_path).values

    n_show = min(48, len(y_true))
    idx = np.random.randint(0, len(y_true) - n_show) if len(y_true) > n_show else 0
    y_seg = y_true[idx: idx + n_show, 0]
    n_history = n_show // 3
    n_forecast = n_show - n_history

    samp_seg = samples[:, idx: idx + n_show, 0]

    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.6))

    t = np.arange(n_show)
    q = np.percentile(samp_seg, [5, 10, 25, 50, 75, 90, 95], axis=0)

    ax.fill_between(t, q[0], q[-1], alpha=0.15, color="#1f77b4", label="90% CI")
    ax.fill_between(t, q[1], q[-2], alpha=0.25, color="#2c8ad4", label="80% CI")
    ax.fill_between(t, q[2], q[-3], alpha=0.35, color="#3a9ee6", label="50% CI")
    ax.plot(t, q[3], color="#d62728", linestyle="--", linewidth=0.8, label="Median")
    ax.plot(t, y_seg, color="#333333", linewidth=0.7, label="Actual")
    ax.axvline(x=n_history - 0.5, color="gray", linestyle=":", linewidth=0.6)
    ax.text(0.17, 0.95, "History",
            ha="center", va="top", fontsize=6, style="italic", transform=ax.transAxes)
    ax.text(0.67, 0.95, "Forecast",
            ha="center", va="top", fontsize=6, style="italic", transform=ax.transAxes)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("Value (normalized)", fontsize=8)
    ax.set_title(f"{best_row['target_label']} — {_short_model_name(best_row['model'])}", fontsize=9, pad=2)
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.7, ncol=1)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "example_forecast.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_dir / 'example_forecast.pdf'} (model={model})")


def plot_pit_histograms_2x2(df: pd.DataFrame, output_dir: Path) -> None:
    """2×2 grid of PIT histograms, best model per target."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    setup_plotting_style()
    fig, axes = plt.subplots(2, 2, figsize=(COL_WIDTH, COL_WIDTH * 0.95),
                             sharex=True, sharey=True,
                             gridspec_kw=dict(hspace=0.50, wspace=0.15))

    targets_sorted = sorted(TARGETS.keys())
    for ax, target in zip(axes.flatten(), targets_sorted):
        best = get_best_model(df, target)
        eval_dir = RESULTS_DIR / target / best["model"] / "evaluation"

        pit_file = eval_dir / "pit_values.feather"
        if pit_file.exists():
            pit = pd.read_feather(pit_file).values.flatten()
        else:
            samples = load_samples(eval_dir)
            if samples is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            processed_dir = Path("data/processed") / target
            y_test_path = processed_dir / "y_test.feather"
            if not y_test_path.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            y_true = pd.read_feather(y_test_path).values
            y_true_exp = y_true[np.newaxis, :, :]
            pit = np.mean(samples < y_true_exp, axis=0).flatten()

        ax.hist(pit, bins=20, range=(0, 1), density=True,
                color="#1f77b4", alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axhline(y=1.0, color="#d62728", linestyle="--", linewidth=0.6, label="Uniform")
        ax.set_title(f"{TARGETS[target]} — {_short_model_name(best['model'])}", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        if ax in axes[-1, :]:
            ax.set_xlabel("PIT value", fontsize=7)
        if ax in axes[:, 0]:
            ax.set_ylabel("Density", fontsize=7)
        ax.legend(fontsize=5, loc="upper right", framealpha=0.7)

    fig.suptitle("PIT Histograms — Best model per target", fontsize=8, y=0.99)
    fig.savefig(fig_dir / "pit_histograms.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "pit_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_dir / 'pit_histograms.pdf'}")


def plot_nll_bar_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """4×3 grid: targets × model families, each panel sorted by NLL."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    setup_plotting_style()
    targets_sorted = sorted(TARGETS.keys())
    n_targets = len(targets_sorted)
    n_families = len(FAMILY_ORDER)

    # per-column xlim: same for all targets in a given family
    col_xlim = {}
    for fi, family in enumerate(FAMILY_ORDER):
        fam_models = MODEL_FAMILIES[family]
        all_vals = []
        for target in targets_sorted:
            tdf = df[df["target"] == target].dropna(subset=["nll"])
            tdf = tdf[~np.isinf(tdf["nll"])]
            fdf = tdf[tdf["model"].isin(fam_models)]
            if not fdf.empty:
                all_vals.extend(fdf["nll"].values)
        if all_vals:
            gmin = min(all_vals)
            gmax = max(all_vals)
            grange = max(gmax - gmin, 1.0)
            col_xlim[family] = (gmin - 0.08 * grange, gmin + 1.15 * grange)
        else:
            col_xlim[family] = (0, 1)

    fig, axes = plt.subplots(n_targets, n_families,
                             figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.65),
                             sharex=False, sharey=False,
                             gridspec_kw=dict(wspace=0.55, hspace=0.40))

    for ti, target in enumerate(targets_sorted):
        tdf = df[df["target"] == target].dropna(subset=["nll"]).copy()
        tdf = tdf[~np.isinf(tdf["nll"])].copy()
        if tdf.empty:
            for fi in range(n_families):
                axes[ti, fi].text(0.5, 0.5, "—", ha="center", va="center", fontsize=8)
            continue
        nll_min = tdf["nll"].min()
        nll_range = max(tdf["nll"].max() - nll_min, 1.0)

        for fi, family in enumerate(FAMILY_ORDER):
            ax = axes[ti, fi]
            fam_models = MODEL_FAMILIES[family]
            fdf = tdf[tdf["model"].isin(fam_models)]
            fdf = fdf[~np.isinf(fdf["nll"])].sort_values("nll")
            if fdf.empty:
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=8, transform=ax.transAxes)
                ax.set_frame_on(False)
                continue

            labels = [_short_model_name(m) for m in fdf["model"]]
            vals = fdf["nll"].values
            colors_list = ["#d62728" if abs(v - nll_min) < 1e-6 else FAMILY_COLORS[family] for v in vals]
            ax.barh(range(len(vals)), vals, color=colors_list, height=0.55)
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels(labels, fontsize=5.5)
            ax.set_xlim(*col_xlim[family])

            if ti == 0:
                ax.set_title(family, fontsize=7, fontweight="bold", pad=3)
            if ti == n_targets - 1:
                ax.set_xlabel("NLL", fontsize=6.5)
            else:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=5.5)
            ax.grid(axis="x", alpha=0.2, linewidth=0.3)
            ax.set_axisbelow(True)
            if nll_range < 100:
                ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

            # target label as ylabel on leftmost column
            if fi == 0:
                ax.set_ylabel(TARGETS[target], fontsize=7.5, fontweight="bold", labelpad=2)

    fig.suptitle("Test NLL per target (lower is better, red = best per target)", fontsize=9, y=0.99)
    fig.savefig(fig_dir / "nll_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / "nll_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote {fig_dir / 'nll_comparison.pdf'}")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("PER-TARGET BEST MODELS (by NLL)")
    print("=" * 80)
    for target in sorted(df["target"].unique()):
        tdf = df[df["target"] == target].sort_values("nll")
        print(f"\n  {target}:")
        for i, (_, row) in enumerate(tdf.iterrows()):
            prefix = "🏆" if i == 0 else " "
            nll_s = f"{row['nll']:.2f}" if pd.notna(row['nll']) else "N/A"
            rmse_s = f"{row['rmse']:.4f}" if pd.notna(row['rmse']) else "N/A"
            ci90_s = f"{row['mean_90_ci_width']:.3f}" if pd.notna(row['mean_90_ci_width']) else "N/A"
            print(f"  {prefix} {row['model']:<45} NLL={nll_s:<8} RMSE={rmse_s:<8} CI90={ci90_s}")

    print("\nCROSS-TARGET WINNERS")
    for target in sorted(df["target"].unique()):
        best = get_best_model(df, target)
        print(f"  {best['target_label']:<15} → {best['model']:<45} NLL={best['nll']:.2f}")
    print()


def main():
    global COL_WIDTH, TEXT_WIDTH
    COL_WIDTH = 3.3

    parser = argparse.ArgumentParser(description="Generate publication figures and tables.")
    parser.add_argument("--output-dir", default="paper_figures", help="Output directory")
    parser.add_argument("--col-width", type=float, default=COL_WIDTH, help="LaTeX column width (in)")
    args = parser.parse_args()

    COL_WIDTH = args.col_width
    TEXT_WIDTH = COL_WIDTH * 2 + 0.3

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation metrics ...")
    df = load_all_metrics()
    if df.empty:
        print("ERROR: No evaluation results. Run DVC repro first.")
        sys.exit(1)
    print(f"  Found {len(df)} (target, model) pairs")
    print(f"  Targets: {sorted(df['target'].unique())}")
    print(f"  Models:  {sorted(df['model'].unique())}")

    print("\nGenerating LaTeX tables ...")
    generate_latex_tables(df, output_dir)

    print("\nGenerating example forecast figure ...")
    best_dla = get_best_model(df, "dla")
    plot_example_forecast(best_dla, output_dir)

    print("\nGenerating PIT histograms 2×2 ...")
    plot_pit_histograms_2x2(df, output_dir)

    print("\nGenerating NLL comparison bar chart ...")
    plot_nll_bar_chart(df, output_dir)

    print_summary(df)
    print(f"\nAll outputs → {output_dir.resolve()}")


if __name__ == "__main__":
    main()
