import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import dvc.api
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
import yaml
from hybrid_flows.models import DensityRegressionModel
from hybrid_flows.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)

from dcp_nf_forecast.utils import load_data, setup_logging

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 0.8,
        "figure.figsize": (8, 4.5),
    }
)


def sample_predictions(
    model: DensityRegressionModel,
    x: np.ndarray,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    n = len(x)
    all_samples = []
    pad = 0
    remainder = n % batch_size
    if remainder:
        pad = batch_size - remainder
        x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size]
        dist = model(batch, training=False)
        samples = dist.sample(n_samples)
        all_samples.append(np.array(samples, dtype=np.float32))
    result = np.concatenate(all_samples, axis=1)
    if pad:
        result = result[:, :n, :]
    del all_samples
    return result


def plot_forecast_with_intervals(
    y_true: np.ndarray,
    samples: np.ndarray,
    n_show: int = 48,
    title: str = "",
    alpha: float = 0.3,
) -> plt.Figure:
    n_steps = y_true.shape[1]
    n_cols = 4
    n_rows = int(np.ceil(n_steps / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5), sharex=True, sharey=True
    )
    axes = axes.flatten()
    n_show = min(n_show, len(y_true))
    quantiles = [5, 10, 25, 50, 75, 90, 95]
    q = np.percentile(samples[:, :n_show], quantiles, axis=0)

    for step in range(n_steps):
        ax = axes[step]
        t = np.arange(n_show)
        y = y_true[:n_show, step]
        ax.plot(t, y, "k-", label="Actual", linewidth=0.8)
        ax.plot(t, q[3, :, step], "b-", label="Median", linewidth=0.8)
        ax.fill_between(
            t,
            q[0, :, step],
            q[-1, :, step],
            alpha=alpha,
            color="b",
            label="90% CI",
        )
        ax.fill_between(
            t,
            q[1, :, step],
            q[-2, :, step],
            alpha=alpha * 1.5,
            color="b",
            label="80% CI",
        )
        ax.fill_between(
            t,
            q[2, :, step],
            q[-3, :, step],
            alpha=alpha * 2,
            color="b",
            label="50% CI",
        )
        if step == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"Step +{step + 1}", fontsize=10)

    for j in range(n_steps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.supxlabel("Test sample index")
    fig.supylabel("Value")
    fig.tight_layout()
    return fig


def plot_pit_histogram(
    y_true: np.ndarray,
    samples: np.ndarray,
    n_bins: int = 20,
    title: str = "",
) -> plt.Figure:
    n_steps = y_true.shape[1]
    n_cols = min(4, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    for step in range(n_steps):
        ax = axes[step]
        y = y_true[:, step]
        s = samples[:, :, step]
        pit = np.mean(s < y, axis=0)
        ax.hist(
            pit,
            bins=n_bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8)
        ax.set_title(f"Step +{step + 1}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2.5)

    for j in range(n_steps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"PIT Histogram \u2013 {title}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_calibration(
    y_true: np.ndarray,
    samples: np.ndarray,
    title: str = "",
) -> plt.Figure:
    n_steps = y_true.shape[1]
    n_cols = min(4, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_steps > 1 else [axes]

    nominal = np.linspace(0, 1, 21)
    for step in range(n_steps):
        ax = axes[step]
        y = y_true[:, step]
        s = samples[:, :, step]
        empirical = []
        for q in nominal:
            lower = np.percentile(s, 100 * (1 - q) / 2, axis=0)
            upper = np.percentile(s, 100 * (1 + q) / 2, axis=0)
            coverage = np.mean((y >= lower) & (y <= upper))
            empirical.append(coverage)
        ax.plot(
            nominal, empirical, "o-", markersize=3, linewidth=0.8, color="steelblue"
        )
        ax.plot([0, 1], [0, 1], "r--", linewidth=0.8)
        ax.set_title(f"Step +{step + 1}", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    for j in range(n_steps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Calibration Plot \u2013 {title}", fontsize=12)
    fig.supxlabel("Nominal coverage")
    fig.supylabel("Empirical coverage")
    fig.tight_layout()
    return fig


def compute_metrics(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> dict:
    median = np.percentile(samples, 50, axis=0)
    rmse = float(np.sqrt(np.mean((y_true - median) ** 2)))
    mae = float(np.mean(np.abs(y_true - median)))
    q_skill = np.percentile(samples, [5, 95], axis=0)
    interval_score = float(np.mean(q_skill[1] - q_skill[0]))
    return {
        "rmse": rmse,
        "mae": mae,
        "mean_90_ci_width": interval_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained normalizing flow model on test data"
    )
    parser.add_argument("--processed-dir", required=True, type=str)
    parser.add_argument("--target-name", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--data-format", required=True, type=str)
    parser.add_argument("--prediction-horizon", required=True, type=int)
    parser.add_argument("--n-samples", required=True, type=int)
    parser.add_argument("--stage-name", required=True, type=str)
    parser.add_argument("--test-mode", required=True, type=str)
    parser.add_argument("--log-level", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    test_mode = args.test_mode.lower() in ("true", "1", "yes")
    run_name = f"eval_{args.model}_{args.target_name}"

    processed_dir = Path(args.processed_dir)
    results_dir = Path("results") / args.target_name / args.model

    params = dvc.api.params_show(stages=args.stage_name)
    prefix = f"params/models/{args.target_name}/{args.model}.yaml:"
    model_kwargs = params[f"{prefix}model_kwargs"]

    x_test, y_test = load_data(processed_dir, "test", args.data_format)
    covariate_dim = x_test.shape[1]

    pk = model_kwargs["parameters_fn_kwargs"]
    pk["conditional_event_shape"] = covariate_dim
    model_kwargs["parameters_fn_kwargs"] = pk

    n_eval = min(len(x_test), 100 if test_mode else 150)
    x_test, y_test = x_test[:n_eval], y_test[:n_eval]
    n_samples = 50 if test_mode else args.n_samples
    batch_size = min(n_eval, 16)

    logger.info(
        "Loaded test data: X=%s y=%s (n_eval=%d)", x_test.shape, y_test.shape, n_eval
    )

    dims = args.prediction_horizon
    model = DensityRegressionModel(dims=dims, **model_kwargs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=lambda y, p_y: -p_y.log_prob(y),
    )
    model(x_test[:1], training=False)
    model.load_weights(str(results_dir / "model_weights.h5"))
    logger.info("Loaded weights from %s", results_dir)

    with start_run_with_exception_logging(run_name=run_name):
        log_cfg(vars(args))
        mlflow.log_params({"model_kwargs": str(model_kwargs)})
        mlflow.tensorflow.autolog()

        logger.info("Sampling %d draws from predictive distribution ...", n_samples)
        samples = sample_predictions(
            model, x_test, n_samples=n_samples, batch_size=batch_size
        )
        logger.info("Sampled shape: %s", samples.shape)

        out_dir = results_dir / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating forecast plot ...")
        fig = plot_forecast_with_intervals(
            y_test,
            samples,
            n_show=48,
            title=f"{args.model} \u2013 {args.target_name}",
        )
        log_and_save_figure(fig, str(out_dir), "forecast", "pdf", dpi=300)
        log_and_save_figure(fig, str(out_dir), "forecast", "png", dpi=150)
        plt.close(fig)

        logger.info("Generating PIT histogram ...")
        fig = plot_pit_histogram(
            y_test,
            samples,
            n_bins=20,
            title=f"{args.model} \u2013 {args.target_name}",
        )
        log_and_save_figure(fig, str(out_dir), "pit_histogram", "pdf", dpi=300)
        log_and_save_figure(fig, str(out_dir), "pit_histogram", "png", dpi=150)
        plt.close(fig)

        logger.info("Generating calibration plot ...")
        fig = plot_calibration(
            y_test,
            samples,
            title=f"{args.model} \u2013 {args.target_name}",
        )
        log_and_save_figure(fig, str(out_dir), "calibration", "pdf", dpi=300)
        log_and_save_figure(fig, str(out_dir), "calibration", "png", dpi=150)
        plt.close(fig)

        metrics = compute_metrics(y_test, samples)
        logger.info("Metrics: %s", metrics)

        mlflow.log_metrics(metrics)
        with open(out_dir / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        mlflow.log_artifacts(str(results_dir))
        logger.info("Evaluation saved to %s", out_dir)


if __name__ == "__main__":
    main()
