"""Evaluate a trained normalizing flow forecasting model on test data."""

import matplotlib

matplotlib.use("Agg")

import argparse
import logging
from pathlib import Path

import dvc.api
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
import yaml
from hybrid_flows.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)
from matplotlib.figure import Figure
from probabilistic_forecast_validation import (
    plot_pit_histogram,
    plot_qq,
)

from dcp_nf_forecast.models import build_model
from dcp_nf_forecast.utils import load_data, setup_logging

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 0.8,
        "figure.figsize": (8, 4.5),
        "font.family": "sans-serif",
    }
)


def sample_predictions(
    model: tf.keras.Model,
    x: np.ndarray,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    """Sample from the predictive distribution in batches.

    Parameters
    ----------
    model : tf.keras.Model
        Trained forecasting model.
    x : np.ndarray
        Input covariates, shape ``(n, cov_dim)``.
    n_samples : int
        Number of samples per input.
    batch_size : int
        Batch size for sampling.

    Returns
    -------
    np.ndarray
        Samples, shape ``(n_samples, n, prediction_horizon)``.
    """
    n = len(x)
    all_samples: list[np.ndarray] = []
    pad = 0
    remainder = n % batch_size
    if remainder:
        pad = batch_size - remainder
        x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size]
        dist = model(batch, training=False)
        batch_samples = dist.sample(n_samples)
        all_samples.append(np.array(batch_samples, dtype=np.float32))
    result = np.concatenate(all_samples, axis=1)
    if pad:
        result = result[:, :n, :]
    del all_samples
    return result


def compute_nll(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> float:
    """Compute mean negative log-likelihood on test data.

    Parameters
    ----------
    model : tf.keras.Model
        Trained forecasting model.
    x : np.ndarray
        Input covariates, shape ``(n, cov_dim)``.
    y : np.ndarray
        True target values, shape ``(n, prediction_horizon)``.
    batch_size : int
        Batch size for evaluation.

    Returns
    -------
    float
        Mean negative log-likelihood across all test samples and
        forecast steps.
    """
    n = len(x)
    nlls: list[np.ndarray] = []
    for i in range(0, n, batch_size):
        batch_x = x[i : i + batch_size]
        batch_y = y[i : i + batch_size]
        dist = model(batch_x, training=False)
        batch_nll = -dist.log_prob(batch_y)
        nlls.append(batch_nll.numpy())
    return float(np.mean(np.concatenate(nlls)))


def plot_forecast_with_intervals(
    y_true: np.ndarray,
    samples: np.ndarray,
    n_show: int = 48,
    title: str = "",
    alpha: float = 0.25,
) -> Figure:
    """Plot forecast time series with prediction intervals.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape ``(n, prediction_horizon)``.
    samples : np.ndarray
        Samples from predictive distribution, shape
        ``(n_samples, n, prediction_horizon)``.
    n_show : int, optional
        Number of test samples to show, by default ``48``.
    title : str, optional
        Plot title, by default ``""``.
    alpha : float, optional
        Base fill alpha, by default ``0.25``.

    Returns
    -------
    Figure
        The figure object.
    """
    n_steps = y_true.shape[1]
    n_cols = min(4, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))
    width, height = n_cols * 2.5, n_rows * 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height), sharex=True, sharey=True
    )
    axes = axes.flatten() if n_steps > 1 else [axes]
    n_show = min(n_show, len(y_true))
    quantiles = [5, 10, 25, 50, 75, 90, 95]
    q = np.percentile(samples[:, :n_show], quantiles, axis=0)

    for step in range(n_steps):
        ax = axes[step]
        t = np.arange(n_show)
        y = y_true[:n_show, step]
        ax.plot(t, y, color="#333333", label=r"Actual", linewidth=0.6)
        ax.plot(t, q[3, :, step], color="#1f77b4", label=r"Median", linewidth=0.7)
        ax.fill_between(
            t,
            q[0, :, step],
            q[-1, :, step],
            alpha=alpha,
            color="#1f77b4",
            label=r"90\% CI",
        )
        ax.fill_between(
            t,
            q[1, :, step],
            q[-2, :, step],
            alpha=alpha * 1.6,
            color="#2c8ad4",
            label=r"80\% CI",
        )
        ax.fill_between(
            t,
            q[2, :, step],
            q[-3, :, step],
            alpha=alpha * 2.2,
            color="#3a9ee6",
            label=r"50\% CI",
        )
        if step == 0:
            ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.set_title(rf"$t + {step + 1}$", fontsize=9)

    for j in range(n_steps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.supxlabel(r"Test sample index", fontsize=9)
    fig.supylabel(r"Value", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pit_histogram_grid(
    y_true: np.ndarray,
    samples: np.ndarray,
    n_bins: int = 20,
    title: str = "",
) -> Figure:
    """Plot PIT histograms for each forecast step.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape ``(n, prediction_horizon)``.
    samples : np.ndarray
        Samples from predictive distribution, shape
        ``(n_samples, n, prediction_horizon)``.
    n_bins : int, optional
        Number of histogram bins, by default ``20``.
    title : str, optional
        Plot title, by default ``""``.

    Returns
    -------
    Figure
        The figure object.
    """
    n_steps = y_true.shape[1]
    n_cols = min(4, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))
    width, height = n_cols * 2.8, n_rows * 2.2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height), sharex=True, sharey=True
    )
    axes = axes.flatten() if n_steps > 1 else [axes]

    for step in range(n_steps):
        ax = axes[step]
        plot_pit_histogram(
            observations=y_true[:, step],
            samples=samples[:, :, step],
            n_bins=n_bins,
            ax=ax,
        )
        ax.set_title(rf"$t + {step + 1}$", fontsize=9)

    for j in range(n_steps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(rf"PIT Histogram -- {title}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_qq_grid(
    y_true: np.ndarray,
    samples: np.ndarray,
    n_quantiles: int = 21,
    title: str = "",
) -> Figure:
    """Plot QQ plots for each forecast step.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape ``(n, prediction_horizon)``.
    samples : np.ndarray
        Samples from predictive distribution, shape
        ``(n_samples, n, prediction_horizon)``.
    n_quantiles : int, optional
        Number of quantile levels, by default ``21``.
    title : str, optional
        Plot title, by default ``""``.

    Returns
    -------
    Figure
        The figure object.
    """
    n_steps = y_true.shape[1]
    n_cols = min(4, n_steps)
    n_rows = int(np.ceil(n_steps / n_cols))
    width, height = n_cols * 2.8, n_rows * 2.2
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(width, height), sharex=True, sharey=True
    )
    axes = axes.flatten() if n_steps > 1 else [axes]

    for step in range(n_steps):
        ax = axes[step]
        plot_qq(
            observations=y_true[:, step],
            samples=samples[:, :, step],
            n_quantiles=n_quantiles,
            ax=ax,
        )
        ax.set_title(rf"$t + {step + 1}$", fontsize=9)

    for j in range(n_steps, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(rf"QQ Plot -- {title}", fontsize=11)
    fig.tight_layout()
    return fig


def compute_metrics(
    y_true: np.ndarray,
    samples: np.ndarray,
) -> dict:
    """Compute point forecast metrics from samples.

    Parameters
    ----------
    y_true : np.ndarray
        True observed values, shape ``(n, prediction_horizon)``.
    samples : np.ndarray
        Samples from predictive distribution, shape
        ``(n_samples, n, prediction_horizon)``.

    Returns
    -------
    dict
        Dictionary with ``rmse``, ``mae``, and ``mean_90_ci_width``.
    """
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
    """Run evaluation: load model, compute NLL, sample, plot, log metrics."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained normalizing flow model on test data"
    )
    parser.add_argument("--target-name", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--stage-name", required=True, type=str)
    parser.add_argument("--prediction-horizon", required=True, type=int)
    args = parser.parse_args()
    params = dvc.api.params_show(stages=args.stage_name)

    setup_logging(params["log_level"], params["log_file"])
    test_mode = params["test_mode"]
    run_name = f"eval_{args.model}_{args.target_name}"
    experiment_name = "-".join(
        [params["experiment_name"], args.target_name] + (["test"] if test_mode else [])
    )
    data_format = params["data"]["data_format"]
    n_samples = 50 if test_mode else params["eval"]["n_samples"]

    processed_dir = Path(params["paths"]["data_processed"]) / args.target_name
    results_dir = Path("results") / args.target_name / args.model

    x_test, y_test = load_data(processed_dir, "test", data_format)
    covariate_dim = x_test.shape[1]

    n_eval = min(len(x_test), 10 if test_mode else 150)
    x_test, y_test = x_test[:n_eval], y_test[:n_eval]
    batch_size = min(n_eval, 16)

    logger.info("n_samples=%d test_mode=%s", n_samples, test_mode)

    logger.info(
        "Loaded test data: X=%s y=%s (n_eval=%d)",
        x_test.shape,
        y_test.shape,
        n_eval,
    )

    dims = args.prediction_horizon
    model = build_model(
        dims=dims, covariate_dim=covariate_dim, model_kwargs=params["model_kwargs"]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=lambda y, p_y: -p_y.log_prob(y),
    )
    model(x_test[:1], training=False)
    model.load_weights(str(results_dir / "model_weights.h5"))
    logger.info("Loaded weights from %s", results_dir)

    mlflow.set_experiment(experiment_name)

    parent_run_id_file = results_dir / "parent_run_id.txt"

    with mlflow.start_run(run_id=parent_run_id_file.read_text().strip()):
        with start_run_with_exception_logging(run_name=f"{run_name}_evaluation"):
            mlflow.set_tag("stage", "evaluation")
            log_cfg(params)

            mlflow.tensorflow.autolog()

            logger.info("Computing NLL on test data ...")
            nll = compute_nll(model, x_test, y_test, batch_size)
            logger.info("NLL: %.4f", nll)

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
                title=f"{args.model} -- {args.target_name}",
            )
            log_and_save_figure(fig, str(out_dir), "forecast", "pdf", dpi=300)
            log_and_save_figure(fig, str(out_dir), "forecast", "png", dpi=150)
            plt.close(fig)

            logger.info("Generating PIT histogram ...")
            fig = plot_pit_histogram_grid(
                y_test,
                samples,
                n_bins=20,
                title=f"{args.model} -- {args.target_name}",
            )
            log_and_save_figure(fig, str(out_dir), "pit_histogram", "pdf", dpi=300)
            log_and_save_figure(fig, str(out_dir), "pit_histogram", "png", dpi=150)
            plt.close(fig)

            logger.info("Generating QQ plot ...")
            fig = plot_qq_grid(
                y_test,
                samples,
                n_quantiles=21,
                title=f"{args.model} -- {args.target_name}",
            )
            log_and_save_figure(fig, str(out_dir), "qq_plot", "pdf", dpi=300)
            log_and_save_figure(fig, str(out_dir), "qq_plot", "png", dpi=150)
            plt.close(fig)

            metrics = compute_metrics(y_test, samples)
            metrics["nll"] = nll
            logger.info("Metrics: %s", metrics)

            mlflow.log_metrics(metrics)
            with open(out_dir / "metrics.yaml", "w") as f:
                yaml.dump(metrics, f)

            mlflow.log_artifacts(str(results_dir))
            logger.info("Evaluation saved to %s", out_dir)


if __name__ == "__main__":
    main()
