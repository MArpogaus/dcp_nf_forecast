"""Evaluate a trained normalizing flow forecasting model on test data."""

import argparse
import logging
from pathlib import Path

import dvc.api
import matplotlib

matplotlib.use("Agg")  # noqa: I001
import matplotlib.pyplot as plt  # noqa: I001, E402
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import yaml

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from hybrid_flows.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)
from matplotlib.figure import Figure

from dcp_nf_forecast.models import build_model
from dcp_nf_forecast.utils import read_dataframe, setup_logging, setup_plotting_style
logger = logging.getLogger(__name__)


def sample_predictions(
    model: tf.keras.Model,
    x: np.ndarray,
    n_samples: int,
    batch_size: int,
) -> tf.Tensor:
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
    tf.Tensor
        Samples, shape ``(n_samples, n, prediction_horizon)``,
        still on GPU.

    """
    n = len(x)
    all_samples: list[tf.Tensor] = []
    pad = 0
    remainder = n % batch_size
    if remainder:
        pad = batch_size - remainder
        x = np.pad(x, ((0, pad), (0, 0)), mode="edge")

    @tf.function(reduce_retracing=True)
    def sample_batch(batch: tf.Tensor, k: tf.Tensor) -> tf.Tensor:
        dist = model(batch, training=False)
        return tf.cast(dist.sample(k), tf.float32)

    for i in range(0, len(x), batch_size):
        batch = tf.constant(x[i : i + batch_size])
        samples_i = sample_batch(batch, tf.constant(n_samples))
        all_samples.append(samples_i)

    result = tf.concat(all_samples, axis=1)
    if pad:
        result = result[:, :n, :]
    return result


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


def save_sample_df(
    samples: np.ndarray,
    times: pd.Index,
    out_dir: Path,
    data_format: str,
) -> Path:
    """Save samples as a 2D DataFrame with (time, sample) MultiIndex.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(n_samples, n, prediction_horizon)``.
    times : pd.Index
        Datetime index for each test sample, length ``n``.
    out_dir : Path
        Output directory.
    data_format : str
        ``"feather"`` or ``"csv"``.

    Returns
    -------
    Path
        Path to the saved file.

    """
    n_samples, n_eval, prediction_horizon = samples.shape
    sample_numbers = np.tile(np.arange(n_samples), n_eval)
    sample_times = np.repeat(times, n_samples)
    idx = pd.MultiIndex.from_arrays(
        [sample_times, sample_numbers],
        names=["forecast_origin", "sample_number"],
    )
    flat = samples.transpose(1, 0, 2).reshape(n_eval * n_samples, prediction_horizon)
    cols = [f"step_{s + 1}" for s in range(prediction_horizon)]
    df = pd.DataFrame(flat, index=idx, columns=cols)
    fp = out_dir / "samples"
    df.reset_index().to_feather(fp.with_suffix(".feather"))
    return fp.with_suffix(".feather")


def main() -> None:
    """Run evaluation: load model, compute NLL, sample, plot, log metrics."""
    setup_plotting_style()
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

    x_test_df = read_dataframe(processed_dir / "X_test", data_format)
    y_test_df = read_dataframe(processed_dir / "y_test", data_format)
    x_test = x_test_df.values.astype(np.float32)
    y_test = y_test_df.values.astype(np.float32)
    covariate_dim = x_test.shape[1]

    n_eval = 10 if test_mode else len(x_test)
    x_test, y_test = x_test[:n_eval], y_test[:n_eval]
    y_test_times = y_test_df.index[:n_eval]
    batch_size = min(n_eval, 8)

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
            nll = float(
                model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
            )
            logger.info("NLL: %.4f", nll)

            logger.info("Sampling %d draws from predictive distribution ...", n_samples)
            samples_tf = sample_predictions(
                model, x_test, n_samples=n_samples, batch_size=batch_size
            )
            logger.info("Sampled shape: %s", samples_tf.shape)

            logger.info("Computing metrics on GPU ...")
            y_true_tf = tf.constant(y_test, dtype=tf.float32)
            median = tfp.stats.percentile(samples_tf, 50.0, axis=0)
            rmse = tf.sqrt(tf.reduce_mean((y_true_tf - median) ** 2))
            mae = tf.reduce_mean(tf.abs(y_true_tf - median))
            q95 = tfp.stats.percentile(samples_tf, 95.0, axis=0)
            q05 = tfp.stats.percentile(samples_tf, 5.0, axis=0)
            mean_ci90 = tf.reduce_mean(q95 - q05)
            metrics = {
                "rmse": float(rmse.numpy()),
                "mae": float(mae.numpy()),
                "mean_90_ci_width": float(mean_ci90.numpy()),
                "nll": nll,
            }
            logger.info("Metrics: %s", metrics)

            samples = samples_tf.numpy()
            del samples_tf

            out_dir = results_dir / "evaluation"
            out_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Saving samples to feather ...")
            save_sample_df(samples, y_test_times, out_dir, data_format)

            logger.info("Generating forecast plot ...")
            fig = plot_forecast_with_intervals(
                y_test, samples, n_show=48,
                title=f"{args.model} -- {args.target_name}",
            )
            log_and_save_figure(fig, str(out_dir), "forecast", "pdf", dpi=300)
            log_and_save_figure(fig, str(out_dir), "forecast", "png", dpi=150)
            plt.close(fig)

            mlflow.log_metrics(metrics)
            with open(out_dir / "metrics.yaml", "w") as f:
                yaml.dump(metrics, f)

            mlflow.log_artifacts(str(results_dir))
            logger.info("Evaluation saved to %s", out_dir)


if __name__ == "__main__":
    main()
