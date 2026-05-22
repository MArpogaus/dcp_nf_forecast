"""Generate descriptive statistics and plots from train/val data splits."""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml
from hybrid_flows.utils.mlflow import (
    log_and_save_figure,
    log_cfg,
    start_run_with_exception_logging,
)

from dcp_nf_forecast.utils import read_dataframe, setup_logging

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


def compute_statistics(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Compute descriptive statistics for feature and target arrays.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target matrix.

    Returns
    -------
    dict
        Dictionary containing ``n_samples``, ``n_features``, ``n_targets``,
        and a ``columns`` list with per-column statistics.

    """
    n_cols_x = x.shape[1]
    n_cols_y = y.shape[1]

    columns: list[dict] = []
    for i, col in enumerate(x.T):
        columns.append(
            {
                "name": f"x_{i}",
                "role": "feature",
                "mean": round(float(np.mean(col)), 6),
                "std": round(float(np.std(col)), 6),
                "min": round(float(np.min(col)), 6),
                "q25": round(float(np.percentile(col, 25)), 6),
                "median": round(float(np.median(col)), 6),
                "q75": round(float(np.percentile(col, 75)), 6),
                "max": round(float(np.max(col)), 6),
            }
        )
    for i, col in enumerate(y.T):
        columns.append(
            {
                "name": f"y_{i}",
                "role": "target",
                "mean": round(float(np.mean(col)), 6),
                "std": round(float(np.std(col)), 6),
                "min": round(float(np.min(col)), 6),
                "q25": round(float(np.percentile(col, 25)), 6),
                "median": round(float(np.median(col)), 6),
                "q75": round(float(np.percentile(col, 75)), 6),
                "max": round(float(np.max(col)), 6),
            }
        )

    return {
        "n_samples": int(x.shape[0]),
        "n_features": n_cols_x,
        "n_targets": n_cols_y,
        "columns": columns,
    }


def plot_histograms(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    max_cols: int = 20,
) -> plt.Figure:
    """Plot histograms for each column in the combined feature-target array.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target matrix.
    title : str, optional
        Plot title suffix, by default ``""``.
    max_cols : int, optional
        Maximum number of columns to plot, by default ``20``.

    Returns
    -------
    plt.Figure
        The histogram grid figure.

    """
    combined = np.concatenate([x, y], axis=1)
    n_cols_plot = min(combined.shape[1], max_cols)
    plot_cols = min(4, n_cols_plot)
    plot_rows = int(np.ceil(n_cols_plot / plot_cols))
    fig, axes = plt.subplots(
        plot_rows, plot_cols, figsize=(plot_cols * 3, plot_rows * 2.5)
    )
    axes = axes.flatten() if n_cols_plot > 1 else [axes]

    for i in range(n_cols_plot):
        axes[i].hist(
            combined[:, i], bins=50, color="steelblue", alpha=0.7, edgecolor="white"
        )
        axes[i].set_title(f"Col {i}", fontsize=8)
        axes[i].tick_params(labelsize=7)

    for j in range(n_cols_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Histograms – {title}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_timeseries(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
    max_cols: int = 10,
) -> plt.Figure:
    """Plot time series line plots for each column.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target matrix.
    title : str, optional
        Plot title suffix, by default ``""``.
    max_cols : int, optional
        Maximum number of columns to plot, by default ``10``.

    Returns
    -------
    plt.Figure
        The time series grid figure.

    """
    combined = np.concatenate([x, y], axis=1)
    n_plot = combined.shape[1] if max_cols <= 0 else min(combined.shape[1], max_cols)
    plot_cols = min(3, n_plot)
    plot_rows = int(np.ceil(n_plot / plot_cols))
    fig, axes = plt.subplots(
        plot_rows, plot_cols, figsize=(plot_cols * 4, plot_rows * 2.5), sharex=True
    )
    axes = axes.flatten() if n_plot > 1 else [axes]

    for i in range(n_plot):
        axes[i].plot(combined[:, i], linewidth=0.5, color="steelblue")
        axes[i].set_title(f"Col {i}", fontsize=8)
        axes[i].tick_params(labelsize=7)

    for j in range(n_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Time Series – {title}", fontsize=12)
    fig.supxlabel("Row index")
    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """Plot a correlation matrix heatmap for the combined feature-target array.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target matrix.
    title : str, optional
        Plot title suffix, by default ``""``.

    Returns
    -------
    plt.Figure
        The correlation matrix figure.

    """
    combined = np.concatenate([x, y], axis=1)
    corr = np.corrcoef(combined.T)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title(f"Correlation Matrix – {title}", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def main() -> None:
    """Entry point: parse CLI args, generate statistics and plots, log to MLflow."""
    parser = argparse.ArgumentParser(
        description="Generate descriptive stats and plots from train/val splits"
    )
    parser.add_argument("--processed-dir", required=True, type=str)
    parser.add_argument("--target-name", required=True, type=str)
    parser.add_argument("--data-format", required=True, type=str)
    parser.add_argument("--log-level", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    run_name = f"describe_{args.target_name}"

    processed_dir = Path(args.processed_dir)
    out_dir = Path("results") / args.target_name / "data_description"
    out_dir.mkdir(parents=True, exist_ok=True)

    with start_run_with_exception_logging(run_name=run_name):
        log_cfg(vars(args))

        for split_name in ("train", "val"):
            logger.info("Processing %s split ...", split_name)
            x = read_dataframe(processed_dir / f"X_{split_name}", args.data_format)
            y = read_dataframe(processed_dir / f"y_{split_name}", args.data_format)
            x_arr = np.array(x)
            y_arr = np.array(y)
            logger.info("  X=%s y=%s", x_arr.shape, y_arr.shape)

            stats = compute_statistics(x_arr, y_arr)
            stats["split"] = split_name
            stats_path = out_dir / f"statistics_{split_name}.yaml"
            with open(stats_path, "w") as f:
                yaml.dump(stats, f, default_flow_style=False)
            mlflow.log_artifact(str(stats_path))
            mlflow.log_metrics(
                {
                    f"{split_name}_n_samples": stats["n_samples"],
                    f"{split_name}_n_features": stats["n_features"],
                    f"{split_name}_n_targets": stats["n_targets"],
                }
            )
            logger.info("  Statistics saved to %s", stats_path)

            logger.info("  Generating histograms ...")
            fig = plot_histograms(
                x_arr,
                y_arr,
                title=f"{args.target_name} – {split_name}",
            )
            log_and_save_figure(
                fig, str(out_dir), f"histograms_{split_name}", "pdf", dpi=300
            )
            log_and_save_figure(
                fig, str(out_dir), f"histograms_{split_name}", "png", dpi=150
            )
            plt.close(fig)

            logger.info("  Generating time series plot ...")
            fig = plot_timeseries(
                x_arr,
                y_arr,
                title=f"{args.target_name} – {split_name}",
            )
            log_and_save_figure(
                fig,
                str(out_dir),
                f"timeseries_{split_name}",
                "pdf",
                dpi=300,
            )
            log_and_save_figure(
                fig,
                str(out_dir),
                f"timeseries_{split_name}",
                "png",
                dpi=150,
            )
            plt.close(fig)

            logger.info("  Generating correlation matrix ...")
            fig = plot_correlation_matrix(
                x_arr,
                y_arr,
                title=f"{args.target_name} – {split_name}",
            )
            log_and_save_figure(
                fig,
                str(out_dir),
                f"correlation_{split_name}",
                "pdf",
                dpi=300,
            )
            log_and_save_figure(
                fig,
                str(out_dir),
                f"correlation_{split_name}",
                "png",
                dpi=150,
            )
            plt.close(fig)

        mlflow.log_artifacts(str(out_dir))

        logger.info("Data description saved to %s", out_dir)


if __name__ == "__main__":
    main()
