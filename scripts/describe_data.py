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

from dcp_nf_forecast.utils import read_dataframe, setup_logging, setup_plotting_style

logger = logging.getLogger(__name__)


def compute_statistics(
    x: np.ndarray,
    y: np.ndarray,
    x_colnames: list[str] | None = None,
    y_colnames: list[str] | None = None,
) -> dict:
    """Compute descriptive statistics for feature and target arrays.

    Parameters
    ----------
    x : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target matrix.
    x_colnames : list[str] | None, optional
        Column names for features, by default ``None``.
    y_colnames : list[str] | None, optional
        Column names for targets, by default ``None``.

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
        name = x_colnames[i] if x_colnames else f"x_{i}"
        columns.append(
            {
                "name": name,
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
        name = y_colnames[i] if y_colnames else f"y_{i}"
        columns.append(
            {
                "name": name,
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


def _plot_column_grid(
    data: np.ndarray,
    plot_fn: callable,
    colnames: list[str] | None = None,
    title: str = "",
    n_cols: int = 6,
    figsize_scale: tuple[float, float] = (2, 1.8),
    sharex: bool = False,
) -> plt.Figure:
    """Plot a grid of per-column plots.

    Parameters
    ----------
    data : np.ndarray
        2D array (samples × columns).
    plot_fn : callable
        Function ``plot_fn(ax, values)`` to draw on each subplot.
    colnames : list[str] | None, optional
        Column names, by default ``None``.
    title : str, optional
        Plot title suffix, by default ``""``.
    n_cols : int, optional
        Number of columns per row, by default ``6``.
    figsize_scale : tuple[float, float], optional
        (width, height) per subplot cell, by default ``(2, 1.8)``.
    sharex : bool, optional
        Whether subplots share the x-axis, by default ``False``.

    Returns
    -------
    plt.Figure
        The grid figure.

    """
    n_plot = data.shape[1]
    plot_cols = min(n_cols, n_plot)
    plot_rows = max(1, int(np.ceil(n_plot / plot_cols)))
    fig, axes = plt.subplots(
        plot_rows,
        plot_cols,
        figsize=(plot_cols * figsize_scale[0], plot_rows * figsize_scale[1]),
        sharex=sharex,
    )
    axes = axes.flatten() if n_plot > 1 else [axes]

    for i in range(n_plot):
        plot_fn(axes[i], data[:, i])
        name = colnames[i] if colnames else f"Col {i}"
        axes[i].set_title(name, fontsize=7)
        axes[i].tick_params(labelsize=6)

    for j in range(n_plot, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _plot_histogram(ax: plt.Axes, values: np.ndarray) -> None:
    """Draw a histogram on *ax*."""
    ax.hist(values, bins=50, color="steelblue", alpha=0.7, edgecolor="white")


def _plot_timeseries(ax: plt.Axes, values: np.ndarray) -> None:
    """Draw a line plot on *ax*."""
    ax.plot(values, linewidth=0.3, color="steelblue")


def plot_histograms(
    data: np.ndarray,
    colnames: list[str] | None = None,
    title: str = "",
) -> plt.Figure:
    """Plot histograms for each column.

    Parameters
    ----------
    data : np.ndarray
        2D array (samples × columns).
    colnames : list[str] | None, optional
        Column names, by default ``None``.
    title : str, optional
        Plot title suffix, by default ``""``.

    Returns
    -------
    plt.Figure
        The histogram grid figure.

    """
    return _plot_column_grid(
        data, _plot_histogram, colnames, title=f"Histograms – {title}"
    )


def plot_timeseries(
    data: np.ndarray,
    colnames: list[str] | None = None,
    title: str = "",
) -> plt.Figure:
    """Plot time series line plots for each column.

    Parameters
    ----------
    data : np.ndarray
        2D array (samples × columns).
    colnames : list[str] | None, optional
        Column names, by default ``None``.
    title : str, optional
        Plot title suffix, by default ``""``.

    Returns
    -------
    plt.Figure
        The time series grid figure.

    """
    fig = _plot_column_grid(
        data,
        _plot_timeseries,
        colnames,
        title=f"Time Series – {title}",
        sharex=True,
    )
    fig.supxlabel("Row index", fontsize=8)
    return fig


def plot_correlation_matrix(
    data: np.ndarray,
    colnames: list[str] | None = None,
    title: str = "",
) -> plt.Figure:
    """Plot a correlation matrix heatmap.

    Parameters
    ----------
    data : np.ndarray
        2D array (samples × columns).
    colnames : list[str] | None, optional
        Column names for tick labels, by default ``None``.
    title : str, optional
        Plot title suffix, by default ``""``.

    Returns
    -------
    plt.Figure
        The correlation matrix figure.

    """
    corr = np.corrcoef(data.T)
    n = data.shape[1]
    figsize = (max(6, n * 0.12), max(5, n * 0.12))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title(f"Correlation Matrix – {title}", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    if colnames:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(colnames, fontsize=4, rotation=90)
        ax.set_yticklabels(colnames, fontsize=4)
    else:
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
    setup_plotting_style()
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
            x_colnames = list(x.columns)
            y_colnames = list(y.columns)
            logger.info("  X=%s y=%s", x_arr.shape, y_arr.shape)

            stats = compute_statistics(x_arr, y_arr, x_colnames, y_colnames)
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

            for prefix, arr, cnames in (
                ("features", x_arr, x_colnames),
                ("targets", y_arr, y_colnames),
            ):
                tag = f"{prefix}_{split_name}"

                logger.info("  Generating %s histograms ...", prefix)
                fig = plot_histograms(
                    arr,
                    colnames=cnames,
                    title=f"{args.target_name} – {tag}",
                )
                log_and_save_figure(
                    fig,
                    str(out_dir),
                    f"histograms_{tag}",
                    "pdf",
                    dpi=300,
                )
                log_and_save_figure(
                    fig,
                    str(out_dir),
                    f"histograms_{tag}",
                    "png",
                    dpi=150,
                )
                plt.close(fig)

                logger.info("  Generating %s time series plot ...", prefix)
                fig = plot_timeseries(
                    arr,
                    colnames=cnames,
                    title=f"{args.target_name} – {tag}",
                )
                log_and_save_figure(
                    fig,
                    str(out_dir),
                    f"timeseries_{tag}",
                    "pdf",
                    dpi=300,
                )
                log_and_save_figure(
                    fig,
                    str(out_dir),
                    f"timeseries_{tag}",
                    "png",
                    dpi=150,
                )
                plt.close(fig)

                logger.info("  Generating %s correlation matrix ...", prefix)
                fig = plot_correlation_matrix(
                    arr,
                    colnames=cnames,
                    title=f"{args.target_name} – {tag}",
                )
                log_and_save_figure(
                    fig,
                    str(out_dir),
                    f"correlation_{tag}",
                    "pdf",
                    dpi=300,
                )
                log_and_save_figure(
                    fig,
                    str(out_dir),
                    f"correlation_{tag}",
                    "png",
                    dpi=150,
                )
                plt.close(fig)

        mlflow.log_artifacts(str(out_dir))

        logger.info("Data description saved to %s", out_dir)


if __name__ == "__main__":
    main()
