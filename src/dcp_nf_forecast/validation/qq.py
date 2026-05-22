# %% Description ###############################################################
"""Quantile--Quantile (QQ) plots for probabilistic forecast evaluation.

Compares quantiles of the predictive distribution to the observed
quantiles.  If the forecasts are well-calibrated the points lie
near the diagonal.
"""

# %% imports ###################################################################
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)
DEFAULT_QUANTILES = 21


# %% public functions ##########################################################
def plot_qq(
    observations: np.ndarray,
    samples: np.ndarray,
    n_quantiles: int = DEFAULT_QUANTILES,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot a QQ plot comparing predictive and empirical quantiles.

    For each quantile level ``q`` in ``{0, 1/(M-1), ..., 1}``, the
    predictive quantile is estimated from *samples* and the empirical
    quantile is estimated from *observations*.

    Parameters
    ----------
    observations : np.ndarray
        True observed values, shape ``(n,)``.
    samples : np.ndarray
        Samples from the predictive distribution, shape
        ``(n_samples, n)``.
    n_quantiles : int, optional
        Number of quantile levels to evaluate, by default ``21``
        (i.e. the 0th, 5th, 10th, ..., 100th percentiles).
    ax : plt.Axes | None, optional
        Matplotlib axes to plot on, by default ``None`` (creates a new
        figure).

    Returns
    -------
    Figure
        The figure object.

    Raises
    ------
    ValueError
        If *observations* and *samples* have incompatible shapes.

    """
    if samples.ndim != 2:
        raise ValueError(
            f"samples must be 2-D (n_samples, n), got shape {samples.shape}"
        )
    if len(observations) != samples.shape[1]:
        raise ValueError(
            f"observations length ({len(observations)}) must match "
            f"samples second dimension ({samples.shape[1]})"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    quantile_levels = np.linspace(0.0, 1.0, n_quantiles)
    obs_quantiles = np.quantile(observations, quantile_levels)
    pred_quantiles = np.quantile(samples, quantile_levels, axis=0).mean(axis=1)

    ax.plot(
        pred_quantiles,
        obs_quantiles,
        "o",
        color="#1f77b4",
        markersize=4,
    )
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "--", color="#d62728", linewidth=0.8, label="Perfect")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Predictive quantile")
    ax.set_ylabel("Observed quantile")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
