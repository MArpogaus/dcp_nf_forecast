# %% Description ###############################################################
"""Probability Integral Transform (PIT) histograms.

The PIT is the value of the predictive CDF evaluated at the true
observation.  If the predictive distribution is well-calibrated,
the PIT values follow a uniform distribution on ``[0, 1]``.
"""

# %% imports ###################################################################
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

# %% globals ###################################################################
__LOGGER__ = logging.getLogger(__name__)
DEFAULT_N_BINS = 20


# %% public functions ##########################################################
def plot_pit_histogram(
    observations: np.ndarray,
    samples: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot a PIT histogram to assess calibration.

    The PIT for each observation is computed as the fraction of
    posterior samples that are less than or equal to the observation.
    For a well-calibrated model the resulting values are uniformly
    distributed and the histogram should be flat.

    Parameters
    ----------
    observations : np.ndarray
        True observed values, shape ``(n,)``.
    samples : np.ndarray
        Samples from the predictive distribution, shape
        ``(n_samples, n)``.
    n_bins : int, optional
        Number of histogram bins, by default ``20``.
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
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    pit = np.mean(samples <= observations, axis=0)

    ax.hist(
        pit,
        bins=n_bins,
        density=True,
        alpha=0.75,
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axhline(
        1.0,
        color="#d62728",
        linestyle="--",
        linewidth=0.8,
        label="Uniform",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
