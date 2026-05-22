"""I/O helpers, logging setup, and plotting configuration utilities."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(
    processed_dir: Path,
    split: str,
    data_format: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-processed split data as numpy arrays.

    Parameters
    ----------
    processed_dir : Path
        Directory containing ``X_{split}`` and ``y_{split}`` files.
    split : str
        Split name (``"train"``, ``"val"``, ``"test"``).
    data_format : str
        ``"feather"`` or ``"csv"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y)`` as ``float32`` arrays.

    """
    x = read_dataframe(processed_dir / f"X_{split}", data_format)
    y = read_dataframe(processed_dir / f"y_{split}", data_format)
    return x.values.astype(np.float32), y.values.astype(np.float32)


def setup_logging(log_level: str, log_file: str) -> logging.Logger:
    """Configure logging with stream and optional file handlers.

    Parameters
    ----------
    log_level : str
        Log level name (e.g. ``"info"``, ``"debug"``).
    log_file : str
        Path to the log file.  Parent directories are created if needed.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def _ext(data_format: str) -> str:
    """Return the file extension for a given data format.

    Parameters
    ----------
    data_format : str
        One of ``"feather"`` or ``"csv"``.

    Returns
    -------
    str
        ``".feather"`` or ``".csv"`` (default).

    """
    return ".feather" if data_format == "feather" else ".csv"


def save_dataframe(df: pd.DataFrame, path: Path, data_format: str) -> None:
    """Save a DataFrame to disk in the requested format.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    path : Path
        File path (without extension – the extension is added based on
        *data_format*).
    data_format : str
        ``"feather"`` (faster) or ``"csv"``.

    """
    full_path = path.with_suffix(_ext(data_format))
    if data_format == "feather":
        df.reset_index(drop=True).to_feather(full_path)
    else:
        df.to_csv(full_path, index=False)


def read_dataframe(
    path: Path,
    data_format: str,
    index_col: int | None = None,
    parse_dates: bool = False,
) -> pd.DataFrame:
    """Read a DataFrame from disk.

    Parameters
    ----------
    path : Path
        File path (without extension).
    data_format : str
        ``"feather"`` or ``"csv"``.
    index_col : int | None, optional
        Column index to use as row index (CSV only).
    parse_dates : bool, optional
        Whether to parse date columns (CSV only).

    Returns
    -------
    pd.DataFrame
        Loaded data.

    """
    full_path = path.with_suffix(_ext(data_format))
    if data_format == "feather":
        return pd.read_feather(full_path)  # type: ignore[no-any-return]
    return pd.read_csv(full_path, index_col=index_col, parse_dates=parse_dates)
