import numpy as np
import pandas as pd


def load_raw_data(
    raw_data_path: str,
    datetime_column: str,
    fillna_method: str,
) -> pd.DataFrame:
    """Load raw CSV data into a DataFrame with a datetime index.

    Parameters
    ----------
    raw_data_path : str
        Path to the CSV file.
    datetime_column : str
        Name of the column containing datetime strings.
    fillna_method : str
        Method for filling NaN values (e.g. ``"ffill"``).

    Returns
    -------
    pd.DataFrame
        Data indexed by the parsed datetime column, with NaNs filled.

    Raises
    ------
    ValueError
        If any NaN values remain after filling.
    """
    df = pd.read_csv(raw_data_path, parse_dates=[datetime_column])
    df = df.set_index(datetime_column)
    if fillna_method == "ffill":
        df = df.ffill()
    df = df.bfill()
    nan_count = int(df.isnull().sum().sum())
    if nan_count > 0:
        raise ValueError(f"Found {nan_count} NaNs after filling - data has gaps")
    return df


def cyclical_encode(
    values: np.ndarray,
    n_freqs: int,
) -> np.ndarray:
    """Generate sin/cos cyclical features at multiple frequencies.

    Parameters
    ----------
    values : np.ndarray
        1-D array of normalized values in ``[0, 1]``.
    n_freqs : int
        Number of frequency multiples to encode (each produces sin + cos).

    Returns
    -------
    np.ndarray
        Shape ``(len(values), 2 * n_freqs)`` with sin/cos pairs per frequency.
    """
    if n_freqs <= 0:
        return np.empty((len(values), 0))
    features: list[np.ndarray] = []
    for freq in range(1, n_freqs + 1):
        angular = 2 * np.pi * freq * values
        features.append(np.sin(angular))
        features.append(np.cos(angular))
    return np.column_stack(features)


def _normalize_time_component(
    df: pd.DataFrame,
    component: str,
) -> np.ndarray:
    """Normalize a datetime component to ``[0, 1]``.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a ``DatetimeIndex``.
    component : str
        One of ``"time"``, ``"day_of_week"``, or ``"day_of_year"``.

    Returns
    -------
    np.ndarray
        Float array in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *component* is not recognised.
    """
    if component == "time":
        values = (df.index.hour * 60 + df.index.minute) / (24 * 60)  # type: ignore[attr-defined]
    elif component == "day_of_week":
        values = df.index.dayofweek / 7  # type: ignore[attr-defined]
    elif component == "day_of_year":
        values = (df.index.dayofyear - 1) / 365  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unknown time feature component: {component}")
    return np.asarray(values, dtype=float)


def encode_time_features(
    df: pd.DataFrame,
    components_n_freqs: dict[str, int],
) -> pd.DataFrame:
    """Create sin/cos encoded time features from a datetime index.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a ``DatetimeIndex``.
    components_n_freqs : dict[str, int]
        Maps component names (``"time"``, ``"day_of_week"``,
        ``"day_of_year"``) to their number of frequency multiples.

    Returns
    -------
    pd.DataFrame
        Columns named ``{component}_f{freq}_{sin,cos}``, same index as *df*.
    """
    if not components_n_freqs:
        return pd.DataFrame(index=df.index)
    columns: dict[str, pd.Series] = {}
    for component, n_freqs in components_n_freqs.items():
        normalized = _normalize_time_component(df, component)
        encoded = cyclical_encode(normalized, n_freqs)
        for freq in range(1, n_freqs + 1):
            columns[f"{component}_f{freq}_sin"] = pd.Series(
                encoded[:, (freq - 1) * 2], index=df.index
            )
            columns[f"{component}_f{freq}_cos"] = pd.Series(
                encoded[:, (freq - 1) * 2 + 1], index=df.index
            )
    return pd.DataFrame(columns, index=df.index)


def create_lag_features(
    df: pd.DataFrame,
    column_lags: dict[str, int],
) -> pd.DataFrame:
    """Create lagged copies of specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    column_lags : dict[str, int]
        Maps column names to the number of lag steps to create.

    Returns
    -------
    pd.DataFrame
        Columns named ``{column}_lag_{step}``, same index as *df*.
        Leading rows will be ``NaN`` for each lag step.
    """
    if not column_lags:
        return pd.DataFrame(index=df.index)
    frames: list[pd.DataFrame] = []
    for col, num_lags in column_lags.items():
        for i in range(1, num_lags + 1):
            lagged = df[[col]].shift(i).rename(columns={col: f"{col}_lag_{i}"})
            frames.append(lagged)
    return pd.concat(frames, axis=1)


def build_target(
    df: pd.DataFrame,
    target_column: str,
    prediction_horizon: int,
) -> pd.DataFrame:
    """Build a multi-step target matrix by shifting a column forward.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    target_column : str
        Name of the column to forecast.
    prediction_horizon : int
        Number of future steps to include.

    Returns
    -------
    pd.DataFrame
        Columns named ``target_1`` … ``target_{horizon}``.  Trailing rows
        will be ``NaN`` for each step.
    """
    frames: list[pd.DataFrame] = []
    for step in range(1, prediction_horizon + 1):
        shifted = (
            df[[target_column]]
            .shift(-step)
            .rename(columns={target_column: f"target_{step}"})
        )
        frames.append(shifted)
    return pd.concat(frames, axis=1)


def build_features_and_target(
    df: pd.DataFrame,
    tabular_covariate_columns: list[str],
    components_n_freqs: dict[str, int],
    column_lags: dict[str, int],
    target_column: str,
    prediction_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the feature matrix X and target matrix Y.

    X is formed by concatenating time features, lag features, and tabular
    covariates.  Y contains the future values of the target column.  Rows
    that contain ``NaN`` from lag or shift operations are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Source data with a ``DatetimeIndex``.
    tabular_covariate_columns : list[str]
        Columns to include as-is (current-timestep values).
    components_n_freqs : dict[str, int]
        Time-component / frequency mapping for sin/cos encoding.
    column_lags : dict[str, int]
        Column / lag-count mapping for lag features.
    target_column : str
        Column to predict.
    prediction_horizon : int
        Number of future steps to predict.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(X, Y)`` with no ``NaN`` values.  Rows are a contiguous subset
        of the original index after dropping leading and trailing ``NaN``
        rows from lag and shift operations.
    """
    parts: list[pd.DataFrame] = [encode_time_features(df, components_n_freqs)]
    lag_df = create_lag_features(df, column_lags)
    if not lag_df.empty:
        parts.append(lag_df)
    if tabular_covariate_columns:
        parts.append(df[tabular_covariate_columns])
    df_x = pd.concat(parts, axis=1)
    df_y = build_target(df, target_column, prediction_horizon)
    mask = df_x.notna().all(axis=1) & df_y.notna().all(axis=1)
    return df_x[mask], df_y[mask]


def validate_data(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
) -> None:
    """Check that both X and Y contain no ``NaN`` values.

    Parameters
    ----------
    df_x : pd.DataFrame
        Feature matrix.
    df_y : pd.DataFrame
        Target matrix.

    Raises
    ------
    ValueError
        If any ``NaN`` is found in *df_x* or *df_y*.
    """
    if df_x.isnull().sum().sum() > 0 or df_y.isnull().sum().sum() > 0:
        raise ValueError("Found NaNs in data")


def split_temporal(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Split X and Y along the time axis without shuffling.

    Parameters
    ----------
    df_x : pd.DataFrame
        Feature matrix (must be time-ordered).
    df_y : pd.DataFrame
        Target matrix.
    train_ratio : float
        Fraction of rows for training.
    val_ratio : float
        Fraction of rows for validation.
    test_ratio : float
        Fraction of rows for testing.

    Returns
    -------
    dict[str, tuple[pd.DataFrame, pd.DataFrame]]
        Keys ``"train"``, ``"val"``, ``"test"`` mapping to ``(X, Y)`` pairs.
    """
    n = len(df_x)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": (df_x.iloc[:train_end], df_y.iloc[:train_end]),
        "val": (df_x.iloc[train_end:val_end], df_y.iloc[train_end:val_end]),
        "test": (df_x.iloc[val_end:], df_y.iloc[val_end:]),
    }
