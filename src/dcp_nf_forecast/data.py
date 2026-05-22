"""Data loading, feature engineering, and preprocessing utilities."""

import logging

import holidays as _holidays
import numpy as np
import pandas as pd

__LOGGER__ = logging.getLogger(__name__)


def _parse_date(s: str) -> pd.Timestamp | None:
    if not s or s.strip().lower() in ("none", ""):
        return None
    return pd.Timestamp(s)


def load_raw_data(
    raw_data_path: str,
    datetime_column: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load raw CSV data into a DataFrame with a datetime index.

    Parameters
    ----------
    raw_data_path : str
        Path to the CSV file.
    datetime_column : str
        Name of the column containing datetime strings.
    end_date : str | None, optional
        Optional end date filter (inclusive).  Rows after this date are
        dropped.  Pass ``None`` or ``"none"`` to keep all data.

    Returns
    -------
    pd.DataFrame
        Data indexed by the parsed datetime column.

    """
    df = pd.read_csv(raw_data_path, parse_dates=[datetime_column])
    df = df.set_index(datetime_column)
    end = _parse_date(end_date)
    if end is not None:
        df = df[df.index <= end]  # type: ignore[index]

    # Defensive validations for time-series shifting
    if not df.index.is_unique:
        __LOGGER__.warning(
            "Datetime index contains duplicate timestamps! "
            "This can cause major alignment issues."
        )
    if not df.index.is_monotonic_increasing:
        __LOGGER__.warning(
            "Datetime index is not monotonically increasing! "
            "Sorting to prevent incorrect shifting."
        )
        df = df.sort_index()

    return df


def cyclical_encode(
    values: np.ndarray,
) -> np.ndarray:
    """Generate sin/cos cyclical features at base frequency.

    Parameters
    ----------
    values : np.ndarray
        1-D array of normalized values in ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Shape ``(len(values), 2)`` with sin and cos of the base frequency.

    """
    angular = 2 * np.pi * values
    return np.column_stack([np.sin(angular), np.cos(angular)])


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
    components: list[str],
) -> pd.DataFrame:
    """Create sin/cos encoded time features from a datetime index.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a ``DatetimeIndex``.
    components : list[str]
        Component names (``"time"``, ``"day_of_week"``, ``"day_of_year"``).

    Returns
    -------
    pd.DataFrame
        Columns named ``{component}_sin`` and ``{component}_cos``,
        same index as *df*.

    """
    if not components:
        return pd.DataFrame(index=df.index)
    columns: dict[str, pd.Series] = {}
    for component in components:
        normalized = _normalize_time_component(df, component)
        sin_val, cos_val = cyclical_encode(normalized).T
        columns[f"{component}_sin"] = pd.Series(sin_val, index=df.index)
        columns[f"{component}_cos"] = pd.Series(cos_val, index=df.index)
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


def _sample_noise(
    rng: np.random.Generator,
    noise_type: str,
    noise_scale: float,
    size: int,
) -> np.ndarray:
    """Draw noise samples from *noise_type* distribution using *getattr*.

    Maps the generic *noise_scale* parameter to the appropriate keyword
    for each distribution family.
    """
    dist_fn = getattr(rng, noise_type)
    if noise_type == "lognormal":
        return dist_fn(mean=np.log(noise_scale), sigma=0.5, size=size)
    if noise_type == "uniform":
        return dist_fn(0, noise_scale, size=size)
    return dist_fn(scale=noise_scale, size=size)


def fillna_with_noise(
    df: pd.DataFrame,
    columns: list[str],
    fill_value: float = 0.0,
    noise_type: str | None = None,
    noise_scale: float | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Fill NaN values in specified columns with a constant plus optional noise.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    columns : list[str]
        Column names to fill NaNs in.
    fill_value : float, optional
        Constant value to replace NaNs with.
    noise_type : str | None, optional
        Name of a ``numpy.random.Generator`` method (e.g. ``"lognormal"``,
        ``"uniform"``, ``"exponential"``, ``"normal"``).
    noise_scale : float | None, optional
        Scale parameter passed to the distribution.  Semantics depend on
        *noise_type* (e.g. ``sigma`` for lognormal, ``high`` for uniform,
        ``scale`` for exponential/normal).
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaNs filled in the specified columns.

    """
    if not columns:
        return df
    df = df.copy()
    rng = np.random.default_rng(seed)
    for col in columns:
        if col not in df.columns:
            __LOGGER__.warning("Column '%s' not found, skipping fillna", col)
            continue
        mask = df[col].isna()
        n = int(mask.sum())
        if n == 0:
            continue
        if noise_scale is not None and noise_scale > 0:
            noise = _sample_noise(rng, noise_type, noise_scale, n)
            df.loc[mask, col] = fill_value + noise
        else:
            df.loc[mask, col] = fill_value
        __LOGGER__.info(
            "Filled %d NaNs in '%s' with %s%s",
            n,
            col,
            fill_value,
            f" + {noise_type}(scale={noise_scale})" if noise_scale else "",
        )
    return df


def add_holiday_indicator(
    df: pd.DataFrame,
    country: str,
) -> pd.DataFrame:
    """Add a binary holiday indicator column for the given country.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame indexed by datetime.
    country : str
        ISO country code (e.g. ``"DE"`` or ``"DE-BW"``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* containing an additional ``is_holiday`` column.

    """
    years = set(df.index.year)  # type: ignore[attr-defined]
    parts = country.upper().split("-")
    country_code = parts[0]
    state_code = parts[1] if len(parts) > 1 else None
    cls = getattr(_holidays, country_code)
    kwargs: dict = {"years": years}
    if state_code:
        kwargs["state"] = state_code
    cal = cls(**kwargs)
    is_holiday = pd.Series(df.index.isin(cal), index=df.index, dtype=int)  # type: ignore[arg-type]
    df_out = df.copy()
    df_out["is_holiday"] = is_holiday
    return df_out


def _parse_tabular_with_offset(
    specs: list[tuple[str, int]],
    df: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for col, offset in specs:
        series = df[col]
        if offset > 0:
            series = series.shift(-offset)
            col_name = f"{col}_lead_{offset}"
        elif offset < 0:
            series = series.shift(-offset)
            col_name = f"{col}_lag_{abs(offset)}"
        else:
            col_name = col
        frames.append(series.to_frame(col_name))
    return pd.concat(frames, axis=1)


def _create_lead_features(
    df: pd.DataFrame,
    column_leads: dict[str, int],
) -> pd.DataFrame:
    """Create leading (future) copies of specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    column_leads : dict[str, int]
        Maps column names to the number of future steps to include.

    Returns
    -------
    pd.DataFrame
        Columns named ``{column}_lead_{step}``, same index as *df*.
        Trailing rows will be ``NaN`` for each step.

    """
    if not column_leads:
        return pd.DataFrame(index=df.index)
    frames: list[pd.DataFrame] = []
    for col, num in column_leads.items():
        for i in range(1, num + 1):
            led = df[[col]].shift(-i).rename(columns={col: f"{col}_lead_{i}"})
            frames.append(led)
    return pd.concat(frames, axis=1)


def build_features_and_target(
    df: pd.DataFrame,
    tabular_covariate_columns: list[tuple[str, int]],
    time_components: list[str],
    column_lags: dict[str, int],
    target_column: str,
    prediction_horizon: int,
    column_leads: dict[str, int] | None = None,
    holiday_country: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the feature matrix X and target matrix Y.

    X is formed by concatenating time features, lag features, lead
    (future) features, tabular covariates, and optionally a holiday
    indicator.  Y contains the future values of the target column.
    Rows that contain ``NaN`` from lag / lead / shift operations are
    dropped (counted and logged).

    Parameters
    ----------
    df : pd.DataFrame
        Source data with a ``DatetimeIndex``.
    tabular_covariate_columns : list[tuple[str, int]]
        ``(col_name, offset)`` pairs.  *offset* > 0 shifts forward
        (lead), *offset* < 0 shifts backward (lag), 0 uses current
        timestep.
    time_components : list[str]
        Time-component names for sin/cos encoding.
    column_lags : dict[str, int]
        Column / lag-count mapping for lag features (past values).
    column_leads : dict[str, int] | None, optional
        Column / lead-count mapping for future values.  Positive
        integer means include that many forward-looking steps.
    target_column : str
        Column to predict.
    prediction_horizon : int
        Number of future steps to predict.
    holiday_country : str | None, optional
        ISO country code for holiday indicator (e.g. ``"DE-BW"``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(X, Y)`` with no ``NaN`` values.  Rows are a contiguous subset
        of the original index after dropping leading and trailing ``NaN``
        rows from lag and shift operations.

    """
    if holiday_country is not None:
        df = add_holiday_indicator(df, holiday_country)
        __LOGGER__.info("Holiday indicator added for '%s'", holiday_country)
    parts: list[pd.DataFrame] = [encode_time_features(df, time_components)]
    if holiday_country is not None:
        parts.append(df[["is_holiday"]])
    lag_df = create_lag_features(df, column_lags)
    if not lag_df.empty:
        parts.append(lag_df)
    all_leads = dict(column_leads or {})
    if holiday_country is not None:
        all_leads.setdefault("is_holiday", prediction_horizon)
    lead_df = _create_lead_features(df, all_leads)
    if not lead_df.empty:
        parts.append(lead_df)
    if tabular_covariate_columns:
        parts.append(_parse_tabular_with_offset(tabular_covariate_columns, df))
    df_x = pd.concat(parts, axis=1)
    df_y = build_target(df, target_column, prediction_horizon)
    n_before = len(df_x)
    mask = df_x.notna().all(axis=1) & df_y.notna().all(axis=1)
    n_nan = int((~mask).sum())
    if n_nan > 0:
        __LOGGER__.info(
            "Dropped %d / %d rows with NaNs (%.1f%%)",
            n_nan,
            n_before,
            100.0 * n_nan / n_before,
        )
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
