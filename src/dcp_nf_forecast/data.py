import numpy as np
import pandas as pd


def load_raw_data(
    raw_data_path: str,
    datetime_column: str,
    fillna_method: str,
) -> pd.DataFrame:
    df = pd.read_csv(raw_data_path, parse_dates=[datetime_column])
    df = df.set_index(datetime_column)
    if fillna_method == "ffill":
        df = df.ffill()
    df = df.bfill()
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Found {nan_count} NaNs after filling - data has gaps")
    return df


def cyclical_encode(
    values: np.ndarray,
    n_freqs: int,
) -> np.ndarray:
    if n_freqs <= 0:
        return np.empty((len(values), 0))
    features = []
    for freq in range(1, n_freqs + 1):
        angular = 2 * np.pi * freq * values
        features.append(np.sin(angular))
        features.append(np.cos(angular))
    return np.column_stack(features)


def _normalize_time_component(
    df: pd.DataFrame,
    component: str,
) -> np.ndarray:
    if component == "time":
        values = (df.index.hour * 60 + df.index.minute) / (24 * 60)
    elif component == "day_of_week":
        values = df.index.dayofweek / 7
    elif component == "day_of_year":
        values = (df.index.dayofyear - 1) / 365
    else:
        raise ValueError(f"Unknown time feature component: {component}")
    return np.asarray(values, dtype=float)


def encode_time_features(
    df: pd.DataFrame,
    components_n_freqs: dict[str, int],
) -> pd.DataFrame:
    if not components_n_freqs:
        return pd.DataFrame(index=df.index)
    columns = {}
    for component, n_freqs in components_n_freqs.items():
        normalized = _normalize_time_component(df, component)
        encoded = cyclical_encode(normalized, n_freqs)
        for freq in range(1, n_freqs + 1):
            columns[f"{component}_f{freq}_sin"] = encoded[:, (freq - 1) * 2]
            columns[f"{component}_f{freq}_cos"] = encoded[:, (freq - 1) * 2 + 1]
    return pd.DataFrame(columns, index=df.index)


def create_lag_features(
    df: pd.DataFrame,
    column_lags: dict[str, int],
) -> pd.DataFrame:
    if not column_lags:
        return pd.DataFrame(index=df.index)
    frames = []
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
    frames = []
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
    parts = [encode_time_features(df, components_n_freqs)]
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
    if df_x.isnull().sum().sum() > 0 or df_y.isnull().sum().sum() > 0:
        raise ValueError("Found NaNs in data")


def split_temporal(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    n = len(df_x)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": (df_x.iloc[:train_end], df_y.iloc[:train_end]),
        "val": (df_x.iloc[train_end:val_end], df_y.iloc[train_end:val_end]),
        "test": (df_x.iloc[val_end:], df_y.iloc[val_end:]),
    }
