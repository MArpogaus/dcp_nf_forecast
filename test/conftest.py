import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "target": np.random.randn(n),
            "cov_a": np.random.randn(n) * 2,
            "cov_b": np.arange(n, dtype=float),
        },
        index=idx,
    )


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "raw.csv"
    df = sample_df.reset_index().rename(columns={"index": "time"})
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_csv_with_nans(tmp_path):
    path = tmp_path / "raw_nan.csv"
    df = pd.DataFrame(
        {
            "time": pd.date_range("2023-01-01", periods=10, freq="15min"),
            "target": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0],
            "cov_a": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        }
    )
    df.to_csv(path, index=False)
    return str(path)
