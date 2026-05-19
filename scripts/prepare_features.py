"""Build feature matrix X and target matrix Y from raw CSV data."""

import argparse
from pathlib import Path

from dcp_nf_forecast.data import (
    build_features_and_target,
    load_raw_data,
    validate_data,
)
from dcp_nf_forecast.utils import save_dataframe, setup_logging


def _parse_csv(s: str) -> list[str]:
    """Split a comma-separated string into a list of trimmed tokens.

    Parameters
    ----------
    s : str
        Comma-separated values, e.g. ``"a, b, c"``.

    Returns
    -------
    list[str]
        Non-empty trimmed tokens.
    """
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_kv_csv(s: str) -> dict[str, int]:
    """Parse a comma-separated string of ``key:value`` pairs.

    Parameters
    ----------
    s : str
        E.g. ``"time:4,day_of_week:2"``.

    Returns
    -------
    dict[str, int]
        Parsed key / integer-value mapping.
    """
    result: dict[str, int] = {}
    for item in _parse_csv(s):
        k, v = item.split(":")
        result[k] = int(v)
    return result


def main() -> None:
    """Entry point: parse CLI args, build features, save to disk."""
    parser = argparse.ArgumentParser(
        description="Prepare feature matrix X and target matrix Y"
    )
    parser.add_argument("--raw-data-path", required=True, type=str)
    parser.add_argument("--datetime-column", required=True, type=str)
    parser.add_argument("--fillna-method", required=True, type=str)
    parser.add_argument("--tabular-covariate-columns", required=True, type=str)
    parser.add_argument("--time-components", required=True, type=str)
    parser.add_argument("--target-name", required=True, type=str)
    parser.add_argument("--y-column", required=True, type=str)
    parser.add_argument("--prediction-horizon", required=True, type=int)
    parser.add_argument("--lag-columns", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--data-format", required=True, type=str)
    parser.add_argument("--log-level", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
    args = parser.parse_args()

    logger = setup_logging(args.log_level, args.log_file)

    df = load_raw_data(
        raw_data_path=args.raw_data_path,
        datetime_column=args.datetime_column,
        fillna_method=args.fillna_method,
    )
    logger.info("Loaded %d rows with columns %s", len(df), list(df.columns))

    tab_cov = _parse_csv(args.tabular_covariate_columns)
    time_spec = _parse_kv_csv(args.time_components)
    lag_cols = _parse_kv_csv(args.lag_columns)

    logger.info(
        "Target '%s': time=%s lags=%s covariates=%s",
        args.target_name,
        time_spec,
        lag_cols,
        tab_cov,
    )
    df_x, df_y = build_features_and_target(
        df=df,
        tabular_covariate_columns=tab_cov,
        components_n_freqs=time_spec,
        column_lags=lag_cols,
        target_column=args.y_column,
        prediction_horizon=args.prediction_horizon,
    )
    logger.info("X shape: %s, Y shape: %s", df_x.shape, df_y.shape)

    validate_data(df_x, df_y)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_dataframe(df_x, out / "X", args.data_format)
    save_dataframe(df_y, out / "y", args.data_format)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
