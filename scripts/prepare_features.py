"""Build feature matrix X and target matrix Y from raw CSV data."""

import argparse
from pathlib import Path

from dcp_nf_forecast.data import (
    build_features_and_target,
    fillna_with_noise,
    load_raw_data,
)
from dcp_nf_forecast.utils import save_dataframe, setup_logging


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip() and x.strip() != "none"]


def _parse_tabular_covariates(s: str) -> list[tuple[str, int]]:
    """Parse tabular covariate specs, optionally with an offset.

    Each item can be ``col_name`` (offset 0) or ``col_name:N`` (offset N).
    """
    result: list[tuple[str, int]] = []
    for item in _parse_csv(s):
        parts = item.split(":")
        if len(parts) == 1:
            result.append((parts[0], 0))
        else:
            result.append((parts[0], int(parts[1])))
    return result


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
    parser.add_argument("--holiday-country", required=True, type=str)
    parser.add_argument("--end-date", required=True, type=str)
    parser.add_argument("--fillna-columns", required=True, type=str)
    parser.add_argument("--fillna-value", required=True, type=float)
    parser.add_argument("--fillna-noise-type", required=True, type=str)
    parser.add_argument("--fillna-noise-scale", required=True, type=float)
    parser.add_argument("--future-columns", required=True, type=str)
    args = parser.parse_args()

    logger = setup_logging(args.log_level, args.log_file)

    end_date = (
        args.end_date if args.end_date and args.end_date.lower() != "none" else None
    )
    holiday_country = (
        args.holiday_country
        if args.holiday_country and args.holiday_country.lower() != "none"
        else None
    )

    df = load_raw_data(
        raw_data_path=args.raw_data_path,
        datetime_column=args.datetime_column,
        end_date=end_date,
    )
    logger.info("Loaded %d rows with columns %s", len(df), list(df.columns))

    fillna_cols = _parse_csv(args.fillna_columns)
    if fillna_cols:
        df = fillna_with_noise(
            df,
            columns=fillna_cols,
            fill_value=args.fillna_value,
            noise_type=args.fillna_noise_type,
            noise_scale=args.fillna_noise_scale
            if args.fillna_noise_scale > 0
            else None,
            seed=42,
        )
        logger.info(
            "Filled NaNs in %s with value=%s noise_type=%s noise_scale=%s",
            fillna_cols,
            args.fillna_value,
            args.fillna_noise_type,
            args.fillna_noise_scale,
        )

    tab_cov = _parse_tabular_covariates(args.tabular_covariate_columns)
    time_comps = _parse_csv(args.time_components)
    lag_cols = _parse_kv_csv(args.lag_columns)
    lead_cols = _parse_kv_csv(args.future_columns)

    logger.info(
        "Target '%s': time=%s lags=%s leads=%s covariates=%s",
        args.target_name,
        time_comps,
        lag_cols,
        lead_cols,
        tab_cov,
    )
    df_x, df_y = build_features_and_target(
        df=df,
        tabular_covariate_columns=tab_cov,
        time_components=time_comps,
        column_lags=lag_cols,
        column_leads=lead_cols,
        target_column=args.y_column,
        prediction_horizon=args.prediction_horizon,
        holiday_country=holiday_country,
    )
    logger.info("X shape: %s, Y shape: %s", df_x.shape, df_y.shape)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_dataframe(df_x, out / "X", args.data_format)
    save_dataframe(df_y, out / "y", args.data_format)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
