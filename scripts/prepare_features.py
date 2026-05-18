import argparse
from pathlib import Path

import yaml

from dcp_nf_forecast.data import (
    build_features_and_target,
    load_raw_data,
    validate_data,
)
from dcp_nf_forecast.utils import save_dataframe, setup_logging


def _parse_kv_pairs(raw: list[str]) -> dict[str, int]:
    result = {}
    for item in raw:
        key, _, val = item.partition(":")
        result[key.strip()] = int(val.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare feature matrix X and target matrix Y"
    )
    parser.add_argument("--raw-data-path", required=True, type=str)
    parser.add_argument("--datetime-column", required=True, type=str)
    parser.add_argument("--fillna-method", required=True, type=str)
    parser.add_argument(
        "--tabular-covariate-columns", required=True, nargs="*", type=str
    )
    parser.add_argument("--time-component", required=True, action="append", type=str)
    parser.add_argument("--target-config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--data-format", required=True, type=str)
    parser.add_argument("--log-level", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
    args = parser.parse_args()

    logger = setup_logging(args.log_level, args.log_file)

    with open(args.target_config) as f:
        target_cfg = yaml.safe_load(f)

    logger.info(
        "Loading raw data from %s",
        args.raw_data_path,
    )
    df = load_raw_data(
        raw_data_path=args.raw_data_path,
        datetime_column=args.datetime_column,
        fillna_method=args.fillna_method,
    )
    logger.info("Loaded %d rows with columns %s", len(df), list(df.columns))

    time_spec = _parse_kv_pairs(args.time_component)
    lag_cols = target_cfg.get("lag_columns", {})
    if not isinstance(lag_cols, dict):
        lag_cols = {}

    logger.info(
        "Building features: time=%s lags=%s covariates=%s",
        time_spec,
        lag_cols,
        args.tabular_covariate_columns,
    )
    df_x, df_y = build_features_and_target(
        df=df,
        tabular_covariate_columns=args.tabular_covariate_columns or [],
        components_n_freqs=time_spec,
        column_lags=lag_cols,
        target_column=target_cfg["y_column"],
        prediction_horizon=target_cfg["prediction_horizon"],
    )
    logger.info("X shape: %s, Y shape: %s", df_x.shape, df_y.shape)

    validate_data(df_x, df_y)
    logger.info("Validation passed")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_dataframe(df_x, out / "X", args.data_format)
    save_dataframe(df_y, out / "y", args.data_format)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
