"""Split feature/target data into temporally ordered train/val/test sets."""

import argparse
from pathlib import Path

from dcp_nf_forecast.data import split_temporal
from dcp_nf_forecast.utils import read_dataframe, save_dataframe, setup_logging


def main() -> None:
    """Entry point: parse CLI args, split, save splits to disk."""
    parser = argparse.ArgumentParser(
        description="Split feature/target data into train/val/test sets"
    )
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--train-ratio", required=True, type=float)
    parser.add_argument("--val-ratio", required=True, type=float)
    parser.add_argument("--test-ratio", required=True, type=float)
    parser.add_argument("--data-format", required=True, type=str)
    parser.add_argument("--log-level", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
    args = parser.parse_args()

    logger = setup_logging(args.log_level, args.log_file)

    in_dir = Path(args.input_dir)
    df_x = read_dataframe(in_dir / "X", args.data_format)
    df_y = read_dataframe(in_dir / "y", args.data_format)
    logger.info("Loaded X=%s Y=%s", df_x.shape, df_y.shape)

    datasets = split_temporal(
        df_x=df_x,
        df_y=df_y,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, (x, y) in datasets.items():
        logger.info("Split %s: X=%s Y=%s", name, x.shape, y.shape)
        save_dataframe(x, out / f"X_{name}", args.data_format)
        save_dataframe(y, out / f"y_{name}", args.data_format)

    logger.info("Splits saved to %s", out)


if __name__ == "__main__":
    main()
