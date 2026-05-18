import logging
from pathlib import Path

import pandas as pd


def setup_logging(log_level: str, log_file: str) -> logging.Logger:
    level = getattr(logging, log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
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
    return ".feather" if data_format == "feather" else ".csv"


def save_dataframe(df: pd.DataFrame, path: Path, data_format: str) -> None:
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
    full_path = path.with_suffix(_ext(data_format))
    if data_format == "feather":
        return pd.read_feather(full_path)
    return pd.read_csv(full_path, index_col=index_col, parse_dates=parse_dates)
