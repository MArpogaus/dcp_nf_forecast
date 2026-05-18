import pandas as pd
import pytest

from dcp_nf_forecast.utils import (
    _ext,
    read_dataframe,
    save_dataframe,
    setup_logging,
)


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging("info", "/dev/null")
        assert logger.name == "dcp_nf_forecast.utils"


class TestExt:
    @pytest.mark.parametrize(
        "fmt,expected",
        [("feather", ".feather"), ("csv", ".csv"), ("unknown", ".csv")],
    )
    def test_extension(self, fmt, expected):
        assert _ext(fmt) == expected


class TestSaveAndReadDataframe:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    def test_save_and_read_feather(self, tmp_path, df):
        path = tmp_path / "test"
        save_dataframe(df, path, "feather")
        assert (tmp_path / "test.feather").exists()
        result = read_dataframe(path, "feather", index_col=None, parse_dates=False)
        pd.testing.assert_frame_equal(result, df)

    def test_save_and_read_csv(self, tmp_path, df):
        path = tmp_path / "test"
        save_dataframe(df, path, "csv")
        assert (tmp_path / "test.csv").exists()
        result = read_dataframe(
            tmp_path / "test", "csv", index_col=None, parse_dates=False
        )
        pd.testing.assert_frame_equal(result, df)

    def test_feather_resets_index(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.RangeIndex(10, 13))
        save_dataframe(df, tmp_path / "test", "feather")
        result = read_dataframe(tmp_path / "test", "feather")
        assert list(result["a"]) == [1, 2, 3]

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_dataframe(
                tmp_path / "noexist", "csv", index_col=None, parse_dates=False
            )
