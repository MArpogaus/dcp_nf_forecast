import numpy as np
import pandas as pd
import pytest

from dcp_nf_forecast.data import (
    _create_lead_features,
    _normalize_time_component,
    _parse_tabular_with_offset,
    _sample_noise,
    add_holiday_indicator,
    build_features_and_target,
    build_target,
    create_lag_features,
    cyclical_encode,
    encode_time_features,
    fillna_with_noise,
    load_raw_data,
    split_temporal,
    validate_data,
)


class TestLoadRawData:
    def test_loads_csv_with_datetime_index(self, sample_csv, sample_df):
        result = load_raw_data(
            raw_data_path=sample_csv,
            datetime_column="time",
        )
        assert isinstance(result.index, pd.DatetimeIndex)
        assert list(result.columns) == list(sample_df.columns)
        assert len(result) == len(sample_df)

    def test_end_date_filter(self, sample_csv, sample_df):
        end = sample_df.index[50]
        result = load_raw_data(
            raw_data_path=sample_csv,
            datetime_column="time",
            end_date=str(end),
        )
        assert len(result) <= 51
        assert result.index[-1] <= end

    def test_none_end_date_keeps_all(self, sample_csv, sample_df):
        result = load_raw_data(
            raw_data_path=sample_csv,
            datetime_column="time",
            end_date=None,
        )
        assert len(result) == len(sample_df)

    def test_end_date_filter_inclusive(self, sample_csv):
        result = load_raw_data(
            raw_data_path=sample_csv,
            datetime_column="time",
            end_date="2023-01-01 00:00:00",
        )
        assert len(result) == 1


class TestFillnaWithNoise:
    def test_no_columns(self):
        df = pd.DataFrame({"a": [np.nan, 1.0]})
        result = fillna_with_noise(df, [])
        assert result["a"].isna().sum() == 1

    def test_fill_constant(self):
        df = pd.DataFrame({"a": [np.nan, 1.0, np.nan]})
        result = fillna_with_noise(df, ["a"], fill_value=0.0)
        assert result["a"].iloc[0] == 0.0
        assert result["a"].iloc[2] == 0.0

    def test_fill_with_noise_reproducible(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        r1 = fillna_with_noise(
            df,
            ["a"],
            fill_value=0.0,
            noise_type="uniform",
            noise_scale=0.1,
            seed=42,
        )
        r2 = fillna_with_noise(
            df,
            ["a"],
            fill_value=0.0,
            noise_type="uniform",
            noise_scale=0.1,
            seed=42,
        )
        assert r1["a"].iloc[0] == r2["a"].iloc[0]

    def test_no_nans_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = fillna_with_noise(df, ["a"], fill_value=0.0)
        assert list(result["a"]) == [1.0, 2.0, 3.0]

    def test_log_normal_noise_positive(self):
        df = pd.DataFrame({"a": [np.nan] * 1000})
        result = fillna_with_noise(
            df,
            ["a"],
            fill_value=0.0,
            noise_type="lognormal",
            noise_scale=0.01,
            seed=42,
        )
        assert (result["a"] >= 0).all()

    def test_unknown_column_skipped(self):
        df = pd.DataFrame({"a": [np.nan]})
        result = fillna_with_noise(df, ["b"], fill_value=1.0)
        assert np.isnan(result["a"].iloc[0])


class TestSampleNoise:
    def test_lognormal(self):
        rng = np.random.default_rng(42)
        samples = _sample_noise(rng, "lognormal", 0.01, 1000)
        assert (samples > 0).all()

    def test_uniform(self):
        rng = np.random.default_rng(42)
        samples = _sample_noise(rng, "uniform", 0.1, 1000)
        assert (samples >= 0).all() and (samples <= 0.1).all()

    def test_exponential(self):
        rng = np.random.default_rng(42)
        samples = _sample_noise(rng, "exponential", 1.0, 1000)
        assert (samples >= 0).all()


class TestAddHolidayIndicator:
    def test_returns_extra_column(self):
        df = pd.DataFrame(
            {"v": [1] * 5},
            index=pd.date_range("2023-12-24", periods=5, freq="D"),
        )
        result = add_holiday_indicator(df, "DE")
        assert "is_holiday" in result.columns

    def test_christmas_is_holiday(self):
        df = pd.DataFrame(
            {"v": [1]},
            index=pd.DatetimeIndex(["2023-12-25"]),
        )
        result = add_holiday_indicator(df, "DE")
        assert result["is_holiday"].iloc[0] == 1

    def test_weekday_not_holiday(self):
        df = pd.DataFrame(
            {"v": [1]},
            index=pd.DatetimeIndex(["2023-01-02"]),  # Monday
        )
        result = add_holiday_indicator(df, "DE")
        assert result["is_holiday"].iloc[0] == 0


class TestCreateLeadFeatures:
    def test_returns_empty_for_empty_dict(self, sample_df):
        result = _create_lead_features(sample_df, {})
        assert result.empty
        assert list(result.index) == list(sample_df.index)

    def test_single_column_leads(self, sample_df):
        result = _create_lead_features(sample_df, {"target": 3})
        assert list(result.columns) == [
            "target_lead_1",
            "target_lead_2",
            "target_lead_3",
        ]

    def test_last_rows_are_nan(self, sample_df):
        n = len(sample_df)
        result = _create_lead_features(sample_df, {"target": 3})
        assert np.isnan(result["target_lead_1"].iloc[n - 1])
        assert not np.isnan(result["target_lead_1"].iloc[n - 4])

    def test_lead_values_match_source(self, sample_df):
        result = _create_lead_features(sample_df, {"target": 2})
        assert result["target_lead_1"].iloc[5] == sample_df["target"].iloc[6]
        assert result["target_lead_2"].iloc[5] == sample_df["target"].iloc[7]


class TestParseTabularWithOffset:
    def test_zero_offset(self, sample_df):
        specs = [("cov_a", 0), ("cov_b", 0)]
        result = _parse_tabular_with_offset(specs, sample_df)
        assert list(result.columns) == ["cov_a", "cov_b"]

    def test_positive_offset(self, sample_df):
        specs = [("cov_a", 2)]
        result = _parse_tabular_with_offset(specs, sample_df)
        assert "cov_a_lead_2" in result.columns
        assert not np.isnan(result["cov_a_lead_2"].iloc[0])
        assert np.isnan(result["cov_a_lead_2"].iloc[-2])

    def test_negative_offset(self, sample_df):
        specs = [("cov_a", -2)]
        result = _parse_tabular_with_offset(specs, sample_df)
        assert "cov_a_lag_2" in result.columns
        assert np.isnan(result["cov_a_lag_2"].iloc[0])
        assert not np.isnan(result["cov_a_lag_2"].iloc[5])


class TestBuildFeaturesAndTarget:
    def test_includes_all_feature_types(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 0), ("cov_b", 0)],
            time_components=["time"],
            column_lags={"target": 4},
            column_leads=None,
            target_column="target",
            prediction_horizon=3,
        )
        assert "time_sin" in df_x.columns
        assert "time_cos" in df_x.columns
        assert "target_lag_1" in df_x.columns
        assert "cov_a" in df_x.columns
        assert "target_1" in df_y.columns

    def test_includes_lead_features(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[],
            time_components=[],
            column_lags={},
            column_leads={"cov_a": 2},
            target_column="target",
            prediction_horizon=1,
        )
        assert "cov_a_lead_1" in df_x.columns
        assert "cov_a_lead_2" in df_x.columns

    def test_no_nans_in_output(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 0), ("cov_b", 0)],
            time_components=["time"],
            column_lags={"target": 4},
            column_leads=None,
            target_column="target",
            prediction_horizon=3,
        )
        assert not df_x.isnull().any().any()
        assert not df_y.isnull().any().any()

    def test_sample_count(self, sample_df):
        n = len(sample_df)
        max_lag = 4
        horizon = 3
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 0)],
            time_components=[],
            column_lags={"target": max_lag},
            column_leads=None,
            target_column="target",
            prediction_horizon=horizon,
        )
        assert len(df_x) == n - max_lag - horizon
        assert len(df_y) == n - max_lag - horizon

    def test_no_covariates(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[],
            time_components=[],
            column_lags={"target": 2},
            column_leads=None,
            target_column="target",
            prediction_horizon=1,
        )
        assert "cov_a" not in df_x.columns
        assert df_x.shape[1] == 2  # target_lag_1, target_lag_2

    def test_no_lags(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 0)],
            time_components=[],
            column_lags={},
            column_leads=None,
            target_column="target",
            prediction_horizon=1,
        )
        assert "target_lag_1" not in df_x.columns
        assert df_x.shape[1] == 1  # just cov_a
        assert len(df_x) == len(sample_df) - 1

    def test_no_time_features(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 0)],
            time_components=[],
            column_lags={"target": 2},
            column_leads=None,
            target_column="target",
            prediction_horizon=1,
        )
        assert "time_sin" not in df_x.columns

    def test_x_and_y_aligned(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 0)],
            time_components=["time"],
            column_lags={"target": 2},
            column_leads=None,
            target_column="target",
            prediction_horizon=1,
        )
        assert list(df_x.index) == list(df_y.index)
        assert df_x.index[0] > sample_df.index[0]
        assert df_x.index[-1] < sample_df.index[-1]

    def test_holiday_indicator_included(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[],
            time_components=[],
            column_lags={},
            column_leads=None,
            target_column="target",
            prediction_horizon=1,
            holiday_country="DE",
        )
        assert "is_holiday" in df_x.columns

    def test_no_future_leakage(self):
        """Per row: all features (past) < all targets (future) for increasing data."""
        n = 200
        df = pd.DataFrame(
            {"target": np.arange(n, dtype=float)},
            index=pd.date_range("2023-01-01", periods=n, freq="15min"),
        )
        df_x, df_y = build_features_and_target(
            df=df,
            tabular_covariate_columns=[],
            time_components=[],
            column_lags={"target": 10},
            column_leads=None,
            target_column="target",
            prediction_horizon=3,
        )
        for i in range(len(df_x)):
            x_max = df_x.iloc[i].max()
            y_min = df_y.iloc[i].min()
            assert x_max < y_min, f"Row {i}: max feature {x_max} >= min target {y_min}"

    def test_holiday_auto_generates_leads(self, sample_df):
        """is_holiday leads are auto-generated for full prediction_horizon."""
        df_x, _ = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[],
            time_components=[],
            column_lags={},
            column_leads=None,
            target_column="target",
            prediction_horizon=5,
            holiday_country="DE",
        )
        assert "is_holiday" in df_x.columns
        for k in range(1, 6):
            assert f"is_holiday_lead_{k}" in df_x.columns

    def test_tabular_with_lead_offset(self, sample_df):
        df_x, df_y = build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 3)],
            time_components=[],
            column_lags={},
            column_leads=None,
            target_column="target",
            prediction_horizon=1,
        )
        assert "cov_a_lead_3" in df_x.columns


class TestCyclicalEncode:
    def test_sin_cos_in_bounds(self):
        values = np.linspace(0, 1, 100)
        encoded = cyclical_encode(values)
        assert np.all(np.abs(encoded) <= 1.0 + 1e-15)

    def test_output_shape(self):
        values = np.array([0.0, 0.5, 1.0])
        encoded = cyclical_encode(values)
        assert encoded.shape == (3, 2)

    def test_periodicity(self):
        values = np.array([0.25, 0.75])
        encoded = cyclical_encode(values)
        assert abs(encoded[0, 0]) > 0
        assert encoded[0, 1] > 0


class TestNormalizeTimeComponent:
    @pytest.fixture
    def df_midnight(self):
        return pd.DataFrame(
            {"v": [1]},
            index=pd.DatetimeIndex(["2023-01-01 00:00:00"]),
        )

    @pytest.fixture
    def df_noon(self):
        return pd.DataFrame(
            {"v": [1]},
            index=pd.DatetimeIndex(["2023-01-01 12:00:00"]),
        )

    def test_time_midnight(self, df_midnight):
        val = _normalize_time_component(df_midnight, "time")
        assert val[0] == 0.0

    def test_time_noon(self, df_noon):
        val = _normalize_time_component(df_noon, "time")
        assert abs(val[0] - 0.5) < 1e-6

    def test_day_of_week(self):
        df = pd.DataFrame(
            {"v": [1]},
            index=pd.DatetimeIndex(["2023-01-02"]),  # Monday
        )
        val = _normalize_time_component(df, "day_of_week")
        assert val[0] == 0.0

    def test_day_of_year(self):
        df = pd.DataFrame(
            {"v": [1]},
            index=pd.DatetimeIndex(["2023-01-01"]),
        )
        val = _normalize_time_component(df, "day_of_year")
        assert val[0] == 0.0

    def test_raises_on_unknown(self, df_midnight):
        with pytest.raises(ValueError, match="Unknown time feature component"):
            _normalize_time_component(df_midnight, "fortnite")


class TestEncodeTimeFeatures:
    def test_no_components(self, sample_df):
        df = sample_df
        result = encode_time_features(df, components=[])
        assert result.empty
        assert list(result.columns) == []

    def test_single_component(self, sample_df):
        df = sample_df
        result = encode_time_features(df, components=["time"])
        expected_cols = ["time_sin", "time_cos"]
        assert list(result.columns) == expected_cols
        assert np.all(np.abs(result.values) <= 1.0 + 1e-15)

    def test_multiple_components(self, sample_df):
        df = sample_df
        components = ["time", "day_of_week"]
        result = encode_time_features(df, components=components)
        expected_cols = [
            "time_sin",
            "time_cos",
            "day_of_week_sin",
            "day_of_week_cos",
        ]
        assert list(result.columns) == expected_cols

    def test_index_preserved(self, sample_df):
        result = encode_time_features(sample_df, components=["time"])
        assert result.index.equals(sample_df.index)


class TestCreateLagFeatures:
    def test_returns_empty_for_empty_dict(self, sample_df):
        result = create_lag_features(sample_df, {})
        assert result.empty
        assert list(result.index) == list(sample_df.index)

    def test_single_column_lags(self, sample_df):
        result = create_lag_features(sample_df, {"target": 3})
        assert list(result.columns) == ["target_lag_1", "target_lag_2", "target_lag_3"]

    def test_multiple_columns(self, sample_df):
        result = create_lag_features(sample_df, {"target": 2, "cov_a": 1})
        assert list(result.columns) == [
            "target_lag_1",
            "target_lag_2",
            "cov_a_lag_1",
        ]

    def test_first_rows_are_nan(self, sample_df):
        result = create_lag_features(sample_df, {"target": 3})
        assert result["target_lag_1"].iloc[0] != result["target_lag_1"].iloc[0]
        assert not np.isnan(result["target_lag_1"].iloc[3])

    def test_lag_values_match_source(self, sample_df):
        result = create_lag_features(sample_df, {"target": 2})
        assert result["target_lag_1"].iloc[5] == sample_df["target"].iloc[4]
        assert result["target_lag_2"].iloc[5] == sample_df["target"].iloc[3]


class TestBuildTarget:
    def test_single_step(self, sample_df):
        result = build_target(sample_df, target_column="target", prediction_horizon=1)
        assert list(result.columns) == ["target_1"]
        assert result["target_1"].iloc[0] == sample_df["target"].iloc[1]

    def test_multi_step(self, sample_df):
        result = build_target(sample_df, target_column="target", prediction_horizon=3)
        assert list(result.columns) == ["target_1", "target_2", "target_3"]
        assert result["target_1"].iloc[0] == sample_df["target"].iloc[1]
        assert result["target_2"].iloc[0] == sample_df["target"].iloc[2]
        assert result["target_3"].iloc[0] == sample_df["target"].iloc[3]

    def test_last_rows_are_nan(self, sample_df):
        n = len(sample_df)
        result = build_target(sample_df, target_column="target", prediction_horizon=3)
        assert np.isnan(result["target_1"].iloc[n - 1])
        assert np.isnan(result["target_3"].iloc[n - 3])
        assert not np.isnan(result["target_1"].iloc[n - 4])


class TestValidateData:
    def test_valid_data_passes(self):
        df_x = pd.DataFrame({"a": [1.0, 2.0]})
        df_y = pd.DataFrame({"b": [3.0, 4.0]})
        validate_data(df_x, df_y)

    def test_nan_in_x_raises(self):
        df_x = pd.DataFrame({"a": [1.0, np.nan]})
        df_y = pd.DataFrame({"b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="NaNs"):
            validate_data(df_x, df_y)

    def test_nan_in_y_raises(self):
        df_x = pd.DataFrame({"a": [1.0, 2.0]})
        df_y = pd.DataFrame({"b": [3.0, np.nan]})
        with pytest.raises(ValueError, match="NaNs"):
            validate_data(df_x, df_y)


class TestSplitTemporal:
    def _build(self, sample_df):
        return build_features_and_target(
            df=sample_df,
            tabular_covariate_columns=[("cov_a", 0)],
            time_components=[],
            column_lags={"target": 2},
            column_leads=None,
            target_column="target",
            prediction_horizon=1,
        )

    def test_correct_split_sizes(self, sample_df):
        df_x, df_y = self._build(sample_df)
        result = split_temporal(
            df_x=df_x,
            df_y=df_y,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        n = len(df_x)
        assert len(result["train"][0]) == int(n * 0.7)
        assert len(result["val"][0]) == int(n * 0.15)
        assert len(result["test"][0]) == n - int(n * 0.7) - int(n * 0.15)

    def test_no_overlap(self, sample_df):
        df_x, df_y = self._build(sample_df)
        result = split_temporal(
            df_x=df_x,
            df_y=df_y,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        train_idx = set(result["train"][0].index)
        val_idx = set(result["val"][0].index)
        test_idx = set(result["test"][0].index)
        assert len(train_idx & val_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx & test_idx) == 0

    def test_returns_all_splits(self, sample_df):
        df_x, df_y = self._build(sample_df)
        result = split_temporal(
            df_x=df_x,
            df_y=df_y,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        assert set(result.keys()) == {"train", "val", "test"}

    def test_x_y_aligned_across_splits(self, sample_df):
        df_x, df_y = self._build(sample_df)
        result = split_temporal(
            df_x=df_x,
            df_y=df_y,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        for name, (x, y) in result.items():
            assert list(x.index) == list(y.index)
