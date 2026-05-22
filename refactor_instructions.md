# Refactoring Instructions

This document provides a detailed specification for refactoring the `dcp_nf_forecast` repository. It covers fixing a critical math bug, robustly resolving data preparation pitfalls, enabling rigorous docstring lint checks with Ruff, removing code duplications, and leveraging built-in Keras APIs.

---

## Part 1: Configuration & Docstring Setup

### 1.1 Update `pyproject.toml`
Enforce the docstring checks (`D` rules) via Ruff, but exclude all files in the `test/` folder since unit tests do not require public docstrings under Python conventions.

**Modify `pyproject.toml` as follows:**
```toml
[tool.ruff.lint]
select = ["I", "E", "F", "D", "UP"]

[tool.ruff.lint.per-file-ignores]
"test/**/*.py" = ["D"]
```

### 1.2 Automated Docstring Formatting
Before adding missing docstrings, run the following command to automatically correct blank line formatting violations (rule `D413`) after `Returns` or `Raises` sections across the core code:
```bash
ruff check . --fix
```

---

## Part 2: Math Bug Fix & Core Docstrings

### 2.1 Fix Log-Normal Base Distribution
*   **File:** `src/dcp_nf_forecast/distributions.py`
*   **Issue:** The log-normal distribution uses `Softplus()` instead of `Exp()`. This limits the target parameters range severely and contradicts configured bounds (e.g. `low: 0.007, high: 148` matching $[e^{-5}, e^{5}]$).
*   **Action:** Replace `Softplus` with `Exp` bijector.

**Before:**
```python
    def dist(parameters: tf.Tensor) -> tfd.Distribution:
        mv_normal = mv_normal_dist(parameters)
        return tfp.distributions.TransformedDistribution(
            mv_normal, tfp.bijectors.Softplus()
        )
```

**After:**
```python
    def dist(parameters: tf.Tensor) -> tfd.Distribution:
        mv_normal = mv_normal_dist(parameters)
        return tfp.distributions.TransformedDistribution(
            mv_normal, tfp.bijectors.Exp()
        )
```

### 2.2 Add Missing Module/Package Docstrings
Add descriptive double-quoted docstrings (`"""..."""`) at the very top of the following files:

*   **`src/dcp_nf_forecast/__init__.py`**:
    ```python
    """Package for normalizing flow forecasting models."""
    ```
*   **`src/dcp_nf_forecast/distributions.py`**:
    ```python
    """Custom probability distributions for normalizing flows."""
    ```
*   **`src/dcp_nf_forecast/models.py`**:
    ```python
    """Model builders and normalizing flow architectures."""
    ```
*   **`src/dcp_nf_forecast/utils.py`**:
    ```python
    """I/O helpers, logging setup, and plotting configuration utilities."""
    ```
*   **`scripts/train.py`**:
    ```python
    """Train a normalizing flow forecasting model on a target dataset."""
    ```

### 2.3 Add Missing Function Docstrings (NumPy Style)
Add NumPy-formatted docstrings to the following public functions:

*   **`add_holiday_indicator`** in `src/dcp_nf_forecast/data.py`:
    ```python
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
    ```

*   **`build_model`** in `src/dcp_nf_forecast/models.py`:
    ```python
    def build_model(
        dims: int, covariate_dim: int, model_kwargs: dict
    ) -> DensityRegressionModel:
        """Build a DensityRegressionModel with conditional bijectors.

        Parameters
        ----------
        dims : int
            Dimensionality of the target variables.
        covariate_dim : int
            Dimensionality of the conditioning features.
        model_kwargs : dict
            Keyword arguments passed to DensityRegressionModel constructor.

        Returns
        -------
        DensityRegressionModel
            Configured normalizing flow regression model.
        """
    ```

---

## Part 3: Robustness & Performance in Data Prep

### 3.1 Silence Datetime Comparison Warning
*   **File:** `src/dcp_nf_forecast/data.py` (inside `add_holiday_indicator`)
*   **Action:** Explicitly extract the `.date` attribute to avoid comparing mismatched `DatetimeIndex` and `datetime.date` types.

**Before:**
```python
    is_holiday = pd.Series(df.index.isin(cal), index=df.index, dtype=int)  # type: ignore[arg-type]
```

**After:**
```python
    is_holiday = pd.Series(df.index.date, index=df.index).isin(cal).astype(int)
```

### 3.2 Add Chronological Order & Index Guard Checks
*   **File:** `src/dcp_nf_forecast/data.py` (inside `load_raw_data`)
*   **Action:** Defensively log warnings for duplicates and ensure index is sorted to guarantee correctness of `.shift` operations.

**Before returning `df` in `load_raw_data`:**
```python
    end = _parse_date(end_date)
    if end is not None:
        df = df[df.index <= end]  # type: ignore[index]
    return df
```

**After:**
```python
    end = _parse_date(end_date)
    if end is not None:
        df = df[df.index <= end]  # type: ignore[index]

    # Defensive validations for time-series shifting
    if not df.index.is_unique:
        __LOGGER__.warning("Datetime index contains duplicate timestamps! This can cause major alignment issues during feature engineering.")
    if not df.index.is_monotonic_increasing:
        __LOGGER__.warning("Datetime index is not monotonically increasing! Sorting to prevent incorrect shifting.")
        df = df.sort_index()

    return df
```

### 3.3 Optimize Lag/Lead Feature Performance
*   **File:** `src/dcp_nf_forecast/data.py`
*   **Action:** Refactor `create_lag_features`, `_create_lead_features`, and `build_target` to populate a Python dictionary rather than invoking repeated `pd.concat` on small DataFrames in loops.

**Optimized `create_lag_features`:**
```python
def create_lag_features(
    df: pd.DataFrame,
    column_lags: dict[str, int],
) -> pd.DataFrame:
    if not column_lags:
        return pd.DataFrame(index=df.index)
    frames: dict[str, pd.Series] = {}
    for col, num_lags in column_lags.items():
        for i in range(1, num_lags + 1):
            frames[f"{col}_lag_{i}"] = df[col].shift(i)
    return pd.DataFrame(frames, index=df.index)
```

**Optimized `_create_lead_features`:**
```python
def _create_lead_features(
    df: pd.DataFrame,
    column_leads: dict[str, int],
) -> pd.DataFrame:
    if not column_leads:
        return pd.DataFrame(index=df.index)
    frames: dict[str, pd.Series] = {}
    for col, num in column_leads.items():
        for i in range(1, num + 1):
            frames[f"{col}_lead_{i}"] = df[col].shift(-i)
    return pd.DataFrame(frames, index=df.index)
```

**Optimized `build_target`:**
```python
def build_target(
    df: pd.DataFrame,
    target_column: str,
    prediction_horizon: int,
) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for step in range(1, prediction_horizon + 1):
        frames[f"target_{step}"] = df[target_column].shift(-step)
    return pd.DataFrame(frames, index=df.index)
```

---

## Part 4: Code Duplication & Plotting Centralization

### 4.1 Create Centralized Matplotlib Configuration
*   **File:** `src/dcp_nf_forecast/utils.py`
*   **Action:** Add a centralized `setup_plotting_style()` helper function.

**Add to `src/dcp_nf_forecast/utils.py`:**
```python
def setup_plotting_style() -> None:
    """Set standard global Matplotlib parameters for consistent paper figures."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 0.8,
            "figure.figsize": (8, 4.5),
            "font.family": "sans-serif",
        }
    )
```

### 4.2 Refactor Scripts to Reuse Central Style
*   **File:** `scripts/evaluate.py` (lines 30-45) and `scripts/describe_data.py` (lines 24-38)
*   **Action:** Remove the raw `plt.rcParams.update(...)` dictionaries and replace them with:
    ```python
    from dcp_nf_forecast.utils import setup_plotting_style
    setup_plotting_style()
    ```

---

## Part 5: Refactor Evaluation to use Keras `model.evaluate`

### 5.1 Replace manual loop with `model.evaluate`
*   **File:** `scripts/evaluate.py`
*   **Action:** Replace the custom, manually batched `compute_nll` call with Keras's native `.evaluate(...)` method.

**Before:**
```python
            logger.info("Computing NLL on test data ...")
            nll = compute_nll(model, x_test, y_test, batch_size)
            logger.info("NLL: %.4f", nll)
```

**After:**
```python
            logger.info("Computing NLL on test data ...")
            nll = float(model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0))
            logger.info("NLL: %.4f", nll)
```

### 5.2 Clean up unused NLL code
Delete the entire `def compute_nll(...)` function definition (lines 91-125) from `scripts/evaluate.py`.

---

## Part 6: Formatting Line Lengths

Format any long comment or docstring lines in `test/test_data.py` exceeding the 88-character limit. For example:

*   **Line 317:** Wrap the docstring `"""Features (past) must be < targets (future) per row..."""`.
*   **Line 340:** Wrap the docstring `"""When holiday_country is set, is_holiday leads..."""`.

---

## Part 7: Quality Verification Protocol

After implementing all changes, run the following verification steps:

1.  **Run Linter:**
    ```bash
    ruff check .
    ```
    *Expectation:* Command exits cleanly with no errors or warnings.
2.  **Run Unit Tests:**
    ```bash
    pytest
    ```
    *Expectation:* All 70 unit tests pass cleanly.

---

## Part 8: Comprehensive Testing Specification (Data Prep Safeguards Covered)

Add the following three test cases to `test/test_data.py` (for instance, inside the `TestLoadRawData` and `TestAddHolidayIndicator` classes) to verify chronological ordering, duplicate warning logs, and subregion holidays:

```python
    def test_unsorted_raw_data_gets_sorted(self, tmp_path):
        """Verify unsorted raw index gets sorted monotonically increasing."""
        path = tmp_path / "unsorted.csv"
        df = pd.DataFrame(
            {"target": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2023-01-02", "2023-01-01"])
        )
        df.reset_index().rename(columns={"index": "time"}).to_csv(path, index=False)
        result = load_raw_data(path, "time")
        assert result.index.is_monotonic_increasing
        assert result["target"].iloc[0] == 2.0  # Jan 1st value should be first

    def test_duplicate_raw_data_logs_warning(self, tmp_path, caplog):
        """Verify loading duplicate timestamps logs a warning."""
        import logging
        path = tmp_path / "duplicates.csv"
        df = pd.DataFrame(
            {"target": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2023-01-01", "2023-01-01"])
        )
        df.reset_index().rename(columns={"index": "time"}).to_csv(path, index=False)
        with caplog.at_level(logging.WARNING):
            load_raw_data(path, "time")
        assert any("contains duplicate timestamps" in record.message for record in caplog.records)

    def test_holiday_indicator_with_subregion(self):
        """Verify holiday subregion DE-BW is parsed correctly."""
        df = pd.DataFrame(
            {"v": [1]},
            index=pd.DatetimeIndex(["2023-01-06"]),  # Epiphany, holiday only in BW, BY, ST
        )
        result = add_holiday_indicator(df, "DE-BW")
        assert result["is_holiday"].iloc[0] == 1
```
