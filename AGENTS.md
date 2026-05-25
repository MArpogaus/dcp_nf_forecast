# Agent conventions for dcp_nf_forecast

## Project structure

- `params/models/<target>/<model>.yaml` — per-target per-model configs
- `scripts/` — CLI entry points (train, evaluate, prepare_features, split_data)
- `src/dcp_nf_forecast/` — package code (data, models, utils)
- `params.yaml` — global DVC pipeline params
- `dvc.yaml` — DVC pipeline definition (foreach over targets × models)
- `hpo_study.md` — HPO phase log

## Config rules

- Bernstein `domain: [0.0, 1.0]` always (matches data in [0,1])
- `parameters_constraint_fn_kwargs.low`/`.high` must match base distribution support:
  - Normal base: `low: -5.0, high: 5.0`
  - LogNormal base: `low: 0.007, high: 148`
- Spline domain = `[range_min, range_min + interval_width]`. Must match base distribution support.
  - Normal base: `range_min: -4, interval_width: 8` → domain [-4, 4]
  - LogNormal base: `range_min: 0` → domain [0, interval_width]
  - TruncatedNormal base: `range_min` should match `low`, `interval_width = high - low`
- **LogNormal base problem**: unbounded upper tail → linear spline extrapolation → extreme samples (CI90 >> 1). Fix: use `truncated_normal` with bounded support instead.

## Environment

- TensorFlow, DVC and all project deps installed in `/app/.venv` (Python 3.11).
- **Always activate the venv first:** `source /app/.venv/bin/activate`
- After activation, `dvc repro` and `python scripts/...` both use the correct env.
- Available GPU: NVIDIA TITAN RTX (2 × 22GB)

## Pipeline

```bash
# Activate venv first
source /app/.venv/bin/activate

# Run via DVC (now works correctly):
dvc repro train@dataset<N>-<model>

# Or run directly:
python scripts/train.py --model <model> --stage-name train@dataset<N>-<model> --target-name <target>
python scripts/evaluate.py --model <model> --stage-name evaluate@dataset<N>-<model> --target-name <target> --prediction-horizon 48

# Stage names: train@dataset<N>-<model> where N is the target index (0=dla, 1=ofen_g_koks, 2=ofen_f_koks, 3=pl2)
```

## Data prep

- `prepare_features.py` receives all params via CLI args (no yaml loading)
- Binary holiday indicator covariate: `--holiday-country <ISO>` (optional, e.g. `DE-BW`). Generates column via `holidays` library.
- Time range filter: `--end-date <date>` to truncate data (must be quoted string in params.yaml)
- No NaN filling during loading. NaNs from lag/shift operations are counted, logged, and dropped.
- Raw data stored at `data/raw/norm_fondium_15_min_data_2023.csv`

## Git

- Conventional commits on `dev` branch
- `main` only for merge commits

## Targets

| Index | Name | y_column | prediction_horizon |
|-------|------|----------|-------------------|
| 0 | dla | dla_stromverbrauch_kwh | 48 |
| 1 | ofen_g_koks | ofen_g_koks_kg | 48 |
| 2 | ofen_f_koks | ofen_f_koks_kg | 48 |
| 3 | pl2 | pl2_stromverbrauch_kwh | 48 |

## MLflow

- Experiment name: `dcp_nf_forecast-<target>` (suffixed `-test` in test mode)
- Run name: `<model>_<target>`
- Log dir: `mlruns/` (gitignored)
- Tags: `stage: training`, `stage: evaluation`
