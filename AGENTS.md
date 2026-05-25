# Agent conventions for dcp_nf_forecast

## Project structure

- `params/models/<target>/<model>.yaml` — per-target per-model configs
- `scripts/` — CLI entry points (train, evaluate, prepare_features, split_data)
- `src/dcp_nf_forecast/` — package code (data, models, utils)
- `params.yaml` — global DVC pipeline params
- `dvc.yaml` — DVC pipeline definition (foreach over targets × models)
- `hpo_study.md` — HPO phase log

## Model naming

Models use suffix indicating base distribution:
- `*_truncated` = `truncated_normal(0, 5)` base (replaces old `*_lognormal`)
- `*` (no suffix) = `normal(0, 1)` base
- `*_baseline` = simple diagonal distributions (not NF)

## Active models (6)

- `normal_baseline` — diagonal multivariate normal
- `truncated_baseline` — diagonal truncated_normal(0,5) (best CI, competitive NLL)
- `spline_nf` — normal(0,1) base, RationalQuadraticSpline
- `spline_nf_truncated` — truncated_normal(0,5) base, RationalQuadraticSpline (best NLL on DLA/pl2)
- `spline_nf_scale` — normal(0,1) base, Scale + RationalQuadraticSpline
- `spline_nf_scale_truncated` — truncated_normal(0,5) base, Scale + Spline (DLA only, NaN on other targets)

**Dropped:** Bernstein variants (consistently underperformed vs spline), `truncated_baseline` replaced `lognormal_baseline`.

## Config rules

- `truncated_normal(0, 5)` is the standard truncated base for all NF models
- `parameters_constraint_fn_kwargs.low`/`.high` must match base distribution support:
  - Normal base: `low: -5.0, high: 5.0`
  - TruncatedNormal base: `low: 0.0, high: 5.0`
- Spline domain = `[range_min, range_min + interval_width]`. Must match base distribution support:
  - Normal base: `range_min: -4, interval_width: 8` → domain [-4, 4]
  - TruncatedNormal base: `range_min: 0, interval_width: 5` → domain [0, 5]
- **Scale + TruncatedNormal(0,5) fails** on non-DLA targets (NaN at init). Use non-scale `spline_nf_truncated` or `truncated_baseline` for ofen_g/ofen_f/pl2.
- `truncated_baseline` uses custom `multivariate_truncated_normal` distribution (hard-codes low/high).

## Environment

- TensorFlow, DVC and all project deps installed in `/app/.venv` (Python 3.11).
- **Always activate the venv first:** `source /app/.venv/bin/activate`
- After activation, `dvc repro` and `python scripts/...` both use the correct env.
- Available GPU: NVIDIA TITAN RTX (2 × 22GB)

## Pipeline

```bash
# Activate venv first
source /app/.venv/bin/activate

# Run via DVC:
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
