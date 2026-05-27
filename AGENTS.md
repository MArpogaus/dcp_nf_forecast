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

## Active models (12)

- `normal_baseline` — diagonal multivariate normal
- `truncated_baseline` — diagonal truncated_normal(0,5) (best CI, competitive NLL)
- `spline_nf` — normal(0,1) base, RationalQuadraticSpline
- `spline_nf_truncated` — truncated_normal(0,5) base, RationalQuadraticSpline (best NLL on DLA/pl2)
- `spline_nf_scale` — normal(0,1) base, Scale + RationalQuadraticSpline
- `spline_nf_scale_truncated` — truncated_normal(0,5) base, Scale + RationalQuadraticSpline
- `spline_nf_scale_shift` — normal(0,1) base, Scale + Shift + RationalQuadraticSpline
- `spline_nf_scale_shift_truncated` — truncated_normal(0,5) base, Scale + Shift + RQS
- `bernstein_nf_scale` — normal(0,1) base, Scale + BernsteinPolynomial
- `bernstein_nf_scale_truncated` — truncated_normal(0,5) base, Scale + BernsteinPolynomial
- `bernstein_nf_scale_shift` — normal(0,1) base, Scale + Shift + BernsteinPolynomial
- `bernstein_nf_scale_shift_truncated` — truncated_normal(0,5) base, Scale + Shift + BernsteinPolynomial

**Dropped:** plain `bernstein_nf`, `bernstein_nf_truncated`, `lognormal_baseline`.

## Config rules

- `truncated_normal(0, 5)` is the standard truncated base for all NF models
- `parameters_constraint_fn_kwargs.low`/`.high` must match base distribution support:
  - Normal base: `low: -5.0, high: 5.0`
  - TruncatedNormal base: `low: 0.0, high: 5.0`
- Spline domain = `[range_min, range_min + interval_width]`. Must match base distribution support:
  - Normal base: `range_min: -5, interval_width: 10` → domain [-5, 5] (covers 5σ of Normal(0,1))
  - TruncatedNormal base: `range_min: 0, interval_width: 5` → domain [0, 5]
- **Scale + TruncatedNormal(0,5)** requires `clipped_softplus_constrain_fn` with `min_value: 0.5` (not 0.01). Root cause: MADE initialization produces near-zero Scale, causing `RQS.forward(y)/scale` to amplify outside TruncatedNormal(0,5) support [0,5]. Fix: constrain scale to [0.5, 1.0]. Applied to all 14 NaN-affected configs.
- Use non-scale `spline_nf_truncated` or `truncated_baseline` only if NaN persists despite min_value=0.5.
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

## Summary docs

- `docs/hpo_summary.md` — publication-oriented results summary: dataset, model set, experiment setup, per-target results tables, cross-target winners, discussion
- `docs/model_architecture.md` — detailed architecture description of all model variants (two-layer MAF MADE + nested FC parameterization, parameter routing, bijector composition, per-variant breakdown with flow diagrams)
- `scripts/report_hpo_results.py` — queries MLflow, prints terminal tables, generates LaTeX (`--latex --output tables.tex`)

### Updating summaries after new evaluations

After running new model evaluations (e.g., the Scale+Shift models from Phase 8), update both summary docs:

1. **Query MLflow and regenerate LaTeX tables:**
   ```bash
   source /app/.venv/bin/activate && python scripts/report_hpo_results.py --latex --output tables.tex
   ```
   Review the terminal output for new entries.

2. **Update `docs/hpo_summary.md`:**
   - Add new rows to the per-target results tables (Section 4.1–4.4) for any newly evaluated models. Keep ranking by NLL.
   - If a new model becomes the best for any target, update the winner row in that target's section and the Cross-Target Winners Summary (Section 4.5).
   - Move newly-evaluated models from "Models Without Evaluation Results" (Section 4.6) into the main results tables.
   - Update the discussion (Section 5) if new findings emerge (e.g., Shift bijector impact, Scale+Shift on ofen_g_koks).
   - Run `scripts/report_hpo_results.py` without flags to capture any new results for the terminal view.

3. **Update `docs/model_architecture.md`** only if model architectures change (not needed for new evaluations of existing architectures).

4. **MLflow run IDs for reference:**
   - Experiment 85: `dcp_nf_forecast-dla`
   - Experiment 86: `dcp_nf_forecast-ofen_g_koks`
   - Experiment 87: `dcp_nf_forecast-ofen_f_koks`
   - Experiment 88: `dcp_nf_forecast-pl2`

## Parallel evaluation

- `scripts/dvc_repro_parallel.sh` runs all evaluate stages across both GPUs.
- Default: `dvc repro <stage>` (full repro checking deps).
- `--force-evaluation` flag: `dvc repro --single-item --force <stage>` (re-eval only, no training).
- Splits stages evenly across GPU 0 and GPU 1; runs both batches in background.
- Logs written to `logs/gpu*`.

```bash
./scripts/dvc_repro_parallel.sh              # full repro
./scripts/dvc_repro_parallel.sh --force-evaluation  # force re-eval
```

- After config changes, always disable test mode (`test_mode: false` in params.yaml) before running the full repro.
