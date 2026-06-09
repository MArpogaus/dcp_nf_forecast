# Agent conventions for dcp_nf_forecast

## Project structure

- `params/models/<target>/<model>.yaml` — per-target per-model configs
- `scripts/` — CLI entry points (train, evaluate, prepare_features, split_data, describe_data)
- `src/dcp_nf_forecast/` — package code (data, models, utils)
- `params.yaml` — global DVC pipeline params
- `dvc.yaml` — DVC pipeline definition (foreach over targets × models)
- `hpo_study.md` — HPO phase log

## Model naming

Models use suffix indicating base distribution:
- `*_truncated` = `truncated_normal(0, 5)` base
- `*_lognormal` = `LogNormal(0, 1)` base
- `*` (no suffix) = `normal(0, 1)` base
- `*_baseline` = simple diagonal distributions (not NF)

## Active models (21)

- `normal_baseline` — diagonal multivariate normal
- `truncated_baseline` — diagonal truncated_normal(0,5)
- `lognormal_baseline` — diagonal multivariate lognormal
- `spline_nf` — normal(0,1) base, RationalQuadraticSpline
- `spline_nf_truncated` — truncated_normal(0,5) base, RationalQuadraticSpline
- `spline_nf_lognormal` — LogNormal(0,1) base, RationalQuadraticSpline
- `spline_nf_scale` — normal(0,1) base, Scale + RationalQuadraticSpline
- `spline_nf_scale_truncated` — truncated_normal(0,5) base, Scale + RationalQuadraticSpline
- `spline_nf_scale_lognormal` — LogNormal(0,1) base, Scale + RationalQuadraticSpline
- `spline_nf_scale_shift` — normal(0,1) base, Scale + Shift + RationalQuadraticSpline
- `spline_nf_scale_shift_truncated` — truncated_normal(0,5) base, Scale + Shift + RQS
- `spline_nf_scale_shift_lognormal` — LogNormal(0,1) base, Scale + Shift + RQS
- `bernstein_nf` — normal(0,1) base, BernsteinPolynomial
- `bernstein_nf_truncated` — truncated_normal(0,5) base, BernsteinPolynomial
- `bernstein_nf_lognormal` — LogNormal(0,1) base, BernsteinPolynomial
- `bernstein_nf_scale` — normal(0,1) base, Scale + BernsteinPolynomial
- `bernstein_nf_scale_truncated` — truncated_normal(0,5) base, Scale + BernsteinPolynomial
- `bernstein_nf_scale_lognormal` — LogNormal(0,1) base, Scale + BernsteinPolynomial
- `bernstein_nf_scale_shift` — normal(0,1) base, Scale + Shift + BernsteinPolynomial
- `bernstein_nf_scale_shift_truncated` — truncated_normal(0,5) base, Scale + Shift + BernsteinPolynomial
- `bernstein_nf_scale_shift_lognormal` — LogNormal(0,1) base, Scale + Shift + BernsteinPolynomial

## Config rules

- `truncated_normal(0, 5)` is the standard truncated base for all NF models
- `parameters_constraint_fn_kwargs.low`/`.high` must match base distribution support:
  - Normal base: `low: -5.0, high: 5.0`
  - TruncatedNormal base: `low: 0.0, high: 5.0`
- Spline domain = `[range_min, range_min + interval_width]`. Must match base distribution support:
  - Normal base: `range_min: -5, interval_width: 10` → domain [-5, 5] (covers 5σ of Normal(0,1))
  - TruncatedNormal base: `range_min: 0, interval_width: 5` → domain [0, 5]
  - LogNormal base: `range_min: 0.006738, interval_width: 148.406` → domain [exp(-5), exp(5)] (covers 5σ of LogNormal(0,1))
- BernsteinPolynomial domain must match for LogNormal base: `domain: [0.006738, 148.413]`, `high: 148.413`, `low: 0.006738`
- Shift constraint for LogNormal base: `tensorflow.math.softplus` (positive shift only — no upper bound, so max_shift not needed)
- **Scale + TruncatedNormal(0,5)** requires `clipped_softplus_constrain_fn` with `min_value: 0.3` (raised from 0.5 for more flexibility; originally 0.01 caused NaN). Root cause: MADE initialization produces near-zero Scale, causing `RQS.forward(y)/scale` to exceed [0,5] support. Fix: constrain scale to [0.3, 1.0]. Applied to all 14 truncated configs.
- **Shift + TruncatedNormal(0,5)** requires `shift_constrain_fn` with `max_shift: 0.5`. The composite flow output `(RQS.forward(y) - shift) / scale` stays in [0,5] only when shift ∈ (-0.5, 0). Derivation:
  - Upper bound (data=1, scale=0.3): `(1 - shift) / 0.3 ≤ 5 → shift ≥ -0.5`
  - Lower bound (data=0): `(0 - shift) / scale ≥ 0 → shift ≤ 0`
  - `max_shift=5.0` (old value) allowed shift=-5 → `(1 + 5)/0.3 = 20` ≫ 5 → `-inf` log_prob.
  - `max_shift=0.5` keeps output at or below `(1 + 0.5)/0.3 = 5.0`, exactly at [0,5] edge.
- Shift **must be negative** for TruncatedNormal base. Positive shift causes data=0 → `-shift/scale < 0` → `-inf` log_prob (verified empirically).
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

### Launch (persistent)

The script can take hours. The shell session must outlive the Bash tool's timeout. Use Python `subprocess` with `os.setsid()` to survive:

```bash
source /app/.venv/bin/activate
python -c "
import subprocess, os
p = subprocess.Popen(['bash', 'scripts/dvc_repro_parallel.sh'],
    stdout=open('logs/repro_full.log','a'),
    stderr=subprocess.STDOUT,
    preexec_fn=lambda: os.setsid())
with open('/tmp/dvc_repro_pid','w') as f: f.write(str(p.pid))
print(f'Launched PID {p.pid}')
"
```

Alternatively use the helper at `scripts/run_repro_background.py`.

### Monitor progress

```bash
# Quick status:
python scripts/check_repro.py

# Tail both GPU logs:
tail -f logs/gpu*_evaluate@*.log

# Check which stages completed today:
source /app/.venv/bin/activate
python -c "
from pathlib import Path; import time
today = time.strftime('%b %d')
for f in sorted(Path('logs').glob('gpu*_evaluate@*.log')):
    t = time.strftime('%b %d', time.gmtime(f.stat().st_mtime))
    if t == today and 'Generating PIT histogram' in (f.read_text() if f.stat().st_size < 5e5 else ''): print('✅', f.name)
"

# Check active DVC processes:
ps aux | grep "dvc repro" | grep -v grep | awk '{for(i=11;i<=NF;i++) printf \"%s \", \$i; print \"\"}'

# Expected: 48 evaluate stages total
```

### Post-repro workflow

After all 48 stages complete:

1. **Query best configs per target:**
   ```bash
   python scripts/query_best_configs.py
   ```

2. **Generate publication figures and tables:**
   ```bash
   python scripts/generate_paper_figures.py --output-dir paper_figures
   ```
   Outputs: `example_forecast.pdf`, `pit_histograms.pdf` (2×2 grid),
   `nll_comparison.pdf`, LaTeX tables (`model_comparison.tex`,
   `cross_target_winners.tex`).

3. **Commit updated dvc.lock:**
   ```bash
   git add dvc.lock AGENTS.md hpo_study.md
   git commit -m "chore: repro all configs with Scale/Shift NaN fix
   
   All 48 configs re-trained with Scale min_value=0.3 and Shift
   max_shift=0.5 constraints. No NaN/INF in training loss."
   ```

- After config changes, always disable test mode (`test_mode: false` in params.yaml) before running the full repro.

## Evaluate script refactor plan

Current `evaluate.py` does 3 things in one pass: (A) sample from model, (B) compute metrics, (C) generate plots.

Proposed split:

| Script | Responsibility | Invocation |
|--------|---------------|------------|
| `sample.py` | Load model, sample posterior → save `samples.feather` + `pit_values.feather` | `python scripts/sample.py --model X --target Y` |
| `eval_samples.py` | Load samples, compute metrics, generate plots → save `metrics.yaml`, `forecast.pdf`, `pit_histogram.pdf` | `python scripts/eval_samples.py --model X --target Y` |
| `evaluate.py` | Thin wrapper: `sample.py` → `eval_samples.py` (backwards compat) | Unchanged CLI |

**Benefits:**
- Separate GPU sampling (step A) from CPU plotting (steps B/C) — iterate on plots without re-sampling
- Decouple `generate_paper_figures.py` from `evaluate.py` — paper script reads pre-saved samples directly
- Parallel: sample all models first, then batch-plot in paper script

**DVC impact:** Would add a `sample@dataset<N>-<model>` stage between `train@` and `evaluate@`, keeping `evaluate@` as the current eval+plot stage. Not required for Phase 9 — defer to Phase 10.
