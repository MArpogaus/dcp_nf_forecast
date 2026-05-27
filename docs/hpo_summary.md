# HPO Study Summary: Probabilistic Multi-Step Time Series Forecasting with Normalizing Flows

## 1. Dataset

**Source:** Industrial process data from a foundry, recorded at 15-minute intervals throughout 2023 (January 1 – October 31, ~35k timesteps). Raw data stored in `data/raw/norm_fondium_15_min_data_2023.csv`.

**Four targets**, each modelled independently with a separate multivariate model. The output variable **y** is a 48-dimensional vector (the prediction horizon = 12 hours), i.e., each training example is `(x_covariates, y_{t+1:48})`:

| Index | Name | y Column | Description |
|-------|------|----------|-------------|
| 0 | `dla` | `dla_stromverbrauch_kwh` | Electrical power consumption of a pressure-chamber system (Druckluftanlage) |
| 1 | `ofen_g_koks` | `ofen_g_koks_kg` | Coke consumption of furnace G |
| 2 | `ofen_f_koks` | `ofen_f_koks_kg` | Coke consumption of furnace F |
| 3 | `pl2` | `pl2_stromverbrauch_kwh` | Electrical power consumption of the pouring-line system (PL2) |

**Data preprocessing:**
- **Covariates:** Time features (sin/cos of hour, day-of-week, day-of-year), holiday indicator (binary, Germany/Baden-Württemberg via `holidays` library), lag features of the target and `ofen_metall_kg` (96 steps = 24 h history), lead (future) features of `ofen_metall_kg` (48 steps = 12 h ahead).
- **NaN handling:** Missing values in `ofen_g_koks_kg` and `ofen_f_koks_kg` filled with 0 plus lognormal noise (scale 0.005). DLA and PL2 have no missing values.
- **Normalization:** All columns min-max scaled to [0, 1].
- **Split:** 80/10/10 temporal split (no shuffle). Training set: Jan 1 – mid-September 2023 (~28k steps). Validation: mid-September to early October. Test: October 1–31.

## 2. Model Set

All models are **multivariate conditional normalizing flows** that jointly model the 48-dimensional distribution of the forecast horizon. Each (target, model variant) is trained independently.

A **detailed architectural description** is in [`docs/model_architecture.md`](model_architecture.md). What follows is a summary of the architecture pattern.

### 2.1 Architecture Pattern

All flow-based models share a **two-layer parameterization**:

```
conditional_input (covariates, shape [N, cov_dim])
    │
    ├──► MAF MADE (tfb.AutoregressiveNetwork)
    │       autoregressive mask over 48 dims: p(y_d | y_{<d}, x)
    │       conditional_input injected at hidden layers
    │       output: [N, 48, num_parameters]   ← parent_parameters
    │
    └──► per-bijector FC networks (parameter_vector_or_simple_network)
            input: conditional_input
            output: [N, 48, slice_size]   ← residual adjustment
                        +                   (ADDED to parent slice)
            parent_parameters sliced by parameters_slice_size
            │
            ▼
        tfb.Chain([bijector_1, bijector_2, ...])
            │  forward: b_k( ... b_2(b_1(base_sample)) )
            ▼
     tfd.TransformedDistribution(
         distribution=tfd.Sample(base_dist, [48]),  i.i.d. over 48 dims
         bijector=tfb.Chain([MAF_layer_1, ...])     ← contains nested chain
     )
```

Key points:

- **MAF outer layer:** A `tfb.AutoregressiveNetwork` (MADE) with `hidden_units=[256,256]`, ReLU activation, and an autoregressive mask over the 48 forecast dimensions. It outputs `(batch, 48, num_parameters)` — one parameter vector per forecast step, each depending only on earlier steps.
- **Nested bijectors:** Each nested bijector (Scale, Shift, Spline, Bernstein) receives a slice of the MAF output (via `parameters_slice_size`). Its own FC network (also `[256,256]`, ReLU) produces a *residual adjustment* that is **added** to the MAF slice. The addition lets the covariates directly modulate each bijector's parameters beyond the MAF's autoregressive structure.
- **Chain order:** Nested bijectors are composed via `tfb.Chain`. Forward direction (generating samples from base) applies them right-to-left in the config list. Config order: `[Scale, {Shift,} Spline/Bernstein]` → forward: `Spline(Shift(Scale(base)))`.
- **Baselines** (`normal_baseline`, `truncated_baseline`, `lognormal_baseline`) skip the MAF entirely — a plain FC network directly predicts `loc` and `scale` vectors (each length 48) for a diagonal multivariate distribution. No flow bijector, no autoregressive structure.

### 2.1 Model Variants (12 active + 3 dropped legacy)

#### 2.1.1 Baselines (no normalizing flow)

| Model | Distribution | Active | Notes |
|-------|-------------|--------|-------|
| `normal_baseline` | Diagonal multivariate Normal(48) | Yes | FC → `Normal(loc, scale)` |
| `truncated_baseline` | Diagonal multivariate TruncatedNormal(48, low=0, high=5) | Yes | Custom distribution; FC → `TruncatedNormal(loc, scale, low=0, high=5)` |
| `lognormal_baseline` | Diagonal multivariate LogNormal(48) | **Dropped** | FC → `LogNormal(loc, scale)`; best NLL on DLA (−193.81) but CI90=5.04 and NaN on non-DLA targets |

#### 2.1.2 Spline-based Normalizing Flows

| Model | Base Distribution | Bijector(s) | \#Params/dim | Active |
|-------|-------------------|-------------|-------------|--------|
| `spline_nf` | Normal(0, 1) | RationalQuadraticSpline(domain=[−5, 5], nbins=12) | 35 | Yes |
| `spline_nf_truncated` | TruncatedNormal(0, 5, low=0, high=5) | RQS(domain=[0, 5], nbins=12) | 35 | Yes |
| `spline_nf_scale` | Normal(0, 1) | Scale(n=48) + RQS(domain=[−5, 5], nbins=12) | 36 | Yes |
| `spline_nf_scale_truncated` | TruncatedNormal(0, 5) | Scale(n=48) + RQS(domain=[0, 5], nbins=8) | 24 | Yes |
| `spline_nf_scale_shift` | Normal(0, 1) | Scale(n=48) + Shift(n=48) + RQS(domain=[−4, 4], nbins=12) | 37 | Yes |
| `spline_nf_scale_shift_truncated` | TruncatedNormal(0, 5) | Scale(n=48) + Shift(n=48) + RQS(domain=[0, 5], nbins=8) | 25 | Yes (DLA only; NaN on non-DLA targets) |
| `spline_nf_lognormal` (alias) | ≈TruncatedNormal(0,5) | RQS(domain=[0,5], nbins=12) | 35 | Subsumed by `spline_nf_truncated` |

**Note on parameter counts and architecture:** `num_parameters` is the MAF output size *per forecast dimension*, sliced among nested bijectors via `parameters_slice_size`. Each nested bijector's own FC network (same `hidden_units` as the MAF, `parameter_vector_or_simple_network`) produces a residual added to its MAF slice. Models *without* `nested_bijectors` (plain `spline_nf`, `spline_nf_truncated`) pass the full MAF output directly to their single bijector without any per-bijector FC network. For example, `spline_nf_scale` has MAF→36 values/dim: slice [0:1]→Scale (1 param/dim, plus FC_residual added), slice [1:36]→Spline (35 params/dim, plus FC_residual added). See [`docs/model_architecture.md`](model_architecture.md) for full details.

#### 2.1.3 Bernstein-based Normalizing Flows

| Model | Base Distribution | Bijector(s) | \#Params/dim | Active |
|-------|-------------------|-------------|-------------|--------|
| `bernstein_nf` | Normal(0, 1) | BernsteinPolynomial(order=8, domain=[0, 1]) | 8 | **Dropped** (NLL ~ 42–288, unusable) |
| `bernstein_nf_lognormal` | ≈TruncatedNormal(0,5) | BernsteinPolynomial(order=8) | 8 | Subsumed by truncated variants |
| `bernstein_nf_scale` | Normal(0, 1) | Scale(n=48) + BernsteinPolynomial(order=8) | 9 | Yes |
| `bernstein_nf_scale_truncated` | TruncatedNormal(0, 5) | Scale(n=48) + BernsteinPolynomial(order=8) | 17 | Yes |
| `bernstein_nf_scale_shift` | Normal(0, 1) | Scale(n=48) + Shift(n=48) + BernsteinPolynomial(order=8) | 14 | Yes |
| `bernstein_nf_scale_shift_truncated` | TruncatedNormal(0, 5) | Scale(n=48) + Shift(n=48) + BernsteinPolynomial(order=8) | 18 | Yes (DLA only; NaN on non-DLA targets) |

### 2.2 Base Distribution Design

The choice of base distribution is the single most important architectural decision in this study. The base is 48-dimensional (independent across dimensions, then transformed by the MAF + bijectors into a dependent joint distribution).

- **Normal(0, 1):** Symmetric, unbounded. Requires spline domain [−5, 5] (5σ coverage). Works well for targets with near-Gaussian residuals after the flow, but limits the flow's ability to model heavy upper tails.
- **LogNormal(0, 1):** Positive, heavy upper tail. Early HPO phases used this for all NF models targeting positively-skewed industrial data. **Problem:** The unbounded upper tail produces samples >> the spline's [0, 8] domain, forcing linear extrapolation → extreme predicted values → CI90 >> 1 (sometimes 8673 or NaN).
- **TruncatedNormal(0, 5, low=0, high=5):** The resolution (Phase 6). Bounded [0, 5] matches the spline domain [0, 5] exactly. No extrapolation needed. All NF models were migrated from LogNormal → TruncatedNormal. For normal-base models the domain is [−5, 5] (5σ of Standard Normal).

**Rule:** `parameters_constraint_fn_kwargs.low`/`.high` must match base distribution support; spline `range_min`/`interval_width` must match domain.

## 3. Experiment Setup

### 3.1 Pipeline

Orchestrated via DVC with four stages per target:
1. `prepare` — feature engineering (lags, leads, time features, holiday, NaN filling)
2. `split` — temporal train/val/test split (80/10/10)
3. `train` — train the flow model, log to MLflow
4. `evaluate` — compute test-set metrics, log to MLflow, save PIT/QQ figures

Each (target, model) combination is a separate DVC stage: 12 active models × 4 targets = 48 stages.

### 3.2 Training Configuration

**Global defaults (from `params.yaml`):**
- `batch_size: 256`
- `epochs: 200` → increased to 400 after Phase 4
- `early_stopping_patience: 10` → increased to 20 after Phase 4
- `learning_rate: 0.001` (constant or CosineDecay with `decay_steps = 22000`)
- `hidden_units: [128, 128]` → increased to `[256, 256]` after Phase 5
- Optimizer: Adam (defaults)
- Loss: `-p_y.log_prob(y)` — the joint log-probability of all 48 dimensions
- Seed: 42

**Hyperparameter search space:**
| Parameter | Values tested | Final best |
|-----------|--------------|------------|
| `learning_rate` | {1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3} | 0.0005 or 0.001 |
| `lr_schedule` | {constant, CosineDecay} | model-dependent (Spline: constant; baseline/scale: CosineDecay) |
| `hidden_units` | {[128,128], [128,64], [256,128], [256,256]} | [256, 256] (universal) |
| `epochs` | {200, 400} | 400 (universal) |
| `early_stopping_patience` | {10, 20} | 20 (universal) |
| `nbins` (Spline) | {3, 8, 10, 12, 16, 24} | 12 (universal) |
| `interval_width` (Spline) | {3, 5, 8, 10, 16} | 5 for truncated, 8 for normal |
| `order` (Bernstein) | {8, 12} | 8 (order increase didn't help) |

### 3.3 HPO Process

**Eight phases** (~68 iterations total):

| Phase | Focus | Key Decision |
|-------|-------|-------------|
| 1 | Constant LR benchmark (old data) | Established baseline hierarchy |
| 2 | CosineDecay LR | CosineDecay helps baselines but hurts NFs |
| 3 | Capacity increase (nbins 8→12, hidden units) | Marginal gains |
| — | Data update (holiday indicator, NaN fill, new raw data) | Dramatic improvement for baselines |
| 4 | **Targeted DLA HPO** (spline_nf_lognormal) | Discovered hidden=[256,256], epochs=400 → beats lognormal_baseline |
| 5 | **Unified config** all targets × 10 models | h=[256,256], ep=400 is universal; each target has unique best model |
| 6 | **Domain fix** (LogNormal → TruncatedNormal) | Solved CI90 explosion across all 48 forecast steps |
| 7 | Cross-target validation of fixed models | Fixed models work on all 4 targets |
| 8 | Add Scale+Shift bijectors, evaluate all models | Shift degrades all models (6–58 NLL loss); NaN Scale+Truncated pattern confirmed |

**Search strategy:** One parameter change per iteration. Start with learning regime → capacity → regularization → budget.

**MLflow tracking:** Experiment `dcp_nf_forecast-<target>`. Parent run stores full config. Child runs: training metrics (per-epoch) and evaluation metrics (test-set). Evaluation metrics are the definitive results.

### 3.4 Evaluation Metrics

| Metric | Description | Computation |
|--------|-------------|-------------|
| **NLL** | Negative log-likelihood on test set | `mean(-log p(y_true))` across samples and 48 dims; **joint density** of the full 48-step forecast vector |
| **RMSE** | Root mean squared error | `sqrt(mean((y_true - median)^2))` over all samples × 48 dims |
| **MAE** | Mean absolute error | `mean(|y_true - median|)` over all samples × 48 dims |
| **CI90** | Mean 90% prediction interval width | `mean(perc95 - perc05)` across samples, then averaged over 48 dims |
| PIT | Per-step (per-dimension) PIT histogram | One subplot per forecast step; uniform = well-calibrated at that horizon |
| QQ | Per-step quantile-quantile plot | One subplot per step; diagonal = well-calibrated |

**Interpretation nuance:** The NLL is the *joint* density of the 48-dimensional forecast. A model that correctly captures cross-step dependencies (via the MAF autoregressive structure) will have a better (lower) NLL than a model that treats each step independently — even if per-step marginal distributions are similar. The CI90 is averaged over all 48 steps, so a single poorly-calibrated step can inflate it.

## 4. Results

### 4.1 Target: DLA (dla_stromverbrauch_kwh)

*Electrical power consumption; lowest variance of all targets.*

| Rank | Model | NLL | RMSE | MAE | CI90 | LR | Hidden | Epochs |
|------|-------|-----|------|-----|------|----|--------|--------|
| 1 | **spline_nf_truncated** | **−197.66** | 0.0304 | 0.0268 | 0.6978 | 0.0005 | [256,256] | 400 |
| 2 | spline_nf_lognormal (trunc base) | −196.23 | 0.0299 | 0.0265 | 0.1283 | 0.0005 | [256,256] | 400 |
| 3 | lognormal_baseline | −193.81 | 0.3878 | 0.3669 | 5.0364 | 0.001 Cos | [128,128] | 200 |
| 4 | spline_nf | −178.20 | 0.0361 | 0.0299 | 0.0623 | 0.001 | [256,256] | 400 |
| 5 | bernstein_nf_scale_lognormal | −168.64 | 0.0197 | 0.0140 | 0.0555 | 0.001 | [128,128] | 200 |
| 6 | bernstein_nf_scale | −167.31 | 0.0142 | 0.0094 | 0.0237 | 0.001 | [128,128] | 200 |
| 7 | spline_nf_scale | −166.40 | 0.0107 | 0.0077 | 0.0227 | 0.001 | [128,128] | 200 |
| 8 | bernstein_nf_lognormal | −158.55 | 0.0386 | 0.0321 | 0.1448 | 0.001 Cos | [256,256] | 400 |
| 9 | spline_nf_scale_lognormal | −157.98 | 116.67 | 29.61 | 8673 | 0.001 Cos | [128,128] | 200 |
| 10 | truncated_baseline | −147.07 | 0.0127 | 0.0089 | 0.0420 | 0.001 | [256,256] | 400 |
| 11 | spline_nf_scale_shift | −144.61 | 0.0162 | 0.0117 | 0.0326 | 0.0005 | [256,256] | 400 |
| 12 | bernstein_nf_scale_shift | −134.95 | 0.0152 | 0.0115 | 0.0310 | 0.0005 | [256,256] | 400 |
| 13 | spline_nf_scale_truncated | −123.51 | 0.0265 | 0.0224 | 0.0568 | 0.001 Cos | [256,256] | 400 |
| 14 | bernstein_nf_scale_truncated | −114.94 | 0.0302 | 0.0265 | 0.0750 | 0.0005 | [256,256] | 400 |
| 15 | normal_baseline | −111.62 | 0.1011 | 0.0789 | 1.2419 | 0.001 Cos | [128,128] | 200 |
| 16 | bernstein_nf | 42.34 | 0.0511 | 0.0446 | 0.0767 | 0.001 Cos | [256,256] | 400 |

**Winner:** `spline_nf_truncated` (−197.66 NLL, 35 params/dim, MAF + RQS with truncated_normal(0,5) base). Two top models share identical architecture; the constant-LR variant (rank 1, −197.66) substantially outperforms CosineDecay (rank 2, −183.72, visible in per-model variation table). Constant LR is strongly preferred on DLA.

**Training dynamics:** The best iteration trained all 400 epochs without early stopping; the CosineDecay variant peaked at epoch 183. The spline NF benefits from sustained high LR throughout training.

**CI90 anomaly:** The winner's CI90 (0.698) is 5.5× wider than the runner-up (0.128) despite both using identical config except LR schedule. The CosineDecay variant achieves CI90=0.093. This suggests the constant-LR solution may have wider intervals on some of the 48 forecast steps even though joint density is higher.

**Scale+Shift impact:** Adding a Shift bijector to `spline_nf_scale` (−166.40 → −144.61, −21.8 NLL) or `bernstein_nf_scale` (−167.31 → −134.95, −32.4 NLL) substantially degrades performance on DLA. The 48 additional shift parameters overparameterize the flow, and the extra bijector layer adds optimization difficulty — consistent with the `spline_nf_scale_truncated` observation that more bijectors do not always help.

### 4.2 Target: ofen_f_koks

*Coke mass for furnace F; very sparse (most values are 0, intermittent positive).*

| Rank | Model | NLL | RMSE | MAE | CI90 | LR | Hidden | Epochs |
|------|-------|-----|------|-----|------|----|--------|--------|
| 1 | **spline_nf_scale_lognormal (trunc)** | **−195.93** | 0.0115 | 0.0058 | 0.0229 | 0.001 Cos | [256,256] | 400 |
| 2 | spline_nf | −191.67 | 0.0288 | 0.0278 | 0.1136 | 0.001 | [256,256] | 400 |
| 3 | truncated_baseline | −188.06 | 0.0127 | 0.0058 | 0.0228 | 0.001 | [256,256] | 400 |
| 4 | spline_nf_scale | −187.81 | 0.0111 | 0.0059 | 0.0135 | 0.001 | [128,128] | 200 |
| 5 | bernstein_nf_scale_lognormal | −171.04 | 0.0112 | 0.0052 | 0.0164 | 0.0005 | [128,128] | 200 |
| 6 | bernstein_nf_scale | −165.58 | 0.0124 | 0.0051 | 0.0110 | 0.0005 | [128,128] | 200 |
| 7 | spline_nf_lognormal (trunc) | −165.47 | 0.0507 | 0.0478 | 2.6513 | 0.0005 | [256,256] | 400 |
| 8 | spline_nf_truncated | −164.78 | 0.0505 | 0.0476 | 2.4398 | 0.0005 | [256,256] | 400 |
| 9 | bernstein_nf_lognormal | −161.35 | 0.0262 | 0.0253 | 0.0939 | 0.0005 Cos | [256,256] | 400 |
| 10 | normal_baseline | −120.61 | 0.1138 | 0.0824 | 1.2765 | 0.001 Cos | [256,256] | 400 |
| 11 | bernstein_nf | 287.98 | 0.0727 | 0.0698 | 0.0767 | 0.0005 Cos | [256,256] | 400 |

| 8 | bernstein_nf_scale_shift | −146.60 | 0.0272 | 0.0170 | 0.0232 | 0.0005 | [256,256] | 400 |
| 9 | spline_nf_scale_shift | −141.65 | 0.0196 | 0.0134 | 0.0188 | 0.0005 | [256,256] | 400 |
| 10 | bernstein_nf_lognormal | −161.35 | 0.0262 | 0.0253 | 0.0939 | 0.0005 Cos | [256,256] | 400 |
| 11 | normal_baseline | −120.61 | 0.1138 | 0.0824 | 1.2765 | 0.001 Cos | [256,256] | 400 |
| 12 | bernstein_nf | 287.98 | 0.0727 | 0.0698 | 0.0767 | 0.0005 Cos | [256,256] | 400 |

**Winner:** `spline_nf_scale_lognormal_ofen_f` (−195.93 NLL, 24 params/dim). Despite its name, this model uses a **truncated_normal(0,5)** base (migrated in Phase 7). The Scale bijector (1 per-dim parameter) is crucial: without it (rank 7, plain spline_nf_lognormal), CI90 jumps from 0.023 to 2.65 — a 100× widening. The 48 learnable scale parameters apparently adapt the flow output to the sparse coke-injection pattern.

**Baseline competitiveness:** The `truncated_baseline` (diagonal truncated normal, rank 3) achieves CI90=0.023 and NLL=−188.06 with zero flow parameters. On this sparse target, the 48-dimensional joint distribution may be roughly diagonal truncated-normal shaped — the MAF's autoregressive dependency modeling adds limited value.

**Scale+Shift impact:** Both `spline_nf_scale_shift` (−141.65) and `bernstein_nf_scale_shift` (−146.60) are ranked lower than their Scale-only counterparts (−187.81 and −165.58 respectively). The Shift bijector adds 48 parameters (1 per forecast dimension) but does not improve the fit — the target's conditional mean is already well-captured by the Scale + MAF structure.

### 4.3 Target: ofen_g_koks

*Coke mass for furnace G; less sparse than ofen_f but still with long zero periods.*

| Rank | Model | NLL | RMSE | MAE | CI90 | LR | Hidden | Epochs |
|------|-------|-----|------|-----|------|----|--------|--------|
| 1 | **bernstein_nf_scale_lognormal** | **−179.87** | 0.0547 | 0.0132 | 0.0405 | 0.0005 | [128,128] | 200 |
| 2 | spline_nf_scale | −170.80 | 0.0551 | 0.0129 | 0.0338 | 0.001 | [128,128] | 200 |
| 3 | bernstein_nf_lognormal | −163.73 | 0.0961 | 0.0772 | 0.2355 | 0.0005 Cos | [256,256] | 400 |
| 4 | spline_nf | −155.17 | 0.1384 | 0.1288 | 0.2334 | 0.001 | [256,256] | 400 |
| 5 | bernstein_nf_scale | −149.61 | 0.0319 | 0.0146 | 0.0365 | 0.0005 | [128,128] | 200 |
| 6 | bernstein_nf_scale_shift | −142.78 | 0.0328 | 0.0157 | 0.0324 | 0.0005 | [256,256] | 400 |
| 7 | spline_nf_scale_shift | −139.25 | 0.0334 | 0.0163 | 0.0307 | 0.0005 | [256,256] | 400 |
| 8 | truncated_baseline | −135.66 | 0.0578 | 0.0168 | 0.0636 | 0.001 | [256,256] | 400 |
| 9 | spline_nf_truncated | −127.05 | 0.1432 | 0.1332 | 0.1751 | 0.0005 | [256,256] | 400 |
| 10 | spline_nf_lognormal (trunc) | −82.66 | 0.1455 | 0.1352 | 0.1712 | 0.0005 | [256,256] | 400 |
| 11 | normal_baseline | −80.82 | 0.1247 | 0.1038 | 0.6704 | 0.001 Cos | [256,256] | 400 |
| 12 | bernstein_nf | 258.07 | 0.1408 | 0.1311 | 0.1603 | 0.0005 Cos | [256,256] | 400 |

**Winner:** `bernstein_nf_scale_lognormal` (−179.87 NLL, 13 params/dim). This is the only target where a Bernstein-based model outperforms all spline-based models. The best config uses **smaller** capacity ([128,128], 200 epochs) — the second HPO iteration with larger capacity regressed to −179.62. The 9 per-dim parameters (1 scale + 8 Bernstein coefficients) apparently capture the ofen_g distribution shape more efficiently than spline's 36.

**Note:** If CI90 is prioritized over NLL, `spline_nf_scale` offers better intervals (0.034 vs 0.041) at a small NLL cost (−170.80 vs −179.87).

**Scale+Shift impact:** Adding Shift to `bernstein_nf_scale` (−149.61 → −142.78, −6.8 NLL) and `spline_nf_scale` (−170.80 → −139.25, −31.6 NLL) degrades performance. The shift parameters are not beneficial on this target, consistent with the pattern across all four targets.

### 4.4 Target: pl2

*Electrical power consumption of the pouring-line system; highest variance, most Gaussian-like distribution.*

| Rank | Model | NLL | RMSE | MAE | CI90 | LR | Hidden | Epochs |
|------|-------|-----|------|-----|------|----|--------|--------|
| 1 | **bernstein_nf_scale** | **−205.28** | 0.0129 | 0.0078 | 0.0234 | 0.0005 | [256,256] | 400 |
| 2 | spline_nf_scale | −204.79 | 0.0099 | 0.0066 | 0.0235 | 0.001 | [128,128] | 200 |
| 3 | bernstein_nf_scale_lognormal | −203.94 | 0.0124 | 0.0078 | 0.0243 | 0.0005 | [128,128] | 200 |
| 4 | spline_nf_lognormal (trunc) | −202.86 | 0.0387 | 0.0360 | 0.0464 | 0.0005 | [256,256] | 400 |
| 5 | spline_nf | −199.30 | 0.0340 | 0.0318 | 0.1671 | 0.001 | [256,256] | 400 |
| 6 | spline_nf_scale_lognormal (trunc) | −193.59 | 0.0136 | 0.0083 | 0.0262 | 0.001 Cos | [256,256] | 400 |
| 7 | spline_nf_truncated | −186.86 | 0.0403 | 0.0375 | 0.0399 | 0.0005 | [256,256] | 400 |
| 8 | bernstein_nf_lognormal | −176.45 | 0.0345 | 0.0322 | 0.0742 | 0.0005 Cos | [256,256] | 400 |
| 9 | spline_nf_scale_shift | −155.59 | 0.0134 | 0.0102 | 0.0395 | 0.0005 | [256,256] | 400 |
| 10 | spline_nf_scale_truncated | −148.26 | 0.0162 | 0.0132 | 0.0478 | 0.001 Cos | [256,256] | 400 |
| 11 | bernstein_nf_scale_shift | −147.72 | 0.0139 | 0.0102 | 0.0364 | 0.0005 | [256,256] | 400 |
| 12 | bernstein_nf_scale_truncated | −127.00 | 0.0164 | 0.0134 | 0.0662 | 0.0005 | [256,256] | 400 |
| 13 | truncated_baseline | −142.00 | 0.0178 | 0.0123 | 0.0806 | 0.001 | [256,256] | 400 |
| 14 | normal_baseline | −131.67 | 0.0278 | 0.0251 | 0.2572 | 0.001 Cos | [256,256] | 400 |
| 15 | bernstein_nf | 184.26 | 0.0660 | 0.0636 | 0.0767 | 0.0005 Cos | [256,256] | 400 |

**Winner:** `bernstein_nf_scale` (−205.28 NLL, 9 params/dim) — the simplest NF model: normal(0,1) base + Scale bijector + Bernstein polynomial, totalling 9 parameters per forecast dimension. Top 4 models are within 2.5 NLL points — a **flat optimum**. CI90 values are near-identical (0.023–0.047), suggesting the prediction intervals are inherently narrow on this target regardless of architecture.

**Overfitting signal:** The second-best variant uses only 200 epochs on [128,128] and beats its 400/256 successor. Larger capacity causes regression.

**Scale+Shift impact:** On pl2, adding Shift degrades both `spline_nf_scale` (−204.79 → −155.59, −49.2 NLL) and `bernstein_nf_scale` (−205.28 → −147.72, −57.6 NLL). This is the largest relative degradation across all targets — the high-variance Gaussian-like pl2 distribution is particularly harmed by the extra 48 shift parameters.

### 4.5 Cross-Target Winners Summary

| Target | Winner | \#Params/dim | NLL | CI90 | Architecture Pattern |
|--------|--------|-------------|-----|------|---------------------|
| DLA | spline_nf_truncated | 35 | −197.66 | 0.698 | Spline + TruncatedNormal, constant LR |
| ofen_f_koks | spline_nf_scale_lognormal (trunc) | 24 | −195.93 | 0.023 | Spline + Scale + TruncatedNormal, CosineDecay |
| ofen_g_koks | bernstein_nf_scale_lognormal | 13 | −179.87 | 0.041 | Bernstein + Scale + LogNormal, small capacity |
| pl2 | bernstein_nf_scale | 9 | −205.28 | 0.023 | Bernstein + Scale + Normal, high capacity |

**Each target requires a different model.** The winning models span both spline and Bernstein families, both normal and truncated base distributions, and both LR schedules. The parameter count per dimension ranges from 9 (pl2, simplest) to 35 (DLA, most complex).

### 4.6 Models Without Evaluation Results (Updated post-Phase 8 Evaluation)

All 12 active models have now been evaluated on all 4 targets. Key findings from the Scale+Shift models:

- **Shift bijector degrades performance universally.** On every target and every base model (spline, bernstein), adding the Shift bijector reduces NLL by 6–58 points. The 48 additional shift parameters (1 per forecast dimension) overparameterize the flow and create optimization challenges.
- **Scale + TruncatedNormal NaN issue persists.** Models using Scale + TruncatedNormal(0,5) on non-DLA targets (ofen_g_koks, ofen_f_koks, pl2) produce NLL=inf (NaN at initialization). Only DLA supports this combination. This applies to all 4 affected variants: `spline_nf_scale_truncated`, `spline_nf_scale_shift_truncated`, `bernstein_nf_scale_truncated`, `bernstein_nf_scale_shift_truncated`.
- **On DLA**, the truncated variants produce valid results but rank lower than non-truncated counterparts (ranks 11–15).

## 5. Discussion

### 5.1 Key Architectural Findings

1. **TruncatedNormal(0,5) base was the single most important improvement** (Phase 6). Replacing LogNormal resolved the CI90 explosion across all 48 forecast steps while maintaining or improving joint NLL. On DLA, NLL improved from −189 to −196 (+7 points) and CI90 dropped from 5.14 to 0.13 (40×).

2. **Capacity scaling: bigger is better (up to a point).** Increasing `hidden_units` from [128,128] to [256,256] and `epochs` from 200 to 400 improved NLL by 5–30 points on most models. However, several models (bernstein_nf_scale on pl2, bernstein_nf_scale_lognormal on ofen_g_koks) peaked at lower capacity and regressed — the additional MAF network weights caused overfitting in the 48-dimensional autoregressive structure.

3. **Scale bijector is target-dependent.** The 48 learnable scale parameters (1 per forecast dimension) improve the joint density when the conditional variance is heteroscedastic across forecast steps but hurt when it is near-constant:
   - **Helps:** ofen_g_koks (spline_nf_scale: −170.80 vs spline_nf: −155.17, +15.6 NLL), pl2 (spline_nf_scale: −204.79 vs spline_nf: −199.30, +5.5 NLL)
   - **Hurts:** DLA (spline_nf_scale: −166.40 vs spline_nf: −178.20, −11.8 NLL), ofen_f_koks (spline_nf_scale: −187.81 vs spline_nf: −191.67, −3.9 NLL)

   **Hypothesis:** The scale parameters add flexibility at step level. On targets with non-stationary variance across the 12-hour horizon, this helps. On targets with stable variance, the extra parameters (and their softplus constraint) introduce an optimization challenge that harms the joint density.

4. **Spline vs Bernstein.** RationalQuadraticSpline (35 params/dim) consistently beats BernsteinPolynomial (8 params/dim) when parameter counts are unequal. Adding Scale helps both, but Spline+Scale (36 params/dim) generally beats Bernstein+Scale (9 params/dim) by 5–15 NLL points — except on:
   - **ofen_g_koks:** Bernstein+Scale wins outright (−179.87 vs −170.80)
   - **pl2:** Bernstein+Scale ties with Spline+Scale (−205.28 vs −204.79)

   The 9-param Bernstein+Scale apparently captures the 48-dimensional joint distribution more parsimoniously on these targets, suggesting the true density is smooth enough for low-order Bernstein approximation.

5. **Baselines are surprisingly competitive.** The `truncated_baseline` (diagonal truncated normal, 0 flow parameters) achieves rank 3 on ofen_f_koks (−188.06 NLL, CI90=0.023). The `lognormal_baseline` was best on DLA until Phase 6. This suggests the 48-dimensional joint distribution may be approximately diagonal on some targets — the MAF's autoregressive dependency modeling (which captures cross-step correlations) may not always be necessary.

6. **Bernstein pure (no scale) is unusable.** `bernstein_nf` with order=8 gives NLL +42 to +288 across all targets. The 8 Bernstein coefficients per dimension produce degenerate densities — likely because the domain [0, 1] is too narrow for the Normal(0,1) base.

7. **Shift bijector degrades all models (Phase 8 evaluation).** Adding a Shift bijector (48 location parameters) after the Scale bijector reduces NLL by 6–58 points across all 8 model/target combinations. The MAF's autoregressive structure already captures the conditional mean effectively; the extra location parameters introduce redundancy and optimization difficulty. The Shift bijector is not beneficial for this dataset and should be dropped from future configs.

8. **Scale + TruncatedNormal is unsolvable on non-DLA targets.** The combination of a learnable scale (via softplus) with a truncated_normal(0,5) base produces NaN at initialization on ofen_g_koks, ofen_f_koks, and pl2. The softplus-constrained scale values push the flow output outside the base distribution's [0, 5] support. On DLA, the same combination produces valid results but ranks lower than non-truncated variants. This is a fundamental architectural limitation: use either the Scale bijector with Normal(0,1) base, or the TruncatedNormal base without Scale.

### 5.2 HPO Methodology

- Single-parameter-at-a-time search proved effective. The 8-phase, ~68-iteration process converged to near-optimal configurations.
- The most impactful changes were qualitative (base distribution, Scale bijector) rather than quantitative (LR schedule, nbins).
- The unified config approach (Phase 5) was a useful heuristic but missed target-specific optima. Final per-target winners differ in architecture family, not just hyperparameters.
- Per-dimension parameter count (`num_parameters`) is a useful complexity metric: winners range from 9 (pl2, simplest) to 35 (DLA, most complex).

### 5.3 Open Questions

1. **Why does `spline_nf_truncated` on DLA have CI90=0.698 with constant LR vs CI90=0.093 with CosineDecay, despite nearly identical NLL (−197.66 vs −183.72)?** The joint density is better with constant LR, but the prediction intervals are wider on some steps. Per-step PIT analysis needed.

2. **Why does the Shift bijector consistently degrade all models across all targets?** Adding 48 shift parameters (1 per forecast step) reduces NLL by 6–58 points universally. The MAF's autoregressive structure already captures the conditional mean; the extra Shift layer may interfere with the gradient flow through the chain.

3. **Why does `spline_nf_scale_truncated` fail on non-DLA targets?** NaN at initialization — likely the Scale bijector's softplus constraint pushes values outside the truncated base support [0, 5] for some of the 48 dimensions.

4. **What is the real-world value of NLL differences on pl2?** The top 4 models are within 2.5 points. In practice, per-step CI90 widths and calibration (PIT/QQ) should drive deployment decisions, not raw joint density.

5. **How important is the MAF autoregressive structure?** The `truncated_baseline` (diagonal, no flow) is competitive on ofen_f_koks. An ablation study removing the MAF from NF models would quantify the value of modeling cross-step dependencies.

6. **Can the NaN issue with Scale + TruncatedNormal on non-DLA targets be resolved?** Potential approaches: clip softplus output to reduce extreme scale values, initialize with smaller scales, use a bounded scale transformation, or drop the Scale bijector entirely for truncated-base models on non-DLA targets.

### 5.4 Supplementary Materials

- **LaTeX tables:** `/app/tables.tex` (36 tables: per-model best, winners, all HPO variations)
- **HPO log:** `/app/hpo_study.md` (566 lines, 8 phases)
- **Query script:** `/app/scripts/report_hpo_results.py` (queries MLflow, generates tables)
- **Configs:** `/app/params/models/<target>/<model>.yaml` (48 files)
- **Global params:** `/app/params.yaml`
- **Data:** `/app/data/raw/norm_fondium_15_min_data_2023.csv` (35k rows, 17 columns)
- **MLflow:** Experiments 85 (DLA), 86 (ofen_g_koks), 87 (ofen_f_koks), 88 (pl2)
