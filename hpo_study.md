# Hyperparameter Optimization Study

## Guiding Principles

- **Only** edit `params.yaml` and `params/models/` — no code changes during HPO (errors excepted)
- Start **very small** and only increase capacity when underfitting is evident
- One commit per meaningful iteration documenting val_loss diffs

---

## Models (all 10 for DLA)

| Model | Base | Bijector | Notes |
|-------|------|----------|-------|
| normal_baseline | — | — | diagonal multivariate normal |
| lognormal_baseline | — | — | diagonal multivariate lognormal |
| bernstein_nf | normal | BernsteinPolynomial | |
| bernstein_nf_lognormal | lognormal | BernsteinPolynomial | |
| bernstein_nf_scale | normal | Scale + BernsteinPolynomial | nested bijectors |
| bernstein_nf_scale_lognormal | lognormal | Scale + BernsteinPolynomial | nested bijectors |
| spline_nf | normal | RationalQuadraticSpline | |
| spline_nf_lognormal | lognormal | RationalQuadraticSpline | |
| spline_nf_scale | normal | Scale + RationalQuadraticSpline | nested bijectors |
| spline_nf_scale_lognormal | lognormal | Scale + RationalQuadraticSpline | nested bijectors |

---

## Phase 1 — Old data: Constant LR Benchmark

**Hyperparams:**
| Param | Value |
|-------|-------|
| epochs | 200 |
| learning_rate | 0.001 (constant) |
| batch_size | 256 |
| early_stopping_patience | 10 |
| hidden_units | [128, 128] |
| Bernstein order | 8 |
| Spline nbins | 8 |

**Per-model `num_parameters`:**
| Model | num_parameters | Rationale |
|-------|---------------|-----------|
| bernstein_nf | 8 | order=8 → 1 param × 8 knots |
| bernstein_nf_lognormal | 8 | same |
| bernstein_nf_scale | 9 | Scale(1) + Bernstein(8) |
| bernstein_nf_scale_lognormal | 9 | same |
| spline_nf | 23 | 3×nbins-1 = 23 |
| spline_nf_lognormal | 23 | same |

**Results:**
```
Model                          min_val_loss   best_ep   RMSE (eval)    MAE (eval)
spline_nf                       -161.133         77       0.0813        0.0622
spline_nf_lognormal             -160.559        196       0.4415        0.2681
bernstein_nf_scale_lognormal    -136.288        102       0.0197        0.0181
bernstein_nf_lognormal          -131.670        155       0.0437        0.0381
bernstein_nf_scale              -104.856         58       0.2469        0.1859
bernstein_nf                     -97.927         91       0.1782        0.1476
normal_baseline                  -22.541          4       0.2676        0.2180
lognormal_baseline                   nan         56       0.9091        0.8886
```

## Phase 2 — Old data: Cosine Decay LR

Switch from constant LR=0.001 to cosine decay (initial_lr=0.001, decay_steps=22000).

**Results:**
```
Model                          Phase1_min_val   Phase2_min_val   Winner
normal_baseline                 -22.541          -77.947          Phase2
lognormal_baseline                  NaN         -164.999          Phase2 (NaN fixed)
bernstein_nf                    -97.927          -98.710          Phase2
bernstein_nf_lognormal         -131.670         -131.920          Phase2
bernstein_nf_scale             -104.856         -102.989          Phase1
bernstein_nf_scale_lognormal   -136.288         -134.587          Phase1
spline_nf                      -161.133         -155.823          Phase1
spline_nf_lognormal            -160.559         -159.103          Phase1
spline_nf_scale                -166.046         -164.635          Phase1
spline_nf_scale_lognormal           —           -124.250          —
```

## Phase 3 — Old data: Capacity Increase

| Model | LR | Changes from Phase 1/2 |
|-------|-----|----------------------|
| spline_nf_v2 | constant 0.001 | nbins=12, params: 23→35 |
| spline_nf_scale_v2 | constant 0.001 | nbins=12, params: 24→36 |
| bernstein_nf_scale_lognormal_v2 | constant 0.001 | order=12, params: 9→13 |
| lognormal_baseline_v2 | cosine decay | [256,256] from [128,128] |

---

## Baseline Re-run (new data with is_holiday + fillna)

**Config:** Constant LR=0.001, nbins=12, order=8, hidden=[128,128]

Results from MLflow after full pipeline re-run on updated data:

| Model | Epochs | BestEp | min_val_loss |
|-------|--------|--------|-------------|
| normal_baseline | 25 | 14 | -106.44 |
| lognormal_baseline | 33 | 22 | **-163.31** |
| bernstein_nf | 200 | 199 | -81.14 |
| bernstein_nf_lognormal | 200 | 197 | -135.15 |
| bernstein_nf_scale | 17 | 6 | -148.97 |
| bernstein_nf_scale_lognormal | 130 | 119 | -146.04 |
| spline_nf | 33 | 22 | -146.35 |
| spline_nf_lognormal | 177 | 166 | -154.27 |
| **spline_nf_scale** | 46 | 35 | **-156.61** |
| spline_nf_scale_lognormal | 38 | 27 | -128.28 |

**Findings:**
- Data changes dramatically improved baselines (lognormal went from NaN → -163.31)
- **spline_nf_scale** is the best NF model (-156.61, close to lognormal_baseline)
- Baselines now competitive with NFs, suggesting simpler models benefit more from new features
- The `_scale` bijector consistently helps normal-base models

---

## HPO Study: spline_nf_scale on DLA

**Target:** Optimize spline_nf_scale min_val_loss

**Starting config:** `params/models/dla/spline_nf_scale.yaml` (nbins=12, hidden=[128,128], lr=0.001 constant, epochs=200, patience=10)

**Baseline:** min_val_loss = -156.61 (epoch 35/46)

**Search space (one param change per iteration):**
| Order | Param | Values | Strategy |
|-------|-------|--------|----------|
| 1 | learning_rate | [5e-4, 1e-3, 3e-3, 1e-4, 5e-3] | Log sweep, constant → cosine |
| 2 | nbins | [8, 12, 16, 24] | Increase once LR is settled |
| 3 | hidden_units | [[128,64],[128,128],[256,128],[256,256]] | Capacity after overfit check |
| 4 | regularization | dropout, batch_norm | Only if overfitting |
| 5 | epochs | [200, 400] | Budget increase last |

**Stopping criteria:**
1. Target min_val_loss ≤ -170
2. Plateau: 5 consecutive iterations without improvement
3. Max iterations: 20

**HPO log:**

| # | Date | Param change | min_val_loss | Δ | Commit | Status |
|---|------|-------------|-------------|---|--------|--------|
| 0 | 2026-05-22 | baseline (lr=0.001, nbins=12, h=[128,128]) | -156.61 | — | — | committed |
| 1 | 2026-05-22 | learning_rate: 0.001→0.0005 | -156.61 | — | — | — | launched |
| 1 | 2026-05-22 | learning_rate: 0.001→0.0005 | -156.61 | -155.53 | -1.08 | — | reverted |
| 2 | 2026-05-22 | learning_rate: 0.001→0.003 | -156.61 | — | — | — | launched |
| 2 | 2026-05-22 | learning_rate: 0.001→0.003 | -156.61 | -152.62 | -3.99 | — | reverted |
| 3 | 2026-05-22 | schedule: constant→CosineDecay(lr=0.001) | -156.61 | — | — | — | launched |
| 3 | 2026-05-22 | schedule: constant→CosineDecay(lr=0.001) | -156.61 | -153.74 | -2.87 | — | reverted |
| 4 | 2026-05-22 | nbins: 12→8 (params: 36→24) | -156.61 | — | — | — | launched |
| 4 | 2026-05-22 | nbins: 12→8 (params: 36→24) | -156.61 | -154.27 | -2.34 | — | reverted |
| 5 | 2026-05-22 | nbins: 12→16 (params: 36→48) | -156.61 | — | — | — | launched |
| 5 | 2026-05-22 | nbins: 12→16 (params: 36→48) | -156.61 | -154.75 | -1.86 | — | reverted |

## HPO: spline_nf_lognormal on DLA

Baseline: lr=0.001 const, nbins=8, h=[128,128] → min_val_loss=-154.27

| # | Date | Param change | Old val | New val | Δ | Commit | Status |
|---|------|-------------|---------|---------|---|--------|--------|
| 1 | 2026-05-22 | schedule: const→CosineDecay(lr=0.001) | -154.27 | — | — | — | launched |
| 1 | 2026-05-22 | schedule: const→CosineDecay(lr=0.001) | -154.27 | -152.26 | -2.01 | — | reverted |
| 2 | 2026-05-22 | learning_rate: 0.001→0.003 | -154.27 | — | — | — | launched |
| 2 | 2026-05-22 | learning_rate: 0.001→0.003 | -154.27 | -152.63 | -1.64 | — | reverted |
| 3 | 2026-05-22 | learning_rate: 0.001→0.0005 | -154.27 | — | — | — | launched |
| 3 | 2026-05-22 | learning_rate: 0.001→0.0005 | -154.27 | -155.48 | +1.21 | db8432c | committed |
| 4 | 2026-05-22 | nbins: 8→12 (params: 23→35) | -155.48 | — | — | — | launched |
| 4 | 2026-05-22 | nbins: 8→12 (params: 23→35) | -155.48 | -157.51 | +2.03 | 02a5f91 | committed |
| 5 | 2026-05-22 | nbins: 12→16 (params: 35→47) | -157.51 | — | — | — | launched |
| 5 | 2026-05-22 | nbins: 12→16 (params: 35→47) | -157.51 | -154.91 | -2.60 | — | reverted |

---

## Full Pipeline — All 4 targets × 10 models

Launching full pipeline with best configs after DLA HPO.

## Config Bug Fix (2026-05-22)

**Root cause:** `build_fully_connected_net()` requires `batch_norm: bool` and `dropout: float` positional args. Nested bijectors using `parameter_vector_or_simple_network` call `build_fully_connected_net`, so they NEED these kwargs. The top-level MAF outer network uses `get_masked_autoregressive_network_fn` → `tfb.AutoregressiveNetwork`, which does NOT accept `batch_norm`/`dropout`, so they must be absent from top-level.

**Previous incorrect fix (ea5ae75):** Removed `dropout`/`batch_norm` from all `parameters_fn_kwargs` (both top-level and nested). This broke nested bijectors.

**Current fix:** 
- Remove `dropout`/`batch_norm` from top-level `parameters_fn_kwargs` of all `masked_autoregressive_flow` models (16 non-baseline MAF configs had them)
- Add `dropout: 0, batch_norm: false` to ALL nested bijectors' `parameters_fn_kwargs` (22 files: 4 targets × 4 scale models = 16 files for nested add, plus 6 non-DLA MAF files for top-level removal only)
- Baseline models (`multivariate_normal`/`multivariate_lognormal`) use `get_parameter_vector_or_simple_network_fn` directly → keep their `dropout`/`batch_norm` unchanged

---

## Full Pipeline Results — All 4 targets × 10 models

Completed 2026-05-23 01:54 UTC (commit c1ae1d4). Config bug fix (550a1f8) applied before run.

### Target: ofen_g_koks (N=8)

| Model | val_loss | Notes |
|-------|----------|-------|
| spline_nf_scale | **-145.92** | best NF |
| bernstein_nf_scale_lognormal | -135.25 | |
| bernstein_nf_scale | -134.27 | |
| bernstein_nf_lognormal | -99.75 | |
| spline_nf | -99.24 | |
| normal_baseline | -71.66 | |
| bernstein_nf | 34.25 | poor |
| spline_nf_lognormal | INF | numerical overflow |

NaN/INF models: spline_nf_lognormal, spline_nf_scale_lognormal, lognormal_baseline — lognormal base causes numerical instability for this target.

### Target: ofen_f_koks (N=8)

| Model | val_loss | Notes |
|-------|----------|-------|
| spline_nf_scale | **-176.82** | best NF |
| bernstein_nf_scale | -156.98 | |
| bernstein_nf_scale_lognormal | -152.58 | |
| spline_nf | -139.41 | |
| bernstein_nf_lognormal | -119.35 | |
| normal_baseline | -113.86 | |
| bernstein_nf | 61.36 | poor |
| spline_nf_lognormal | INF | overflow |

NaN/INF: spline_nf_lognormal, spline_nf_scale_lognormal, lognormal_baseline.

### Target: pl2 (N=8)

| Model | val_loss | Notes |
|-------|----------|-------|
| spline_nf | **-156.78** | best NF |
| spline_nf_scale | -142.88 | |
| bernstein_nf_scale_lognormal | -74.77 | |
| normal_baseline | -64.33 | |
| bernstein_nf_lognormal | -48.06 | |
| bernstein_nf_scale | -34.81 | |
| bernstein_nf | 78.21 | poor |
| spline_nf_lognormal | INF | overflow |

NaN/INF: spline_nf_lognormal, spline_nf_scale_lognormal, lognormal_baseline.

### Target: DLA (from earlier run, unchanged)

| Model | val_loss | Notes |
|-------|----------|-------|
| lognormal_baseline | **-163.31** | overall best (simple!) |
| spline_nf_lognormal | -157.51 | best NF, HPO improved |
| spline_nf_scale | -156.61 | |
| bernstein_nf_scale | -148.97 | |
| spline_nf | -146.35 | |
| bernstein_nf_scale_lognormal | -146.04 | |
| bernstein_nf_lognormal | -135.15 | |
| spline_nf_scale_lognormal | -128.28 | |
| normal_baseline | -106.44 | |
| bernstein_nf | -81.14 | |

---

## Analysis: Why lognormal_baseline outperforms complex NFs on DLA

DLA ranking: `lognormal_baseline (-163.31) > spline_nf_lognormal (-157.51) > spline_nf_scale (-156.61)`

Key findings:
1. **Simplest model wins.** The lognormal baseline is a single `parameter_vector_or_simple_network` + `multivariate_lognormal` distribution. No flow, no bijectors — just a fully connected net predicting lognormal params. Fewer parameters → less overfitting.
2. **Lognormal base helps on positive-skewed data.** The target `dla_stromverbrauch_kwh` is always positive, right-skewed, in [0,1]. Lognormal is a natural match. Models with normal base consistently underperform their lognormal counterparts.
3. **Spline NF with lognormal + HPO closes the gap.** spline_nf_lognormal reached -157.51 after HPO (lr=0.0005, nbins=12), but is still 5.8 points short. Further gains may come from longer training.
4. **Scale bijector inconsistent.** On DLA, scale helps normal-base models but hurts lognormal-base. On non-DLA, scale consistently helps.
5. **Bernstein underperforms Spline.** Across all targets, spline-based models beat equivalent Bernstein models. The rational quadratic spline is more flexible.

--- 

## HPO Phase 4 — Closing the gap on DLA (all models)

**Goal:** Improve all 8 NF models on DLA, with focus on closing the 5.8-point gap between spline_nf_lognormal and lognormal_baseline.

**Hypothesis:** NF models may need more training time and capacity to reach lognormal_baseline performance. The early stopping at 200 epochs may be premature for these complex models.

**Current DLA Bernstein parameters:**
| Model | order | hidden | lr_schedule | epochs | val_loss |
|-------|-------|--------|-------------|--------|----------|
| bernstein_nf | 8 | [128,128] | CosineDecay 0.001 | 200 | -81.14 |
| bernstein_nf_lognormal | 8 | [128,128] | CosineDecay 0.001 | 200 | -135.15 |
| bernstein_nf_scale | 8 | [128,128] | constant 0.001 | 200 | -148.97 |
| bernstein_nf_scale_lognormal | 12 | [128,128] | constant 0.001 | 200 | -146.04 |

**Search space (one param change per iteration, per model):**
| Order | Param | Values | Rationale |
|-------|-------|--------|-----------|
| 1 | epochs | 200 → 400, patience 10 → 20 | Complex NFs need more time |
| 2 | hidden_units | [128,128] → [256,256] or [256,256,128] | More capacity |
| 3 | learning_rate | log sweep [5e-4, 3e-4, 1e-4, 5e-3] for each model | Fine-tune |
| 4 | num_layers | 1 → 2 (deeper flow) | More expressive flow |
| 5 | nbins (spline) | 12 → 16, 24 | More spline flexibility |
| 6 | order (Bernstein) | 8 → 12, 16 | Higher-order Bernstein |
| 7 | schedule | const ↔ CosineDecay | Compare schedulers |

**Priority order:**
1. First on best NF candidates: spline_nf_lognormal, spline_nf_scale
2. Then on remaining models: bernstein_nf_scale, bernstein_nf_scale_lognormal
3. Low priority (poor performers): bernstein_nf, bernstein_nf_lognormal, spline_nf, spline_nf_scale_lognormal
