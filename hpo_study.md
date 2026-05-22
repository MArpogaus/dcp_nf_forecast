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
