# Hyperparameter Optimization Study

## Guiding Principles

- **Only** edit `params.yaml` and `params/models/` — no code changes during HPO (errors excepted)
- Start **very small** and only increase capacity when underfitting is evident
- One commit per meaningful iteration documenting val_loss diffs

---

## Models (all 8 for DLA)

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

## Phase 1 — Constant LR Benchmark

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

**Commands:**
```bash
nohup dvc repro train@dataset0-normal_baseline train@dataset0-lognormal_baseline train@dataset0-bernstein_nf train@dataset0-bernstein_nf_lognormal train@dataset0-bernstein_nf_scale train@dataset0-bernstein_nf_scale_lognormal train@dataset0-spline_nf train@dataset0-spline_nf_lognormal > phase1_training.log 2>&1 &
# Followed by: dvc repro evaluate@dataset0-* (auto-runs after training stages complete)
```

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

**Findings:**
- **spline_nf** (Normal base, [128,128], order=8, nbins=8): best NLL (-161.133) but 3rd best RMSE (0.0813)
- **bernstein_nf_scale_lognormal**: best RMSE (0.0197) and tight 90% CI (width=0.067) — excellent point forecasts
- **spline_nf_lognormal**: near-best NLL but poor RMSE (0.44) and very wide CIs (width=4.42) — overconfident uncertainty
- **lognormal_baseline**: NaN — full-covariance lognormal + Exp bijector unstable with small data
- **Scale bijector helps LogNormal base** (RMSE 0.020 vs 0.044) but **hurts Normal base** (RMSE 0.247 vs 0.178)

**Decision:** Continue Phase 2 with all 8 models. Cosine decay may help models that plateaued early (spline_nf at ep 77, bernstein_nf_scale at ep 58).

---

## Phase 2 — Cosine Decay LR

Switch from constant LR=0.001 to cosine decay (initial_lr=0.001, decay_steps=22000 = 200ep × 110 batches).

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

**Findings:**
- Cosine decay helps baselines (normal: 3.5× better, lognormal: NaN → -165)
- Spline models prefer constant LR (simpler models benefit from fine-tuning)
- Bernstein models are mixed (slight preference for cosine)
- Top-3 NLL: spline_nf_scale(constant) -166.0, lognormal_baseline(cosine) -165.0, spline_nf_scale(cosine) -164.6

**Per-model LR policy for Phase 3:**
- Spline variants, normal baseline → constant LR (0.001)
- Bernstein variants → mixed (pick winner per model)
- Lognormal baseline → cosine decay (fixed NaN)

---

## Phase 3 — Capacity Increase

Apply to top-4 models with increased capacity:

| Model | LR | Changes from Phase 1/2 |
|-------|-----|----------------------|
| spline_nf_v2 | constant 0.001 | nbins=12, params: 23→35 |
| spline_nf_scale_v2 | constant 0.001 | nbins=12, params: 24→36 |
| bernstein_nf_scale_lognormal_v2 | constant 0.001 | order=12, params: 9→13 |
| lognormal_baseline_v2 | cosine decay | [256,256] from [128,128] |

Further steps if underfitting:

| Step | Bernstein order | Spline nbins | Hidden units |
|------|----------------|--------------|--------------|
| 3b | 16 | 16 | [256, 128] |
| 3c | 16 | 16 | [256, 256] |
| 3d | — | — | increase epochs to 400 |

---

## Phase 4 — Export Best Params to All Targets

Once optimal DLA params are found, apply to `ofen_g_koks`, `ofen_f_koks`, `pl2` and run full pipeline. Commit final results.

## DLA Full Run — 2026-05-22 14:49:10 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | RUNNING | -52.7867 |
| lognormal_baseline | PENDING | - |
| bernstein_nf | FINISHED | -142.7449 |
| bernstein_nf_lognormal | PENDING | - |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | RUNNING | -148.4261 |
| spline_nf_lognormal | RUNNING | -148.4261 |
| spline_nf_scale | RUNNING | -128.9820 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 14:51:10 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | RUNNING | -52.7867 |
| lognormal_baseline | PENDING | - |
| bernstein_nf | FINISHED | -142.7449 |
| bernstein_nf_lognormal | PENDING | - |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | RUNNING | -152.4902 |
| spline_nf_lognormal | RUNNING | -152.4902 |
| spline_nf_scale | RUNNING | -128.9820 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 14:53:10 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -53.7762 |
| lognormal_baseline | PENDING | - |
| bernstein_nf | RUNNING | -125.3643 |
| bernstein_nf_lognormal | RUNNING | -125.3643 |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | FINISHED | -153.9878 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | RUNNING | -128.9820 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 14:55:10 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -53.7762 |
| lognormal_baseline | PENDING | - |
| bernstein_nf | RUNNING | -134.8092 |
| bernstein_nf_lognormal | RUNNING | -134.8092 |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | FINISHED | -153.9878 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | RUNNING | -128.9820 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 14:57:10 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -53.7762 |
| lognormal_baseline | PENDING | - |
| bernstein_nf | FINISHED | -135.1432 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | RUNNING | -131.3480 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | RUNNING | -128.9820 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 14:59:10 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -53.7762 |
| lognormal_baseline | PENDING | - |
| bernstein_nf | FINISHED | -135.1432 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | RUNNING | -151.9763 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | RUNNING | -151.9763 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 15:01:11 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -159.9537 |
| lognormal_baseline | FINISHED | -159.9537 |
| bernstein_nf | RUNNING | -80.4127 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | FINISHED | -153.8930 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | FINISHED | -153.8930 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 15:03:11 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -159.9537 |
| lognormal_baseline | FINISHED | -159.9537 |
| bernstein_nf | RUNNING | -81.1339 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | FINISHED | -153.8930 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | FINISHED | -153.8930 |
| spline_nf_scale_lognormal | RUNNING | -128.9820 |


## DLA Full Run — 2026-05-22 15:05:11 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -159.9537 |
| lognormal_baseline | FINISHED | -159.9537 |
| bernstein_nf | FINISHED | -81.1403 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | FINISHED | -142.7449 |
| bernstein_nf_scale_lognormal | PENDING | - |
| spline_nf | RUNNING | -126.3985 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | RUNNING | -126.3985 |
| spline_nf_scale_lognormal | RUNNING | -126.3985 |


## DLA Full Run — 2026-05-22 15:07:11 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -159.9537 |
| lognormal_baseline | FINISHED | -159.9537 |
| bernstein_nf | RUNNING | -141.1975 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | RUNNING | -141.1975 |
| bernstein_nf_scale_lognormal | RUNNING | -141.1975 |
| spline_nf | FINISHED | -126.0304 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | FINISHED | -126.0304 |
| spline_nf_scale_lognormal | FINISHED | -126.0304 |


## DLA Full Run — 2026-05-22 15:09:11 — Status: RUNNING

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -159.9537 |
| lognormal_baseline | FINISHED | -159.9537 |
| bernstein_nf | RUNNING | -145.4636 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | RUNNING | -145.4636 |
| bernstein_nf_scale_lognormal | RUNNING | -145.4636 |
| spline_nf | FINISHED | -126.0304 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | FINISHED | -126.0304 |
| spline_nf_scale_lognormal | FINISHED | -126.0304 |


## DLA Full Run — 2026-05-22 15:11:11 — Status: COMPLETE

| Model | Status | Min Val Loss |
|-------|--------|-------------|
| normal_baseline | FINISHED | -159.9537 |
| lognormal_baseline | FINISHED | -159.9537 |
| bernstein_nf | FINISHED | -145.8355 |
| bernstein_nf_lognormal | FINISHED | -135.1432 |
| bernstein_nf_scale | FINISHED | -145.8355 |
| bernstein_nf_scale_lognormal | FINISHED | -145.8355 |
| spline_nf | FINISHED | -126.0304 |
| spline_nf_lognormal | FINISHED | -153.9878 |
| spline_nf_scale | FINISHED | -126.0304 |
| spline_nf_scale_lognormal | FINISHED | -126.0304 |

