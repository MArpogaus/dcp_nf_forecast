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

## Phase 2 — Cosine Decay LR

Same as Phase 1, but switch `learning_rate` scalar → dict:

```yaml
learning_rate:
  scheduler_name: cosine_decay
  scheduler_kwargs:
    initial_learning_rate: 0.001
    decay_steps: 200
```

Compare val_loss vs constant LR for each model.

---

## Phase 3 — Capacity Increase (if underfitting)

Apply only to models that plateau early or have poor val_loss:

| Step | Bernstein order | Spline nbins | Hidden units |
|------|----------------|--------------|--------------|
| 3a | 12 | 12 | [128, 128] |
| 3b | 16 | 16 | [256, 128] |
| 3c | 16 | 16 | [256, 256] |
| 3d | — | — | increase epochs to 400 |

---

## Phase 4 — Export Best Params to All Targets

Once optimal DLA params are found, apply to `ofen_g_koks`, `ofen_f_koks`, `pl2` and run full pipeline. Commit final results.
