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
dvc repro train@dataset0-normal_baseline train@dataset0-lognormal_baseline train@dataset0-bernstein_nf train@dataset0-bernstein_nf_lognormal train@dataset0-bernstein_nf_scale train@dataset0-bernstein_nf_scale_lognormal train@dataset0-spline_nf train@dataset0-spline_nf_lognormal
dvc repro evaluate@dataset0-normal_baseline evaluate@dataset0-lognormal_baseline evaluate@dataset0-bernstein_nf evaluate@dataset0-bernstein_nf_lognormal evaluate@dataset0-bernstein_nf_scale evaluate@dataset0-bernstein_nf_scale_lognormal evaluate@dataset0-spline_nf evaluate@dataset0-spline_nf_lognormal
```

---

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
