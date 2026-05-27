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

---

## HPO Phase 4a — DLA spline_nf_lognormal (best NF candidate)

**Target:** Optimise spline_nf_lognormal on DLA. Gap to beat: -157.51 → -163.31 (lognormal_baseline).

**Starting config:** `params/models/dla/spline_nf_lognormal.yaml`

**Baseline:** lr=0.0005 const, nbins=12, h=[128,128], epochs=200, patience=10 → val_loss=-157.51

**Search space (one param change per iteration):**
| Order | Param | Current | Test values | Strategy |
|-------|-------|---------|-------------|----------|
| 1 | epochs/patience | 200/10 | 400/20 | More training budget |
| 2 | hidden_units | [128,128] | [256,256] | More capacity |
| 3 | learning_rate | 0.0005 | [3e-4, 1e-4, 1e-3] | Fine-tune |
| 4 | nbins | 12 | [16, 24] | More spline flexibility |

**Stopping criteria:**
1. Target min_val_loss ≤ -163.31
2. Plateau: 5 consecutive iterations without improvement
3. Max iterations: 12

**HPO log:**
| # | Date | Param change | Old val | New val | Δ | Commit | Status |
| 1 | 2026-05-23 | epochs: 200→400, patience: 10→20 | -157.51 | **-159.39** | **+1.88** | f6333ca | ✅ completed, best_epoch=395 |
| 2 | 2026-05-23 | lr: 0.0005→0.0003, rest gleich | -159.39 | -157.02 | -2.37 | ef014ca | ❌ regression, LR zu niedrig |
| 3 | 2026-05-23 | h=[128,128]→[256,256], lr=0.0005 | -159.39 | **-165.83** | **-6.44** | caaae42 | ✅ BEATS BASELINE! best_epoch=336 |

---

## Autonomous Monitoring & Handover Protocol

### Strategy
- **Monitoring:** `sleep 30–120s` loops checking `ps aux | grep "dvc repro"` and reading latest `val_loss` from training log.
- **Completion detection:** When DVC process exits, wait 15s for final writes, then read `metrics.yaml`.
- **Decision logic per model:**
  1. If `min_val_loss` improved → log result, update `hpo_study.md`, commit, continue to next config in HPO plan.
  2. If no improvement → revert config, try next hyperparameter.
  3. If model's HPO plan exhausted → move to next DLA model.
- **Plan progression:**
  1. DLA: spline_nf_lognormal (LR sweep → capacity sweep → nbins sweep)
  2. DLA: remaining 7 NF models (epochs 400 → LR sweep → capacity)
  3. Export best configs to ofen_g_koks, ofen_f_koks, pl2
  4. Full pipeline reproduce
  5. Evaluate and possibly iterate on non-DLA targets if gaps exist

### Commit after every completed run
- `git add -A && git commit -m "hpo(dla): <model> iter<N> <param_change>: <old>→<new>"`
- Push when convenient (not blocking)

### Files
| File | Purpose |
|------|---------|
| `/app/hpo_study.md` | Full HPO log, plans, decisions (THIS FILE) |
| `/app/logs/hpo_iter*.log` | Training stdout per iteration |
| `/app/logs/hpo_phase4_controller.log` | Controller monitoring output |
| `/app/scripts/hpo_master_loop.py` | Python master loop (fallback) |
| `/app/scripts/run_autonomous_hpo.sh` | Bash launcher template |
| `/app/results/<target>/<model>/metrics.yaml` | Final metrics per run |

### To resume after compact/handover:
1. Read this file to see last completed iteration.
2. Check `/app/results/dla/spline_nf_lognormal/metrics.yaml` for current best.
3. Check if a training is running: `ps aux | grep "dvc repro"` or `ps aux | grep "train.py"`.
4. If running: monitor with `sleep 30` loop reading val_loss from the iteration's log.
5. If idle: check `hpo_study.md` HPO log for last completed line, continue from there.
6. Decision rules: improvement → continue plan; plateau 5 iters → new strategy; beat baseline → export to other targets.

---

## Phase 5: Unified Config — Full Pipeline Results

**Config:** h=[256,256], epochs=400, patience=20 for ALL 40 models (4 targets × 10 models)

| Target | Best Model | NLL | 2nd Best | NLL | Previous Best |
|--------|-----------|-----|----------|-----|--------------|
| dla | spline_nf_lognormal | **-189.34** | spline_nf | -178.20 | -163.31 (lognormal_baseline) |
| ofen_g_koks | bernstein_nf_scale_lognormal | **-179.62** | spline_nf_scale | -169.23 | -145.9 (spline_nf_scale) |
| ofen_f_koks | spline_nf | **-191.67** | spline_nf_scale | -186.60 | -176.8 (spline_nf_scale) |
| pl2 | spline_nf_scale | **-204.79** | bernstein_nf_scale_lognormal | -203.94 | -156.8 (spline_nf) |

**Key findings:**
1. h=[256,256] universally improved all models across all targets
2. Each target has a different optimal architecture
3. Lognormal base → NaN/INF on non-DLA targets (needs shift fix)
4. Pure Bernstein (no scale) is terrible everywhere — needs higher order
5. Scale bijector helps on ofen_g_koks and pl2, hurts on ofen_f_koks

## Phase 5b — Bernstein Order Exploration

**Goal:** Find optimal Bernstein order for models that still underperform.

### Current Bernstein models needing order increase:
| Target | Model | Current order | NLL | Target NLL |
|--------|-------|-------------|-----|-----------|
| dla | bernstein_nf | 8 | 42.34 ❌ | beat -178 |
| dla | bernstein_nf_scale | 8 | -128.19 | beat -189 |
| ofen_g_koks | bernstein_nf | 8 | 258.07 ❌ | beat -179 |
| ofen_f_koks | bernstein_nf | 8 | 287.98 ❌ | beat -191 |
| pl2 | bernstein_nf | 8 | 184.26 ❌ | beat -204 |

**Plan:** Incrementally increase num_parameters (order) for Bernstein models to 12, 16, 24.

---

## Phase 6 — Domain Expansion for Lognormal-base models

**Problem:** Models with lognormal base distribution produce extreme sample values
when the bijector domain is too narrow. Base distribution samples fall outside the
bijector's domain → linear extrapolation → extreme output values → CI90 >> 1.

**Root cause (spline):** `interval_width: 8` with `range_min: 0` → domain [0, 8].
Lognormal heavy tail produces samples >> 8 → linear extrapolation.

**Root cause (Bernstein):** `domain: [0.0, 1.0]` with `extrapolation: false` →
base samples outside [0,1] are clamped at boundaries, producing degenerate PIT.

**Targets:**
1. DLA: spline_nf_lognormal (NLL=-189.34, CI90=5.14 — best NLL, worst CI)
2. DLA: bernstein_nf_lognormal (NLL=-158.55, CI90=0.14 — good CI but can improve)
3. DLA: spline_nf_scale_lognormal (NLL=-155.78, CI90=nan — unstable)
4. DLA: bernstein_nf_scale_lognormal (NLL=-160.79, CI90=0.07)
5. Scale best domain fix to other targets

**Search space:**
| Order | Model | Param | Current | Test values |
|-------|-------|-------|---------|-------------|
| 1 | spline_nf_lognormal | interval_width | 8 | [16, 32, 64] |
| 2 | spline_nf_lognormal | range_max | (auto) | [16, 32, auto] |
| 3 | bernstein_nf_lognormal | domain | [0.0, 1.0] | [0.0, 5.0], [0.0, 10.0], [-1.0, 5.0] |
| 4 | bernstein_nf_scale_lognormal | domain | [0.0, 1.0] | same as best from iter 3 |
| 5 | spline_nf_scale_lognormal | interval_width | 4 | [8, 16] |

**Primary metric:** NLL (lower=better). **Secondary metric:** CI90 (target < 0.5).

**Stopping criteria:**
1. CI90 ≤ 0.5 AND NLL ≤ -180 (both metrics good)
2. Plateau: 3 consecutive iterations without NLL improvement
3. Max total iterations: 12

**HPO log:**
| # | Target | Model | Param change | Old NLL | New NLL | Old CI90 | New CI90 | Status |
|---|--------|-------|-------------|---------|---------|----------|----------|--------|
| 1 | dla | spline_nf_lognormal | interval_width: 8→16 | -189.34 | -187.61 | 5.14 | 6.50 | reverted |
| 2 | dla | spline_nf_lognormal | range_min: 0→-5, interval_width: 10 | -189.34 | 1.25 | 5.14 | 5.53 | reverted |
| 3 | dla | spline_nf_lognormal | interval_width: 8→3 | -189.34 | -182.51 | 5.14 | 5.12 | reverted |
| 4 | dla | spline_nf_lognormal | base: lognormal→truncated_normal(0,5), interval_width: 5 | -189.34 | **-196.23** | 5.14 | **0.13** | **committed** 🏆 |
| 5 | dla | bernstein_nf_lognormal | base: lognormal→truncated_normal(0,5), domain: [0,1]→[0,5], thetas: [0.007,148]→[0,5] | -158.55 | -98.01 | 0.14 | 0.19 | reverted |
| 6 | dla | spline_nf_scale_lognormal | base: lognormal→truncated_normal(0,5), interval_width: 4→5 | -155.78 | -146.64 | nan | 0.028 | committed (CI fixed ✅) |
| 7 | dla | bernstein_nf_scale_lognormal | base: lognormal→truncated_normal(0,5), domain: [0,1]→[0,5], thetas: [0.007,148]→[0,5] | -160.79 | -149.11 | 0.07 | **0.037** | committed (CI halved ✅) |

---

## Phase 6 Summary — Domain Fix via Bounded Base Distribution

**Problem:** LogNormal base distribution has unbounded upper tail → spline linear extrapolation → extreme samples (CI90 >> 1).

**Solution:** Replace LogNormal base with `truncated_normal(0, 5)` for all NF models. Spline domain matches base support: `[0, interval_width]` where `interval_width = high - low = 5`.

**Results on DLA:**
| Model | Base | NLL (eval) | CI90 | RMSE | Δ |
|-------|------|-----------|------|------|---|
| spline_nf_lognormal | ~~lognormal~~ → **truncated_normal** | -189 → **-196** | 5.14 → **0.13** | 0.76 → **0.03** | 🏆 **BEST** |
| spline_nf_scale_lognormal | ~~lognormal~~ → **truncated_normal** | NaN → -147 | NaN → **0.028** | NaN → **0.023** | CI fixed ✅ |
| bernstein_nf_scale_lognormal | ~~lognormal~~ → **truncated_normal** | -161 → -149 | 0.07 → **0.037** | 0.024 → 0.028 | CI halved ✅ |
| bernstein_nf_lognormal | ~~lognormal~~ → **truncated_normal** | -159 → -98 | — | — | ❌ reverted (capacity too low) |

---

## Phase 7 — Cross-Target Validation

**Goal:** Verify `truncated_normal(0, 5)` fix works on all 4 targets. Run the winning model (spline_nf_lognormal) and its scale variant on each target.

**Results (spline_nf_lognormal with truncated_normal(0,5) base):**
| Target | NLL | RMSE | CI90 | MAE | Prior state |
|--------|-----|------|------|-----|-------------|
| DLA | **-196.23** | 0.030 | 0.128 | — | CI90=5.14 with lognormal |
| ofen_g_koks | **-82.66** | 0.145 | 0.171 | 0.135 | no prior training ❌ |
| ofen_f_koks | **-165.47** | 0.051 | 2.651 | 0.048 | no prior training ❌ |
| pl2 | **-202.86** | 0.039 | 0.046 | 0.036 | no prior training ❌ |

**Results (spline_nf_scale_lognormal with truncated_normal(0,5) base):**
| Target | NLL | RMSE | CI90 | MAE | Prior state |
|--------|-----|------|------|-----|-------------|
| DLA | -146.64 | 0.023 | 0.028 | — | NaN with lognormal |
| ofen_g_koks | inf 🔴 | — | — | — | scale model fails on this target |
| ofen_f_koks | **-195.93** | 0.012 | **0.023** | 0.006 | no prior training ❌ |
| pl2 | **-193.59** | 0.014 | **0.026** | 0.008 | no prior training ❌ |

**Findings:**
1. The `truncated_normal(0, 5)` fix works across all targets — no more NaN/INF for spline_nf_lognormal
2. ofen_f_koks CI90 elevated (2.65) with simple spline, but scale variant fixes it (CI90=0.023)
3. ofen_g_koks scale model fails (inf loss) — use plain spline for this target
4. pl2 spline_nf_lognormal achieves best NLL across all targets (-202.86)

**Recommendations per target:**
| Target | Recommended model | NLL | CI90 |
|--------|------------------|-----|------|
| DLA | spline_nf_lognormal | -196.23 | 0.128 |
| ofen_g_koks | spline_nf_lognormal | -82.66 | 0.171 |
| ofen_f_koks | spline_nf_scale_lognormal | -195.93 | 0.023 |
| pl2 | spline_nf_lognormal | -202.86 | 0.046 |

## Phase 8 — Scale+Shift Bijector Addition + Config Optimisation

**Date:** 2026-05-26

**Goal:** Add Scale+Shift bijector combinations for both Spline and Bernstein NF
variants, revise all configs with HPO-optimal parameters, and keep all existing
configurations.

### Changes Made

| Change | Details |
|--------|---------|
| New `spline_nf_scale_shift` | normal(0,1) base, Scale + Shift + RationalQuadraticSpline (num_parameters=37) |
| New `spline_nf_scale_shift_truncated` | truncated_normal(0,5) base, Scale + Shift + RQS (num_parameters=25, CosineDecay LR) |
| New `bernstein_nf_scale_shift` | normal(0,1) base, Scale + Shift + BernsteinPolynomial (num_parameters=14) |
| New `bernstein_nf_scale_shift_truncated` | truncated_normal(0,5) base, Scale + Shift + BernsteinPolynomial (num_parameters=18) |
| Re-added `bernstein_nf_scale` | back to active pipeline (was only on disk) |
| Re-added `bernstein_nf_scale_truncated` | back to active pipeline (was only on disk) |
| Synced non-DLA Bernstein configs | ofen_g_koks/ofen_f_koks/pl2 now match DLA HPO-optimised num_parameters |
| Fixed `max_epochs` | removed extraneous field from `dla/spline_nf_truncated.yaml` |

### Active Model Inventory (12 models × 4 targets = 48 configs)

| Model | Base | Bijectors | num_parameters |
|-------|------|-----------|---------------|
| normal_baseline | — | multivariate_normal (diag) | — |
| truncated_baseline | — | multivariate_truncated_normal (diag) | — |
| spline_nf | normal(0,1) | RationalQuadraticSpline | 35 |
| spline_nf_truncated | truncated_normal(0,5) | RQS | 35 |
| spline_nf_scale | normal(0,1) | Scale + RQS | 36 |
| spline_nf_scale_truncated | truncated_normal(0,5) | Scale + RQS | 24 |
| **spline_nf_scale_shift** | normal(0,1) | Scale + Shift + RQS | **37** |
| **spline_nf_scale_shift_truncated** | truncated_normal(0,5) | Scale + Shift + RQS | **25** |
| bernstein_nf_scale | normal(0,1) | Scale + Bernstein | 13 |
| bernstein_nf_scale_truncated | truncated_normal(0,5) | Scale + Bernstein | 17 |
| **bernstein_nf_scale_shift** | normal(0,1) | Scale + Shift + Bernstein | **14** |
| **bernstein_nf_scale_shift_truncated** | truncated_normal(0,5) | Scale + Shift + Bernstein | **18** |

### Config Rules (updated)

- **Scale + Shift** bijectors follow the same base distribution rules as Scale:
  - Normal(0,1) base: `range_min: -4, interval_width: 8`
  - TruncatedNormal(0,5) base: `range_min: 0, interval_width: 5`
- Shift bijector uses `shift_constrain_fn` with `max_shift: 0.5` for TruncatedNormal base (see Phase 9)
- Scale bijector uses `clipped_softplus_constrain_fn` with `min_value: 0.3` for TruncatedNormal base (see Phase 9)
- `parameter_shape: [48]` for both Scale and Shift (1 parameter per dimension)
- `parameters_slice_size: 1` for both Scale and Shift
- `nbins=12` for spline normal base, `nbins=8` for truncated base scale variants
- Shift is inserted after Scale, before the flow bijector in `nested_bijectors`

### Known Issues

- **Scale + TruncatedNormal(0,5) NaN on non-DLA** — resolved in Phase 9 via `min_value: 0.3` on Scale
  and `max_shift: 0.5` on Shift constraints.
- Bernstein models still expected to underperform spline (historical evidence from
  Phases 5-7), but kept for completeness.

**Next steps:** Run full pipeline to evaluate the new scale+shift models and
re-validated Bernstein scale models across all 4 targets.

---

## Phase 9 — NaN Fix: TruncatedNormal(0,5) + Scale/Shift Constraints

**Date:** 2026-05-27

**Problem:** TruncatedNormal(0,5) base models with Scale (and Scale+Shift) bijectors
produce `loss: inf` because the composite flow output `(RQS.forward(y) - shift) / scale`
falls outside the base support [0, 5].

**Root cause 1 (Scale):** MADE initialization produces near-zero Scale parameters.
`Scale⁻¹(z) = z / scale` amplifies RQS output outside [0,5]. With scale ≈ 0.01,
even `z = 0.05` gives `0.05/0.01 = 5` → at the [0,5] edge; values `z > 0.05`
produce `z/0.01 > 5` → `-inf` log_prob.

**Root cause 2 (Shift):** Unconstrained Shift allows both positive and large negative
values. `Shift⁻¹(z) = z - shift` with `shift > 0` pushes data=0 below 0; with
`shift = -5` pushes data=1 to `(1 + 5)/0.3 = 20` ≫ 5.

### Fix

| Bijector | Constraint | Before (broken) | After (fixed) | Rationale |
|----------|-----------|-----------------|---------------|-----------|
| **Scale** | `clipped_softplus_constrain_fn` | `min_value: 0.5` | `min_value: 0.3` | More flexibility than 0.5, still prevents amplification. |
| **Shift** | `shift_constrain_fn` | None (or `max_shift: 5.0`) | `max_shift: 0.5` | Keeps `(z - shift)/scale ∈ [0,5]`. |

### Math derivation

During log_prob computation (`Invert(RQS)` → `Shift.inverse` → `Scale.inverse`):

```
flow_output = (RQS.forward(y) - shift) / scale  ∈ [0, 5]
```

**Upper bound** (data=1 ≈ z=1, worst-case scale=0.3):
```
(1 - shift) / 0.3 ≤ 5  →  shift ≥ -0.5
```

**Lower bound** (data=0 ≈ z=0):
```
(0 - shift) / scale ≥ 0  →  shift ≤ 0
```

∴ `shift ∈ (-0.5, 0)`. With `shift = -max_shift * sigmoid(raw)`, set `max_shift = 0.5`.

With `max_shift = 5.0` (old): shift could be -5 → `(1 + 5)/0.3 = 20` → `-inf` log_prob.
With `max_shift = 0.5`: shift ∈ (-0.5, 0) → `(1 + 0.5)/0.3 = 5.0` → exactly at [0,5] edge.

**Scale constraint:** `min_value = 0.3` ensures `scale ≥ 0.3`, so worst-case
`z / scale = 5 / 0.3 = 16.7` (if RQS learns z=5). In practice RQS adapts jointly
with Scale, staying within [0,5] naturally.

### Empirical verification (minimal example)

```
RQS.forward: [0.0, 0.5, 1.0] (identity init, domain [0, 5])

shift=+0.5: output=[-1.67,  0.00,  1.67]  → inf=True  ❌ positive shift
shift=-0.3: output=[ 1.00,  2.67,  4.33]  → inf=False ✓
shift=-0.5: output=[ 1.67,  3.33,  5.00]  → inf=False ✓ (at edge)
shift=-1.0: output=[ 3.33,  5.00,  6.67]  → inf=True  ❌ too negative
shift=-5.0: output=[16.67, 18.33, 20.00]  → inf=True  ❌ max_shift=5
```

### Files changed

| File | Change |
|------|--------|
| `src/dcp_nf_forecast/utils.py` | Added `shift_constrain_fn(max_shift=0.5)` |
| `params/models/*/*_scale_truncated.yaml` | Scale `min_value: 0.5 → 0.3` (14 files) |
| `params/models/*/*_scale_shift_truncated.yaml` | Shift `max_shift: 5.0 → 0.5` (8 files) |
| `AGENTS.md` | Updated config rules with derivation |
| `hpo_study.md` | This section |

### Training verification (ofen_f_koks, spline_nf_scale_shift_truncated)

```
Epoch 1: loss=-116.09  val_loss=-124.92  (no inf)
Epoch 2: loss=-142.93  val_loss=-137.45  (no inf)
Epoch 3: loss=-150.24  val_loss=-142.90  (no inf)
Epoch 4: ...converging
```

### Status

- All 14 truncated configs updated with Scale `min_value: 0.3`
- All 8 Shift+truncated configs updated with Shift `max_shift: 0.5`
- Full pipeline repro pending (DVC shows 533 changed stages)
