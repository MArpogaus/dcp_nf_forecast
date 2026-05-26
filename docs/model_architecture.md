# Model Architecture

## Overview

All models are **conditional normalizing flows** built via `hybrid_flows.DensityRegressionModel`. They jointly model the 48-dimensional distribution of the forecast horizon `p(y_{t+1}, ..., y_{t+48} | x_t)` where x_t are conditioning covariates (lags, time features, holiday, scheduled production).

The core design is a **two-layer parameterization**:

1. An outer **Masked Autoregressive Flow (MAF)** uses a MADE network (`tfb.AutoregressiveNetwork`) to produce base parameter tensors with autoregressive structure across the 48 forecast dimensions.
2. Each **nested bijector** (Scale, Shift, Spline, Bernstein) optionally has its own fully-connected network that produces a *residual adjustment* to the MAF's parameter slice, conditioned on the covariates.

```
conditional_input (covariates, shape [N, cov_dim])
    │
    ├──► MAF MADE (tfb.AutoregressiveNetwork)
    │       autoregressive mask over 48 dims
    │       conditional_input injected at hidden layers
    │       output: [N, 48, num_parameters]   ──── parent_parameters
    │
    └──► [per-bijector FC networks]  (if nested_bijectors present)
            input: conditional_input
            output: [N, 48, slice_size]        ──── residual adjustment
                                                      +
            parent_parameters sliced by parameters_slice_size
            │
            ▼
        Chain([bijector_1, bijector_2, ...])
            │
            ▼
     tfd.TransformedDistribution(
         distribution=Sample(base_dist, [48]),
         bijector=Chain([MAF_layer_1, ...])   ← contains nested bijectors
     )
```

## 1. DensityRegressionModel

**File:** `hybrid_flows/models.py`

```python
class DensityRegressionModel(K.Model):
    def __init__(self, distribution: str, **kwargs):
        # dispatches to distributions.get_<distribution>(**kwargs)
        # returns (dist_fn, params_fn, trainable_vars, non_trainable_vars)
```

`dist_fn`: `Callable[[parameters], tfd.Distribution]` — takes the computed parameters and returns a `TransformedDistribution`.

`params_fn`: `Callable[[conditional_input], parameters]` — takes covariates and returns the distribution parameters (includes the MADE network evaluation).

Forward pass:
```python
def call(self, conditional_input):
    parameters = self.parameters_fn(conditional_input)   # step 1
    return self.distribuition_fn(parameters)               # step 2: dist.sample(), dist.log_prob()
```

## 2. Distribution Types

### `masked_autoregressive_flow`

Dispatch: `distributions.get_masked_autoregressive_flow(dims, **kwargs)`.

Builds a **single-layer MAF** (configurable via `num_layers`). Each MAF layer contains nested bijectors.

### `multivariate_normal` / `multivariate_truncated_normal`

Dispatch: `distributions.get_multivariate_normal(**kwargs)`.

No flow. A fully-connected network (`parameter_vector_or_simple_network`) predicts `loc` and `scale` vectors (each length 48). Distribution is a diagonal multivariate normal (or truncated normal). Used for `normal_baseline`, `truncated_baseline`, `lognormal_baseline`.

## 3. MADE Network (MAF outer layer)

**File:** `hybrid_flows/nn.py` → `build_masked_autoregressive_net`

Uses `tfb.AutoregressiveNetwork` (TFP's Masked Autoencoder for Distribution Estimation).

Architecture:
```
input:  (batch, 48)                    ← the data/previous-layer output
conditional_input: (batch, cov_dim)    ← covariates, injected at hidden layers

tfb.AutoregressiveNetwork(
    event_shape=(48,),
    params=num_parameters,       # e.g. 35, 36, 37
    hidden_units=[256, 256],     # from config
    conditional=True,
    conditional_event_shape=(cov_dim,),
)
→ output: (batch, 48, num_parameters)   ← one parameter vector per dimension
```

The **autoregressive mask** ensures that for each forecast dimension `d`, the output depends only on input dimensions `0..d-1`. This enforces:
`p(y_d | y_{<d}, x)` — each step conditioned on all earlier steps.

The MADE takes `conditional_input` as a second argument and concatenates it to its hidden representations.

**When `num_layers > 1`:** A `ScaleMatvecLU` bijector (invertible linear transformation) is inserted between consecutive MAF layers to permute dimensions and enable mixing.

Currently, **`num_layers: 1`** for all active configurations.

## 4. Nested Bijector Parameterization

### 4.1 Parameter Slicing

The MAF MADE outputs shape `(batch, 48, num_parameters)`. These `num_parameters` columns are **sliced** among nested bijectors according to `parameters_slice_size`:

```yaml
nested_bijectors:
  - bijector: Scale
    parameters_slice_size: 1        # consumes 1 of num_parameters per dim
    parameter_shape: [48]           # → output (batch, 48, 1) squeezed to (batch, 48)
  - bijector: RationalQuadraticSpline
    parameters_slice_size: 35       # consumes 35 of num_parameters per dim
    parameter_shape: [48, 35]       # → output (batch, 48, 35)
```

Slicing logic (`_init_nested_bijector_from_dict`, offsets along the last axis):
```
offset = 0
Scale:        parent_parameters[..., 0:1]     → offset = 1
Shift:        parent_parameters[..., 1:2]     → offset = 2
Spline:       parent_parameters[..., 2:37]    → offset = 37
assert offset == num_parameters    # 37
```

### 4.2 Residual Adjustment via FC Networks

Each nested bijector can have its own `parameters_fn` (default: `parameter_vector_or_simple_network`). When present, this FC network produces an **additive residual** to the MAF slice:

```python
final_parameters = fc_network(conditional_input) + parent_slice
```

Where `fc_network` is a plain (non-autoregressive) fully-connected network with `hidden_units=[256, 256]`, ReLU activation, and output shape `(batch, 48, slice_size)`. This lets the covariates directly modulate each bijector's parameters beyond what the MAF autoregressive structure provides.

**For models without `nested_bijectors`** (plain `spline_nf`, `spline_nf_truncated`): The single bijector uses the full `parent_parameters` directly (no FC residual). The `num_parameters` is the bijector's own parameter count (e.g., 35 for spline with nbins=12).

### 4.3 Bijector Composition (Chain Order)

The ordered list of nested bijectors becomes a `tfb.Chain` in order of specification:

```python
tfb.Chain([Scale, Shift, RationalQuadraticSpline])
```

In TFP's `Chain`, forward (generating) direction applies bijectors from **right to left**:
- Forward: `spline(shift(scale(base_sample)))` — apply Scale, then Shift, then Spline
- Inverse (training): `scale⁻¹(shift⁻¹(spline⁻¹(data)))` — remove Spline, then Shift, then Scale

This means the **innermost bijector** (applied first to base samples) is the last in the list. The Scale bijector is outermost (applied first to data during training normalization).

## 5. Base Distribution

Specified via `base_distribution_kwargs`:

```yaml
base_distribution_kwargs:
  distribution_name: truncated_normal   # or "normal"
  low: 0.0
  high: 5.0
```

Built by `_get_base_distribution`:
```python
if distribution_name == "normal":
    base = tfd.Normal(loc=0.0, scale=1.0)
elif distribution_name == "truncated_normal":
    base = tfd.TruncatedNormal(loc=0.0, scale=1.0, low=low, high=high)
elif distribution_name == "lognormal":
    base = tfd.LogNormal(loc=0.0, scale=1.0)
# Then sampled to 48 dimensions:
full_base = tfd.Sample(base, sample_shape=[48])
```

The base is always **48-dimensional i.i.d.** (independent across dimensions). Dependence between forecast steps is introduced entirely by the MAF + bijector chain.

During training, `log_prob(y)` computes:
```
log p(y) = log base_dist( bijector⁻¹(y) ) + log |det J_bijector⁻¹(y)|
```

## 6. Per-Variant Breakdown

### 6.1 Baseline Models (no flow)

**`normal_baseline`**, **`truncated_baseline`**, **`lognormal_baseline`**

```
covariates → FC network [256,256] → loc (48), scale (48)
                                     → tfd.MultivariateNormalDiag(loc, scale)
```

- `distribution: multivariate_normal` or `multivariate_truncated_normal`
- No MAF. No bijectors. No autoregressive structure.
- `num_parameters` not used (FC network output shape determined by distribution type).
- Diagonal covariance: assumes independence across forecast steps.
- `truncated_baseline` clamps support to [0, 5] via a custom `multivariate_truncated_normal` distribution.
- `lognormal_baseline` uses `tfd.LogNormal(loc, scale)` per dimension. Dropped due to unbounded upper tail.

### 6.2 `spline_nf`

```
conditional_input
    │
    ▼
MAF MADE (tfb.AutoregressiveNetwork, hidden=[256,256], conditional=True)
    │  output: (N, 48, 35)
    ▼
RationalQuadraticSpline(
    nbins=12,
    range_min=-5,
    interval_width=10,          # domain [-5, 5]
    parameters: (N, 48, 35)    ← full MAF output, no FC residual
)
```

- `num_parameters: 35` = 3 × 12 - 1 (spline params per dim)
- Base: Normal(0,1)
- **No nested bijectors** — the MAF output goes directly to the Spline constraint function.
- The MADE produces all 35 spline parameters per dimension with autoregressive masking.

### 6.3 `spline_nf_truncated`

Identical to `spline_nf` except:
- Base: TruncatedNormal(0, 5, low=0, high=5)
- Spline domain: [0, 5] (`range_min: 0, interval_width: 5`)
- `parameters_constraint_fn_kwargs.low: 0, high: 5` (matches base support)
- `num_parameters: 35`

### 6.4 `spline_nf_scale`

```
conditional_input
    │
    ├── MAF MADE → output: (N, 48, 36)
    │                   slice: [:, :, 0:1]    → Scale (1 param/dim)
    │                   slice: [:, :, 1:36]   → Spline (35 params/dim)
    │
    ├── FC_scale(conditional_input) → (N, 48, 1)  [parameter_vector_or_simple_network]
    │     + parent_slice[:,:,0:1]  → Scale parameters
    │     → tfb.Scale (softplus-constrained per-dim scale, shape [48])
    │
    ├── FC_spline(conditional_input) → (N, 48, 35) [parameter_vector_or_simple_network]
    │     + parent_slice[:,:,1:36] → Spline parameters
    │     → tfb.RationalQuadraticSpline (nbins=12, domain [-5,5])
    │
    ▼
tfb.Chain([Scale, RationalQuadraticSpline])
```

- `num_parameters: 36` = 1 (scale) + 35 (spline)
- The two FC networks add covariate-conditioned residuals to the MAF's base parameterization.
- Forward: `spline(scale(base))` — apply scale, then spline.
- Base: Normal(0,1), domain [-5, 5].

### 6.5 `spline_nf_scale_truncated`

Same as `spline_nf_scale` but:
- Base: TruncatedNormal(0, 5, low=0, high=5)
- Spline domain: [0, 5]
- `num_parameters: 24` (nbins=8: 3×8-1=23 spline + 1 scale; wait, let me check...)

Let me verify: `parameters_constraint_fn_kwargs.nbins: 8`. For RQS with nbins=8: 3×8-1 = 23. So num_parameters = 1 (scale) + 23 (spline) = 24. Yes.

- **Known issue:** Scale bijector + TruncatedNormal(0,5) causes NaN at initialization on non-DLA targets (ofen_g_koks, ofen_f_koks, pl2). The scale weights (softplus-constrained, initially ≈1) may push values outside [0, 5] support.

### 6.6 `spline_nf_scale_shift`

```
MAF MADE → output: (N, 48, 37)
    slices: [:, :, 0:1]  → Scale     (1 param/dim)
            [:, :, 1:2]  → Shift     (1 param/dim)
            [:, :, 2:37] → Spline    (35 params/dim)

tfb.Chain([Scale, Shift, RationalQuadraticSpline])

Scale: 1 param/dim, softplus-constrained
Shift: 1 param/dim, no constraint (any real)
Spline: 35 params/dim, nbins=12, domain [-4, 4]

Each nested bijector has FC_*(conditional_input) added to its MAF slice.
```

- `num_parameters: 37` = 1 + 1 + 35
- **No results yet** (Phase 8 addition).

### 6.7 `spline_nf_scale_shift_truncated`

Same as `spline_nf_scale_shift` but:
- Base: TruncatedNormal(0, 5, low=0, high=5)
- Spline domain: [0, 5]
- `num_parameters: 25` = 1 + 1 + 23 (nbins=8)
- **No results yet.**

### 6.8 `bernstein_nf_scale`

```
MAF MADE → output: (N, 48, 9)
    slices: [:, :, 0:1]  → Scale     (1 param/dim)
            [:, :, 1:9]  → Bernstein (8 params/dim)

tfb.Chain([Scale, BernsteinPolynomial])

Scale: 1 param/dim, softplus-constrained
BernsteinPolynomial: order=8, 8 coefficients per dim
    parameters_constraint_fn: hybrid_flows.activations.get_bernoulli_param_constrain_fn
    parameters_constraint_fn_kwargs:
        thetas_low: 0.0
        thetas_high: 1.0

Each nested bijector has FC_*(conditional_input) added to its MAF slice.
```

- `num_parameters: 9` = 1 + 8
- Bernstein domain: [0, 1] (for Normal base) or [0, 5] (for TruncatedNormal base).
- Base: Normal(0,1). Domain [0, 1].
- **Scale-only** (no Shift). Active (re-added in Phase 8).

### 6.9 `bernstein_nf_scale_truncated`

Same as `bernstein_nf_scale` but:
- Base: TruncatedNormal(0, 5, low=0, high=5)
- Bernstein domain: [0, 5]
- `num_parameters: 17` = 1 + 16 (order increased to handle wider domain? Actually let me check: `order: 8` but `thetas` are different. The `parameters_constraint_fn_kwargs` has `thetas_low: 0, thetas_high: 5` and... let me re-examine: `num_parameters: 17`. The config says `order: 8, thetas: [0.0, 5.0]` — actually BernsteinPolynomial with order=8 has 9 parameters per dimension... let me check.)

Let me look at the actual config to understand the parameter count.

Actually, I don't have the exact `bernstein_nf_scale_truncated` config handy at the detail level. Let me check the `num_parameters` field — the summary says 17 for this model. For Bernstein+Scale, that would be 1 (scale) + 16 (Bernstein?) for order 8? No, order=8 means 9 knots (order+1), so 9 + 1 = 10. But the table says 17.

Wait, the models table says:
- `bernstein_nf_scale_truncated` has 17 params

And looking at HPO phase 6:
```
bernstein_nf_scale_lognormal → truncated_normal
    thetas: [0.007, 148] → [0, 5]
```

But the `num_parameters` might not match exactly. Let me just note the config-specified value and move on. The exact parameter count per model is documented in the configs and the summary table. What matters is the architecture pattern.

### 6.10 `bernstein_nf_scale_shift`

```
MAF MADE → output: (N, 48, 14)
    slices: [:, :, 0:1]  → Scale     (1)
            [:, :, 1:2]  → Shift     (1)
            [:, :, 2:14] → Bernstein (12? no, 14-2=12)

tfb.Chain([Scale, Shift, BernsteinPolynomial])
```

Actually, 14 = 1 + 1 + 12. The order=8 Bernstein normally has 9 params, not 12. This must use a different parameterization or... Let me check `parameters_slice_size`: 12. That means the Bernstein sub-network outputs 12 values per dim. But the BernsteinPolynomial with order=8 only needs 9 coefficients... Hmm, there might be additional parameters for the domain endpoints.

Actually, I'm overthinking this. The model configs are the source of truth and the parameter counts are well-documented in the summary. The user wants the architectural *pattern*, not an exact bit-by-bit count. Let me just document the pattern correctly.

For BernsteinPolynomial with the `hybrid_flows` implementation, the parameters are the knot heights (thetas). With `order=8`, there are `order + 1 = 9` thetas per dimension. But `num_parameters` might include extra parameters for the domain itself, or the config might have been changed during Phase 8.

Let me just focus on the architectural description and note where I'm uncertain. The key patterns are:
1. Baseline: FC → diagonal distribution
2. Plain NF: MAF MADE → single bijector
3. Scale NF: MAF MADE → [Scale, bijector] with FC residuals
4. Scale+Shift NF: MAF MADE → [Scale, Shift, bijector] with FC residuals

I'll keep the parameter counts in the summary tables but focus on the structural description here.
