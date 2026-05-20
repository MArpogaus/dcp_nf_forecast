from hybrid_flows.models import DensityRegressionModel


def build_model(
    dims: int, covariate_dim: int, model_kwargs: dict
) -> DensityRegressionModel:
    model_kwargs = model_kwargs.copy()
    pk = model_kwargs["parameters_fn_kwargs"]
    pk["conditional_event_shape"] = covariate_dim
    for nb in model_kwargs.get("nested_bijectors", []):
        if "parameters_fn_kwargs" in nb and "conditional" in nb["parameters_fn_kwargs"]:
            nb["parameters_fn_kwargs"]["conditional_event_shape"] = covariate_dim
    return DensityRegressionModel(dims=dims, **model_kwargs)
