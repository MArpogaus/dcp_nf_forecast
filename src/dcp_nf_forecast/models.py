from hybrid_flows.models import DensityRegressionModel


def create_bernstein_model(
    dims: int,
    covariate_dim: int,
    cfg: dict,
) -> DensityRegressionModel:
    """Build a Bernstein-polynomial normalizing flow model.

    Parameters
    ----------
    dims : int
        Output dimensionality (prediction horizon).
    covariate_dim : int
        Number of conditional input features.
    cfg : dict
        Model configuration (``num_layers``, ``polynomial_order``,
        ``hidden_units``, ``activation``).

    Returns
    -------
    DensityRegressionModel
        Uncompiled Keras model.
    """
    return DensityRegressionModel(
        distribution="masked_autoregressive_flow",
        dims=dims,
        num_layers=cfg["num_layers"],
        num_parameters=cfg["polynomial_order"] * dims,
        bijector="BernsteinPolynomial",
        bijector_kwargs={"domain": [0, 1], "extrapolation": False},
        invert=True,
        base_distribution_kwargs={"distribution_name": "lognormal"},
        parameters_constraint_fn="hybrid_flows.activations.get_thetas_constrain_fn",
        parameters_constraint_fn_kwargs={
            "allow_flexible_bounds": False,
            "bounds": "linear",
            "high": -4,
            "low": 4,
        },
        parameters_fn_kwargs={
            "hidden_units": cfg["hidden_units"],
            "activation": cfg["activation"],
            "conditional": True,
            "conditional_event_shape": [covariate_dim],
        },
    )


def create_spline_model(
    dims: int,
    covariate_dim: int,
    cfg: dict,
) -> DensityRegressionModel:
    """Build a rational-quadratic-spline normalizing flow model.

    Parameters
    ----------
    dims : int
        Output dimensionality (prediction horizon).
    covariate_dim : int
        Number of conditional input features.
    cfg : dict
        Model configuration (``num_layers``, ``num_bins``,
        ``hidden_units``, ``activation``).

    Returns
    -------
    DensityRegressionModel
        Uncompiled Keras model.
    """
    return DensityRegressionModel(
        distribution="masked_autoregressive_flow",
        dims=dims,
        num_layers=cfg["num_layers"],
        num_parameters=cfg["num_bins"] * 3 - 1,
        bijector="RationalQuadraticSpline",
        bijector_kwargs={"range_min": -4},
        base_distribution_kwargs={"distribution_name": "lognormal"},
        parameters_constraint_fn="hybrid_flows.activations.get_spline_param_constrain_fn",
        parameters_constraint_fn_kwargs={
            "interval_width": 8,
            "min_slope": 0.001,
            "min_bin_width": 0.001,
            "nbins": cfg["num_bins"],
        },
        parameters_fn_kwargs={
            "hidden_units": cfg["hidden_units"],
            "activation": cfg["activation"],
            "conditional": True,
            "conditional_event_shape": [covariate_dim],
        },
    )
