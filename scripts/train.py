"""Train a normalizing flow model for a given target."""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from hybrid_flows.models import DensityRegressionModel

from dcp_nf_forecast.utils import (
    load_model_params,
    read_dataframe,
    setup_logging,
)


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


def load_data(
    processed_dir: Path,
    split: str,
    data_format: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-processed split data as numpy arrays.

    Parameters
    ----------
    processed_dir : Path
        Directory containing X_{split} and y_{split} files.
    split : str
        Split name (``"train"``, ``"val"``, ``"test"``).
    data_format : str
        ``"feather"`` or ``"csv"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) as float32 arrays.
    """
    x = read_dataframe(processed_dir / f"X_{split}", data_format)
    y = read_dataframe(processed_dir / f"y_{split}", data_format)
    return x.values.astype(np.float32), y.values.astype(np.float32)


def main() -> None:
    """Entry point: parse CLI args, build model, train, save weights + metrics."""
    parser = argparse.ArgumentParser(
        description="Train a normalizing flow forecasting model"
    )
    parser.add_argument("--processed-dir", required=True, type=str)
    parser.add_argument("--target-name", required=True, type=str)
    parser.add_argument(
        "--model", required=True, type=str, choices=["bernstein_nf", "spline_nf"]
    )
    parser.add_argument("--data-format", required=True, type=str)
    parser.add_argument("--prediction-horizon", required=True, type=int)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--early-stopping-patience", required=True, type=int)
    parser.add_argument("--test-mode", required=True, type=str)
    parser.add_argument("--test-epochs", required=True, type=int)
    parser.add_argument("--test-batch-size", required=True, type=int)
    parser.add_argument("--log-level", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
    args = parser.parse_args()

    logger = setup_logging(args.log_level, args.log_file)

    processed_dir = Path(args.processed_dir)
    test_mode = args.test_mode.lower() in ("true", "1", "yes")

    x_train, y_train = load_data(processed_dir, "train", args.data_format)
    x_val, y_val = load_data(processed_dir, "val", args.data_format)
    covariate_dim = x_train.shape[1]

    model_cfg = load_model_params(args.target_name, args.model)

    logger.info(
        "Creating %s_%s: dims=%d cov=%d cfg=%s",
        args.model,
        args.target_name,
        args.prediction_horizon,
        covariate_dim,
        model_cfg,
    )

    if "bernstein" in args.model:
        model = create_bernstein_model(
            dims=args.prediction_horizon,
            covariate_dim=covariate_dim,
            cfg=model_cfg,
        )
    else:
        model = create_spline_model(
            dims=args.prediction_horizon,
            covariate_dim=covariate_dim,
            cfg=model_cfg,
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg["learning_rate"]),
        loss=lambda y, p_y: -p_y.log_prob(y),
    )

    epochs = args.test_epochs if test_mode else args.epochs
    batch_size = args.test_batch_size if test_mode else model_cfg["batch_size"]

    logger.info(
        "Training %s_%s: epochs=%d batch_size=%d test_mode=%s",
        args.model,
        args.target_name,
        epochs,
        batch_size,
        test_mode,
    )

    callbacks: list[tf.keras.callbacks.Callback] = []
    if not test_mode:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=args.early_stopping_patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    results_dir = Path("results") / args.target_name / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(results_dir / "model_weights.h5"))

    metrics: dict[str, float] = {
        "final_train_loss": float(history.history["loss"][-1]),
    }
    if "val_loss" in history.history:
        metrics["final_val_loss"] = float(history.history["val_loss"][-1])

    with open(results_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f)

    logger.info("Saved weights + metrics to %s", results_dir)


if __name__ == "__main__":
    main()
