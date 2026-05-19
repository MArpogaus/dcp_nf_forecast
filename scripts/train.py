import argparse
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
import yaml
from hybrid_flows.utils.mlflow import (
    log_cfg,
    start_run_with_exception_logging,
)

from dcp_nf_forecast.models import (
    create_bernstein_model,
    create_spline_model,
)
from dcp_nf_forecast.utils import (
    load_data,
    load_model_params,
    setup_logging,
)


def main() -> None:
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
    run_name = f"{args.model}_{args.target_name}"
    results_dir = Path("results") / args.target_name / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_data(processed_dir, "train", args.data_format)
    x_val, y_val = load_data(processed_dir, "val", args.data_format)
    covariate_dim = x_train.shape[1]
    model_cfg = load_model_params(args.target_name, args.model)

    epochs = args.test_epochs if test_mode else args.epochs
    batch_size = args.test_batch_size if test_mode else model_cfg["batch_size"]

    logger.info(
        "Creating %s: dims=%d cov=%d cfg=%s",
        run_name,
        args.prediction_horizon,
        covariate_dim,
        model_cfg,
    )

    with start_run_with_exception_logging(run_name=run_name):
        log_cfg(vars(args) | {"model_params": model_cfg})
        mlflow.tensorflow.autolog(checkpoint_save_weights_only=True)

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
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=model_cfg["learning_rate"]
            ),
            loss=lambda y, p_y: -p_y.log_prob(y),
        )

        logger.info(
            "Training %s: epochs=%d batch_size=%d test_mode=%s",
            run_name,
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

        model.save_weights(str(results_dir / "model_weights.h5"))

        hist = history.history
        min_idx = int(np.argmin(hist["val_loss"]))
        min_loss = float(hist["loss"][min_idx])
        min_val_loss = float(hist["val_loss"][min_idx])

        mlflow.log_metric("best_epoch", min_idx)
        mlflow.log_metric("final_epoch", len(hist["loss"]))
        mlflow.log_metric("min_loss", min_loss)
        mlflow.log_metric("min_val_loss", min_val_loss)

        metrics = {
            "final_train_loss": float(hist["loss"][-1]),
            "final_val_loss": float(hist["val_loss"][-1]),
            "best_epoch": min_idx,
            "min_loss": min_loss,
            "min_val_loss": min_val_loss,
        }
        with open(results_dir / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        mlflow.log_artifacts(str(results_dir))

        logger.info("Saved weights + metrics to %s", results_dir)


if __name__ == "__main__":
    main()
