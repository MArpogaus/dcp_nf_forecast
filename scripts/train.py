import argparse
from pathlib import Path

import dvc.api
import mlflow
import numpy as np
import tensorflow as tf
import yaml
from hybrid_flows.models import DensityRegressionModel
from hybrid_flows.utils.mlflow import (
    log_cfg,
    start_run_with_exception_logging,
)

from dcp_nf_forecast.utils import load_data, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a normalizing flow forecasting model"
    )
    parser.add_argument("--processed-dir", required=True, type=str)
    parser.add_argument("--target-name", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--data-format", required=True, type=str)
    parser.add_argument("--prediction-horizon", required=True, type=int)
    parser.add_argument("--stage-name", required=True, type=str)
    parser.add_argument("--experiment-name", required=True, type=str)
    parser.add_argument("--test-mode", required=True, type=str)
    parser.add_argument("--log-level", required=True, type=str)
    parser.add_argument("--log-file", required=True, type=str)
    args = parser.parse_args()

    logger = setup_logging(args.log_level, args.log_file)
    test_mode = args.test_mode.lower() in ("true", "1", "yes")
    run_name = f"{args.model}_{args.target_name}"
    results_dir = Path("results") / args.target_name / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = args.experiment_name + ("_test" if test_mode else "")

    params = dvc.api.params_show(stages=args.stage_name)
    prefix = f"params/models/{args.target_name}/{args.model}.yaml:"
    compile_kwargs = params.get(f"{prefix}compile_kwargs", {})
    fit_kwargs = params[f"{prefix}fit_kwargs"]
    model_kwargs = params[f"{prefix}model_kwargs"]

    if test_mode:
        fit_kwargs["epochs"] = 1

    x_train, y_train = load_data(Path(args.processed_dir), "train", args.data_format)
    x_val, y_val = load_data(Path(args.processed_dir), "val", args.data_format)
    covariate_dim = x_train.shape[1]

    dims = args.prediction_horizon

    pk = model_kwargs["parameters_fn_kwargs"]
    pk["conditional_event_shape"] = covariate_dim
    model_kwargs["parameters_fn_kwargs"] = pk

    logger.info(
        "Creating %s: dims=%d cov=%d model_kwargs=%s",
        run_name,
        dims,
        covariate_dim,
        model_kwargs,
    )

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id_file = results_dir / "parent_run_id.txt"
        with open(parent_run_id_file, "w") as f:
            f.write(parent_run.info.run_id)

        with start_run_with_exception_logging(run_name=f"{run_name}_training"):
            mlflow.set_tag("stage", "training")
            mlflow.log_dict(params, "dvc_params.yaml")
            log_cfg(
                vars(args)
                | {
                    f"{prefix}model_kwargs": model_kwargs,
                    f"{prefix}fit_kwargs": fit_kwargs,
                    f"{prefix}compile_kwargs": compile_kwargs,
                }
            )
            mlflow.tensorflow.autolog(checkpoint_save_weights_only=True)

            model = DensityRegressionModel(dims=dims, **model_kwargs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=fit_kwargs["learning_rate"]
                ),
                loss=lambda y, p_y: -p_y.log_prob(y),
                **compile_kwargs,
            )

            logger.info(
                "Training %s: epochs=%d batch_size=%d test_mode=%s",
                run_name,
                fit_kwargs["epochs"],
                fit_kwargs["batch_size"],
                test_mode,
            )

            callbacks: list[tf.keras.callbacks.Callback] = []
            patience = fit_kwargs["early_stopping_patience"]
            if not test_mode and patience > 0:
                callbacks.append(
                    tf.keras.callbacks.EarlyStopping(
                        patience=patience,
                        restore_best_weights=True,
                    )
                )

            history = model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_val, y_val),
                epochs=fit_kwargs["epochs"],
                batch_size=fit_kwargs["batch_size"],
                callbacks=callbacks,
                verbose=int(fit_kwargs["verbose"]),
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
