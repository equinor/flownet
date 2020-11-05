import pathlib
from functools import partial
import argparse

from hyperopt import fmin, tpe
import mlflow

from ..ahm import run_flownet_history_matching
from ..config_parser._config_parser_hyperparam import create_ahm_config


def run_flownet_hyperparameter(
    args: argparse.Namespace, hyperparameters: list, n_runs=1
):
    """
    Run flownet in hyper paramater exploration or optimization mode.

    Args:
        args: The argparse namespace given by the user
        hyperparameters: Dictionary with hyper parameters


    Returns:
        Nothing

    """
    fmin_objective = partial(flownet_ahm_run, args=args)
    fmin(fn=fmin_objective, space=hyperparameters, algo=tpe.suggest, max_evals=n_runs)


def flownet_ahm_run(x: list, args: argparse.Namespace):

    config = create_ahm_config(base_config=args.config, hyperparameter_values=x)

    mlflow.set_tracking_uri(str(args.output_folder))
    mlflow.set_experiment(f"{config.name}")
    with mlflow.start_run(run_name=config.name):
        args.output_folder = pathlib.Path(
            mlflow.get_artifact_uri().rsplit("artifacts")[0] + "flownet_run"
        )
        run_flownet_history_matching(config, args)

        # Here the metric needs to be read....!
        mlflow.log_param("MAE", 1)
