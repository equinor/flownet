import copy
import pathlib
from functools import partial
import argparse

import yaml
from hyperopt import fmin, tpe, atpe, rand, STATUS_OK, STATUS_FAIL
import mlflow
from mlflow.entities import RunStatus
import pandas as pd

from ..ahm import run_flownet_history_matching
from ..config_parser._config_parser_hyperparam import (
    create_ahm_config,
    list_hyperparameters_names,
)


def run_flownet_hyperparameter(args: argparse.Namespace, hyperparameters: list):
    """
    Run flownet in hyperparamater exploration or optimization mode.

    Args:
        args: The argparse namespace given by the user
        hyperparameters: Dictionary with hyperparameters

    Returns:
        Nothing

    """
    fmin_objective = partial(flownet_ahm_run, args=args)

    raw_config = yaml.safe_load(args.config.read_text())

    if raw_config["flownet"]["hyperopt"]["mode"] == "tpe":
        algo = tpe.suggest
    elif raw_config["flownet"]["hyperopt"]["mode"] == "adaptive_tpe":
        algo = atpe.suggest
    elif raw_config["flownet"]["hyperopt"]["mode"] == "random":
        algo = rand.suggest
    else:
        raise NotImplementedError(
            f"Hyperopt mode '{args.flownet.hyperopt.mode}' not implemented."
        )

    fmin(
        fn=fmin_objective,
        space=hyperparameters,
        algo=algo,
        max_evals=raw_config["flownet"]["hyperopt"]["n_runs"],
    )


def flownet_ahm_run(x: list, args: argparse.Namespace):
    """
    Run individual ahm using the actual hyperparameter values for the run.

    Args:
        x: Actual values for the hyperparameters.
        args: The argparse namespace given by the user.

    Returns:
        Nothing

    """
    config = create_ahm_config(
        base_config=args.config,
        hyperparameter_values=x,
        update_config=args.update_config,
    )

    mlflow.set_tracking_uri(str(args.output_folder))
    mlflow.set_experiment(f"{config.name}")
    mlflow.start_run(run_name=config.name)

    run_args = copy.deepcopy(args)
    run_args.output_folder = pathlib.Path(
        mlflow.get_artifact_uri().rsplit("artifacts")[0] + "flownet_run"
    )
    try:
        parameters = list_hyperparameters_names(
            yaml.safe_load(args.config.read_text()), []
        )

        for (parameter, param_value) in zip(parameters, x):
            mlflow.log_param(key=parameter, value=param_value)

        run_flownet_history_matching(config, run_args)

        df_analytics = pd.read_csv(
            (run_args.output_folder / config.ert.analysis[0].outfile).with_suffix(
                ".csv"
            )
        ).drop_duplicates()

        hyperopt_loss = 0.0
        for _, row in df_analytics.iterrows():
            for i, metric in enumerate(df_analytics.columns[2:]):
                key = f"{row[0]}_{metric}"
                mlflow.log_metric(
                    key=key.replace(":", "."),
                    value=row[i + 2],
                    step=row[1],
                )

                if (
                    row[1] == df_analytics["iteration"].max()
                    and row[0] in config.flownet.hyperopt.loss.keys
                    and metric == config.flownet.hyperopt.loss.metric
                ):
                    hyperopt_loss += (
                        row[i + 2]
                        * config.flownet.hyperopt.loss.factors[
                            config.flownet.hyperopt.loss.keys.index(row[0])
                        ]
                    )

        mlflow.log_metric("hyperopt_loss", value=hyperopt_loss)

        mlflow.end_run(status=RunStatus.to_string(RunStatus.FINISHED))
        return {"loss": hyperopt_loss, "status": STATUS_OK}

    except Exception as exception:  # pylint: disable=broad-except
        print(exception)
        mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))
        return {"status": STATUS_FAIL}
