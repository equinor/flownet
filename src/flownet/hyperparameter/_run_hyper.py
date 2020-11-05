import mlflow


def run_flownet_hyperparameter(config, args):
    """
    Run flownet in hyper paramater exploration or optimization mode.

    Args:
        config: A configsuite instance.
        args: The argparse namespace given by the user

    Returns:
        Nothing

    """
    mlflow.set_tracking_uri(str(args.output_folder))
    mlflow.set_experiment(f"{config.name}")

    with mlflow.start_run(run_name=config.name):
        mlflow.log_param("MAE", 1)
