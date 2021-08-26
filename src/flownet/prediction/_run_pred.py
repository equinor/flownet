import pickle
import shutil

import jinja2

from ..ert import create_ert_setup, run_ert_subprocess

_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)


def run_flownet_prediction(config, args):
    """
    Create prediction ERT setup, and runs it.

    Args:
        config: A configsuite instance.
        args: The argparse namespace given by the user

    Returns:
        Nothing

    """
    with open(args.ahm_folder / "network.pickled", "rb") as fh:
        network = pickle.load(fh)

    with open(args.ahm_folder / "schedule.pickled", "rb") as fh:
        schedule = pickle.load(fh)

    with open(args.ahm_folder / "parameters.pickled", "rb") as fh:
        parameters = pickle.load(fh)

    create_ert_setup(
        args,
        network,
        schedule,
        config,
        parameters,
        prediction_setup=True,
    )
    shutil.copyfile(
        args.ahm_folder / "parameters_iteration-latest.parquet.gzip",
        args.output_folder / "parameters.parquet",
    )

    with open(args.output_folder / "webviz_config.yml", "w", encoding="utf8") as fh:
        fh.write(
            _TEMPLATE_ENVIRONMENT.get_template("webviz_pred_config.yml.jinja2").render(
                {"runpath": config.ert.runpath}
            )
        )

    run_ert_subprocess(
        "ert ensemble_experiment pred_config.ert",
        cwd=args.output_folder,
        runpath=config.ert.runpath,
        timeout=config.ert.timeout,
    )
