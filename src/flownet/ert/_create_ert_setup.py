import os
import pathlib
import argparse
import pickle
import shutil
from typing import List

import jinja2
import numpy as np

from ._create_synthetic_refcase import create_synthetic_refcase
from ..parameters import Parameter
from ..realization import Schedule


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)
_TEMPLATE_ENVIRONMENT.globals["isnan"] = np.isnan
_MODULE_FOLDER = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


def _create_observation_file(
    schedule: Schedule,
    obs_file: pathlib.Path,
    training_set_fraction: float = 1,
    yaml: bool = False,
):
    # pylint: disable=no-self-use
    """
    Creates an ERT observation file from a given schedule instance.
    It has not yet been decided if schedule is to be given at creation or should be part
    of self.

    Args:
        schedule: FlowNet Schedule instance to create observations from.
        obs_file: Path to store the observation file.
        training_set_fraction: Fraction of observations in schedule to use in training set
        yaml: Flag to indicate whether a yaml observation file is to be stored. False means ertobs.

    Returns:
        Nothing

    """
    num_training_dates = round(len(schedule.get_dates()) * training_set_fraction)

    if yaml:
        template = _TEMPLATE_ENVIRONMENT.get_template("observations.yamlobs.jinja2")
        with open(obs_file, "w") as fh:
            fh.write(template.render({"schedule": schedule}))
    else:
        template = _TEMPLATE_ENVIRONMENT.get_template("observations.ertobs.jinja2")
        with open(obs_file, "w") as fh:
            fh.write(
                template.render(
                    {"schedule": schedule, "num_training_dates": num_training_dates}
                )
            )


def _create_ert_parameter_file(
    parameters: List[Parameter], output_folder: pathlib.Path
) -> None:
    """
    Takes in the parameters prior distribution as a dataframe,
    and outputs them in an ert parameter definition file

    Args:
        parameters: List with Paratemers
        output_folder: Path to the output_folder

    Returns:
        Nothing

    """
    with open(output_folder / "parameters.ertparam", "w") as fh:
        index = 0  # Prepend index in parameter name to enforce uniqueness in ERT
        for parameter in parameters:
            for random_variable in parameter.random_variables:
                variable_name = str(index) + "_" + parameter.__class__.__name__.lower()
                fh.write(f"{variable_name} {random_variable.ert_gen_kw}\n")
                index += 1

    # This is an ERT workaround. ERT parameter configuration needs a template file
    # which we don't use. Call the template file "EMPTYFILE" and let be it empty.
    open(output_folder / "EMPTYFILE", "a").close()


def create_ert_setup(  # pylint: disable=too-many-arguments
    args: argparse.Namespace,
    network,
    schedule: Schedule,
    ert_config: dict,
    parameters=None,
    random_seed=None,
    perforation_strategy: str = None,
    reference_simulation: str = None,
    training_set_fraction: float = 1,
    prediction_setup: bool = False,
):
    """
        Create a ready-to-go ERT setup in output_folder.

        Args:
            schedule: FlowNet Schedule instance to create ERT setup from
            args: Arguments given to FlowNet at execution
            training_set_fraction: Fraction of observations to be used for model training (default = 1)

        Returns:
            Nothing

        """

    # Create output folders if they don't exist
    output_folder = pathlib.Path(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Derive absolute path to reference simulation case
    if reference_simulation:
        path_ref_sim = pathlib.Path(reference_simulation).resolve()
    else:
        path_ref_sim = pathlib.Path(".").resolve()

    if prediction_setup:
        ert_config_file = output_folder / "pred_config.ert"
        template = _TEMPLATE_ENVIRONMENT.get_template("pred_config.ert.jinja2")
    else:
        ert_config_file = output_folder / "ahm_config.ert"
        template = _TEMPLATE_ENVIRONMENT.get_template("ahm_config.ert.jinja2")

    # Pickle network
    with open(output_folder / "network.pickled", "wb") as fh:
        pickle.dump(network, fh)

    # Pickle schedule
    with open(output_folder / "schedule.pickled", "wb") as fh:
        pickle.dump(schedule, fh)

    with open(output_folder / "parameters.pickled", "wb") as fh:
        pickle.dump(parameters, fh)

    with open(ert_config_file, "w") as fh:  # type: ignore[assignment]
        fh.write(  # type: ignore[call-overload]
            template.render(
                {
                    "pickled_network": output_folder.resolve() / "network.pickled",
                    "pickled_schedule": output_folder.resolve() / "schedule.pickled",
                    "pickled_parameters": output_folder.resolve()
                    / "parameters.pickled",
                    "random_seed": random_seed,
                    "perforation_strategy": perforation_strategy,
                    "reference_simulation": path_ref_sim,
                    "ert_config": ert_config,
                    "debug": args.debug if hasattr(args, "debug") else False,
                    "pred_schedule_file": getattr(
                        ert_config, "pred_schedule_file", None
                    ),
                }
            )
        )

    shutil.copyfile(
        _MODULE_FOLDER / ".." / "static" / "CREATE_FLOWNET_MODEL",
        output_folder / "CREATE_FLOWNET_MODEL",
    )

    shutil.copyfile(
        _MODULE_FOLDER / ".." / "static" / "DELETE_IN_CURRENT_ITERATION",
        output_folder / "DELETE_IN_CURRENT_ITERATION",
    )

    shutil.copyfile(
        _MODULE_FOLDER / ".." / "static" / "SAVE_ITERATION_PARAMETERS_WORKFLOW",
        output_folder / "SAVE_ITERATION_PARAMETERS_WORKFLOW",
    )

    shutil.copyfile(
        _MODULE_FOLDER / ".." / "static" / "SAVE_ITERATION_PARAMETERS_WORKFLOW_JOB",
        output_folder / "SAVE_ITERATION_PARAMETERS_WORKFLOW_JOB",
    )

    static_path = (
        getattr(ert_config, "static_include_files")
        if hasattr(ert_config, "static_include_files")
        else ert_config["static_include_files"]
    )

    shutil.copyfile(
        _MODULE_FOLDER / ".." / "static" / "SAVE_ITERATION_ANALYTICS_WORKFLOW",
        output_folder / "SAVE_ITERATION_ANALYTICS_WORKFLOW",
    )

    shutil.copyfile(
        _MODULE_FOLDER / ".." / "static" / "SAVE_ITERATION_ANALYTICS_WORKFLOW_JOB",
        output_folder / "SAVE_ITERATION_ANALYTICS_WORKFLOW_JOB",
    )

    for section in ["RUNSPEC", "PROPS", "SOLUTION", "SCHEDULE"]:
        static_source_path = pathlib.Path(static_path) / f"{section}.inc"
        if static_source_path.is_file():
            # If there is a static file for this section, for this field, copy it.
            shutil.copyfile(static_source_path, output_folder / f"{section}.inc")
        else:
            # Otherwise create an empty one.
            (output_folder / f"{section}.inc").touch()

    if parameters is not None:
        _create_observation_file(
            schedule, output_folder / "observations.ertobs", training_set_fraction,
        )

        _create_observation_file(
            schedule, output_folder / "observations.yamlobs", yaml=True
        )

        _create_ert_parameter_file(parameters, output_folder)

    create_synthetic_refcase(output_folder / "SYNTHETIC_REFCASE", schedule)
