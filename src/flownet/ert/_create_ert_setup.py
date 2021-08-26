import os
import subprocess
import pathlib
import argparse
import pickle
import shutil
from typing import Optional, List, IO

from configsuite import ConfigSuite
import jinja2
import numpy as np
import pandas as pd

from ._create_synthetic_refcase import create_synthetic_refcase
from ..parameters import Parameter
from ..realization import Schedule


_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)
_TEMPLATE_ENVIRONMENT.globals["isnan"] = np.isnan
_MODULE_FOLDER = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


def resample_schedule_dates(schedule: Schedule, resampling: Optional[str]) -> List:
    """
    Resampling dates based on requested frequency without introducing interpolated dates,
    i.e., keeps nearest existing date

    Args:
        schedule: FlowNet Schedule instance to create observations from.
        resampling: Pandas resampling frequency (str)

    Returns:
        dates: list of resampled dates

    """
    df_dates_schedule = pd.to_datetime(schedule.get_dates())

    if resampling:
        df_dates_resampled = pd.date_range(
            schedule.get_dates()[0],
            schedule.get_dates()[-1],
            freq=resampling,
        )
        df_schedule = pd.DataFrame(
            data=range(len(df_dates_schedule)), index=df_dates_schedule
        )
        idx = np.zeros(len(df_dates_resampled), dtype=int)
        for i, k in enumerate(df_dates_resampled):
            idx[i] = df_schedule.index.get_loc(k, method="nearest")
        dates = [
            d.date() for d in df_schedule.iloc[np.unique(idx)].index.to_pydatetime()
        ]
    else:
        dates = [d.date() for d in df_dates_schedule[1:]]

    return dates


def create_observation_file(
    schedule: Schedule,
    obs_file: pathlib.Path,
    config: ConfigSuite.snapshot,
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
        config: Information from the FlowNet config yaml
        training_set_fraction: Fraction of observations in schedule to use in training set
        yaml: Flag to indicate whether a yaml observation file is to be stored. False means ertobs.

    Returns:
        Nothing

    """
    dates = resample_schedule_dates(schedule, config.flownet.data_source.resampling)

    num_dates = len(dates)
    num_training_dates = round(num_dates * training_set_fraction)

    export_settings = [
        ["_complete", 0, num_dates],
        ["_training", 0, num_training_dates],
        ["_test", num_training_dates + 1, num_dates],
    ]

    file_root = os.path.splitext(obs_file)[0]

    for setting in export_settings:
        if yaml:
            obs_export_type = "yamlobs"
        else:
            obs_export_type = "ertobs"
        export_filename = f"{file_root}{setting[0]}.{obs_export_type}"
        template = _TEMPLATE_ENVIRONMENT.get_template(
            f"observations.{obs_export_type}.jinja2"
        )
        with open(export_filename, "w", encoding="utf8") as fh:
            fh.write(
                template.render(
                    {
                        "dates": dates,
                        "schedule": schedule,
                        "error_config": config.flownet.data_source.vectors,
                        "num_beginning_date": setting[1],
                        "num_end_date": setting[2],
                        "last_training_date": dates[num_training_dates - 1],
                    }
                )
            )


def _create_ert_parameter_file(
    parameters: List[Parameter], output_folder: pathlib.Path
) -> None:
    """
    Takes in the parameters prior distribution as a dataframe,
    and outputs them in an ert parameter definition file

    Args:
        parameters: List with Parameters
        output_folder: Path to the output_folder

    Returns:
        Nothing

    """
    with open(output_folder / "parameters.ertparam", "w", encoding="utf8") as fh:
        index = 0  # Prepend index in parameter name to enforce uniqueness in ERT
        for parameter in parameters:
            for random_variable in parameter.random_variables:
                variable_name = str(index) + "_" + parameter.__class__.__name__.lower()
                fh.write(f"{variable_name} {random_variable.ert_gen_kw}\n")
                index += 1

    # This is an ERT workaround. ERT parameter configuration needs a template file
    # which we don't use. Call the template file "EMPTYFILE" and let be it empty.
    with open(output_folder / "EMPTYFILE", "a", encoding="utf8") as fh:
        fh.close()


def create_ert_setup(  # pylint: disable=too-many-arguments
    args: argparse.Namespace,
    network,
    schedule: Schedule,
    config: ConfigSuite.snapshot,
    parameters=None,
    training_set_fraction: float = 1,
    prediction_setup: bool = False,
):
    """
    Create a ready-to-go ERT setup in output_folder.

    Args:
        schedule: FlowNet Schedule instance to create ERT setup from
        args: Arguments given to FlowNet at execution
        network: FlowNet network instance
        config: Information from the FlowNet config yaml
        parameters: List with parameters (default = None)
        training_set_fraction: Fraction of observations to be used for model training (default = 1)
        prediction_setup: Set to true if it is a prediction run (default = False)

    Returns:
        Nothing

    """

    # Create output folders if they don't exist
    output_folder = pathlib.Path(args.output_folder)
    os.makedirs(output_folder, exist_ok=True)

    if prediction_setup and config.ert.ref_sim:
        path_ref_sim = pathlib.Path(config.ert.ref_sim).resolve()
        mode = "pred"
    elif not prediction_setup:
        mode = "ahm"
        # Derive absolute path to reference simulation case
        if config.flownet.data_source.simulation.input_case:
            path_ref_sim = pathlib.Path(
                config.flownet.data_source.simulation.input_case
            ).resolve()
        else:
            path_ref_sim = pathlib.Path(".").resolve()

    if prediction_setup:
        ert_config_file = output_folder / "pred_config.ert"
        template = _TEMPLATE_ENVIRONMENT.get_template("pred_config.ert.jinja2")
    else:
        ert_config_file = output_folder / "ahm_config.ert"
        template = _TEMPLATE_ENVIRONMENT.get_template("ahm_config.ert.jinja2")

    # Pickle network
    with open(output_folder / "network.pickled", "wb") as fh:  # type: IO
        pickle.dump(network, fh)

    # Pickle schedule
    with open(output_folder / "schedule.pickled", "wb") as fh:
        pickle.dump(schedule, fh)

    with open(output_folder / "parameters.pickled", "wb") as fh:
        pickle.dump(parameters, fh)

    configuration = {
        "pickled_network": output_folder.resolve() / "network.pickled",
        "pickled_schedule": output_folder.resolve() / "schedule.pickled",
        "pickled_parameters": output_folder.resolve() / "parameters.pickled",
        "config": config,
        "random_seed": None,
        "debug": args.debug if hasattr(args, "debug") else False,
        "pred_schedule_file": getattr(config.ert, "pred_schedule_file", None),
    }

    if not prediction_setup:
        configuration.update(
            {
                "random_seed": config.flownet.random_seed,
                "reference_simulation": path_ref_sim,
                "perforation_strategy": config.flownet.perforation_handling_strategy,
            }
        )

    with open(ert_config_file, "w", encoding="utf8") as fh:
        fh.write(template.render(configuration))

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
        getattr(config.ert, "static_include_files")
        if hasattr(config.ert, "static_include_files")
        else config.ert.static_include_files
    )

    shutil.copyfile(
        _MODULE_FOLDER / ".." / "static" / "SAVE_ITERATION_ANALYTICS_WORKFLOW_JOB",
        output_folder / "SAVE_ITERATION_ANALYTICS_WORKFLOW_JOB",
    )

    analytics_workflow_template = _TEMPLATE_ENVIRONMENT.get_template(
        "SAVE_ITERATION_ANALYTICS_WORKFLOW.jinja2"
    )
    if hasattr(config.ert, "analysis"):
        for i, analysis_item in enumerate(config.ert.analysis):
            with open(
                output_folder / f"SAVE_ITERATION_ANALYTICS_WORKFLOW_{i}",
                "w",
                encoding="utf8",
            ) as fh:
                fh.write(
                    analytics_workflow_template.render(
                        {
                            "mode": mode,
                            "reference_simulation": path_ref_sim,
                            "run_path": config.ert.runpath,
                            "ecl_base": config.ert.eclbase,
                            "analysis_start": analysis_item.start,
                            "analysis_end": analysis_item.end,
                            "analysis_quantity": "["
                            + ",".join(list(analysis_item.quantity))
                            + "]",
                            "analysis_metric": "["
                            + ",".join(list(analysis_item.metric))
                            + "]",
                            "analysis_outfile": analysis_item.outfile,
                        }
                    )
                )

    shutil.copyfile(args.config, output_folder / args.config.name)
    with open(
        os.path.join(output_folder, "pipfreeze.output"), "w", encoding="utf8"
    ) as fh:
        subprocess.call(["pip", "freeze"], stdout=fh)

    for section in ["RUNSPEC", "PROPS", "SOLUTION", "SCHEDULE"]:
        static_source_path = pathlib.Path(static_path) / f"{section}.inc"
        if static_source_path.is_file():
            # If there is a static file for this section, for this field, copy it.
            shutil.copyfile(static_source_path, output_folder / f"{section}.inc")
        else:
            # Otherwise create an empty one.
            (output_folder / f"{section}.inc").touch()

    if not prediction_setup:
        if parameters is not None:
            create_observation_file(
                schedule,
                output_folder / "observations.ertobs",
                config,
                training_set_fraction,
            )

            create_observation_file(
                schedule,
                output_folder / "observations.yamlobs",
                config,
                training_set_fraction,
                yaml=True,
            )

        _create_ert_parameter_file(parameters, output_folder)

    create_synthetic_refcase(output_folder / "SYNTHETIC_REFCASE", schedule)
