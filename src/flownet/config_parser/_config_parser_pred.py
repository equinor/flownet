import os
import pathlib
from typing import Dict, Optional, List, Union

import yaml
import configsuite
from configsuite import types, MetaKeys as MK, ConfigSuite

from ._merge_configs import merge_configs


def create_schema(config_folder: Optional[pathlib.Path] = None) -> Dict:
    """
    Returns a configsuite type schema, where configuration value types are defined, together
    with which values are optional and/or has default values.

    Args:
        _to_abs_path: Use absolute path transformation

    Returns:
        Dictionary to be used as configsuite type schema

    """

    @configsuite.transformation_msg("Tries to convert input to absolute path")
    def _to_abs_path(path: Optional[str]) -> str:
        """
        Helper function for the configsuite. Take in a path as a string and
        attempts to convert it to an absolute path.

        Args:
            path: A relative or absolute path

        Returns:
            Absolute path

        """
        if path is None:
            return ""
        if pathlib.Path(path).is_absolute():
            return str(pathlib.Path(path).resolve())
        return str((config_folder / pathlib.Path(path)).resolve())  # type: ignore

    @configsuite.transformation_msg("Convert string to upper case")
    def _to_upper(input_data: Union[List[str], str]) -> Union[List[str], str]:
        if isinstance(input_data, str):
            return input_data.upper()

        return [x.upper() for x in input_data]

    return {
        MK.Type: types.NamedDict,
        MK.Content: {
            "name": {MK.Type: types.String},
            "ert": {
                MK.Type: types.NamedDict,
                MK.Content: {
                    "runpath": {
                        MK.Type: types.String,
                        MK.Default: "output/runpath/realization-%d/pred",
                    },
                    "pred_schedule_file": {
                        MK.Type: types.String,
                        MK.Transformation: _to_abs_path,
                    },
                    "static_include_files": {
                        MK.Type: types.String,
                        MK.Transformation: _to_abs_path,
                        MK.Default: pathlib.Path(
                            os.path.dirname(os.path.realpath(__file__))
                        )
                        / ".."
                        / "static",
                    },
                    "enspath": {MK.Type: types.String, MK.Default: "output/storage"},
                    "eclbase": {
                        MK.Type: types.String,
                        MK.Default: "./eclipse/model/FLOWNET_REALIZATION",
                    },
                    "realizations": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "num_realizations": {MK.Type: types.Integer},
                            "required_success_percent": {
                                MK.Type: types.Number,
                                MK.Default: 20,
                            },
                            "max_runtime": {MK.Type: types.Integer, MK.Default: 300},
                        },
                    },
                    "queue": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "system": {MK.Type: types.String},
                            "name": {MK.Type: types.String},
                            "server": {MK.Type: types.String, MK.AllowNone: True},
                            "max_running": {MK.Type: types.Integer},
                        },
                    },
                    "ref_sim": {
                        MK.Type: types.String,
                        MK.Transformation: _to_abs_path,
                        MK.Description: "Reference simulation to be used in analysis",
                        MK.AllowNone: True,
                    },
                    "analysis": {
                        MK.Type: types.List,
                        MK.Description: "List of analysis workflows to run.",
                        MK.Content: {
                            MK.Item: {
                                MK.Type: types.NamedDict,
                                MK.Description: "Definitions of the analysis workflow.",
                                MK.Content: {
                                    "metric": {
                                        MK.Type: types.List,
                                        MK.Content: {
                                            MK.Item: {
                                                MK.Type: types.String,
                                                MK.AllowNone: True,
                                            }
                                        },
                                        MK.Transformation: _to_upper,
                                        MK.Description: "List of accuracy metrics to be computed "
                                        "in FlowNet analysis workflow",
                                    },
                                    "quantity": {
                                        MK.Type: types.List,
                                        MK.Content: {
                                            MK.Item: {
                                                MK.Type: types.String,
                                                MK.AllowNone: True,
                                            }
                                        },
                                        MK.Transformation: _to_upper,
                                        MK.Description: "List of summary vectors for which accuracy "
                                        "is to be computed",
                                    },
                                    "start": {
                                        MK.Type: types.Date,
                                        MK.AllowNone: True,
                                        MK.Description: "Start date in YYYY-MM-DD format.",
                                    },
                                    "end": {
                                        MK.Type: types.Date,
                                        MK.AllowNone: True,
                                        MK.Description: "End date in YYYY-MM-DD format.",
                                    },
                                    "outfile": {
                                        MK.Type: types.String,
                                        MK.AllowNone: True,
                                        MK.Description: "The filename of the output of the workflow. "
                                        "In case multiple analysis workflows are run this name should be unique.",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def parse_pred_config(
    base_config: pathlib.Path, update_config: pathlib.Path = None
) -> ConfigSuite.snapshot:
    """
    Takes in path to a yaml configuration file, parses it, populates with default values
    where that is defined and the has not provided his/her own value. Also error checks input
    arguments, and making sure they are of expected type.

    Args:
        base_config: Path to the main configuration file.
        update_config: Optional configuration file with
            key/values to override in main configuration file.

    Returns:
        Parsed config, where values can be extracted like e.g. 'config.ert.queue.system'.

    """
    if update_config is None:
        input_config = yaml.safe_load(base_config.read_text())
    else:
        input_config = merge_configs(
            yaml.safe_load(base_config.read_text()),
            yaml.safe_load(update_config.read_text()),
        )

    suite = ConfigSuite(
        input_config,
        create_schema(config_folder=base_config.parent),
        deduce_required=True,
    )

    if not suite.valid:
        raise ValueError(
            "The configuration is not valid:"
            + ", ".join([error.msg for error in suite.errors])
        )

    config = suite.snapshot

    if config.ert.queue.system.upper() != "LOCAL" and (
        config.ert.queue.name is None or config.ert.queue.server is None
    ):
        raise ValueError(
            "Queue name and server needs to be provided if system is not 'LOCAL'."
        )

    if config.ert.analysis and not config.ert.ref_sim:
        raise ValueError(
            "Path to the folder of a reference simulation (ref_sim), "
            "required for the analytics workflow is missing."
        )

    return config
