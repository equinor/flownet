import os
import pathlib
from typing import Dict

import yaml
import configsuite
from configsuite import types, MetaKeys as MK, ConfigSuite


def create_schema(_to_abs_path) -> Dict:
    """
    Returns a configsuite type schema, where configuration value types are defined, together
    with which values are optional and/or has default values.

    Args:
        _to_abs_path: Use absolute path transformation

    Returns:
        Dictionary to be used as configsuite type schema

    """
    return {
        MK.Type: types.NamedDict,
        MK.Content: {
            "name": {MK.Type: types.String},
            "ert": {
                MK.Type: types.NamedDict,
                MK.Content: {
                    "runpath": {MK.Type: types.String, MK.Required: False},
                    "pred_schedule_file": {
                        MK.Type: types.String,
                        MK.Transformation: _to_abs_path,
                    },
                    "static_include_files": {
                        MK.Type: types.String,
                        MK.Transformation: _to_abs_path,
                        MK.Required: False,
                    },
                    "enspath": {MK.Type: types.String, MK.Required: False},
                    "eclbase": {MK.Type: types.String, MK.Required: False},
                    "realizations": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "num_realizations": {MK.Type: types.Integer},
                            "required_success_percent": {
                                MK.Type: types.Number,
                                MK.Required: False,
                            },
                            "max_runtime": {MK.Type: types.Integer, MK.Required: False},
                        },
                    },
                    "simulator": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "name": {
                                MK.Type: types.String,
                                MK.Required: False,
                                MK.Transformation: lambda name: name.lower(),
                            },
                            "version": {MK.Type: types.String, MK.Required: False},
                        },
                    },
                    "queue": {
                        MK.Type: types.NamedDict,
                        MK.Content: {
                            "system": {MK.Type: types.String},
                            "name": {MK.Type: types.String, MK.Required: False},
                            "server": {MK.Type: types.String, MK.Required: False},
                            "max_running": {MK.Type: types.Integer},
                        },
                    },
                },
            },
        },
    }


DEFAULT_VALUES = {
    "ert": {
        "runpath": "output/runpath/realization-%d/pred",
        "enspath": "output/storage",
        "eclbase": "./eclipse/model/FLOWNET_REALIZATION",
        "static_include_files": pathlib.Path(
            os.path.dirname(os.path.realpath(__file__))
        )
        / ".."
        / "static",
        "realizations": {"max_runtime": 300, "required_success_percent": 20},
        "simulator": {"name": "flow"},
    },
}


def parse_pred_config(configuration_file: pathlib.Path) -> ConfigSuite.snapshot:
    """
    Takes in path to a yaml configuration file, parses it, populates with default values
    where that is defined and the has not provided his/her own value. Also error checks input
    arguments, and making sure they are of expected type.

    Args:
        configuration_file: Path to configuration file.

    Returns:
        Parsed config, where values can be extracted like e.g. 'config.ert.queue.system'.

    """
    input_config = yaml.safe_load(configuration_file.read_text())

    @configsuite.transformation_msg("Tries to convert input to absolute path")
    def _to_abs_path(path: str) -> str:
        """
        Helper function for the configsuite. Take in a path as a string and
        attempts to convert it to an absolute path.

        Args:
            path: A relative or absolute path

        Returns:
            Absolute path

        """
        return str((configuration_file.parent / pathlib.Path(path)).resolve())

    suite = ConfigSuite(
        input_config, create_schema(_to_abs_path=_to_abs_path), layers=(DEFAULT_VALUES,)
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

    return config
