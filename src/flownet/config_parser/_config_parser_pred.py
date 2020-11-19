import os
import pathlib
from typing import Dict

import yaml
import configsuite
from configsuite import types, MetaKeys as MK, ConfigSuite

from ._merge_configs import merge_configs


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
        if path is None:
            return ""
        if pathlib.Path(path).is_absolute():
            return str(pathlib.Path(path).resolve())
        return str((base_config.parent / pathlib.Path(path)).resolve())

    suite = ConfigSuite(
        input_config, create_schema(_to_abs_path=_to_abs_path), deduce_required=True
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
