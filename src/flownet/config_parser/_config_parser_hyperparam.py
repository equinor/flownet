from typing import Tuple, List
import pathlib

from configsuite import ConfigSuite
from hyperopt import hp
from hyperopt.pyll.base import Apply
import yaml

from ._config_parser import create_schema
from ._merge_configs import merge_configs


def create_hyperopt_space(key: str, name: str, values: list) -> Apply:
    """Function to create a hyperopt search space for a single hyperparameter.

    Args:
        key: Key in yaml file for the hyperparameter search space
        name: Name of the search space type
        values: Range of, or choices in the search space

    Raises:
        ValueError: If the search space type does not exists a value error will be raised

    Returns:
        A hyperopt search space for a single hyperparameter.

    """
    if name in ("UNIFORM_CHOICE", "CHOICE"):
        result = hp.choice(key, values)
    elif name == "UNIFORM":
        result = hp.uniform(key, *values)
    else:
        raise ValueError(f"'{name}' is not a supported search space for '{key}'.")

    return result


def list_hyperparameters_names(
    config_dict: dict, hyperparameters: List[str], in_config: str = ""
) -> List[str]:
    """List all hyperparameter names defined in a yaml configuration file.

    Args:
        config_dict: configuration as dictionary
        hyperparameters: list of hyperparameters already found (used for
                         recursive calling of the function.)
        in_config: path in the config file

    Returns:
        Return a list of strings of names of the hyperparameters

    """
    for key, value in config_dict.items():
        if isinstance(value, dict):
            hyperparameters += list_hyperparameters_names(
                value, hyperparameters=[], in_config=f"{key}."
            )
        if isinstance(value, list):
            if value[0] in ["UNIFORM_CHOICE", "UNIFORM", "CHOICE"]:
                hyperparameters.append(f"{in_config}{key}")

    return hyperparameters


def list_hyperparameters(
    config_dict: dict, hyperparameters: list, in_config: str = ""
) -> list:
    """List all hyperparameters defined in a yaml configuration file.

    Args:
        config_dict: configuration as dictionary
        hyperparameters: list of hyperparameters already found (used for
                         recursive calling of the function.)
        in_config: path in the config file

    Returns:
        Return a list of all hyperparameters in the config.

    """
    for key, value in config_dict.items():
        if isinstance(value, dict):
            hyperparameters += list_hyperparameters(
                value, hyperparameters=[], in_config=f"{key}."
            )
        if isinstance(value, list):
            if value[0] in ["UNIFORM_CHOICE", "UNIFORM", "CHOICE"]:
                value = create_hyperopt_space(
                    key=f"{in_config}{key}", name=value[0], values=value[1:]
                )
                hyperparameters.append(value)

    return hyperparameters


def parse_hyperparam_config(base_config: pathlib.Path) -> list:
    """Parse a flownet configuration file for hyperparameter tuning. This function
    will not parse the entire file and check for errors. It will merely extract
    the hyperparameters.

    Args:
        base_config: Path to the hyperparameter config file.

    Returns:
        List of hyperparameters.

    """
    with open(base_config, encoding="utf8") as file:
        hyper_config = yaml.load(file, Loader=yaml.FullLoader)

    return list_hyperparameters(hyper_config, hyperparameters=[])


def update_hyper_config(hyper_dict, hyperparameter_values, i=0) -> Tuple[dict, int]:
    for key, value in hyper_dict.items():
        if isinstance(value, dict):
            value, i = update_hyper_config(value, hyperparameter_values, i=i)
        if isinstance(value, list):
            if value[0] in ["UNIFORM_CHOICE", "UNIFORM", "CHOICE"]:
                hyper_dict[key] = hyperparameter_values[i]
                i += 1

    return hyper_dict, i


def create_ahm_config(
    base_config: pathlib.Path, hyperparameter_values: list, update_config: pathlib.Path
) -> ConfigSuite.snapshot:
    """Create a flownet ahm config file from a hyperparameter config file and
    the known drawn values for the hyperparameters.

    Args:
        base_config: Path to the hyperparameter config file.
        hyperparameter_values: List of actual hyperparameter values to be run.
        update_config: Path to config used for updating, like CI.

    Raises:
        ValueError: If the resulting ConfigSuite is invalid.

    Returns:
        A validated ConfigSuite with filled-in hyperparameter values ready to be
        run in flownet ahm.
    """
    if update_config is None:
        hyper_config = yaml.safe_load(base_config.read_text())
    else:
        hyper_config = merge_configs(
            yaml.safe_load(base_config.read_text()),
            yaml.safe_load(update_config.read_text()),
        )
    hyper_config = update_hyper_config(hyper_config, hyperparameter_values)[0]

    suite = ConfigSuite(
        hyper_config,
        create_schema(config_folder=base_config.parent),
        deduce_required=True,
    )

    if not suite.valid:
        raise ValueError(
            "The configuration is not valid:"
            + ", ".join([error.msg for error in suite.errors])
        )

    return suite.snapshot
