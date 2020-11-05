import pathlib

from configsuite import ConfigSuite
from hyperopt import hp
import yaml

from ._config_parser import create_schema


def create_hyperopt_space(key: str, name: str, values: list):
    if name in ("UNIFORM_CHOICE", "CHOICE"):
        result = hp.choice(key, values)
    elif name == "UNIFORM":
        result = hp.uniform(key, *values)
    else:
        raise ValueError(f"'{name}' is not a supported search space for '{key}'.")

    return result


def list_hyperparameters(config_dict: dict, hyper_dict: list):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            hyper_dict += list_hyperparameters(value, hyper_dict=[])
        if isinstance(value, list):
            if value[0] in ["UNIFORM_CHOICE", "UNIFORM"]:
                value = create_hyperopt_space(key=key, name=value[0], values=value[1:])
                hyper_dict.append(value)

    return hyper_dict


def parse_hyperparam_config(base_config: pathlib.Path):
    with open(base_config) as file:
        hyper_config = yaml.load(file, Loader=yaml.FullLoader)

    return list_hyperparameters(hyper_config, hyper_dict=[])


def update_hyper_config(hyper_dict, hyperparameter_values, i=0) -> dict:
    for key, value in hyper_dict.items():
        if isinstance(value, dict):
            value, i = update_hyper_config(value, hyperparameter_values, i=i)
        if isinstance(value, list):
            if value[0] in ["UNIFORM_CHOICE", "UNIFORM"]:
                hyper_dict[key] = hyperparameter_values[i]
                i += 1

    return hyper_dict, i


def create_ahm_config(base_config: pathlib.Path, hyperparameter_values: list):
    with open(base_config) as file:
        hyper_config = yaml.load(file, Loader=yaml.FullLoader)
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