import pathlib
import os.path
from datetime import datetime, date
import collections

import numpy as np
import pandas as pd
import yaml
import jinja2

from flownet.realization import Schedule
from flownet.ert import create_observation_file
from flownet.realization._simulation_keywords import WCONHIST, WCONINJH

_OBSERVATION_FILES = pathlib.Path("./tests/observation_files")
_PRODUCTION_DATA_FILE_NAME = pathlib.Path(_OBSERVATION_FILES / "ProductionData.csv")
_TRAINING_SET_FRACTION = 0.75

_MIN_ERROR = 10
_REL_ERROR = 0.05

_RESAMPLING = None

_TEMPLATE_ENVIRONMENT = jinja2.Environment(
    loader=jinja2.PackageLoader("flownet", "templates"),
    undefined=jinja2.StrictUndefined,
)
_TEMPLATE_ENVIRONMENT.globals["isnan"] = np.isnan


def read_ert_obs(ert_obs_file_name: pathlib.Path) -> dict:
    """This function reads the content of a ERT observation file and returns the information in a dictionary.

    Args:
        ert_obs_file_name: path to the ERT observation file
    Returns:
        ert_obs: dictionary that contains the information in a ERT observation file
    """
    assert os.path.exists(ert_obs_file_name) == 1
    ert_obs = {}
    text = ""
    with open(ert_obs_file_name, "r") as a_ert_file:
        for line in a_ert_file:
            text = text + line

    text = text.replace(" ", "")
    text = text.split("};")
    for item in text:
        if "SUMMARY_OBSERVATION" in item:
            tmp = item.split("{")[1].split(";")
            dic = {}
            for var in tmp:
                tmp2 = var.split("=")
                if len(tmp2) > 1:
                    dic[tmp2[0]] = tmp2[1]
            if not dic["KEY"] in ert_obs:
                ert_obs[dic["KEY"]] = [[], [], []]
            ert_obs[dic["KEY"]][0].append(
                datetime.strptime(dic["DATE"], "%d/%m/%Y").toordinal()
            )
            ert_obs[dic["KEY"]][1].append(float(dic["VALUE"]))
            ert_obs[dic["KEY"]][2].append(float(dic["ERROR"]))

    return ert_obs


def read_yaml_obs(yaml_obs_file_name: pathlib.Path) -> dict:
    """This function reads the content of a YAML observation file and returns the information in a dictionary.

    Args:
        yaml_obs_file_name: path to the YAML observation file
    Returns:
        dictionary that contains the information in a YAML observation file
    """
    assert os.path.exists(yaml_obs_file_name) == 1
    a_yaml_file = open(yaml_obs_file_name, "r")
    yaml.allow_duplicate_keys = True

    return yaml.load(a_yaml_file, Loader=yaml.FullLoader)


def compare(ert_obs_dict: dict, yaml_obs_dict: dict) -> None:
    """This function compares if the given dictionaries: ert_obs_dict and yaml_obs_dict contain the same information.

    Args:
        ert_obs_dict: dictionary that contains the information in a ERT observation file
        yaml_obs_dict: dictionary that contains the information in a YAML observation file
    Returns:
        None: the function stops by assert functions if both dictionaries have diferent information.
    """
    yaml_obs = {}
    for item in yaml_obs_dict:
        for list_item in yaml_obs_dict[item]:
            for lost_item in list_item["observations"]:
                if not list_item["key"] in yaml_obs:
                    yaml_obs[list_item["key"]] = [[], [], []]
                yaml_obs[list_item["key"]][0].append(lost_item["date"].toordinal())
                yaml_obs[list_item["key"]][1].append(float(lost_item["value"]))
                yaml_obs[list_item["key"]][2].append(float(lost_item["error"]))
            assert yaml_obs[list_item["key"]][0] == ert_obs_dict[list_item["key"]][0]
            assert yaml_obs[list_item["key"]][1] == ert_obs_dict[list_item["key"]][1]
            assert yaml_obs[list_item["key"]][2] == ert_obs_dict[list_item["key"]][2]


def test_check_obsfiles_ert_yaml() -> None:
    """
    This function checks if the observation files (complete, training, and test) in ERT and YAML version are equal.

    Returns:
        Nothing
    """

    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-statements
    # pylint: disable=maybe-no-member
    config = collections.namedtuple("configuration", "flownet")
    config.flownet = collections.namedtuple("flownet", "data_source")
    config.flownet.data_source = collections.namedtuple("data_source", "simulation")
    config.flownet.data_source.simulation = collections.namedtuple(
        "simulation", "vectors"
    )
    config.flownet.data_source.simulation.vectors = collections.namedtuple(
        "vectors", "WTHP"
    )
    config.flownet.data_source.simulation.vectors.WOPR = collections.namedtuple(
        "WOPR", "min_error"
    )
    config.flownet.data_source.simulation.vectors.WOPR.min_error = _MIN_ERROR
    config.flownet.data_source.simulation.vectors.WOPR.rel_error = _REL_ERROR

    config.flownet.data_source.simulation.vectors.WGPR = collections.namedtuple(
        "WGPR", "min_error"
    )
    config.flownet.data_source.simulation.vectors.WGPR.min_error = _MIN_ERROR
    config.flownet.data_source.simulation.vectors.WGPR.rel_error = _REL_ERROR

    config.flownet.data_source.simulation.vectors.WWPR = collections.namedtuple(
        "WWPR", "min_error"
    )
    config.flownet.data_source.simulation.vectors.WWPR.min_error = _MIN_ERROR
    config.flownet.data_source.simulation.vectors.WWPR.rel_error = _REL_ERROR

    config.flownet.data_source.simulation.vectors.WBHP = collections.namedtuple(
        "WBHP", "min_error"
    )
    config.flownet.data_source.simulation.vectors.WBHP.min_error = _MIN_ERROR
    config.flownet.data_source.simulation.vectors.WBHP.rel_error = _REL_ERROR

    config.flownet.data_source.simulation.vectors.WTHP = collections.namedtuple(
        "WTHP", "min_error"
    )
    config.flownet.data_source.simulation.vectors.WTHP.min_error = _MIN_ERROR
    config.flownet.data_source.simulation.vectors.WTHP.rel_error = _REL_ERROR

    config.flownet.data_source.simulation.vectors.WGIR = collections.namedtuple(
        "WGIR", "min_error"
    )
    config.flownet.data_source.simulation.vectors.WGIR.min_error = _MIN_ERROR
    config.flownet.data_source.simulation.vectors.WGIR.rel_error = _REL_ERROR

    config.flownet.data_source.simulation.vectors.WWIR = collections.namedtuple(
        "WWIR", "min_error"
    )
    config.flownet.data_source.simulation.vectors.WWIR.min_error = _MIN_ERROR
    config.flownet.data_source.simulation.vectors.WWIR.rel_error = _REL_ERROR

    config.flownet.data_source.simulation.resampling = _RESAMPLING
    # Load production

    headers = [
        "date",
        "WOPR",
        "WGPR",
        "WWPR",
        "WBHP",
        "WTHP",
        "WGIR",
        "WWIR",
        "WSTAT",
        "WELL_NAME",
        "PHASE",
        "TYPE",
        "date",
    ]
    df_production_data: pd.DataFrame = pd.read_csv(
        _PRODUCTION_DATA_FILE_NAME, usecols=headers
    )

    df_production_data["date"] = pd.to_datetime(df_production_data["date"])

    # Create schedule
    schedule = Schedule()

    # Feed schedule with production data
    start_date = date(2005, 10, 1)
    for _, value in df_production_data.iterrows():
        if value["TYPE"] == "WI" and start_date and value["date"] >= start_date:
            schedule.append(
                WCONINJH(
                    date=value["date"],
                    well_name=value["WELL_NAME"],
                    inj_type="WATER",
                    status=value["WSTAT"],
                    rate=value["WWIR"],
                    bhp=value["WBHP"],
                    thp=value["WTHP"],
                    inj_control_mode="RATE",
                )
            )
        elif value["TYPE"] == "GI" and start_date and value["date"] >= start_date:
            schedule.append(
                WCONINJH(
                    date=value["date"],
                    well_name=value["WELL_NAME"],
                    inj_type="GAS",
                    status=value["WSTAT"],
                    rate=value["WGIR"],
                    bhp=value["WBHP"],
                    thp=value["WTHP"],
                    inj_control_mode="RATE",
                )
            )
        elif value["TYPE"] == "OP" and start_date and value["date"] >= start_date:
            schedule.append(
                WCONHIST(
                    date=value["date"],
                    well_name=value["WELL_NAME"],
                    status=value["WSTAT"],
                    prod_control_mode="RESV",
                    vfp_table="1*",
                    oil_rate=value["WOPR"],
                    water_rate=value["WWPR"],
                    gas_rate=value["WGPR"],
                    bhp=value["WBHP"],
                    thp=value["WTHP"],
                )
            )

    create_observation_file(
        schedule,
        _OBSERVATION_FILES / "observations.ertobs",
        config,
        _TRAINING_SET_FRACTION,
    )

    create_observation_file(
        schedule,
        _OBSERVATION_FILES / "observations.yamlobs",
        config,
        _TRAINING_SET_FRACTION,
        yaml=True,
    )

    dt_schedule = pd.to_datetime(schedule.get_dates())

    # Resampling dates based on requested frequency with
    # no interpolation: keeps nearest existing date
    if config.flownet.data_source.simulation.resampling:
        dt_resampled = pd.date_range(
            schedule.get_dates()[1],
            schedule.get_dates()[-1],
            freq=config.flownet.data_source.simulation.resampling,
        )
        df_schedule = pd.DataFrame(data=range(len(dt_schedule)), index=dt_schedule)
        idx = np.zeros(len(dt_resampled), dtype=int)
        for i, k in enumerate(dt_resampled):
            idx[i] = df_schedule.index.get_loc(k, method="nearest")
        dates = [
            d.date() for d in df_schedule.iloc[np.unique(idx)].index.to_pydatetime()
        ]
    else:
        dates = schedule.get_dates()

    num_dates = len(dates)
    num_training_dates = round(num_dates * _TRAINING_SET_FRACTION)

    if config.flownet.data_source.simulation.resampling:
        export_settings = [
            ["_complete", 0, num_dates],
            ["_training", 0, num_training_dates],
            ["_test", num_training_dates + 1, num_dates],
        ]
    else:
        export_settings = [
            ["_complete", 1, num_dates],
            ["_training", 1, num_training_dates],
            ["_test", num_training_dates + 1, num_dates],
        ]

    file_root = pathlib.Path(_OBSERVATION_FILES / "observations")
    for setting in export_settings:
        ert_obs_file_name = f"{file_root}{setting[0]}.ertobs"
        yaml_obs_file_name = f"{file_root}{setting[0]}.yamlobs"
        # Comparing the complete observation data
        # Reading ERT file
        ert_obs = read_ert_obs(ert_obs_file_name)
        # Reading YAML file
        parsed_yaml_file = read_yaml_obs(yaml_obs_file_name)
        # Comparing observation data
        compare(ert_obs, parsed_yaml_file)
