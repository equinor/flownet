import os
import pathlib
from datetime import datetime
import yaml


def _read_ert_obs(ert_obs_file_name: pathlib.Path) -> dict:
    """This function reads the content of a ERT observation file and returns the information in a dictionary.

    Args:
        ert_obs_file_name: path to the ERT observation file

    Returns:
        ert_obs: dictionary that contains the information in a ERT observation file
    """
    assert os.path.exists(ert_obs_file_name) == 1
    ert_obs: dict = {}
    text = ""
    with open(ert_obs_file_name, "r", encoding="utf8") as a_ert_file:
        for line in a_ert_file:
            text = text + line

    for item in text.replace(" ", "").split("};"):
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
                datetime.strptime(dic["DATE"], "%d/%m/%Y").date()
            )
            ert_obs[dic["KEY"]][1].append(float(dic["VALUE"]))
            ert_obs[dic["KEY"]][2].append(float(dic["ERROR"]))

    return ert_obs


def _read_yaml_obs(yaml_obs_file_name: pathlib.Path) -> dict:
    """This function reads the content of a YAML observation file and returns the information in a dictionary.

    Args:
        yaml_obs_file_name: path to the YAML observation file

    Returns:
        dictionary that contains the information in a YAML observation file
    """
    assert os.path.exists(yaml_obs_file_name) == 1
    with open(yaml_obs_file_name, "r", encoding="utf8") as a_yaml_file:
        return yaml.load(a_yaml_file, Loader=yaml.FullLoader)
