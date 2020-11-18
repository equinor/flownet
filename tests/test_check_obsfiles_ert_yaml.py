import pathlib

import os.path

from datetime import datetime

import yaml

ERT_OBS_FILE_NAME = pathlib.Path("./tests/observation_files/observations.ertobs")
YAML_OBS_FILE_NAME = pathlib.Path("./tests/observation_files/observations.yamlobs")

TRAINING_ERT_OBS_FILE_NAME = pathlib.Path(
    "./tests/observation_files/observations_training.ertobs"
)
TEST_ERT_OBS_FILE_NAME = pathlib.Path(
    "./tests/observation_files/observations_test.ertobs"
)

TRAINING_YAML_OBS_FILE_NAME = pathlib.Path(
    "./tests/observation_files/observations_training.yamlobs"
)
TEST_YAML_OBS_FILE_NAME = pathlib.Path(
    "./tests/observation_files/observations_test.yamlobs"
)


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


def compare(ert_obs_dict: dict, yaml_obs_dict: dict) -> bool:
    """This function compares if the given dictionaries: ert_obs_dict and yaml_obs_dict contain the same information.
    Args:
        ert_obs_dict: dictionary that contains the information in a ERT observation file
        yaml_obs_dict: dictionary that contains the information in a YAML observation file
    Returns:
        True: If the ert_obs_dict and yaml_obs_dict contains the same information
              Otherwise function stops by assert functions if both dictionaries have diferent information.
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

    return True


def test_check_obsfiles_ert_yaml() -> None:
    """
    This function checks if the observation files (complete, training, and test) in ERT and YAML version are equal.
       Args:
           None. The file names containig the observation files are harcoded as
       Returns:
           None
    """

    # Comparing the complete observation data
    # Reading ERT file
    ert_obs = read_ert_obs(ERT_OBS_FILE_NAME)

    # Reading YAML file
    parsed_yaml_file = read_yaml_obs(YAML_OBS_FILE_NAME)

    assert compare(ert_obs, parsed_yaml_file)

    # Comparing the training observation data
    # Reading ERT file
    training_ert_obs = read_ert_obs(TRAINING_ERT_OBS_FILE_NAME)

    # Reading YAML file
    training_parsed_yaml_file = read_yaml_obs(TRAINING_YAML_OBS_FILE_NAME)

    assert compare(training_ert_obs, training_parsed_yaml_file)

    # Comparing the Test observation data
    # Reading ERT file
    test_ert_obs = read_ert_obs(TEST_ERT_OBS_FILE_NAME)

    # Reading YAML file
    test_parsed_yaml_file = read_yaml_obs(TEST_YAML_OBS_FILE_NAME)

    assert compare(test_ert_obs, test_parsed_yaml_file)
