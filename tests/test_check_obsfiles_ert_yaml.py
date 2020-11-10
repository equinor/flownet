from datetime import datetime

import os.path

# import pytest

import yaml


def read_ert_obs(ert_obs_file_name):
    # Reading ERT file
    assert os.path.exists(ert_obs_file_name) == 1
    a_ert_file = open(ert_obs_file_name, "r")
    ert_obs = {}
    text = ""
    for line in a_ert_file:
        text = text + line
    a_ert_file.close()

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


def read_yaml_obs(yaml_obs_file_name):
    # Reading YALM file
    assert os.path.exists(yaml_obs_file_name) == 1
    a_yaml_file = open(yaml_obs_file_name, "r")
    yaml.allow_duplicate_keys = True

    return yaml.load(a_yaml_file, Loader=yaml.FullLoader)


def compare(ert_obs_dict, yaml_obs_dict):
    yaml_obs = {}
    equal = True
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
            return equal


def test_check_obsfiles_ert_yaml():
    ert_obs_file_name = "./tests/observation_files/observations.ertobs"
    yaml_obs_file_name = "./tests/observation_files/observations.yamlobs"

    training_obs_file_name = "./tests/observation_files/observations_training.ertobs"
    test_ert_obs_file_name = "./tests/observation_files/observations_test.ertobs"

    training_yaml_obs_file_name = (
        "./tests/observation_files/observations_training.yamlobs"
    )
    test_yaml_obs_file_name = "./tests/observation_files/observations_test.yamlobs"

    # Comparing the complete observation data
    # Reading ERT file
    ert_obs = read_ert_obs(ert_obs_file_name)

    # Reading YAML file
    parsed_yaml_file = read_yaml_obs(yaml_obs_file_name)

    assert compare(ert_obs, parsed_yaml_file)

    # Comparing the training observation data
    # Reading ERT file
    ert_obs = read_ert_obs(training_obs_file_name)

    # Reading YAML file
    parsed_yaml_file = read_yaml_obs(training_yaml_obs_file_name)

    assert compare(ert_obs, parsed_yaml_file)

    # Comparing the Test observation data
    # Reading ERT file
    ert_obs = read_ert_obs(test_ert_obs_file_name)

    # Reading YAML file
    parsed_yaml_file = read_yaml_obs(test_yaml_obs_file_name)

    assert compare(ert_obs, parsed_yaml_file)
