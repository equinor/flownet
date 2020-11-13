from datetime import datetime

import os.path

import yaml


def read_ert_obs(ert_obs_file_name: str) -> dict:
    # Reading ERT file
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


def read_yaml_obs(yaml_obs_file_name: str) -> dict:
    # Reading YALM file
    assert os.path.exists(yaml_obs_file_name) == 1
    a_yaml_file = open(yaml_obs_file_name, "r")
    yaml.allow_duplicate_keys = True

    return yaml.load(a_yaml_file, Loader=yaml.FullLoader)


def compare_and_show(ert_obs_dict: dict, yaml_obs_dict: dict) -> None:
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
            if yaml_obs[list_item["key"]][0] != ert_obs_dict[list_item["key"]][0]:
                print("Values Are NOT Equal")
                print("YAML_OBS")
                print(yaml_obs[list_item["key"]][0])
                print("--------------------------------------")
                print("ERT_OBS")
                print(ert_obs_dict[list_item["key"]][0])
                print("--------------------------------------")
                print("--------------------------------------")
                equal = False

            if yaml_obs[list_item["key"]][1] != ert_obs_dict[list_item["key"]][1]:
                print("Values Are NOT Equal")
                print("YAML_OBS")
                print(yaml_obs[list_item["key"]][1])
                print("--------------------------------------")
                print("ERT_OBS")
                print(ert_obs_dict[list_item["key"]][1])
                print("--------------------------------------")
                print("--------------------------------------")
                equal = False

            if yaml_obs[list_item["key"]][2] != ert_obs_dict[list_item["key"]][2]:
                print("Error Are NOT Equal")
                print("YAML_OBS")
                print(yaml_obs[list_item["key"]][2])
                print("--------------------------------------")
                print("ERT_OBS")
                print(ert_obs_dict[list_item["key"]][2])
                print("--------------------------------------")
                print("--------------------------------------")
                equal = False


def compare(ert_obs_dict: dict, yaml_obs_dict: dict) -> bool:
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


def test_check_obsfiles_ert_yaml() -> None:
    ERT_OBS_FILE_NAME = "./tests/observation_files/observations.ertobs"
    YAML_OBS_FILE_NAME = "./tests/observation_files/observations.yamlobs"

    TRAINING_ERT_OBS_FILE_NAME = (
        "./tests/observation_files/observations_training.ertobs"
    )
    TEST_ERT_OBS_FILE_NAME = "./tests/observation_files/observations_test.ertobs"

    TRAINING_YAML_OBS_FILE_NAME = (
        "./tests/observation_files/observations_training.yamlobs"
    )
    TEST_YAML_OBS_FILE_NAME = "./tests/observation_files/observations_test.yamlobs"

    # Comparing the complete observation data
    # Reading ERT file
    ert_obs = read_ert_obs(ERT_OBS_FILE_NAME)

    # Reading YAML file
    parsed_yaml_file = read_yaml_obs(YAML_OBS_FILE_NAME)

    compare_and_show(ert_obs, parsed_yaml_file)
    assert compare(ert_obs, parsed_yaml_file)

    # Comparing the training observation data
    # Reading ERT file
    training_ert_obs = read_ert_obs(TRAINING_ERT_OBS_FILE_NAME)

    # Reading YAML file
    training_parsed_yaml_file = read_yaml_obs(TRAINING_YAML_OBS_FILE_NAME)

    compare_and_show(training_ert_obs, training_parsed_yaml_file)
    assert compare(training_ert_obs, training_parsed_yaml_file)

    # Comparing the Test observation data
    # Reading ERT file
    test_ert_obs = read_ert_obs(TEST_ERT_OBS_FILE_NAME)

    # Reading YAML file
    test_parsed_yaml_file = read_yaml_obs(TEST_YAML_OBS_FILE_NAME)

    compare_and_show(test_ert_obs, test_parsed_yaml_file)
    assert compare(test_ert_obs, test_parsed_yaml_file)
