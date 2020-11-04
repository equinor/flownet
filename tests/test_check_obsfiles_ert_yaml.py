
import pytest

from datetime import datetime

import os.path

import yaml


def test_method():
    ERT_OBS_FILE = "../output_test/observations.ertobs"
    YAML_OBS_FILE = "../output_test/observations.yamlobs"
    
    
    #data =  yaml.load(YAML_OBS_FILE, Loader=yaml.FullLoader)
    #print(data)
    
    #sorted = yaml.dump(data, sort_keys=True)
    #print(sorted)
    
    a_yaml_file = open(YAML_OBS_FILE,'r')
    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)

    print("YAML type var")

    print(type(parsed_yaml_file))

    a_ert_file = open(ERT_OBS_FILE, 'r')
    
    ert_obs = {}
    text =""
    for line in a_ert_file:
        text = text + line
    a_ert_file.close()

    text = text.replace(" ","")
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
                        ert_obs[dic["KEY"]] = [[],[],[]]
                    ert_obs[dic["KEY"]][0].append(datetime.strptime(dic["DATE"], '%d/%m/%Y').toordinal())
                    ert_obs[dic["KEY"]][1].append(float(dic["VALUE"]))
                    ert_obs[dic["KEY"]][2].append(float(dic["ERROR"]))

    print("ERT type var")
    print(type(ert_obs))
    assert os.path.exists(ERT_OBS_FILE) == 1
    assert os.path.exists(YAML_OBS_FILE) == 1    
    
