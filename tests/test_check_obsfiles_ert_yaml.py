
import pytest

import os.path

import yaml


def test_method():
    ERT_OBS_FILE = "../output_test/observations.ertobs"
    YAML_OBS_FILE = "../output_test/observations.yamlobs"
    print("Vergacion")
    
    #data =  yaml.load(YAML_OBS_FILE, Loader=yaml.FullLoader)
    #print(data)
    
    #sorted = yaml.dump(data, sort_keys=True)
    #print(sorted)
    
    a_yaml_file = open(YAML_OBS_FILE)
    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)

    print(parsed_yaml_file)

        
    assert os.path.exists(ERT_OBS_FILE) == 1
    assert os.path.exists(YAML_OBS_FILE) == 1        
