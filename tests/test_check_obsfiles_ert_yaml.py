
import pytest

import os.path

def test_method():
    ERT_OBS_FILE = "../output_test/observations.ertobs"
    YAML_OBS_FILE = "../output_test/observations.yamlobs"

    assert os.path.exists(ERT_OBS_FILE) == 1
    assert os.path.exists(YAML_OBS_FILE) == 1
    
    
