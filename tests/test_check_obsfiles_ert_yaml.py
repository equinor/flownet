import pathlib

import os.path

from datetime import datetime, date

import yaml

from configsuite import ConfigSuite

from flownet.realization import Schedule

from flownet.config_parser import parse_config

#from flownet.ert import _create_observation_file

from flownet.realization._simulation_keywords import Keyword, WCONHIST, WCONINJH

import jinja2

import numpy as np

import pandas as pd

_CONFIG_FILE_NAME = pathlib.Path(
    "/home/manuel/repos/Flownet_October/flownet-testdata/norne_test/config/assisted_history_matching.yml"
)

_PRODUCTION_DATA_FILE_NAME = pathlib.Path(
    "./tests/observation_files/Norne_ProductionData.csv"
)

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

###############################################################################################                   
###############################################################################################                   

    # Load Config
    config = parse_config(_CONFIG_FILE_NAME, None)
    
    #TODO: Create a fake error vector (config.flownet.data_source.simulation.vectors),
    #      it apear that it needs to be named_dict type.
    #vectors = configsuite.config.named_dict
    #print(type(config.flownet.data_source.simulation.vectors))
    #print(config.flownet.data_source.simulation.vectors)
    #print(type(vectors))
    #print(vectors)
###############################################################################################                   


    

    # Load production 
    
    headers = ['date','WOPR','WGPR','WWPR','WBHP','WTHP','WGIR','WWIR','WSTAT','WELL_NAME','PHASE','TYPE','date']           
    df_production_data: pd.DataFrame =  pd.read_csv(_PRODUCTION_DATA_FILE_NAME,
                                                    usecols=headers)
    
    df_production_data["date"] = pd.to_datetime(df_production_data["date"])
    
###############################################################################################                   
###############################################################################################

    #TODO Remormulate Schedules functions: I copied and pasted internal lines in _calculate_wconhist() and _calculate_wconinjh(). 
    #     This functions requires well names and dates already asing by _calculate_compdat()
    #     _calculate_welspecs()
    #     
    
    schedule = Schedule()
    start_date = date(2005,10,1)
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
                    inj_control_mode=schedule._inj_control_mode,
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
                    inj_control_mode=schedule._inj_control_mode,
                )
            )
        elif value["TYPE"] == "OP" and start_date and value["date"] >= start_date:
            schedule.append(
                WCONHIST(
                    date=value["date"],
                    well_name=value["WELL_NAME"],
                    status=value["WSTAT"],
                    prod_control_mode=schedule._prod_control_mode,
                    vfp_table='1*',
                    oil_rate=value["WOPR"],
                    water_rate=value["WWPR"],
                    gas_rate=value["WGPR"],
                    bhp=value["WBHP"],
                    thp=value["WTHP"],
                )
            )
###############################################################################################                   
###############################################################################################                                   
    # TODO We intend to run _create_observation_file but I wasn't able to import the function
    #      
    training_set_fraction = 0.75
    num_dates = len(schedule.get_dates())
    num_training_dates = round(num_dates * training_set_fraction)

    export_settings = [
        ["_complete", 1, num_dates],
        ["_training", 1, num_training_dates],
        ["_test", num_training_dates + 1, num_dates],
    ]



    file_root = pathlib.Path("./tests/observation_files/observations")
    obs_export_types = ["yamlobs", "ertobs"]
    print("WRITING the files")
    for obs_export_type in obs_export_types:
        for setting in export_settings:
            export_filename = f"{file_root}{setting[0]}.{obs_export_type}"
            template = _TEMPLATE_ENVIRONMENT.get_template(
                f"observations.{obs_export_type}.jinja2"
            )
            with open(export_filename, "w") as fh:
                fh.write(
                    template.render(
                        {
                            "schedule": schedule,
                            "error_config": config.flownet.data_source.simulation.vectors,
                            "num_beginning_date": setting[1],
                            "num_end_date": setting[2],
                        }
                    )
                )
                        
###############################################################################################                   

    print("READING the files")

    for setting in export_settings:
        ert_obs_file_name = f"{file_root}{setting[0]}.ertobs"
        yaml_obs_file_name = f"{file_root}{setting[0]}.yamlobs"
        # Comparing the complete observation data
        # Reading ERT file
        ert_obs = read_ert_obs(ert_obs_file_name)
        # Reading YAML file
        parsed_yaml_file = read_yaml_obs(yaml_obs_file_name)
        # Comparing observation data
        assert compare(ert_obs, parsed_yaml_file)
