
import pytest

from datetime import datetime

import os.path

import yaml


def test_method():
    ERT_OBS_FILE = "../output_test/observations.ertobs"
    YAML_OBS_FILE = "../output_test/observations.yamlobs"
    
    
    #Reading ERT file
    assert os.path.exists(ERT_OBS_FILE) == 1
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
    #print(ert_obs)
    print(type(ert_obs))


    #Reading YALM file
    assert os.path.exists(YAML_OBS_FILE) == 1 
    a_yaml_file = open(YAML_OBS_FILE,'r')
    yaml.allow_duplicate_keys = True

    parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)

    print("YAML type var")

    #for item in ert_obs:    
        #print ("{",item, ": ", ert_obs[item][0], " }")
    yaml_obs = {}
    for item in parsed_yaml_file:
        print (item, ": ", type(item))
        for list_item in parsed_yaml_file[item]:
            #print (list_item, ": ", type(list_item))
            print list_item["key"]
            #print ert_obs[list_item["key"]]
            for lost_item in list_item["observations"]:
                #print lost_item
                #print(lost_item["error"])
                #print(lost_item["value"])
                if not list_item["key"] in yaml_obs:
                        yaml_obs[list_item["key"]] = [[],[],[]]
                yaml_obs[list_item["key"]][0].append(lost_item["date"].toordinal())        
                yaml_obs[list_item["key"]][1].append(float(lost_item["value"]))
                yaml_obs[list_item["key"]][2].append(float(lost_item["error"]))
            #assert yaml_obs[list_item["key"]][0] == ert_obs[list_item["key"]][0]
            if yaml_obs[list_item["key"]][0]!=ert_obs[list_item["key"]][0][1:]:
                print("Values Are NOT Equal")
                print("YAML_OBS")
                print(yaml_obs[list_item["key"]][1])
                print("--------------------------------------")
                print("ERT_OBS")
                print(ert_obs[list_item["key"]][1])
                print("--------------------------------------")
                print("--------------------------------------")
                
            if yaml_obs[list_item["key"]][1]!=ert_obs[list_item["key"]][1][1:]:
                print("Values Are NOT Equal")
                print("YAML_OBS")
                print(yaml_obs[list_item["key"]][1])
                print("--------------------------------------")
                print("ERT_OBS")
                print(ert_obs[list_item["key"]][1])
                print("--------------------------------------")
                print("--------------------------------------")

            if yaml_obs[list_item["key"]][2]!=ert_obs[list_item["key"]][2][1:]:
                print("Error Are NOT Equal")
                print("YAML_OBS")
                print(yaml_obs[list_item["key"]][2])
                print("--------------------------------------")
                print("ERT_OBS")
                print(ert_obs[list_item["key"]][2])
                print("--------------------------------------")
                print("--------------------------------------")
            assert yaml_obs[list_item["key"]][0] == ert_obs[list_item["key"]][0][1:]               
            assert yaml_obs[list_item["key"]][1] == ert_obs[list_item["key"]][1][1:]
            assert yaml_obs[list_item["key"]][2] == ert_obs[list_item["key"]][2][1:]

                ##, ": "lost_item["error"])
            ##for second_list_item in parsed_yaml_file[item]:
            ##print list_item["value"]
    #print(yaml_obs)        
    
       
