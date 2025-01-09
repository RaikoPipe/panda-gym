
import os
import yaml

from classes.train_config import TrainConfig

from definitions import PROJECT_PATH

# default path options
default_path = f"run_data"
default_model_location = 'files/model.zip'
default_yaml_location = "files/config.yaml"

def get_group_path(group_name):
    return os.path.join(PROJECT_PATH, default_path, group_name, "wandb")

def get_group_model_paths(group_name):
    group_path = get_group_path(group_name)

    model_paths = []

    # walk through ensemble path
    for path in os.listdir(group_path):
        model_paths.append(f"{group_path}/{path}/{default_model_location}")

    return model_paths

def get_group_yaml_paths(group_name):
    group_path = get_group_path(group_name)

    yaml_paths = []

    # walk through ensemble path
    for path in os.listdir(group_path):
        yaml_paths.append(f"{group_path}/{path}/{default_yaml_location}")

    return yaml_paths

def open_yaml(yaml_path):
    with open(yaml_path, 'r') as stream:
        return yaml.safe_load(stream)

def get_train_config_from_yaml(yaml_path):
    configuration = TrainConfig()
    yaml_config = open_yaml(yaml_path)
    # omit wandb version
    for key, value in yaml_config.items():
        if isinstance(value, dict):
            setattr(configuration, key, value["value"])
    return configuration