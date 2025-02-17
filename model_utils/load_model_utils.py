
import os
import yaml

from classes.train_config import TrainConfig

from definitions import PROJECT_PATH

# default path options
default_path = f"run_data"
default_model_location = 'files/model.zip'
default_replay_buffer_location = 'files/replay_buffer.pkl'
default_yaml_location = "files/config.yaml"

def get_group_path(group_name):
    return os.path.join(PROJECT_PATH, default_path, group_name, "wandb")

def get_model_path_with_seed(group_name, seed):
    group_yaml_paths = get_group_yaml_paths(group_name)
    group_model_paths = get_group_model_paths(group_name)

    for i, yaml_path in enumerate(group_yaml_paths):
        yaml_config = open_yaml(yaml_path)
        if yaml_config["seed"] == seed:
            return group_model_paths[i]


def get_replay_buffer_path_with_seed(group_name, seed):
    group_yaml_paths = get_group_yaml_paths(group_name)
    group_replay_buffer_paths = get_group_replay_buffer_paths(group_name)

    for i, yaml_path in enumerate(group_yaml_paths):
        yaml_config = open_yaml(yaml_path)
        if yaml_config["seed"] == seed:
            return group_replay_buffer_paths[i]

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

def get_group_replay_buffer_paths(group_name):
    group_path = get_group_path(group_name)

    replay_buffer_paths = []

    # walk through ensemble path
    for path in os.listdir(group_path):
        replay_buffer_paths.append(f"{group_path}/{path}/{default_replay_buffer_location}")

    return replay_buffer_paths

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