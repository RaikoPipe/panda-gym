
import os
import yaml

from classes.train_config import TrainConfig

from definitions import PROJECT_PATH, default_path, default_yaml_location, default_model_location, default_replay_buffer_location

def get_group_path(group_name):
    return os.path.join(PROJECT_PATH, default_path, group_name, "wandb")

def get_model_component_path(group_name, seed=None, desired_file="model"):
    group_yaml_paths = get_group_model_paths(group_name, "config")

    for i, yaml_path in enumerate(group_yaml_paths):
        yaml_config = open_yaml(yaml_path)
        if yaml_config["seed"]["value"] == seed or seed is None:
            return get_group_model_paths(group_name, desired_file)[i]

def get_group_model_paths(group_name, desired_file):
    suffix = default_model_location
    if desired_file in ["config", "yaml"]:
        suffix = default_yaml_location
    elif desired_file in ["replay", "buffer", "replay_buffer"]:
        suffix = default_replay_buffer_location

    paths = []
    group_path = get_group_path(group_name)

    # walk through ensemble path
    for dir_name in os.listdir(group_path):
        # if path starts with "run"
        if dir_name.startswith("run"):
            paths.append(f"{group_path}/{dir_name}/{suffix}")

    if not paths:
        raise FileNotFoundError(f"No {desired_file} found for group {group_name}")

    return paths

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