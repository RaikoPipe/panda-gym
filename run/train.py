import time

import numpy as np


import wandb
# from pygame import mixer

import sys
import gymnasium
sys.modules["gym"] = gymnasium

from sb3_contrib import TQC
import panda_gym
import os
from run.learning_methods.learning import learn, get_env

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer, VecHerReplayBuffer
# from stable_baselines3 import HerReplayBuffer


# hyperparameters from rl-baselines3-zoo tuned pybullet defaults

config = {
    "env_name": "PandaReachAO-v3",
    "algorithm": "TQC",
    "reward_type": "sparse",  # sparse; dense
    "goal_distance_threshold": 0.05,
    "max_timesteps": 1_200_000,
    "seed": 1,
    "render": False,  # renders the pybullet env
    "n_substeps": 20, # number of simulation steps before handing control back to agent
    "obs_type": ("ee",), # Robot state to observe
    "control_type": "js",  # Agent Output; js: joint velocities, ee: end effector displacements; jsd: joint velocities (applied directly)
    "limiter": "sim",
    "action_limiter": "clip",
    "show_goal_space": True,
    "replay_buffer": HerReplayBuffer,  # HerReplayBuffer
    "policy_type": "MultiInputPolicy",
    "show_debug_labels": True,
    "n_envs": 1,
    "max_ep_steps": [50],
    "eval_freq": 5_000,
    "stages": ["cube_4"],
    "reward_thresholds": [-1],  # [-7, -10, -12, -17, -20]
    "joint_obstacle_observation": "closest",  # "all": closest distance to any obstacle of all joints is observed;
    "learning_starts": 10_000,
    "prior_steps": 0,
    # "closest": only closest joint distance is observed
}

# hyperparameters are from rl-baselines3 zoo and https://arxiv.org/pdf/2106.13687.pdf

hyperparameters_td3 = {  # 10000,
    "learning_rate": 0.001,
    "gamma": 0.98,
    "tau": 0.005,
    "buffer_size": 200_000,
    "gradient_steps": -1,
    "policy_kwargs": dict(net_arch=[400, 300]),
    # "noise_std": 0.2,
}

hyperparameters_sac = {
    "learning_rate": 0.00073,
    "gamma": 0.98,
    "tau": 0.02,
    "buffer_size": 300_000,
    "gradient_steps": config["n_envs"] * 8,
    "train_freq": config["n_envs"] * 8,
    "ent_coef": "auto",
    "use_sde": True,
    "policy_kwargs": dict(log_std_init=-3, net_arch=[400, 300])
}

if __name__ == "__main__":
    key = os.getenv("wandb_key")
    wandb.login(key=os.getenv("wandb_key"))

    # register envs to gymnasium
    panda_gym.register_envs(config["max_ep_steps"][0])

    if config["algorithm"] in ("TD3", "DDPG"):
        config.update(hyperparameters_td3)
    elif config["algorithm"] in  ("SAC", "TQC"):
        config.update(hyperparameters_sac)

    # env = get_env(config, config["stages"][0])
    # model = TQC.load(r"run_data/wandb/fresh-wood-32/files/model.zip", env=env,
    #                  train_freq=config["n_envs"],
    #                  gradient_steps=config["gradient_steps"])

    model = learn(config=config, algorithm=config["algorithm"])


    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)
