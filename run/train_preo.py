import time

import numpy as np
import wandb
# from pygame import mixer

import panda_gym
import os
from learning_methods.curriculum_learning import learn
from learning_methods.imitation_learning import fill_replay_buffer
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3 import HerReplayBuffer
from learning_methods.curriculum_learning import get_env, get_model
from stable_baselines3 import TD3

# hyperparameters from rl-baselines3-zoo tuned pybullet defaults

config = {
    "env_name": "PandaReachEvadeObstacles-v3",
    "algorithm": "TD3",
    "reward_type": "sparse",  # sparse; dense
    "goal_distance_threshold": 0.02,
    "max_timesteps": 1e6,
    "seed": 23,
    "render": True,  # renders the pybullet env
    "obs_type": "ee",
    "control_type": "js",  # "ee": end effector displacement; "js": joint space
    "limiter": "sim",
    "show_goal_space": True,
    "replay_buffer": None,  # HerReplayBuffer
    "policy_type": "MultiInputPolicy",
    "show_debug_labels": True,
    "n_envs": 1,
    "max_ep_steps": 200,
    "eval_freq": 5_000,
    "stages": ["sphere_2_random"],
    "reward_thresholds": [-10],  # [-7, -10, -12, -17, -20]
    "joint_obstacle_observation": "closest",  # "all": closest distance to any obstacle of all joints is observed;
    "prior_steps": 10_000
    # "closest": only closest joint distance is observed
}

# hyperparameters are from rl-baselines3 zoo and https://arxiv.org/pdf/2106.13687.pdf

hyperparameters_td3 = {
    "learning_starts": 0,  # 10000,
    "learning_rate": 0.001,
    "gamma": 0.98,
    "tau": 0.005,
    "buffer_size": 200_000,
    "gradient_steps": -1,
    "policy_kwargs": dict(net_arch=[400, 300]),
    # "noise_std": 0.2,
}

hyperparameters_sac = {
    "learning_starts": 10000,
    "learning_rate": 0.00073,
    "gamma": 0.98,
    "tau": 0.02,
    "buffer_size": 300_000,
    "gradient_steps": config["n_envs"] * 8,
    "train_freq": config["n_envs"] * 8,
    "ent_coed": "auto",
    "use_sde": True,
    "policy_kwargs": dict(log_std_init=-3, net_arch=[400, 300])
}

# register envs to gymnasium
panda_gym.register_envs(config["max_ep_steps"])

if __name__ == "__main__":
    key = os.getenv("wandb_key")
    wandb.login(key=os.getenv("wandb_key"))

    # register envs to gymnasium
    panda_gym.register_envs(config["max_ep_steps"])

    #env = get_env(config, "sphere_2_random")

    # for algorithm in "PPO":
    if config["algorithm"] in ("TD3", "DDPG"):
        config.update(hyperparameters_td3)
    elif config["algorithm"] == "SAC":
        config.update(hyperparameters_sac)

    # model = TD3.load(r"run_data/wandb/run-20221206_122636-k1fwbcbr/files/model.zip", env=env,
    #                  train_freq=config["n_envs"],
    #                  gradient_steps=config["gradient_steps"])


    model = learn(config=config, algorithm=config["algorithm"], )#initial_model=model)


    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)
