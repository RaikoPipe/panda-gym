import time

import numpy as np
import wandb
from pygame import mixer

import panda_gym
import os
from utils.learning import curriculum_learn
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3 import HerReplayBuffer
from utils.learning import get_env
# hyperparameters from rl-baselines3-zoo tuned pybullet defaults

config = {
    "env_name": "PandaReachEvadeObstacles-v3",
    "algorithm": "TD3",
    "reward_type": "sparse",  # sparse; dense
    "total_timesteps": 400_000,
    "seed": 12,
    "render": False,  # renders the pybullet env
    "control_type": "js",  # "ee": end effector displacement; "js": joint angles
    "limiter": "sim",
    "show_goal_space": True,
    "replay_buffer": None,  # HerReplayBuffer
    "policy_type": "MultiInputPolicy",
    "show_debug_labels": True,
    "n_envs": 4,
    "max_ep_steps": 50,
    "eval_freq": 10_000,
    "stages": ["wall_parkour_1"],  # 0: No obstacles; 1: 1 small cube near ee; 2: 2 small cubes neighboring ee
    "reward_thresholds": [-7],
    "joint_obstacle_observation": "closest",  # "all": closest distance to any obstacle of all joints is observed;
    # "closest": only closest joint distance is observed


}

# hyperparameters are from rl-baselines3 zoo and https://arxiv.org/pdf/2106.13687.pdf

hyperparameters_td3 ={
    "learning_starts": 10000,
    "learning_rate": 0.001,
    "gamma": 0.98,
    "tau": 0.95,
    "buffer_size": 200_000,
    "gradient_steps": -1,
    "policy_kwargs": dict(net_arch=[400, 300]),
    #"noise_std": 0.2,
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
    "use_sde":True,
    "policy_kwargs": dict(log_std_init=-3, net_arch=[400, 300])
}

# register envs to gymnasium
panda_gym.register_envs(config["max_ep_steps"])

if __name__ == "__main__":
    key = os.getenv("wandb_key")
    wandb.login(key=os.getenv("wandb_key"))

    # register envs to gymnasium
    panda_gym.register_envs(config["max_ep_steps"])

    # env = gym.make(config["env_name"], render=config["render"], control_type=config["control_type"],
    #                reward_type=config["reward_type"],
    #                show_goal_space=False, obstacle_layout=1,
    #                show_debug_labels=True)

    # env = get_env(config, "box_3")

    # model = TD3.load(r"run_data/wandb/run-20221120_093738-vpgfx0te/files/model.zip", env=env, train_freq=config["n_envs"],
    #                  gradient_steps=config["gradient_steps"])
    #for algorithm in "PPO":
    if config["algorithm"] in ("TD3", "DDPG"):
        config.update(hyperparameters_td3)
    elif config["algorithm"] == "SAC":
        config.update(hyperparameters_sac)
    model = curriculum_learn(config=config, algorithm=config["algorithm"])  # initial_model=model)

    model.env.close()
    del model

    mixer.init()
    mixer.music.load("learning_complete.mp3")
    mixer.music.set_volume(1.0)
    mixer.music.play()
    time.sleep(2.0)
