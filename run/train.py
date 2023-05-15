import time

import numpy as np

# from pygame import mixer

import sys
import gymnasium
sys.modules["gym"] = gymnasium

from sb3_contrib import TQC
import panda_gym
import os


from stable_baselines3.her.her_replay_buffer import HerReplayBuffer, VecHerReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer


# hyperparameters from rl-baselines3-zoo tuned pybullet defaults

config = {
    "env_name": "PandaReachAO-v3",
    "algorithm": "TQC",
    "reward_type": "sparse",  # sparse; dense
    "goal_distance_threshold": 0.05,
    "max_timesteps": 3_000_000,
    "seed": 1,
    "render": False,  # renders the pybullet env
    "n_substeps": 20, # number of simulation steps before handing control back to agent
    "obs_type": ["ee", "js"], # Robot state to observe
    "control_type": "js",  # Agent Output; js: joint velocities, ee: end effector displacements; jsd: joint velocities (applied directly)
    "limiter": "sim",
    "action_limiter": "clip",
    "show_goal_space": True,
    "replay_buffer": VecHerReplayBuffer,  # HerReplayBuffer
    "policy_type": "MultiInputPolicy",
    "show_debug_labels": True,
    "n_envs": 4,
    "max_ep_steps": [50],
    "eval_freq": 5_000,
    "stages": ["wang_3"],
    "success_thresholds": [0.99],  # [-7, -10, -12, -17, -20]
    "joint_obstacle_observation": "all2",  # "all": closest distance to any obstacle of all joints is observed;
    "learning_starts": 30_000,
    "prior_steps": 0,
    "randomize_robot_pose": False,
    "truncate_episode_on_collision": True
    # "closest": only closest joint distance is observed
}

# register envs to gymnasium
panda_gym.register_envs(config["max_ep_steps"][0])

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
    "gradient_steps": 8 if config["n_envs"] <=1 else 8*config["n_envs"]/2,
    "train_freq": 8 ,
    "ent_coef": "auto",
    "use_sde": True,

    "policy_kwargs": dict(log_std_init=-3, net_arch=[400, 300])
}

hyperparameters_panda = {
    "policy_kwargs": dict(log_std_init=-3, net_arch=[256, 256, 256]),
    "tau" : 0.95
}

hyperparameters_tqc = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "buffer_size": int(1e6),
    "replay_buffer_kwargs": dict(
        goal_selection_strategy='future', n_sampled_goal=4),
    "gamma": 0.95,
    "policy_kwargs": dict(net_arch=[64, 64], n_critics=1),
}

if config["algorithm"] in ("TD3", "DDPG"):
    config["hyperparams"] = hyperparameters_td3
elif config["algorithm"] == "SAC":
    config["hyperparams"] = hyperparameters_sac
elif config["algorithm"] == "TQC":
    config["hyperparams"] = hyperparameters_panda

if __name__ == "__main__":
    from run.learning_methods.learning import learn, get_env
    import wandb
    key = os.getenv("wandb_key")
    wandb.login(key=os.getenv("wandb_key"))

    # env = get_env(config, config["n_envs"], config["stages"][0])
    # model = TQC.load(r"run_data/wandb/cerulean_sky/files/best_model.zip", env=env, replay_buffer_class=config["replay_buffer"])

    model = learn(config=config, algorithm=config["algorithm"])


    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)
