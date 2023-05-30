import time

import numpy as np

# from pygame import mixer

import sys
import gymnasium
sys.modules["gym"] = gymnasium

from sb3_contrib import TQC
from stable_baselines3 import PPO
import panda_gym
import os


from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from typing import Callable

reach_stages = ["reach1", "reach2", "reach3", "reach4"]
reach_max_ep_steps = [50,50,50,50]
reach_succ_thresholds = [1.0,1.0,1.0,1.0]

reach_ao_stages = ["base1", "base2", "wangexp_3"]
reach_ao_max_ep_steps = [50,75,100]
reach_ao_succ_thresholds = [0.9,0.9,0.99]



# hyperparameters from rl-baselines3-zoo tuned pybullet defaults

config = {
    "env_name": "PandaReachAO-v3",
    "algorithm": "TQC",
    "reward_type": "sparse",  # sparse; dense
    "goal_distance_threshold": 0.05,
    "max_timesteps": 300_000,
    "seed": 1,
    "render": False,  # renders the eval env
    "n_substeps": 20, # number of simulation steps before handing control back to agent
    "obs_type": ["ee","js"], # Robot state to observe
    "control_type": "js",  # Agent Output; js: joint velocities, ee: end effector displacements; jsd: joint velocities (applied directly)
    "limiter": "sim",
    "action_limiter": "clip",
    "show_goal_space": False,
    "replay_buffer_class": HerReplayBuffer,  # HerReplayBuffer
    "policy_type": "MultiInputPolicy",
    "show_debug_labels": False,
    "n_envs": 1,
    "eval_freq": 5_000,
    "stages": reach_ao_stages,
    "success_thresholds": reach_ao_succ_thresholds,  # [-7, -10, -12, -17, -20]
    "max_ep_steps": reach_ao_max_ep_steps,
    "joint_obstacle_observation": "vectors",  # "all": closest distance to any obstacle of all joints is observed;
    "learning_starts": 10_000,
    "prior_steps": 0,
    "randomize_robot_pose": False,
    "truncate_episode_on_collision": True,
    "collision_reward": -100
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

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


hyperparameters_sac = {
    "learning_rate": 0.00073, #0.0007, #0.00073 # linear_schedule(0.001)
    "gamma": 0.98,
    "tau": 0.02,
    "buffer_size": int(1e6), # 300_000
    "gradient_steps": 8,
    "train_freq": 8,
    "ent_coef": "auto",
    "use_sde": True,

    "policy_kwargs": dict(log_std_init=-3, net_arch=[256,256]) # 256, 256
}

hyperparameters_fetch = {
    "learning_rate": 0.0007, #0.00073
    "gamma": 0.98,
    "tau": 0.005,
    "batch_size": 512,
    "buffer_size": int(1e6), # 300_000
    # "gradient_steps": 8,
    # "train_freq": 8,
    "ent_coef": "auto",
    "use_sde": True,
    "replay_buffer_kwargs": dict(online_sampling= True, goal_selection_strategy="future", n_sampled_goal=4),

    "policy_kwargs": dict(net_arch=[512, 512, 512], n_critics=2, log_std_init=-3) # 256, 256
}
"""defaults for tqc and pybullet envs"""
hyperparameters_pybullet_defaults = {

    "learning_rate": float(7.3e-4),  # 0.0007, #0.00073 # linear_schedule(0.001)
    "gamma": 0.98,
    "tau": 0.02,
    "buffer_size": 300_000,  # 300_000
    "batch_size": 256,
    "gradient_steps": 8,
    "train_freq": 8,
    "ent_coef": "auto",
    "use_sde": True,

    "policy_kwargs": dict(log_std_init=-3, net_arch=[256, 256])  # 400, 300

}

hyperparameters_tqc = {
    "learning_rate": float(1e-3),
    "batch_size": 2048,
    "buffer_size": int(1e6),
    "replay_buffer_kwargs": dict(
        goal_selection_strategy='future', n_sampled_goal=4),
    "gamma": 0.95,
    "tau": 0.05,
    "policy_kwargs": dict(net_arch=[400, 300]),
    "use_sde": True
}

if config["algorithm"] in ("TD3", "DDPG"):
    config["hyperparams"] = hyperparameters_td3
elif config["algorithm"] == "SAC":
    config["hyperparams"] = hyperparameters_sac
elif config["algorithm"] == "TQC":
    config["hyperparams"] = hyperparameters_pybullet_defaults

def main():
    from run.learning_methods.learning import learn, get_env
    import wandb

    wandb.login(key=os.getenv("wandb_key"))

    env = get_env(config, config["n_envs"], config["stages"][0])
    # model = TQC.load(r"run_data/wandb/morning-grass/files/best_model.zip", env=env, replay_buffer_class=config["replay_buffer_class"],
    #                  custom_objects={"action_space":gymnasium.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),} # workaround
    #                  )

    model = learn(config=config, algorithm=config["algorithm"])


if __name__ == "__main__":
    main()

    # training mode
    # for i in range(5):
    #     config["random_seed"] = i
    #     main()


    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)
