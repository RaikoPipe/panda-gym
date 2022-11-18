import time

import wandb
from pygame import mixer
from stable_baselines3 import TD3, SAC, PPO
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import panda_gym
from utils.learning import curriculum_learn

config = {
    "env_name": "PandaReachEvadeObstacles-v3",
    "reward_type": "sparse",  # sparse; dense
    "total_timesteps": 400_000,
    "seed": 12,
    "gradient_steps": -1,
    "render": False,  # renders the pybullet env
    "control_type": "js",  # "ee": end effector displacement; "js": joint angles
    "show_goal_space": True,
    "replay_buffer": None,  # HerReplayBuffer
    "policy_type": "MultiInputPolicy",
    "show_debug_labels": True,
    "learning_starts": 1000,
    "n_envs": 4,
    "max_ep_steps": 50,
    "eval_freq": 10_000,
    "stages": [1,"wall_parkour_1"],  # 0: No obstacles; 1: 1 small cube near ee; 2: 2 small cubes neighboring ee
    "reward_thresholds": [-7, -7],
    "joint_obstacle_observation": "closest",  # "all": closest distance to any obstacle of all joints is observed;
    # "closest": only closest joint distance is observed
}

# register envs to gymnasium
panda_gym.register_envs(config["max_ep_steps"])

if __name__ == "__main__":
    wandb.login(key="5d65c571cf2a6110b15190696682f6e36ddcdd11")

    # env = gym.make(config["env_name"], render=config["render"], control_type=config["control_type"],
    #                reward_type=config["reward_type"],
    #                show_goal_space=False, obstacle_layout=1,
    #                show_debug_labels=True)

    env = make_vec_env(config["env_name"], n_envs=config["n_envs"],
                       env_kwargs={"render": False, "control_type": config["control_type"],
                                   "reward_type": config["reward_type"],
                                   "show_goal_space": False, "obstacle_layout": 1,
                                   "show_debug_labels": False}, vec_env_cls=SubprocVecEnv)

    model = TD3.load(r"run_data/wandb/run_obs_layout_1_best_08_11/files/model.zip", env=env)

    curriculum_learn(config=config, initial_model= model, starting_stage="wall_parkour_1", )

    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)


