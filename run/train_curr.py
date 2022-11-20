import time

import wandb
#from pygame import mixer
from stable_baselines3 import TD3


import panda_gym
from utils.learning import curriculum_learn
from utils.learning import get_env

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
    "n_envs": 3,
    "max_ep_steps": 50,
    "eval_freq": 10_000,
    "stages": ["box_1"],  # 0: No obstacles; 1: 1 small cube near ee; 2: 2 small cubes neighboring ee
    "reward_thresholds": [-7],
    "joint_obstacle_observation": "closest",  # "all": closest distance to any obstacle of all joints is observed;
    # "closest": only closest joint distance is observed
}

if __name__ == "__main__":
    wandb.login(key="5d65c571cf2a6110b15190696682f6e36ddcdd11")

    # register envs to gymnasium
    panda_gym.register_envs(config["max_ep_steps"])

    # env = gym.make(config["env_name"], render=config["render"], control_type=config["control_type"],
    #                reward_type=config["reward_type"],
    #                show_goal_space=False, obstacle_layout=1,
    #                show_debug_labels=True)

    env = get_env(config, "box_1")

    # model = TD3.load(r"run_data/wandb/run-20221120_093738-vpgfx0te/files/model.zip", env=env, train_freq=config["n_envs"],
    #                  gradient_steps=config["gradient_steps"])


    curriculum_learn(config=config, ) #initial_model=model)

    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)


