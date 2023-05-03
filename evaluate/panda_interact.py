import gymnasium as gym
import panda_gym
import numpy as np
from run.train import config
import time

panda_gym.register_envs(100)

env = gym.make(config["env_name"], render=True, show_goal_space=True, obs_type="js", control_type = "js",
               show_debug_labels=True, scenario="wang_1", reward_type=config["reward_type"])
env.reset()
while True:
    observation, reward, terminated, truncated, info = env.step(np.array([0,0,0,0,0,0,0]))
    time.sleep(3)
    env.reset()


