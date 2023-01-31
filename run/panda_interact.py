import gymnasium as gym
import panda_gym
import numpy as np
from train_preo import config

panda_gym.register_envs(100)

env = gym.make("PandaReachEvadeObstacles-v3", render=True, show_goal_space=True,
               show_debug_labels=True, obstacle_layout="wall_parkour_1", reward_type=config["reward_type"])
env.reset()
while True:
    observation, reward, terminated, truncated, info = env.step(np.array([0,0,0]))

