import gymnasium as gym
import panda_gym
import numpy as np

panda_gym.register_envs(100)

env = gym.make("PandaReachEvadeObstacles-v3", render=True, show_goal_space=True,
               show_debug_labels=True, obstacle_layout="box_1")
env.reset()
while True:
    observation, reward, terminated, truncated, info = env.step(np.array([0,0,0]))

