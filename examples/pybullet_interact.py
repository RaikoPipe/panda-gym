import gymnasium as gym
import panda_gym
import numpy as np

env = gym.make("PandaReachEvadeObstacles-v3", render=True, show_goal_space=True, goal_range=0.3)
env.reset()
while True:
    observation, reward, terminated, truncated, info = env.step(np.array([0,0,0]))

