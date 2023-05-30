import gymnasium as gym
import panda_gym
import numpy as np
from run.train import config
import time

panda_gym.register_envs(100)

env = gym.make(config["env_name"], render=True, show_goal_space=False, obs_type="js", control_type = "js",
               show_debug_labels=True, scenario="base1", reward_type=config["reward_type"], randomize_robot_pose = False,
               joint_obstacle_observation="vectors+all")
env.reset()
while True:
    observation, reward, terminated, truncated, info = env.step(np.array([0,0,0,0,0,0,0]))
    # time.sleep(0.5)
    # env.reset()
    # print(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))



