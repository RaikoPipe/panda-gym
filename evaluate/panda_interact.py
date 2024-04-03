import gymnasium as gym
import panda_gym
import numpy as np
from run.train import configuration
import time
import pybullet

panda_gym.register_envs(100)

env = gym.make(configuration["env_name"], render=True, show_goal_space=False, obs_type="js", control_type ="js",
               show_debug_labels=False, scenario="wall", reward_type="kumar", randomize_robot_pose = False,
               joint_obstacle_observation="vectors+all", terminate_on_success=False, goal_condition="halt")
env.reset()
goals = []
while True:
    observation, reward, terminated, truncated, info = env.step(np.array([0,0,0,0,0,0,0]))
    #print(env.task.goal_reached)
    #time.sleep(0.5)
    #env.reset()
    camera = pybullet.getDebugVisualizerCamera()
    print(
        f"cameraTargetPosition = {camera[11]}\ncameraDistance = {camera[10]} \ncameraPitch = {camera[9]}\ncameraYaw = {camera[8]}", flush=True)
    # time.sleep(0.5)
    #env.reset()
    #print(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))



