import gymnasium as gym
import panda_gym
import numpy as np
from run.train import configuration
import time
import json

panda_gym.register_envs(100)

# env = gym.make(configuration["env_name"], render=False, show_goal_space=True, obs_type="js", control_type ="js",
#                show_debug_labels=True, scenario="narrow_tunnel", reward_type="kumar", randomize_robot_pose = False,
#                joint_obstacle_observation="vectors+all")
# env.reset()
scenario_goals = {}
for scenario in ["wangexp_3", "narrow_tunnel", "workshop", "library2", "wall"]:
    goals = []
    env = gym.make(configuration["env_name"], render=False, show_goal_space=True, obs_type="js", control_type="js",
                   show_debug_labels=True, scenario=scenario, reward_type="kumar", randomize_robot_pose=False,
                   joint_obstacle_observation="vectors+all")
    for i in range(1000):
        env.reset()
        goals.append(tuple(env.task.goal))

    scenario_goals[scenario] = goals
    env.close()

json.dumps(scenario_goals)
with open("scenario_goals", "w") as file:
    file.write(json.dumps(scenario_goals))


    # time.sleep(0.5)
    #env.reset()
    #print(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))



