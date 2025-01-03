import gymnasium as gym
import panda_gym
import numpy as np

from classes.train_config import TrainConfig
import time
import pybullet

panda_gym.register_reach_ao(100)

configuration = TrainConfig(render=True)

configuration.show_debug_labels = True
configuration.show_goal_space = True
configuration.debug_collision = False

scenario = "narrow_tunnel"

# get env
env = gym.make(configuration.env_name,
               render=configuration.render,
    config=configuration,
                          scenario=scenario)

env.unwrapped.task.sim.physics_client.configureDebugVisualizer(pybullet.COV_ENABLE_MOUSE_PICKING, 1)

env.reset()
goals = []

robot = env.unwrapped.robot
joint_sliders = []
joint_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])

# for i in joint_indices:
#     joint_info = pybullet.getJointInfo(robotId, i)
#     joint_name = joint_info[1].decode('utf-8')
#     joint_sliders.append(pybullet.addUserDebugParameter(joint_name, -3.14, 3.14, 0))

while True:
    # Read slider values and set the joint positions
    # for i in range(len(joint_indices)):
    #     slider_value = pybullet.readUserDebugParameter(joint_sliders[i])
    #     pybullet.setJointMotorControl2(bodyIndex=robotId,
    #                             jointIndex=i,
    #                             controlMode=pybullet.POSITION_CONTROL,
    #                             targetPosition=slider_value)

    observation, reward, terminated, truncated, info = env.step(np.array([0,0,0,0,0,0,0]))
    #print(env.task.goal_reached)
    camera = pybullet.getDebugVisualizerCamera()
    print(
        f"cameraTargetPosition = {camera[11]}\ncameraDistance = {camera[10]} \ncameraPitch = {camera[9]}\ncameraYaw = {camera[8]}", flush=True)
    #time.sleep(0.5)
    #env.reset()
    joint_angles = np.array([robot.get_joint_angle(joint=i) for i in range(7)])
    joint_angle_string = ", ".join([f"{angle:.3f}" for angle in joint_angles])
    print(f"Joint angles: [{joint_angle_string}]", flush=True)
    print(info)



