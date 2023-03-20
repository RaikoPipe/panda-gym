import wandb

import sys
import gymnasium
from matplotlib import pyplot as plt

sys.modules["gym"] = gymnasium

from stable_baselines3 import TD3
from sb3_contrib import TQC

import numpy as np
from train_preo import config
from time import sleep

import gym

import panda_gym


from learning_methods.curriculum_learning import get_env

import seaborn as sns

done_events = []
action_diffs = []
manipulabilities = []

end_effector_positions = []
end_effector_velocities = []
end_effector_speeds = []

joint_positions = []
joint_velocities = []

def evaluate(model, num_steps=10_000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    # robot parameters
    # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
    #env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])

    episode_rewards = [0.0]
    obs, _ = env.reset()
    done_events = []
    action_diffs = []
    manipulabilities = []
    # todo: change neutral values of panda

    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)

        obs, reward, done, truncated, info, = env.step(action)
        action_diff = env.task.action_diff
        manipulability = env.task.manipulability
        sleep(0.05) # for human eval

        # Stats
        episode_rewards[-1] += reward
        action_diffs.append(action_diff)
        manipulabilities.append(manipulability)

        end_effector_positions.append(env.robot.get_ee_position())
        ee_velocity = env.robot.get_ee_velocity
        end_effector_velocities.append(ee_velocity)
        #end_effector_speeds.append(np.square(np.linalg.norm(ee_velocity)))

        joint_positions.append(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))
        joint_velocities.append(np.array([env.robot.get_joint_velocity(joint=i) for i in range(7)]))

        if done or truncated:
            #sleep(2)
            if info["is_success"]:
                print("Success!")
                done_events.append(1)
            elif info["is_truncated"]:
                print("Collision...")
                done_events.append(-1)
            else:
                print("Timeout...")
                done_events.append(0)
            obs, _ = env.reset()
            episode_rewards.append(0.0)
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = np.mean(episode_rewards[-100:])

    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    print(f"Success Rate: {done_events.count(1)/len(done_events)}")
    print(f"Collision Rate: {done_events.count(-1) / len(done_events)}")
    print(f"Mean Action Difference: {np.mean(action_diffs)}")
    print(f"Mean Manipulability: {np.mean(manipulabilities)}")

    print(sum(action_diffs))
    print(sum(manipulabilities))

    return mean_100ep_reward

panda_gym.register_envs(100)

#env = get_env(config, "cube_3_random")
if __name__ == "__main__":
    env = gym.make(config["env_name"], render=True, control_type=config["control_type"],
                   obs_type=config["obs_type"], goal_distance_threshold=config["goal_distance_threshold"],
                   reward_type=config["reward_type"], limiter=config["limiter"],
                   show_goal_space=False, scenario="library",
                   show_debug_labels=True)

    model = TQC.load(r"run_data/wandb/incandescent_monkey_18/files/best_model.zip", env=env)

    evaluate(model)

    # Some boilerplate to initialise things
    sns.set()
    plt.figure()

    # This is where the actual plot gets made
    ax = sns.lineplot(data=end_effector_speeds, saturation=0.6)

    # Customise some display properties
    ax.set_title('End Effector Speeds')
    ax.grid(color='#cccccc')
    ax.set_ylabel('Speed')
    ax.set_xlabel("TimeStep")
    #ax.set_xticklabels(df["year"].unique().astype(str), rotation='vertical')

    # Ask Matplotlib to show it
    plt.show()



