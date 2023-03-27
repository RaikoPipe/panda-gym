import wandb

import sys
import gymnasium
from matplotlib import pyplot as plt

sys.modules["gym"] = gymnasium

from stable_baselines3 import TD3
from sb3_contrib import TQC

import numpy as np
from run.train_preo import config
from time import sleep
import pprint

import gym

import panda_gym

import seaborn as sns


def evaluate(model, human=True, num_steps=10_000, goals_to_achieve=None, deterministic=True):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    # robot parameters
    # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
    # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])

    episode_rewards = [0.0]
    obs, _ = env.reset()

    if goals_to_achieve:
        env.task.goal = goals_to_achieve.pop(0)

    done_events = []
    action_diffs = []
    manipulabilities = []
    goals = []
    goals.append(env.task.goal)

    end_effector_positions = []
    end_effector_velocities = []
    end_effector_speeds = []

    joint_positions = []
    joint_velocities = []

    if goals_to_achieve:
        while goals_to_achieve:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info, = env.step(action)
            action_diff = env.task.action_diff
            manipulability = env.task.manipulability
            if human:
                sleep(0.025)  # for human eval
            # Stats
            episode_rewards[-1] += reward
            action_diffs.append(action_diff)
            manipulabilities.append(manipulability)
            end_effector_positions.append(env.robot.get_ee_position())
            ee_velocity = env.robot.get_ee_velocity
            end_effector_velocities.append(ee_velocity)
            # end_effector_speeds.append(np.square(np.linalg.norm(ee_velocity)))
            joint_positions.append(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))
            joint_velocities.append(np.array([env.robot.get_joint_velocity(joint=i) for i in range(7)]))
            if done or truncated:
                # sleep(2)
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

                if goals_to_achieve:
                    # set goal from list
                    env.task.goal = goals_to_achieve.pop(0)

                goals.append(env.task.goal)
                episode_rewards.append(0.0)
            # evaluation_step(action_diffs, done_events, end_effector_positions, end_effector_velocities, episode_rewards,
            #                 goals, human, joint_positions, joint_velocities, manipulabilities, model, obs,
            #                 goals_to_achieve)
    else:

        for num in range(num_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info, = env.step(action)
            action_diff = env.task.action_diff
            manipulability = env.task.manipulability
            if human:
                sleep(0.025)  # for human eval
            # Stats
            episode_rewards[-1] += reward
            action_diffs.append(action_diff)
            manipulabilities.append(manipulability)
            end_effector_positions.append(env.robot.get_ee_position())
            ee_velocity = env.robot.get_ee_velocity
            end_effector_velocities.append(ee_velocity)
            # end_effector_speeds.append(np.square(np.linalg.norm(ee_velocity)))
            joint_positions.append(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))
            joint_velocities.append(np.array([env.robot.get_joint_velocity(joint=i) for i in range(7)]))
            if done or truncated:
                # sleep(2)
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

                if goals_to_achieve:
                    # set goal from list
                    env.task.goal = goals_to_achieve.pop(0)

                goals.append(env.task.goal)
                episode_rewards.append(0.0)
        # evaluation_step(action_diffs, done_events, end_effector_positions, end_effector_velocities, episode_rewards,
        #                 goals, human, joint_positions, joint_velocities, manipulabilities, model, obs, goals_to_achieve)
    # Compute mean reward for the last 100 episodes

    results = {"mean_reward": np.mean(episode_rewards),
              "success_rate": done_events.count(1) / len(done_events),
              "collision_rate": done_events.count(-1) / len(done_events),
              "timeout_rate": done_events.count(0) / len(done_events),
              "num_episodes": len(done_events),
              "mean_action_difference": np.mean(action_diffs),
              "mean_manipulability": np.mean(manipulabilities),
              }

    metrics = {
        "end_effector_positions": end_effector_positions,
        "end_effector_speeds": end_effector_speeds,
        "end_effector_velocities": end_effector_velocities,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,

        "goals": goals
    }

    return results, metrics


# def evaluation_step(action_diffs, done_events, end_effector_positions, end_effector_velocities, episode_rewards, goals,
#                     human, joint_positions, joint_velocities, manipulabilities, model, obs, goals_to_achieve):
#     action, _states = model.predict(obs)
#     obs, reward, done, truncated, info, = env.step(action)
#     action_diff = env.task.action_diff
#     manipulability = env.task.manipulability
#     if human:
#         sleep(0.025)  # for human eval
#     # Stats
#     episode_rewards[-1] += reward
#     action_diffs.append(action_diff)
#     manipulabilities.append(manipulability)
#     end_effector_positions.append(env.robot.get_ee_position())
#     ee_velocity = env.robot.get_ee_velocity
#     end_effector_velocities.append(ee_velocity)
#     # end_effector_speeds.append(np.square(np.linalg.norm(ee_velocity)))
#     joint_positions.append(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))
#     joint_velocities.append(np.array([env.robot.get_joint_velocity(joint=i) for i in range(7)]))
#     if done or truncated:
#         # sleep(2)
#         if info["is_success"]:
#             print("Success!")
#             done_events.append(1)
#         elif info["is_truncated"]:
#             print("Collision...")
#             done_events.append(-1)
#         else:
#             print("Timeout...")
#             done_events.append(0)
#         obs, _ = env.reset()
#
#         if goals_to_achieve:
#             # set goal from list
#             env.task.goal = goals_to_achieve.pop(0)
#
#         goals.append(env.task.goal)
#         episode_rewards.append(0.0)


panda_gym.register_envs(200)

# env = get_env(config, "cube_3_random")
if __name__ == "__main__":
    human = False

    env = gym.make(config["env_name"], render=human, control_type=config["control_type"],
                   obs_type=config["obs_type"], goal_distance_threshold=config["goal_distance_threshold"],
                   reward_type=config["reward_type"], limiter=config["limiter"],
                   show_goal_space=False, scenario="library2",
                   show_debug_labels=True)

    # Compare Models
    model1 = TQC.load(r"../run/run_data/wandb/efficient-fog-124/files/model.zip", env=env)
    model2 = TQC.load(r"../run/run_data/wandb/efficient-fog-124/files/best_model.zip", env=env)

    results, metrics = evaluate(model1, human=human, num_steps=50_000, deterministic=False)
    results2, metrics2 = evaluate(model1, human=human, goals_to_achieve=metrics["goals"], deterministic=True)

    print("Stochastic:")
    pprint.pprint(results)
    print("Deterministic:")
    pprint.pprint(results2)

    # # Some boilerplate to initialise things
    # sns.set()
    # plt.figure()
    #
    # # This is where the actual plot gets made
    # ax = sns.lineplot(data=results["end_effector_speeds"], saturation=0.6)
    #
    # # Customise some display properties
    # ax.set_title('End Effector Speeds')
    # ax.grid(color='#cccccc')
    # ax.set_ylabel('Speed')
    # ax.set_xlabel("TimeStep")
    # # ax.set_xticklabels(df["year"].unique().astype(str), rotation='vertical')
    #
    # # Ask Matplotlib to show it
    # plt.show()
