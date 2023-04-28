import wandb

import sys
import gymnasium
from matplotlib import pyplot as plt

sys.modules["gym"] = gymnasium

from stable_baselines3 import TD3
from sb3_contrib import TQC

import numpy as np
from run.train import config
from time import sleep
import pprint

import gym

import panda_gym

import seaborn as sns

def evaluate_ensemble(models, human=True, num_steps=10_000, goals_to_achieve=None, deterministic=True, strategy="variance_only"):
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
    total_index_count = []
    episode_index_count = []

    # episode_action_sovereignty = 0
    action_sovereignty = None

    model_colors = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 1.0])}

    for num in range(num_steps):
        actions = []
        distribution_variances = []
        variances = []

        for model in models:
            action, _states = model.predict(obs, deterministic=deterministic)
            distribution_variance = model.actor.action_dist.distribution.variance.squeeze().cpu().detach().numpy()
            actions.append(action)
            variances.append(distribution_variance)
            distribution_variances.append(np.sum(distribution_variance))


        if strategy == "sum_actions":
            # trick 17: Act together
            action = np.add(actions[0], actions[1])/2
        elif strategy == "variance_only":
            min_variance = min(distribution_variances)
            action_sovereignty = distribution_variances.index(min_variance)
            action = actions[action_sovereignty]
            env.task.sim.create_debug_text(f"Model Actions",
                                           f">>>Action: Model {action_sovereignty}<<<",
                                           color=model_colors[action_sovereignty])

        total_index_count.append(action_sovereignty)
        episode_index_count.append(action_sovereignty)

        for i in range(len(models)):
            env.task.sim.create_debug_text(f"Model Action {i}",
                                           f">>>Model 0 Variance: {distribution_variances[i]}<<<",
                                           color=model_colors[i])

        # if action_selection_strategy == "first_come":
        #     # The first to reach 5 has sovereignty for the rest of the episode
        #     if len(episode_index_count) == 9:
        #         if episode_index_count.count(0) >= 5:
        #             episode_action_sovereignty = 0
        #         else:
        #             episode_action_sovereignty = 1
        #     elif len(episode_index_count) < 9:
        #         episode_action_sovereignty = action_sovereignty
        #
        #     action_sovereignty = episode_action_sovereignty





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
        ee_velocity = env.robot.get_ee_velocity()
        end_effector_velocities.append(ee_velocity)
        end_effector_speeds.append(np.square(np.linalg.norm(ee_velocity)))
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

            episode_index_count = []
            # episode_action_sovereignty = 0
        # evaluation_step(action_diffs, done_events, end_effector_positions, end_effector_velocities, episode_rewards,
        #                 goals, human, joint_positions, joint_velocities, manipulabilities, model, obs, goals_to_achieve)
    # Compute mean reward for the last 100 episodes

    results = {"mean_reward": np.round(np.mean(episode_rewards), 3),
              "success_rate": np.round(done_events.count(1) / len(done_events), 3),
              "collision_rate": np.round(done_events.count(-1) / len(done_events), 3),
              "timeout_rate": np.round(done_events.count(0) / len(done_events), 3),
              "num_episodes": np.round(len(done_events), 3),
              "mean_action_difference": np.round(np.mean(action_diffs), 4),
              "mean_manipulability": np.round(np.mean(manipulabilities), 4),
              }

    idx = 0
    for _ in models:
        results[f"model_{idx}_action_rate"] = total_index_count.count(idx)/num_steps
        idx = idx+1

    metrics = {
        "end_effector_positions": end_effector_positions,
        "end_effector_speeds": end_effector_speeds,
        "end_effector_velocities": end_effector_velocities,
        "joint_positions": joint_positions,
        "joint_velocities": joint_velocities,

        "goals": goals
    }

    return results, metrics

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
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info, = env.step(action)
            action_diff = env.task.action_diff
            manipulability = env.task.manipulability
            if human:
                pass
                #sleep(0.0075)  # for human eval
            # Stats
            episode_rewards[-1] += reward
            action_diffs.append(action_diff)
            manipulabilities.append(manipulability)
            end_effector_positions.append(env.robot.get_ee_position())
            ee_velocity = env.robot.get_ee_velocity()
            end_effector_velocities.append(ee_velocity)
            end_effector_speeds.append(np.linalg.norm(end_effector_velocities))
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
            action, _states = model.predict(obs, deterministic=deterministic)
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

    results = {"mean_reward": np.round(np.mean(episode_rewards), 3),
              "success_rate": np.round(done_events.count(1) / len(done_events), 3),
              "collision_rate": np.round(done_events.count(-1) / len(done_events), 3),
              "timeout_rate": np.round(done_events.count(0) / len(done_events), 3),
              "num_episodes": np.round(len(done_events), 3),
              "mean_action_difference": np.round(np.mean(action_diffs), 4),
              "mean_manipulability": np.round(np.mean(manipulabilities), 4),
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
    human = True

    env = gym.make(config["env_name"], render=human, control_type=config["control_type"],
                   obs_type=config["obs_type"], goal_distance_threshold=config["goal_distance_threshold"],
                   reward_type=config["reward_type"], limiter=config["limiter"],
                   show_goal_space=False, scenario="library2",
                   show_debug_labels=True, n_substeps=config["n_substeps"])

    # Load Model ensemble
    model1 = TQC.load(r"../run/run_data/wandb/quiet-lion-122/files/best_model.zip", env=env)



    # evaluate ensemble
    results, metrics = evaluate_ensemble([model1], human=human, num_steps=500, deterministic=True, strategy="variance_only")

    # env = gym.make(config["env_name"], render=human, control_type=config["control_type"],
    #                obs_type=("ee",), goal_distance_threshold=config["goal_distance_threshold"],
    #                reward_type=config["reward_type"], limiter=config["limiter"],
    #                show_goal_space=False, scenario="library2",
    #                show_debug_labels=True, n_substeps=config["n_substeps"])
    # model2 = TQC.load(r"../run/run_data/wandb/earnest-feather-1/files/best_model.zip", env=env)
    # #
    # results1, metrics1 = evaluate_ensemble([model2], human=human, num_steps=10000, deterministic=True, strategy="variance_only", goals_to_achieve=metrics["goals"])
    # results2, metrics2 = evaluate(model1, human=human, goals_to_achieve=metrics["goals"], deterministic=True)
    # results3, metrics3 = evaluate(model2, human=human, goals_to_achieve=metrics["goals"], deterministic=True)

    # # Compare Models
    # model1 = TQC.load(r"../run/run_data/wandb/quiet-lion-122/files/model.zip", env=env)
    # model2 = TQC.load(r"../run/run_data/wandb/efficient-fog-124/files/model.zip", env=env)

    # results, metrics = evaluate(model1, model2, human=human, num_steps=50_000, deterministic=True)


    print("Model 1:")
    pprint.pprint(results)
    # print("Model 2:")
    # pprint.pprint(results1)
    # print("library2-expert:")
    # pprint.pprint(results3)

    # Some boilerplate to initialise things
    sns.set()
    plt.figure()

    # This is where the actual plot gets made
    ax = sns.lineplot(data=metrics["end_effector_speeds"])

    # Customise some display properties
    ax.set_title('End Effector Speeds')
    ax.grid(color='#cccccc')
    ax.set_ylabel('Speed')
    ax.set_xlabel("TimeStep")
    # ax.set_xticklabels(df["year"].unique().astype(str), rotation='vertical')

    # Ask Matplotlib to show it
    plt.show()
