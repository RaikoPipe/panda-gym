import sys
import time

from matplotlib import pyplot as plt
import gymnasium

sys.modules["gym"] = gymnasium

from stable_baselines3 import TD3
from sb3_contrib import TQC

import numpy as np
from run.train import configuration
from time import sleep
import pprint
from torch.distributions.normal import Normal
import torch
import pandas as pd
from tabulate import tabulate

import panda_gym
from tqdm import tqdm
import json
from copy import copy

import seaborn as sns


def fuse_controllers(prior_mu, prior_sigma, policy_mu, policy_sigma):
    # The policy mu and sigma are from the stochastic SAC output
    # The sigma from prior is fixed
    mu = (np.power(policy_sigma, 2) * prior_mu + np.power(prior_sigma, 2) * policy_mu) / (
                np.power(policy_sigma, 2) + np.power(prior_sigma, 2))
    sigma = np.sqrt(
        (np.power(prior_sigma, 2) * np.power(policy_sigma, 2)) / (np.power(policy_sigma, 2) + np.power(prior_sigma, 2)))
    return mu, sigma


def evaluate_ensemble(models, env, human=True, num_episodes=1000, goals_to_achieve=None, deterministic=True,
                      strategy="variance_only", scenario_name="", prior_orientation=None):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    # robot parameters
    # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
    # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])

    episode_rewards = [0.0]

    if goals_to_achieve:
        env.task.goal = goals_to_achieve.pop(0)

    done_events = []

    goals = []
    goals.append(env.task.goal)

    end_effector_positions = []
    end_effector_velocities = []
    end_effector_speeds = []

    efforts = []
    manipulabilities = []

    total_index_count = []
    episode_index_count = []
    ep_lengths = []
    ep_length = 0

    # episode_action_sovereignty = 0
    action_sovereignty = None

    model_colors = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 0.0, 1.0])}

    obs, _ = env.reset()
    for _ in tqdm(range(num_episodes), desc=scenario_name):
        episode_reward = 0.0
        ee_pos = []
        ee_speed = []
        total_effort = 0.0
        total_manipulability = 0.0

        # get next goal
        env.task.goal = np.array(goals_to_achieve.pop(0))

        while True:
            actions = []
            distribution_stds = []
            distribution_variances = []
            variances = []

            if isinstance(models[0], str):
                action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles, env.task.collision_detector,
                                                      prior_orientation)
            for model in models:
                action, _states = model.predict(obs, deterministic=deterministic)

                distribution_std = model.actor.action_dist.distribution.stddev.squeeze().cpu().detach().numpy()
                distribution_variance = model.actor.action_dist.distribution.variance.squeeze().cpu().detach().numpy()
                actions.append(action)
                variances.append(distribution_variance)
                distribution_variances.append(np.sum(distribution_variance))
                distribution_stds.append(distribution_std)

            if strategy == "sum_actions":
                # trick 17: Act together
                action = np.add(actions[0], actions[1]) / 2
            elif strategy == "variance_only":
                min_variance = min(distribution_variances)
                action_sovereignty = distribution_variances.index(min_variance)
                action = actions[action_sovereignty]
                env.task.sim.create_debug_text(f"Model Actions",
                                               f">>>Action: Model {action_sovereignty}<<<",
                                               color=model_colors[action_sovereignty])
            elif strategy == "prior_bcf":
                rl_sigma = distribution_stds[0]
                rl_mu = variances[0]

                prior_action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles,
                                                            env.task.collision_detector, "")

                mu_mcf, std_mcf = fuse_controllers(prior_action, 0.4, rl_mu, rl_sigma)

                dist_hybrid = Normal(torch.tensor(mu_mcf).double().detach(), torch.tensor(std_mcf).double().detach())

                action = dist_hybrid.sample()
                action = torch.tanh(action).numpy()
            elif strategy == "prior_sum":
                prior_action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles,
                                                            env.task.collision_detector, "back")
                action = np.add(actions[0], prior_action) / 2
            elif strategy == "prior":
                prior_action = env.robot.compute_action_neo(env.task.goal, env.task.obstacles,
                                                            env.task.collision_detector,
                                                            "")
                zipped = [sorted(zip(i, y)) for i, y in
                          zip([distribution_variances[0], 0.4], [actions[0], prior_action])]
                action = zipped[1][0]

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

            if human:
                pass
                # sleep(0.01/8)  # for human eval

            # add results and metrics
            episode_reward += reward

            total_effort += env.task.get_norm_effort()
            total_manipulability += env.task.manipulability
            ee_pos.append(env.robot.get_ee_position())
            ee_speed.append(np.linalg.norm(env.robot.get_ee_velocity()))
            ep_length += 1


            if done or truncated:
                # sleep(2)
                if info["is_success"]:
                    #print("Success!")
                    done_events.append(1)
                elif info["is_truncated"]:
                    #print("Collision...")
                    done_events.append(-1)
                    if human:
                        sleep(1)
                else:
                    #print("Timeout...")
                    done_events.append(0)
                obs, _ = env.reset()

                episode_index_count = []

                # add episode results and metrics
                episode_rewards.append(episode_reward)
                ep_lengths.append(ep_length)
                end_effector_speeds.append(ee_speed)
                end_effector_positions.append(ee_pos)
                efforts.append(total_effort/ep_length)
                manipulabilities.append(total_manipulability/ep_length)

                # finish episode
                break

            # episode_action_sovereignty = 0
        # evaluation_step(efforts, done_events, end_effector_positions, end_effector_velocities, episode_rewards,
        #                 goals, human, joint_positions, joint_velocities, manipulabilities, model, obs, goals_to_achieve)
    # Compute mean reward for the last 100 episodes

    results = {"mean_reward": np.round(np.mean(episode_rewards), 3),
               "success_rate": np.round(done_events.count(1) / len(done_events), 3),
               "collision_rate": np.round(done_events.count(-1) / len(done_events), 3),
               "timeout_rate": np.round(done_events.count(0) / len(done_events), 3),
               "num_episodes": np.round(len(done_events), 3),
               "mean_ep_length": np.round(np.mean(ep_length), 3),
               "mean_num_sim_steps": np.round(np.mean([i * configuration["n_substeps"] for i in ep_lengths]), 3),
               "mean_effort": np.round(np.mean(efforts), 4),
               "mean_manipulability": np.round(np.mean(manipulabilities), 4),
               }

    idx = 0
    for _ in models:
        results[f"model_{idx}_action_rate"] = total_index_count.count(idx) / num_episodes
        idx = idx + 1

    metrics = {
        "end_effector_positions": end_effector_positions,
        "end_effector_speeds": end_effector_speeds,
        "end_effector_velocities": end_effector_velocities,
        "goals": goals
    }

    return results, metrics


# def evaluate(model, human=True, num_steps=10_000, goals_to_achieve=None, deterministic=True):
#     """
#     Evaluate a RL agent
#     :param model: (BaseRLModel object) the RL Agent
#     :param num_steps: (int) number of timesteps to evaluate it
#     :return: (float) Mean reward for the last 100 episodes
#     """
#     # robot parameters
#     # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
#     # env.robot.neutral_joint_values = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4, 0.00, 0.00])
#
#     episode_rewards = [0.0]
#     obs, _ = env.reset()
#
#     if goals_to_achieve:
#         env.task.goal = goals_to_achieve.pop(0)
#
#     done_events = []
#     action_diffs = []
#     manipulabilities = []
#     goals = []
#     goals.append(env.task.goal)
#
#     end_effector_positions = []
#     end_effector_velocities = []
#     end_effector_speeds = []
#
#     joint_positions = []
#     joint_velocities = []
#
#     if goals_to_achieve:
#         while goals_to_achieve:
#             action, _states = model.predict(obs, deterministic=deterministic)
#             obs, reward, done, truncated, info, = env.step(action)
#             action_diff = env.task.action_diff
#             manipulability = env.task.manipulability
#             if human:
#                 pass
#                 #sleep(0.0075)  # for human eval
#             # Stats
#             episode_rewards[-1] += reward
#             action_diffs.append(action_diff)
#             manipulabilities.append(manipulability)
#             end_effector_positions.append(env.robot.get_ee_position())
#             ee_velocity = env.robot.get_ee_velocity()
#             end_effector_velocities.append(ee_velocity)
#             end_effector_speeds.append(np.linalg.norm(end_effector_velocities))
#             # end_effector_speeds.append(np.square(np.linalg.norm(ee_velocity)))
#             joint_positions.append(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))
#             joint_velocities.append(np.array([env.robot.get_joint_velocity(joint=i) for i in range(7)]))
#             if done or truncated:
#                 # sleep(2)
#                 if info["is_success"]:
#                     print("Success!")
#                     done_events.append(1)
#                 elif info["is_truncated"]:
#                     print("Collision...")
#                     done_events.append(-1)
#                 else:
#                     print("Timeout...")
#                     done_events.append(0)
#                 obs, _ = env.reset()
#
#                 if goals_to_achieve:
#                     # set goal from list
#                     env.task.goal = goals_to_achieve.pop(0)
#
#                 goals.append(env.task.goal)
#                 episode_rewards.append(0.0)
#             # evaluation_step(action_diffs, done_events, end_effector_positions, end_effector_velocities, episode_rewards,
#             #                 goals, human, joint_positions, joint_velocities, manipulabilities, model, obs,
#             #                 goals_to_achieve)
#     else:
#
#         for num in range(num_steps):
#             action, _states = model.predict(obs, deterministic=deterministic)
#             obs, reward, done, truncated, info, = env.step(action)
#             action_diff = env.task.action_diff
#             manipulability = env.task.manipulability
#             # Stats
#             episode_rewards[-1] += reward
#             action_diffs.append(action_diff)
#             manipulabilities.append(manipulability)
#             end_effector_positions.append(env.robot.get_ee_position())
#             ee_velocity = env.robot.get_ee_velocity
#             end_effector_velocities.append(ee_velocity)
#             # end_effector_speeds.append(np.square(np.linalg.norm(ee_velocity)))
#             joint_positions.append(np.array([env.robot.get_joint_angle(joint=i) for i in range(7)]))
#             joint_velocities.append(np.array([env.robot.get_joint_velocity(joint=i) for i in range(7)]))
#             if done or truncated:
#                 # sleep(2)
#                 if info["is_success"]:
#                     print("Success!")
#                     done_events.append(1)
#                 elif info["is_truncated"]:
#                     print("Collision...")
#                     if human:
#                         time.sleep((1))
#                     done_events.append(-1)
#                 else:
#                     print("Timeout...")
#                     done_events.append(0)
#                 obs, _ = env.reset()
#
#                 if goals_to_achieve:
#                     # set goal from list
#                     env.task.goal = goals_to_achieve.pop(0)
#
#                 goals.append(env.task.goal)
#                 episode_rewards.append(0.0)
#         # evaluation_step(action_diffs, done_events, end_effector_positions, end_effector_velocities, episode_rewards,
#         #                 goals, human, joint_positions, joint_velocities, manipulabilities, model, obs, goals_to_achieve)
#     # Compute mean reward for the last 100 episodes
#
#     results = {"mean_reward": np.round(np.mean(episode_rewards), 3),
#               "success_rate": np.round(done_events.count(1) / len(done_events), 3),
#               "collision_rate": np.round(done_events.count(-1) / len(done_events), 3),
#               "timeout_rate": np.round(done_events.count(0) / len(done_events), 3),
#               "num_episodes": np.round(len(done_events), 3),
#               "mean_action_difference": np.round(np.mean(action_diffs), 4),
#               "mean_manipulability": np.round(np.mean(manipulabilities), 4),
#               }
#
#     metrics = {
#         "end_effector_positions": end_effector_positions,
#         "end_effector_speeds": end_effector_speeds,
#         "end_effector_velocities": end_effector_velocities,
#         "joint_positions": joint_positions,
#         "joint_velocities": joint_velocities,
#
#         "goals": goals
#     }
#
#     return results, metrics


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


panda_gym.register_envs(100)

# env = get_env(config, "cube_3_random")

mt_cc_agents = ["gallant-serenity-299", "deep-frog-298", "solar-microwave-297", "revived-serenity-296",
                          "glamorous-resonance-295"]

cc_agents = ["firm-pond-79", "confused-firebrand-91", "rare-moon-92", "gallant-shape-95", "silver-hill-96"]

mt_agents = ["solar-disco-133", "stellar-river-132", "snowy-pine-131", "comic-frost-130", "giddy-darkness-129"]

simple_agents = ["dulcet-plant-105", "restful-bird-103", "graceful-dream-100", "noble-field-99", "easy-lion-98"]

def evaluate_prior(human=False):
    with open("scenario_goals", "r") as file:
        scenario_goals = json.load(file)

    evaluation_results = {}
    for evaluation_scenario, prior_orientation in zip(["wangexp_3", "narrow_tunnel", "library2", "workshop",
                                "wall"], ["ORIENTATIONS"]):  # "wang_3", "library2", "library1", "narrow_tunnel", "wall"
        env = gymnasium.make(configuration["env_name"], render=human, control_type=configuration["control_type"],
                             obs_type=configuration["obs_type"], goal_distance_threshold=0.05,
                             goal_condition="reach",
                             reward_type="sparse", limiter=configuration["limiter"],
                             show_goal_space=True, scenario=evaluation_scenario,
                             randomize_robot_pose=False,  # if evaluation_scenario != "wang_3" else True,
                             joint_obstacle_observation="vectors+all",
                             truncate_episode_on_collision=True,
                             show_debug_labels=True, n_substeps=20)
        print(f"Evaluating {evaluation_scenario}")

        goals_to_achieve = copy(scenario_goals[evaluation_scenario])
        results, metrics = evaluate_ensemble(["prior"], env, human=human, num_episodes=500, deterministic=True,
                                             strategy="variance_only", goals_to_achieve=goals_to_achieve, scenario_name=evaluation_scenario,
                                             prior_orientation=prior_orientation)

        evaluation_results[evaluation_scenario] = {"results": results, "metrics": metrics}
        env.close()


        for key, value in evaluation_results.items():
            results = value["results"]
            print(f"{key}: {results}")

        results = {}
        for key, value in evaluation_results.items():
            results[key] = value["results"]

        table = pd.DataFrame(results)
        table.index.name = "Criterias"
        print(table.to_markdown())
        table.to_excel(f"results/prior.csv")



def evaluate_rl_agent():
    human = False

    # workaround
    init_env = gymnasium.make(configuration["env_name"], render=human, control_type=configuration["control_type"],
                         obs_type=configuration["obs_type"],
                         goal_distance_threshold=configuration["goal_distance_threshold"],
                         reward_type="sparse", limiter=configuration["limiter"],
                         show_goal_space=False, scenario="wang_4",
                         randomize_robot_pose=configuration["randomize_robot_pose"],
                         joint_obstacle_observation="vectors+all",
                         truncate_episode_on_collision=configuration["truncate_episode_on_collision"],
                         show_debug_labels=True, n_substeps=configuration["n_substeps"])

    with open("scenario_goals", "r") as file:
        scenario_goals = json.load(file)

    for model_name in cc_agents:


        # Load model
        model = TQC.load(fr"../run/run_data/wandb/{model_name}/files/best_model.zip", env=init_env,
                         custom_objects={"action_space": gymnasium.spaces.Box(-1.0, 1.0, shape=(7,),
                                                                              dtype=np.float32)})  # for some reason it won't read action space sometimes

        evaluation_results = {}
        for evaluation_scenario in ["wangexp_3", "narrow_tunnel", "library2", "workshop",
                                    "wall"]:  # "wang_3", "library2", "library1", "narrow_tunnel", "wall"
            env = gymnasium.make(configuration["env_name"], render=human, control_type=configuration["control_type"],
                                 obs_type=configuration["obs_type"], goal_distance_threshold=0.05,
                                 goal_condition="reach",
                                 reward_type="sparse", limiter=configuration["limiter"],
                                 show_goal_space=True, scenario=evaluation_scenario,
                                 randomize_robot_pose=False,  # if evaluation_scenario != "wang_3" else True,
                                 joint_obstacle_observation="vectors+all",
                                 truncate_episode_on_collision=True,
                                 show_debug_labels=True, n_substeps=20)
            print(f"Evaluating {evaluation_scenario}")
            model.set_env(env)
            goals_to_achieve = copy(scenario_goals[evaluation_scenario])
            results, metrics = evaluate_ensemble([model], env, human=human, num_episodes=500, deterministic=True,
                                                 strategy="variance_only", goals_to_achieve=goals_to_achieve, scenario_name=evaluation_scenario)

            evaluation_results[evaluation_scenario] = {"results": results, "metrics": metrics}
            env.close()

        for key, value in evaluation_results.items():
            results = value["results"]
            print(f"{key}: {results}")

        results = {}
        for key, value in evaluation_results.items():
            results[key] = value["results"]

        table = pd.DataFrame(results)
        table.index.name = "Criterias"
        print(table.to_markdown())
        table.to_excel(f"results/{model_name}.csv")



if __name__ == "__main__":
    evaluate_rl_agent()
    # evaluate ensemble

    # env = gym.make(config["env_name"], render=human, control_type=config["control_type"],
    #                obs_type=("ee",), goal_distance_threshold=config["goal_distance_threshold"],
    #                reward_type=config["reward_type"], limiter=config["limiter"],
    #                show_goal_space=False, scenario="library2",
    #                show_debug_labels=True, n_substeps=config["n_substeps"])
    # model2 = TQC.load(r"../run/run_data/wandb/earnest-feather-1/files/best_model.zip", env=env)
    # results1, metrics1 = evaluate_ensemble([model2], human=human, num_steps=10000, deterministic=True, strategy="variance_only", goals_to_achieve=metrics["goals"])
    # results2, metrics2 = evaluate(model1, human=human, goals_to_achieve=metrics["goals"], deterministic=True)
    # results3, metrics3 = evaluate(model2, human=human, goals_to_achieve=metrics["goals"], deterministic=True)

    # # Compare Models
    # model1 = TQC.load(r"../run/run_data/wandb/quiet-lion-122/files/model.zip", env=env)
    # model2 = TQC.load(r"../run/run_data/wandb/efficient-fog-124/files/model.zip", env=env)

    # results, metrics = evaluate(model1, model2, human=human, num_steps=50_000, deterministic=True)

    # print("Model 1:")
    # pprint.pprint(results)
    # # print("Model 2:")
    # # pprint.pprint(results1)
    # # print("library2-expert:")
    # # pprint.pprint(results3)
    #
    # # Some boilerplate to initialise things
    # sns.set()
    # plt.figure()
    #
    # # This is where the actual plot gets made
    # ax = sns.lineplot(data=metrics["end_effector_speeds"])
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
