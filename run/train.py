import sys
import time
from copy import copy, deepcopy

import gymnasium
import numpy as np

# from pygame import mixer
sys.modules["gym"] = gymnasium

from stable_baselines3 import PPO, TD3
from sb3_contrib import TQC

import panda_gym
import os

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer, DictReplayBuffer
from typing import Callable

from torch import nn
from time import sleep

reach_stages = ["reach1", "reach2", "reach3", "reach4"]
reach_max_ep_steps = [50, 50, 50, 50]
reach_succ_thresholds = [1.0, 1.0, 1.0, 1.0]

reach_ao_stages = ["base1", "base2", "wangexp_3"]
reach_ao_max_ep_steps = [50, 100, 150]
reach_ao_succ_thresholds = [0.9, 0.9, 1.1]

reach_ao_stages_test1 = ["base1", "base2", "wangexp_3"]
reach_ao_max_ep_steps_test1 = [100, 150, 200]
reach_ao_succ_thresholds_test1 = [0.9, 0.9, 1.1]

reach_optim_stages = ["wangexp_3"]
reach_optim_max_ep_steps = [400]
reach_optim_succ_thresholds = [999]

# hyperparameters from rl-baselines3-zoo tuned pybullet defaults

configuration = {
    "env_name": "PandaReachAO-v3",
    "algorithm": "TQC",
    "reward_type": "sparse",  # sparse; dense
    "goal_distance_threshold": 0.05,
    "max_timesteps": 1_200_000,
    "seed": 8,
    "render": False,  # renders the eval env
    "n_substeps": 50,  # number of simulation steps before handing control back to agent
    "obs_type": ["ee", "js"],  # Robot state to observe
    "control_type": "js",
    # Agent Output; js: joint velocities, ee: end effector displacements; jsd: joint velocities (applied directly)
    "limiter": "sim",
    "action_limiter": "clip",
    "show_goal_space": False,
    "replay_buffer_class": HerReplayBuffer,  # HerReplayBuffer
    "policy_type": "MultiInputPolicy",
    "show_debug_labels": False,
    "n_envs": 8,
    "eval_freq": 5_000,
    "stages": reach_ao_stages_test1,
    "success_thresholds": reach_ao_succ_thresholds_test1,  # [-7, -10, -12, -17, -20]
    "max_ep_steps": reach_ao_max_ep_steps_test1,
    "task_observations": {'obstacles': "vectors", 'prior': "rrmc_neo"},
    # "all": closest distance to any obstacle of all joints is observed; "vectors": directional vectors pointing to closest obstacles
    "learning_starts": 10000,
    "prior_steps": 0,
    "randomize_robot_pose": False,
    "truncate_on_collision": True,
    "terminate_on_success": True,
    "collision_reward": -100,
    "goal_condition": "reach"  # reach; reach_hold #
    # "closest": only closest joint distance is observed
}


# hyperparameters are from rl-baselines3 zoo and https://arxiv.org/pdf/2106.13687.pdf


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


"""defaults pybullet envs"""
hyperparameters_pybullet_defaults_tqc = {  # same as sac

    "learning_rate": linear_schedule(0.001),  # 0.0007, #0.00073 # linear_schedule(0.001)
    "gamma": 0.98,
    "tau": 0.02,
    "buffer_size": 300_000,  # 300_000
    "batch_size": 256,
    "gradient_steps": 8,
    "train_freq": 8,
    "ent_coef": "auto",
    "use_sde": True,

    "policy_kwargs": dict(log_std_init=-3, net_arch=[256, 256])  # 400, 300
}

"""defaults pybullet envs"""
hyperparameters_pybullet_defaults_tqc_optim = {  # same as sac

    "learning_rate": float(0.0001),  # 0.0007, #0.00073 # linear_schedule(0.001)
    "gamma": 0.98,
    "tau": 0.02,
    "buffer_size": 300_000,  # 300_000
    "batch_size": 64,
    "gradient_steps": 8,
    "train_freq": 8,
    "ent_coef": "auto",
    "use_sde": False,

    "policy_kwargs": dict(log_std_init=-3, net_arch=[256, 256])  # 400, 300
}

hyperparameters_her_defaults = {
    "buffer_size": 1_000_000,
    "batch_size": 2048,
    "gamma": 0.95,
    "learning_rate": float(1e-3),
    "tau": 0.05,
    "policy_kwargs": dict(net_arch=[512, 512, 512], n_critics=2)

}

hyperparameters_pybullet_defaults_td3 = {

    "learning_rate": float(1e-3),  # 0.0007, #0.00073 # linear_schedule(0.001)
    "gamma": 0.98,
    "buffer_size": 200_000,  # 300_000
    "gradient_steps": -1,
    "train_freq": (1, "episode"),

    "policy_kwargs": dict(net_arch=[256, 256])  # 400, 300
}

hyperparameters_pybullet_defaults_ppo = {
    "normalize": True,
    "n_envs": 16,
    "policy": 'MlpPolicy',
    "batch_size": 128,
    "n_steps": 512,
    "gamma": 0.99,
    "gae_lambda": 0.9,
    "n_epochs": 20,
    "ent_coef": 0.0,
    "sde_sample_freq": 4,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "learning_rate": float(3e-5),
    "use_sde": True,
    "clip_range": 0.4,
    "policy_kwargs": dict(log_std_init=-2,
                          ortho_init=False,
                          activation_fn=nn.ReLU,
                          net_arch=dict(pi=[256, 256], vf=[256, 256])
                          ),
}

hyperparameters_pybullet_defaults_ddpg = {
    "learning_rate": float(1e-3),  # 0.0007, #0.00073 # linear_schedule(0.001)
    "gamma": 0.98,
    "buffer_size": 200_000,  # 300_000
    "gradient_steps": 1,
    "train_freq": 1,
    "noise_std": 0.1,

    "policy_kwargs": dict(net_arch=[256, 256])  # 400, 300
}

# # register envs to gymnasium
panda_gym.register_envs(configuration["max_ep_steps"][0])

# hyperparameters_tqc = {
#     "learning_rate": float(1e-3),
#     "batch_size": 2048,
#     "buffer_size": int(1e6),
#     "replay_buffer_kwargs": dict(
#         goal_selection_strategy='future', n_sampled_goal=4),
#     "gamma": 0.95,
#     "tau": 0.05,
#     "policy_kwargs": dict(net_arch=[400, 300]),
#     "use_sde": True
# }

if configuration["algorithm"] == "DDPG":
    configuration["hyperparams"] = hyperparameters_pybullet_defaults_ddpg
if configuration["algorithm"] == "TD3":
    configuration["hyperparams"] = hyperparameters_pybullet_defaults_td3
elif configuration["algorithm"] in ["SAC", "TQC"]:
    configuration["hyperparams"] = hyperparameters_pybullet_defaults_tqc
elif configuration["algorithm"] == "PPO":
    configuration.pop("replay_buffer_class", None)  # ppo has no replay buffer
    configuration["hyperparams"] = hyperparameters_pybullet_defaults_ppo


def main():
    # env = get_env(config, config["n_envs"], config["stages"][0])
    # model = TQC.load(r"run_data/wandb/morning-grass/files/best_model.zip", env=env, replay_buffer_class=config["replay_buffer_class"],
    #                  custom_objects={"action_space":gymnasium.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),} # workaround
    #                  )

    model = learn(config=configuration, algorithm=configuration["algorithm"])
    # Delete model after training is finished
    del model


def base_train(config):

    model, run = learn(config=config, algorithm=config["algorithm"])

    return model, run


def optimize_train(path_to_model, config, wandb_run=None):
    config["reward_type"] = "kumar_optim"
    config["stages"] = ["wangexp_3"]
    config["success_thresholds"] = [999]
    config["max_ep_steps"] = [100]
    config["max_timesteps"] = 1_200_000
    config["replay_buffer_class"] = DictReplayBuffer
    config["goal_condition"] = "reach"
    config["n_substeps"] = 20
    config["truncate_on_collision"] = True
    config["terminate_on_success"] = False

    # model.save("temp_model")

    # if config["replay_buffer_class"] in (VecHerReplayBuffer, HerReplayBuffer):
    #     model.save_replay_buffer("temp_buffer")

    # model.env.close()
    env = get_env(config, config["n_envs"], config["stages"][0], )
    model = TQC.load(f"{path_to_model}/files/best_model.zip", env=env,
                     replay_buffer_class=configuration["replay_buffer_class"],
                     )

    # model = model.load("temp_model", env=env, replay_buffer_class = config["replay_buffer_class"])
    # odel.learning_starts = 10_000

    if config["replay_buffer_class"] in (HerReplayBuffer):
        model.load_replay_buffer(f"{path_to_model}/files/replay_buffer.pkl")
        model.replay_buffer.set_env(model.env)
    # else:
    #     model.load_replay_buffer(f"{path_to_model}/files/replay_buffer.pkl")
    #     her_observations = deepcopy(model.replay_buffer.observations)
    #     dict_buffer = DictReplayBuffer(buffer_size=config["hyperparams"]["buffer_size"],
    #                                    action_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
    #                                    observation_space=env.observation_space.spaces["observation"])

    return continue_learning(model, config, wandb_run)


def train_benchmark_scenarios():
    model_names = ["snowy-pine-131", "deep-frog-298", "glamorous-resonance-295", "firm-pond-79"]
    path_names = []
    for name in model_names:
        path_names.append(fr"../run/run_data/wandb/{name}")

    for scenario, model_name in zip(["narrow_tunnel", "library2", "workshop", "wall"], path_names):
        configuration["stages"] = [scenario]

        env = get_env(configuration, 8, scenario)
        model = TQC.load(f"{model_name}/files/best_model.zip", env=env,
                         replay_buffer_class=configuration["replay_buffer_class"],
                         )

        if configuration["replay_buffer_class"] in (HerReplayBuffer):
            model.load_replay_buffer(f"../run/run_data/wandb/glamorous-resonance-295/files/replay_buffer.pkl")
            model.replay_buffer.set_env(model.env)

        # model.learning_starts = 10.000

        model, run = continue_learning(model, configuration)

        run.finish()

        model.env.close()

        del model


def train_base_model():
    for seed in range(0, 1):
        configuration["seed"] = seed
        model, run = base_train(configuration)
            # model, run = optimize_train(model, run, configuration)

        run.finish()

        model.env.close()
        del model


def l2():
    path_names = []
    for path_to_model in ["gallant-serenity-299", "deep-frog-298", "solar-microwave-297", "revived-serenity-296",
                          "glamorous-resonance-295"]:
        path_names.append(fr"../run/run_data/wandb/{path_to_model}")

    for path_to_model, seed in zip(path_names, range(5)):
        # set seed according to number of envs, because gym will increment seed for each vec environment
        configuration["seed"] = seed
        # env = get_env(configuration, configuration["n_envs"], configuration["stages"][0])
        # model = TQC.load(f"{path_to_model}/files/best_model.zip", env=env,
        #                  replay_buffer_class=configuration["replay_buffer_class"],
        #                  custom_objects={
        #                      "action_space": gymnasium.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32), }
        # workaround
        # )
        # model.load_replay_buffer(f"{path_to_model}/files/replay_buffer.pkl")

        # model, run = base_train()
        model, run = optimize_train(path_to_model=path_to_model, config=configuration)

        run.finish()

        model.env.close()
        del model


def train_hf(path_to_model, config, wandb_run=None):
    config["reward_type"] = "sparse"
    config["stages"] = ["wangexp_3"]
    config["max_ep_steps"] = [400]
    config["max_timesteps"] = 300_000
    config["replay_buffer_class"] = DictReplayBuffer
    config["goal_condition"] = "reach"
    config["n_substeps"] = 5
    config["truncate_on_collision"] = True
    config["terminate_on_success"] = True

    # model.save("temp_model")

    # if config["replay_buffer_class"] in (VecHerReplayBuffer, HerReplayBuffer):
    #     model.save_replay_buffer("temp_buffer")

    # model.env.close()
    env = get_env(config, config["n_envs"], config["stages"][0], )
    model = TQC.load(f"{path_to_model}/files/best_model.zip", env=env,
                     replay_buffer_class=configuration["replay_buffer_class"],
                     )

    # model = model.load("temp_model", env=env, replay_buffer_class = config["replay_buffer_class"])
    # odel.learning_starts = 10_000

    if config["replay_buffer_class"] in (HerReplayBuffer):
        model.load_replay_buffer(f"{path_to_model}/files/replay_buffer.pkl")
        model.replay_buffer.set_env(model.env)
    # else:
    #     model.load_replay_buffer(f"{path_to_model}/files/replay_buffer.pkl")
    #     her_observations = deepcopy(model.replay_buffer.observations)
    #     dict_buffer = DictReplayBuffer(buffer_size=config["hyperparams"]["buffer_size"],
    #                                    action_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),
    #                                    observation_space=env.observation_space.spaces["observation"])

    return continue_learning(model, config, wandb_run)


if __name__ == "__main__":
    # print("Sleeping...")
    # time.sleep(3600*8)
    from run.learning_methods.learning import learn, get_env, continue_learning
    import wandb

    wandb.login(key=os.getenv("wandb_key"))

    # train_benchmark_scenarios()
    train_base_model()
    # l2()

    # for seed in range(5,20):
    #     configuration["seed"] = seed
    #     model, run = base_train(configuration)
    #     #model, run = optimize_train(model, run, configuration)
    #
    #     run.finish()
    #
    #     model.env.close()
    #     del model

    # for i in range(5):
    #     configuration["seed"] = i
    #
    #     model, run = base_train(configuration)
    #     #model, run = optimize_train(model, run, configuration)
    #
    #     run.finish()
    #
    #     del model

    # training mode
    # for i in range(1, 5):
    # config["seed"] = 4
    # model, run = base_train()
    # # model.replay_buffer.reset()
    # # model, run = optimize_train(model, run)
    #
    # run.finish()
    #
    # del model

    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)
