import os
import sys
import time

import gymnasium
sys.modules["gym"] = gymnasium

from sb3_contrib import TQC


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import panda_gym
import os

from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from classes.train_config import TrainConfig
from classes.hyperparameters import Hyperparameters

reach_stages = ["reach1", "reach2", "reach3", "reach4"]
reach_max_ep_steps = [50, 50, 50, 50]
reach_succ_thresholds = [1.0, 1.0, 1.0, 1.0]

reach_ao_stages = ["base1", "base2", "wangexp_3"]
reach_ao_max_ep_steps = [50, 100, 150]
reach_ao_succ_thresholds = [0.9, 0.9, 1.1]

reach_ao_stages_test1 = ["base1", "base2", "wangexp_3"]
reach_ao_max_ep_steps_test1 = [100, 150, 200]
reach_ao_succ_thresholds_test1 = [0.9, 0.9, 1.1]

speed_thresholds = [0.5, 0.1, 0.01]

reach_optim_stages = ["wangexp_3"]
reach_optim_max_ep_steps = [400]
reach_optim_succ_thresholds = [999]

# hyperparameters from rl-baselines3-zoo tuned pybullet defaults

configuration = TrainConfig()

# # register envs to gymnasium
panda_gym.register_reach_ao(configuration.max_ep_steps[0])


def main():
    # env = get_env(config, config.n_envs, config.stages[0])
    # model = TQC.load(r"run_data/wandb/morning-grass/files/best_model.zip", env=env, replay_buffer_class=config.replay_buffer_class,
    #                  custom_objects={"action_space":gymnasium.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32),} # workaround
    #                  )

    model = learn(config=configuration, algorithm=configuration.algorithm)
    # Delete model after training is finished
    del model


def base_train(config):
    model, run = learn(config=config)

    return model, run


def train_benchmark_scenarios():
    model_names = ["snowy-pine-131", "deep-frog-298", "glamorous-resonance-295", "firm-pond-79"]
    path_names = []
    for name in model_names:
        path_names.append(fr"../run/run_data/wandb/{name}")

    for scenario, model_name in zip(["narrow_tunnel", "library2", "workshop", "wall"], path_names):
        configuration.stages = [scenario]

        env = get_env(configuration, 8, scenario)
        model = TQC.load(f"{model_name}/files/best_model.zip", env=env,
                         replay_buffer_class=configuration.replay_buffer_class,
                         )

        if configuration.replay_buffer_class in (HerReplayBuffer):
            model.load_replay_buffer(f"../run/run_data/wandb/glamorous-resonance-295/files/replay_buffer.pkl")
            model.replay_buffer.set_env(model.env)

        # model.learning_starts = 10.000

        model, run = continue_learning(model, configuration)

        run.finish()

        model.env.close()

        del model


def train_base_model(config=TrainConfig(), iterations=None):
    for seed in range(0, iterations):
        if config.name == 'default':
            # assign random name
            name = f"base_model_{time.asctime().replace(' ', '_').replace(':', '_')}"
            config.name = f'{name}_{seed}'

        config.seed = seed
        model, run = base_train(config)
        # model, run = optimize_train(model, run, configuration)

        run.finish()

        model.env.close()
        del model


if __name__ == "__main__":
    from training.utils.setup_training import learn, get_env, continue_learning
    import wandb

    wandb.login(key=os.getenv("wandb_key"))

    # adjustments from basic training configuration
    # stages = ["base1", "base2", "wangexp_3", 'wangexp_3', 'wangexp_3', 'wangexp_3', 'wangexp_3']
    # speed_thresholds = [0.5, 0.5, 0.5, 0.1, 0.01, 0.001, 0]
    # ee_error_thresholds = [0.05 for i in range(7)]
    # max_ep_steps = [50, 100, 150, 200, 250, 300, 300]
    # success_thresholds = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0]
    # goal_condition = 'halt'
    #



    # train_config_blind = TrainConfig(
    #     stages=["wangexp_3"],
    #     max_ep_steps=[300],
    #     success_thresholds=[1.0],
    #     ee_error_thresholds=[0.05])
    # train_config_reach = TrainConfig(
    #     stages=["reachao1", "reachao2", "reachao3", "reachao_rand"],
    #     max_ep_steps=[75, 150, 200, 250],
    #     success_thresholds=[0.8, 0.8, 0.8, 0.8],
    #     ee_error_thresholds=[0.05, 0.05, 0.05, 0.05],
    # )



    for configuration in [TrainConfig()]:
        train_base_model(config=configuration, iterations=1)
