import os, sys

import gymnasium

from classes.train_config import TrainConfig

sys.modules["gym"] = gymnasium

from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()


def register_reach_ao(max_ep_steps):
    register(
        id="PandaReachAO-v3",
        entry_point="panda_gym.envs:PandaReachAOEnv",
        max_episode_steps=max_ep_steps,  # default: 50
    )


def register_envs(max_ep_steps):
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            kwargs = {"reward_type": reward_type, "control_type": control_type}

            register(
                id="PandaReach{}{}-v3".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:PandaReachEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )

            register(
                id="MyCobotReach{}{}-v0".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:MyCobotReachEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )

            register(
                id="PandaReachChecker{}{}-v3".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:PandaReachCheckerEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )

            register(
                id="PandaReachAO-v3",
                entry_point="panda_gym.envs:PandaReachAOEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )

            register(
                id="PandaPush{}{}-v3".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:PandaPushEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )

            register(
                id="PandaSlide{}{}-v3".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:PandaSlideEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )

            register(
                id="PandaPickAndPlace{}{}-v3".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:PandaPickAndPlaceEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )

            register(
                id="PandaStack{}{}-v3".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:PandaStackEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 100
            )

            register(
                id="PandaFlip{}{}-v3".format(control_suffix, reward_suffix),
                entry_point="panda_gym.envs:PandaFlipEnv",
                kwargs=kwargs,
                max_episode_steps=max_ep_steps,  # default: 50
            )
