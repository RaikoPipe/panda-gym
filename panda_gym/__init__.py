from importlib.metadata import PackageNotFoundError, version

from gymnasium.envs.registration import register

try:
    __version__ = version("panda_gym")
except PackageNotFoundError:
    __version__ = "dev"

for task in ["Reach", "Push", "Slide", "PickAndPlace", "Stack", "Flip"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""
            register(
                id=f"Panda{task}{control_suffix}{reward_suffix}-v3",
                entry_point=f"panda_gym.envs:Panda{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100 if task == "Stack" else 50,
            )

# Obstacle avoidance: pass scenario=..., config=ReachAOConfig(...), max_episode_steps=... to gym.make
register(id="PandaReachAO-v3", entry_point="panda_gym.envs:PandaReachAOEnv", max_episode_steps=100)
