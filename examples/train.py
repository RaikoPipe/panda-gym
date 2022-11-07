# WARNING, This file will not be functional until stable-baselines3 is compatible
# with gymnasium. See https://github.com/DLR-RM/stable-baselines3/pull/780 for more information.
import gymnasium as gym
from stable_baselines3 import HerReplayBuffer, TD3
from wandb.integration.sb3 import WandbCallback
import wandb
from torch.utils import tensorboard

# noinspection PyUnresolvedReferences
import panda_gym

env = gym.make("PandaReach-v3", render=True, goal_range=0.3, control_type="", show_goal_space=True)

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 100000,
    "env_name": env.__class__.__name__,
    "replay_buffer_class": str(HerReplayBuffer.__name__)
}

wandb.login(key="5d65c571cf2a6110b15190696682f6e36ddcdd11")
#wandb.tensorboard.patch(root_logdir="run_data")
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    dir="run_data",
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

model = TD3(config["policy_type"], env=env, replay_buffer_class=HerReplayBuffer, verbose=1,
            tensorboard_log=f"runs/{run.id}", device="cuda")

# fixme: charts not being visualized? Problem might lie in tensorboard -> probably missing admin priv
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=wandb.run.dir
    ),
)

run.finish()
