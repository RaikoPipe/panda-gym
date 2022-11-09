import gymnasium as gym
from stable_baselines3 import HerReplayBuffer, TD3
from wandb.integration.sb3 import WandbCallback
import wandb
from torch.utils import tensorboard

# noinspection PyUnresolvedReferences
import panda_gym

env_name = "PandaReachEvadeObstacles-v3"
total_timesteps = 400_000
render = True  # renders the pybullet env
goal_range = 0.3  # Size of the cuboid in which the goal is sampled
control_type = ""  # "ee": end effector displacement; "js": joint angles
show_goal_space = False
replay_buffer = HerReplayBuffer
policy_type = "MultiInputPolicy"

obstacle_layout = 1
joint_obstacle_observation = "closest"  # "all": closest distance to any obstacle of all joints is observed;
# "closest": only closest joint distance is observed

env = gym.make(env_name, render=render, goal_range=goal_range, control_type=control_type,
               show_goal_space=show_goal_space, obstacle_layout=1, show_debug_labels=True)

config = {
    "policy_type": policy_type,
    "total_timesteps": total_timesteps,
    "env_name": env_name,
    "replay_buffer_class": str(replay_buffer.__name__),
    "control_type": control_type
}

if __name__ == "__main__":
    wandb.login(key="5d65c571cf2a6110b15190696682f6e36ddcdd11")
    # wandb.tensorboard.patch(root_logdir="run_data")
    run = wandb.init(
        project="panda-gym",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        dir="run_data",
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    model = TD3(policy_type, env=env, replay_buffer_class=HerReplayBuffer, verbose=1,
                tensorboard_log=f"runs/{run.id}", device="cuda")

    # fixme: charts not being visualized? Problem might lie in tensorboard -> probably missing admin priv
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            model_save_path=wandb.run.dir
        ),
    )

    run.finish()
