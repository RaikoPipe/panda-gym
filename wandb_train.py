import time

import numpy as np

# from pygame import mixer

import sys
import gymnasium
sys.modules["gym"] = gymnasium

from sb3_contrib import TQC
import os


from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer

def main():
    from run.learning_methods.learning import learn, get_env
    import wandb
    run = wandb.init()
    wandb.login(key=os.getenv("wandb_key"))
    config=wandb.config


    config["replay_buffer_class"] = HerReplayBuffer
    config["reward_thresholds"] = [-1]
    config["limiter"] = "sim"
    config["success_thresholds"] = [0.99]
    config["render"] = True
    config["eval_freq"] = 20_000


    # env = get_env(config, config["n_envs"], config["stages"][0])
    # model = TQC.load(r"run_data/wandb/cerulean-sky/files/best_model.zip", env=env, replay_buffer_class=config["replay_buffer"],
    #                  custom_objects={"action_space":gymnasium.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)}
    #                  )

    model = learn(config=config, algorithm=config["algorithm"])


if __name__ == "__main__":
    main()


    # mixer.init()
    # mixer.music.load("learning_complete.mp3")
    # mixer.music.set_volume(1.0)
    # mixer.music.play()
    # time.sleep(2.0)
