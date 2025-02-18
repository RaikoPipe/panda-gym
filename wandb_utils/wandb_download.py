import wandb
import os
from definitions import PROJECT_PATH

wandb.login(key=os.getenv("wandb_key"))
api = wandb.Api()

run_paths = [
             "raikowand/panda-gym/lskswrcd",
]

for run_path in run_paths:
    run = api.run(run_path)
    seed = run.config["seed"]
    for file in run.files():
        file.download(f"{PROJECT_PATH}/run_data/final_simba/wandb/final_{seed}")