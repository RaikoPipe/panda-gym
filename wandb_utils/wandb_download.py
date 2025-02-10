import wandb
import os

wandb.login(key=os.getenv("wandb_key"))
api = wandb.Api()

run_paths = ["raikowand/panda-gym/i8o63h1t",
             "raikowand/panda-gym/fpixmsxz",
             "raikowand/panda-gym/gqtmswrp",
             "raikowand/panda-gym/9u0mdumc",
             "raikowand/panda-gym/1uf7l3i2"]

for run_path in run_paths:
    run = api.run(run_path)
    seed = run.config["seed"]
    for file in run.files():
        file.download(f"final_sb3c/wandb/final_{seed}")