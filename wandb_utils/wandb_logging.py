import wandb
import time
import collections
import os

# start wandb

def get_ordered_hypers(variables):
    HYPERS = collections.OrderedDict()

    for name, value in variables.items():
        if isinstance(value, (str,int,float)):
            HYPERS[name] = value

    return HYPERS
def start_wandb(project_log, task, seed, method, hypers):

    wandb.login(key=os.getenv("wandb_key"))
    wandb.init(project=project_log)

    t = time.asctime().replace(' ', '_').replace(":", "_")

    name = f"{task}_{str(seed)}_{project_log}_{method}_{t}"

    wandb.run.name = name

    wandb.config.update(hypers)

    return name

def watch_agent(ac):
    wandb.watch(ac)


def write_logs(logs, t):
    wandb.log(logs, t)
