import wandb
import time
import collections

# start wandb

def get_ordered_hypers(variables):
    HYPERS = collections.OrderedDict()

    for name, value in variables.items():
        if isinstance(value, (str,int,float)):
            HYPERS[name] = value

    return HYPERS
def start_wandb(project_log, task, seed, method, hypers):

    wandb.login(key="5d65c571cf2a6110b15190696682f6e36ddcdd11")
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
