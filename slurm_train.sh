#!/usr/bin/env bash
# Submit: sbatch slurm_train.sh PandaPush-v3 --steps 600000
# All train.py arguments are passed through. Logs: runs/slurm-<jobid>.out
#SBATCH --job-name=panda-gym
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu[01-03]
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/slurm-%j.out
set -eu
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
module load python/python-3.13.2  # the venv needs this module's libpython3.13.so
.venv/bin/python train.py "$@"
