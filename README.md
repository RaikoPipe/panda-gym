# panda-gym (obstacle-avoidance fork)

PyBullet Franka Panda environments for [gymnasium](https://gymnasium.farama.org): the classic
[panda-gym](https://github.com/qgallouedec/panda-gym) manipulation tasks plus **PandaReachAO**, a collision-free
reaching task with randomized obstacles and benchmark scenes (library, workshop, narrow tunnel, ...).

Originally built for a master thesis on collision-free real-time trajectory planning with RL
([thesis (German)](https://1drv.ms/b/s!Ala_n6z0JphEg9kD0aIXblv2X2x7Jg?e=p5Xr7S), [thesis training results](https://api.wandb.ai/links/raikowand/9tkqy7b9),
[defense (German)](https://1drv.ms/p/s!Ala_n6z0JphEg7oyeWd9JYOaw3Hb9Q?e=xmuDDT)).
The thesis investigated how well agents trained with curriculum learning and domain randomization generalize.

## Install

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[train]"
```

## Environments

```python
import gymnasium as gym
import panda_gym

env = gym.make("PandaPush-v3", render_mode="human")  # also: Reach, Slide, PickAndPlace, Stack, Flip (+Joints, +Dense)
env = gym.make("PandaReachAO-v3", scenario="library2", max_episode_steps=150, render_mode="rgb_array")
```

PandaReachAO scenarios: curriculum stages `reachao1`–`reachao3`, `reachao_rand`, `wangexp-<n>` and the benchmark scenes
`library1`, `library2`, `narrow_tunnel`, `tunnel`, `workshop`, `workshop2`, `industrial`, `kasys`, `bookshelves`,
`kitchen`, `countertop`, `raised_shelves`, `tabletop`, `tabletop2`. Further settings live in `panda_gym.envs.ReachAOConfig`.
`rgb_array` rendering uses PyBullet's EGL plugin (GPU, shadows) when available.

## Training

Agents are trained with **CrossQ** ([Bhatt et al., ICLR 2024](https://openreview.net/forum?id=PczQtTsTIX)) and
Hindsight Experience Replay, via stable-baselines3 / sb3-contrib. CrossQ removes target networks and uses
batch-renormalized critics, which gives high sample efficiency at a low update-to-data ratio. That matters here
because training is bound by gradient-step throughput. TQC + HER (the previous setup, with rl-baselines3-zoo
hyperparameters) is available as `--algo tqc` for comparison.

Head-to-head on PandaPush (8 envs, UTD 0.5, both runs sharing the GPU for 42 min):

| | eval success @ ~200k steps |
|---|---|
| TQC + HER (previous setup) | 72 % |
| **CrossQ + HER** | **90–95 %** |

```bash
python train.py PandaPush-v3 --steps 600000
python train.py PandaReachAO-v3 --curriculum reachao1:50:0.9 reachao2:75:0.9 reachao3:100 --steps 1200000
python train.py PandaReachAO-v3 --init runs/ReachAO/final_reachao3.zip --curriculum library2:150 --steps 150000
tensorboard --logdir runs
```

`--curriculum` takes `scenario:max_episode_steps[:success_threshold]` stages. A stage ends once the evaluation
success rate reaches its threshold, and the last stage uses up the remaining step budget.

## Videos

```bash
python record.py runs/Push PandaPush-v3            # videos/Push_training.mp4 + videos/Push_deploy.mp4
./run_all.sh                                       # overnight: train everything, then record all videos
```

`*_training.mp4` plays the checkpoints saved during training in order, so viewers can watch the behavior emerge.
`*_deploy.mp4` shows the final agent on fresh episodes.

## Tests

```bash
.venv/bin/pip install -e ".[develop]" && .venv/bin/pytest -q test
```
