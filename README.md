# Extended panda-gym version with collision-avoidance scenarios, curriculum learning pipeline, trajectory visualization and more

Implemeneted for Master thesis on collision free real time trajectory planning using reinforcement learning (graded 1.0)
Goal was to investigate maximum generalisation capabilities of RL-agents trained with curriculum learning and domain randomization.
Uses vectorized version of Hindsight Experience Replay for faster training and domain randomization techniques.
Includes [analytical motion planner](https://github.com/petercorke/robotics-toolbox-python) for comparison. While the RL-based motion controller cannot outcompete current analytical approaches (see cuMotion), it shows moderate generalisation capabilities with only around 300.000 episode steps (performed in around 2 hours training time) when using parallelized environments.

# panda-gym

## [Training results](https://api.wandb.ai/links/raikowand/9tkqy7b9)

https://github.com/RaikoPipe/panda-gym/assets/74252023/89374c7d-927b-4bd3-bc3c-c5db5ba30182

## [Thesis (German)](https://1drv.ms/b/s!Ala_n6z0JphEg9kD0aIXblv2X2x7Jg?e=p5Xr7S)

## [Defense (German)](https://1drv.ms/p/s!Ala_n6z0JphEg7oyeWd9JYOaw3Hb9Q?e=xmuDDT)

