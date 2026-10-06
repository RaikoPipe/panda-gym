from typing import Optional

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.flip import Flip
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.tasks.push import Push
from panda_gym.envs.tasks.reach import Reach
from panda_gym.envs.tasks.reach_ao import ReachAO, ReachAOConfig
from panda_gym.envs.tasks.slide import Slide
from panda_gym.envs.tasks.stack import Stack
from panda_gym.pybullet import PyBullet

# camera for rgb_array rendering of the table-top tasks
TABLE_CAMERA = dict(target_position=np.array([-0.15, 0.0, 0.15]), distance=1.35, yaw=45, pitch=-28)


class PandaTaskEnv(RobotTaskEnv):
    """Classic panda-gym task: robot base at x=-0.6, table-top scene."""

    def __init__(self, task_cls, block_gripper, render_mode, reward_type, control_type) -> None:
        sim = PyBullet(render=render_mode == "human", egl=render_mode == "rgb_array")
        robot = Panda(sim, block_gripper=block_gripper, base_position=np.array([-0.6, 0.0, 0.0]),
                      control_type=control_type)
        kwargs = {"get_ee_position": robot.get_ee_position} if task_cls is Reach else {}
        task = task_cls(sim, reward_type=reward_type, **kwargs)
        super().__init__(robot, task, render_mode=render_mode, render_kwargs=TABLE_CAMERA)


def PandaReachEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Reach, True, render_mode, reward_type, control_type)


def PandaPushEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Push, True, render_mode, reward_type, control_type)


def PandaSlideEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Slide, True, render_mode, reward_type, control_type)


def PandaPickAndPlaceEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(PickAndPlace, False, render_mode, reward_type, control_type)


def PandaStackEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Stack, False, render_mode, reward_type, control_type)


def PandaFlipEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Flip, False, render_mode, reward_type, control_type)


class PandaReachAOEnv(RobotTaskEnv):
    """Reach a goal with the Panda while avoiding obstacles (benchmark scenes, curriculum stages).

    Args:
        render_mode: None, "human" or "rgb_array".
        scenario: Scenario name, e.g. "reachao1", "library2", "narrow_tunnel", "wangexp-3".
        ee_error_threshold: Goal tolerance in meters.
        config: Further environment settings.
    """

    def __init__(self, render_mode: Optional[str] = None, scenario: str = "reachao1",
                 ee_error_threshold: float = 0.05, config: Optional[ReachAOConfig] = None) -> None:
        config = config or ReachAOConfig()
        sim = PyBullet(render=render_mode == "human", n_substeps=config.n_substeps, egl=render_mode == "rgb_array")
        robot = Panda(sim, block_gripper=True, base_position=np.zeros(3), control_type=config.control_type,
                      obs_type=config.obs_type, action_limiter=config.action_limiter, n_substeps=config.n_substeps)
        task = ReachAO(sim, robot, get_ee_position=robot.get_ee_position, scenario=scenario,
                       ee_error_threshold=ee_error_threshold, config=config)
        camera = dict(target_position=np.array(task.cameraTargetPosition), distance=task.cameraDistance,
                      yaw=task.cameraYaw, pitch=task.cameraPitch)
        super().__init__(robot, task, config.terminate_on_success, render_mode=render_mode, render_kwargs=camera)
