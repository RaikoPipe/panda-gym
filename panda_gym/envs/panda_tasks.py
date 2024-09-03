from typing import Optional

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.robots.mycobot import MyCobot
from panda_gym.envs.tasks.flip import Flip
from panda_gym.envs.tasks.pick_and_place import PickAndPlace
from panda_gym.envs.tasks.push import Push
from panda_gym.envs.tasks.reach import Reach
from panda_gym.envs.tasks.slide import Slide
from panda_gym.envs.tasks.stack import Stack
from panda_gym.envs.tasks.reach_ao import ReachAO
from panda_gym.pybullet import PyBullet

from classes.train_config import TrainConfig


class PandaFlipEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Flip(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaPickAndPlaceEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PickAndPlace(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaPushEnv(RobotTaskEnv):
    """Push task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "js") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Push(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, realtime: bool = False, reward_type: str = "sparse",
                 control_type: str = "js",
                 goal_range=0.3, show_goal_space=False) -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, goal_range=goal_range,
                     show_goal_space=show_goal_space)
        super().__init__(robot, task)


class MyCobotReachEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, realtime: bool = False, reward_type: str = "sparse",
                 control_type: str = "js",
                 goal_range=0.3, show_goal_space=False) -> None:
        sim = PyBullet(render=render)
        robot = MyCobot(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, goal_range=goal_range,
                     show_goal_space=show_goal_space)
        super().__init__(robot, task)


class PandaReachCheckerEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, realtime: bool = False, reward_type: str = "sparse",
                 control_type: str = "js",
                 goal_range=0.3, show_goal_space=False) -> None:
        sim = PyBullet(render=False)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type,
                      use_robotics_toolbox=False, action_limiter="clip")
        task = Reach(sim, reward_type=reward_type, get_ee_position=robot.get_ee_position, goal_range=goal_range,
                     show_goal_space=show_goal_space)
        super().__init__(robot, task)


class PandaReachAOEnv(RobotTaskEnv):
    """Reach task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, ee_error_threshold: float = 0.05,
                 speed_threshold=0.1,
                 scenario: str = "wangexp_3",
                 config=TrainConfig()
                 ) -> None:
        sim = PyBullet(render=render, n_substeps=config.n_substeps)
        robot = Panda(sim, block_gripper=True, base_position=np.array([0.0, 0.0, 0.0]),
                      control_type=config.control_type,
                      obs_type=config.obs_type,
                      limiter=config.limiter, action_limiter=config.action_limiter, n_substeps=config.n_substeps)
        task = ReachAO(sim, robot,
                       ee_error_threshold=ee_error_threshold,
                       speed_threshold=speed_threshold,
                       scenario=scenario,
                       get_ee_position=robot.get_ee_position,
                       config=config
                       )
        super().__init__(robot, task, config.terminate_on_success)


class PandaSlideEnv(RobotTaskEnv):
    """Slide task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=True, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Slide(sim, reward_type=reward_type)
        super().__init__(robot, task)


class PandaStackEnv(RobotTaskEnv):
    """Stack task wih Panda robot.

    Args:
        render (bool, optional): Activate rendering. Defaults to False.
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
    """

    def __init__(self, render: bool = False, reward_type: str = "sparse", control_type: str = "ee") -> None:
        sim = PyBullet(render=render)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = Stack(sim, reward_type=reward_type)
        super().__init__(robot, task)
