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

    def __init__(self, task_cls, block_gripper, render_mode, reward_type, control_type, grasp_start_prob=0.0,
                 grasp_object="object") -> None:
        self.grasp_start_prob, self.grasp_object = grasp_start_prob, grasp_object
        sim = PyBullet(render=render_mode == "human", egl=render_mode == "rgb_array")
        robot = Panda(sim, block_gripper=block_gripper, base_position=np.array([-0.6, 0.0, 0.0]),
                      control_type=control_type)
        kwargs = {"get_ee_position": robot.get_ee_position} if task_cls is Reach else {}
        task = task_cls(sim, reward_type=reward_type, **kwargs)
        super().__init__(robot, task, render_mode=render_mode, render_kwargs=TABLE_CAMERA)

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.grasp_start_prob and self.np_random.random() < self.grasp_start_prob:
            # Training aid (Nair et al., ICRA 2018): start with the cube held at a random point above the table,
            # so lifting and carrying are experienced long before random exploration would find a grasp.
            target = self.np_random.uniform([-0.15, -0.15, 0.05], [0.15, 0.15, 0.2])
            half = self.task.object_size / 2
            arm = self.robot.inverse_kinematics(self.robot.ee_link, target, np.array([1.0, 0.0, 0.0, 0.0]))[:7]
            self.robot.set_joint_angles(np.concatenate([arm, [half, half]]))
            self.sim.set_base_pose(self.grasp_object, self.robot.get_ee_position(), np.array([0.0, 0.0, 0.0, 1.0]))
            observation = self._get_obs()
            info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_goal())}
        return observation, info


def PandaReachEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Reach, True, render_mode, reward_type, control_type)


def PandaPushEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Push, True, render_mode, reward_type, control_type)


def PandaSlideEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee"):
    return PandaTaskEnv(Slide, True, render_mode, reward_type, control_type)


def PandaPickAndPlaceEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee",
                         grasp_start_prob: float = 0.0):
    return PandaTaskEnv(PickAndPlace, False, render_mode, reward_type, control_type, grasp_start_prob)


def PandaStackEnv(render_mode: Optional[str] = None, reward_type: str = "sparse", control_type: str = "ee",
                  grasp_start_prob: float = 0.0):
    return PandaTaskEnv(Stack, False, render_mode, reward_type, control_type, grasp_start_prob, "object2")


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
                      obs_type=config.obs_type, action_limiter=config.action_limiter, n_substeps=config.n_substeps,
                      urdf="custom")
        task = ReachAO(sim, robot, get_ee_position=robot.get_ee_position, scenario=scenario,
                       ee_error_threshold=ee_error_threshold, config=config)
        camera = dict(target_position=np.array(task.cameraTargetPosition), distance=task.cameraDistance,
                      yaw=task.cameraYaw, pitch=task.cameraPitch)
        super().__init__(robot, task, config.terminate_on_success, render_mode=render_mode, render_kwargs=camera)
