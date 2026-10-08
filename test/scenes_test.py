import gymnasium as gym
import numpy as np
import pybullet as p
import pytest

import panda_gym  # noqa: F401


def make(scene):
    return gym.make("PandaReachAO-v3", scenario=scene)


@pytest.mark.parametrize("scene", ["workshop", "wang-3"])
def test_goals_are_reachable(scene):
    env = make(scene)
    task = env.unwrapped.task
    for seed in range(10):
        env.reset(seed=seed)
        assert task.goal_reachable(task.goal)
    env.close()


def test_goal_reachable_rejects_far_goals_and_restores_the_pose():
    env = make("reachao3")
    env.reset(seed=0)
    task = env.unwrapped.task
    q = task.robot.get_joint_angles(task.robot.joint_indices)
    assert not task.goal_reachable(np.array([1.5, 0.0, 0.5]))
    assert task.goal_reachable(task.robot.get_ee_position() + np.array([0.05, 0.0, 0.0]))
    np.testing.assert_allclose(task.robot.get_joint_angles(task.robot.joint_indices), q)
    env.close()


@pytest.mark.parametrize("scene", ["reachao_rand_start", "reachao_dr", "reachao_dr_plr"])
def test_episodes_start_collision_free(scene):
    env = make(scene)
    task = env.unwrapped.task
    for seed in range(20):
        env.reset(seed=seed)
        p.performCollisionDetection(physicsClientId=task.sim_id)
        assert not task.check_collided(), (seed, task.collision_link)
    env.close()


def test_showcase_builds():
    make("showcase").close()
