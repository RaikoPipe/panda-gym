import gymnasium as gym
import numpy as np
import pybullet as p
import pytest

import panda_gym  # noqa: F401
from panda_gym.envs.tasks.reach_ao import ReachAOConfig


def positions(env):
    task = env.unwrapped.task
    return np.array([p.getBasePositionAndOrientation(u, physicsClientId=task.sim_id)[0]
                     for u in task.obstacles.values()])


@pytest.mark.parametrize("scene,speed", [("dynamic_cross_slow", 0.1), ("dynamic_cross_fast", 0.3)])
def test_spheres_cross_at_the_set_speed_from_outside_the_reach_envelope(scene, speed):
    env = gym.make("PandaReachAO-v3", scenario=scene)
    task = env.unwrapped.task
    for seed in range(5):
        env.reset(seed=seed)
        start = positions(env)
        assert np.all(np.linalg.norm(start - task.SHOULDER, axis=1) > 0.99)
        assert not any(task.check_collision("robot", n, margin=0.05) for n in task.obstacles)
        for _ in range(10):
            env.step(np.zeros(7, dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(positions(env) - start, axis=1), 10 * env.unwrapped.sim.dt * speed,
                                   rtol=1e-3)
    env.close()


def test_trajectories_are_seeded():
    runs = []
    for _ in range(2):
        env = gym.make("PandaReachAO-v3", scenario="dynamic_cross_fast")
        env.reset(seed=7)
        for _ in range(5):
            env.step(np.zeros(7, dtype=np.float32))
        runs.append(positions(env))
        env.close()
    np.testing.assert_array_equal(*runs)


def test_holding_the_start_pose_collides_sometimes():
    """Design T9 acceptance: a policy holding q0 collides in 20-60 % of episodes (slow level; 32 % on 100 episodes,
    the fast level collides in ~80 %, see CHANGELOG_paper.md)."""
    collisions = 0
    for seed in range(30):
        env = gym.make("PandaReachAO-v3", scenario="dynamic_cross_slow", max_episode_steps=150)
        env.reset(seed=90000 + seed)
        for _ in range(150):
            _, _, term, trunc, info = env.step(np.zeros(7, dtype=np.float32))
            if term or trunc:
                break
        collisions += info["is_collided"]
        env.close()
    assert 0.1 <= collisions / 30 <= 0.7


def test_history_observation_has_a_fixed_shape():
    env = gym.make("PandaReachAO-v3", scenario="dynamic_cross_slow", config=ReachAOConfig(obstacle_obs="vectors+past"))
    shape = env.observation_space["observation"].shape
    obs, _ = env.reset(seed=0)
    assert obs["observation"].shape == shape
    for _ in range(5):
        assert env.step(env.action_space.sample())[0]["observation"].shape == shape
    env.close()
