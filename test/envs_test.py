import gymnasium as gym
import numpy as np

import panda_gym


def run_env(env):
    """Tests running panda gym envs."""
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()
    env.close()


def test_reach():
    env = gym.make("PandaReach-v3")
    run_env(env)


def test_slide():
    env = gym.make("PandaSlide-v3")
    run_env(env)


def test_push():
    env = gym.make("PandaPush-v3")
    run_env(env)


def test_pickandplace():
    env = gym.make("PandaPickAndPlace-v3")
    run_env(env)


def test_stack():
    env = gym.make("PandaStack-v3")
    run_env(env)


def test_flip():
    env = gym.make("PandaFlip-v3")
    run_env(env)


def test_dense_reach():
    env = gym.make("PandaReachDense-v3")
    run_env(env)


def test_dense_slide():
    env = gym.make("PandaSlideDense-v3")
    run_env(env)


def test_dense_push():
    env = gym.make("PandaPushDense-v3")
    run_env(env)


def test_dense_pickandplace():
    env = gym.make("PandaPickAndPlaceDense-v3")
    run_env(env)


def test_dense_stack():
    env = gym.make("PandaStackDense-v3")
    run_env(env)


def test_dense_flip():
    env = gym.make("PandaFlipDense-v3")
    run_env(env)


def test_reach_joints():
    env = gym.make("PandaReachJoints-v3")
    run_env(env)


def test_slide_joints():
    env = gym.make("PandaSlideJoints-v3")
    run_env(env)


def test_push_joints():
    env = gym.make("PandaPushJoints-v3")
    run_env(env)


def test_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJoints-v3")
    run_env(env)


def test_stack_joints():
    env = gym.make("PandaStackJoints-v3")
    run_env(env)


def test_flip_joints():
    env = gym.make("PandaFlipJoints-v3")
    run_env(env)


def test_dense_reach_joints():
    env = gym.make("PandaReachJointsDense-v3")
    run_env(env)


def test_dense_slide_joints():
    env = gym.make("PandaSlideDense-v3")
    run_env(env)


def test_dense_push_joints():
    env = gym.make("PandaPushJointsDense-v3")
    run_env(env)


def test_dense_pickandplace_joints():
    env = gym.make("PandaPickAndPlaceJointsDense-v3")
    run_env(env)


def test_dense_stack_joints():
    env = gym.make("PandaStackJointsDense-v3")
    run_env(env)


def test_dense_flip_joints():
    env = gym.make("PandaFlipJointsDense-v3")
    run_env(env)


def test_reach_ao_scenarios():
    for scenario in ["reachao1", "reachao3", "reachao_dr", "reachao_dr_plr", "library2", "narrow_tunnel"]:
        run_env(gym.make("PandaReachAO-v3", scenario=scenario))


def test_reach_ao_her_reward_uses_stored_collisions():
    """HER relabels batches of transitions: the collision penalty must come from each stored info."""
    env = gym.make("PandaReachAO-v3", scenario="reachao1")
    env.reset(seed=0)
    goal = np.zeros((2, 3))
    infos = np.array([{"is_collided": False}, {"is_collided": True}])
    rewards = env.unwrapped.compute_reward(goal, goal, infos)
    assert rewards.tolist() == [0.0, -101.0]
    env.close()
