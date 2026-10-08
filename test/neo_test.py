import gymnasium as gym
import numpy as np
import pybullet as p

import evaluate as E
import panda_gym  # noqa: F401
from panda_gym.baselines.neo import NEO
from panda_gym.envs.tasks.reach_ao import ReachAOConfig


def make(scene):
    env = gym.make("PandaReachAO-v3", scenario=scene, config=ReachAOConfig(control_type="jsd"))
    env.reset(seed=3)
    neo = NEO()
    neo.bind(env.unwrapped)
    return env, neo


def test_jacobian_matches_finite_differences_and_state_is_untouched():
    """pybullet's calculateJacobian takes the point in the link's URDF frame (not the centre-of-mass frame its docs
    suggest). Central differences use a 1e-3 rad step: pybullet's kinematics are single precision, so steps around
    1e-6 rad drown in rounding."""
    env, neo = make("reachao3")
    for _ in range(3):
        env.step(env.action_space.sample())  # non-zero joint velocities
    before = [s[:2] for s in p.getJointStates(neo.body, neo.movable, physicsClientId=neo.cid)]
    q = neo._q()
    for name in ("panda_link6", "panda_hand", "panda_rightfinger"):
        link = neo.robot.link_names.index(name)
        com, orn = p.getLinkState(neo.body, link, computeForwardKinematics=True, physicsClientId=neo.cid)[:2]
        point = np.array(com) + np.array([0.02, -0.01, 0.03])
        attached = p.multiplyTransforms(*p.invertTransform(com, orn), point.tolist(), [0, 0, 0, 1])[0]
        jt, _ = neo._jacobian(link, neo._local(link, point), q)

        def at(qq):
            for j, qj in zip(neo.movable, qq):
                p.resetJointState(neo.body, j, qj, physicsClientId=neo.cid)
            c, o = p.getLinkState(neo.body, link, computeForwardKinematics=True, physicsClientId=neo.cid)[:2]
            return np.array(p.multiplyTransforms(c, o, attached, [0, 0, 0, 1])[0])

        fd = np.stack([(at(q + e) - at(q - e)) / 2e-3 for e in np.eye(len(q)) * 1e-3], axis=1)
        for j, (qj, vj) in zip(neo.movable, before):
            p.resetJointState(neo.body, j, qj, vj, physicsClientId=neo.cid)
        np.testing.assert_allclose(jt, fd, atol=1e-4)
    neo(env.unwrapped._get_obs())  # a control step must not move the robot or change its velocities
    np.testing.assert_allclose([s[:2] for s in p.getJointStates(neo.body, neo.movable, physicsClientId=neo.cid)],
                               before)
    env.close()


def test_actions_are_finite_velocities_with_obstacle_constraints():
    env, neo = make("library2")
    obs = env.reset(seed=3)[0]
    for _ in range(5):
        a, member = neo(obs)
        assert a.shape == (7,) and np.all(np.isfinite(a)) and member is None
        obs = env.step(a)[0]
    env.close()


def test_neo_runs_on_shared_episodes():
    r = E._work((("neo", ("neo",), None), "PandaReachAO-v3", "reachao3", 200, "thesis", [12000, 12001], True, None))
    policy_hashes = [x["hash"] for x in E._work((("neo", ("neo",), None), "PandaReachAO-v3", "reachao3", 5, "strict",
                                                [12000, 12001], True, None))]
    assert [x["hash"] for x in r] == policy_hashes
    assert all(x["outcome"] in ("success", "collision", "timeout") for x in r)
