import gymnasium as gym
import numpy as np
import pybullet as p
import pytest

import panda_gym  # noqa: F401
from panda_gym.envs.tasks.reach_ao import COLLISION_DEFS, THESIS_IGNORE_LINKS, ReachAOConfig

def make(config):
    env = gym.make("PandaReachAO-v3", scenario="reach1", config=config)  # no obstacles of its own
    env.reset(seed=0)
    return env


def touching_links(task, body):
    pts = p.getClosestPoints(task.robot.id, body, 0.0, physicsClientId=task.sim_id)
    return {task.robot.link_names[c[3]] for c in pts if c[3] >= 0}


def add_probe_touching(env, link_name):
    """A 5 mm static sphere that touches `link_name` (it must have collision geometry) and no other link."""
    task = env.unwrapped.task
    task.create_obstacle_sphere(name="probe", radius=0.005, position=np.array([0.0, 0.0, -3.0]))
    probe = task.bodies["probe"]
    link = task.robot.link_names.index(link_name)
    center = np.array(p.getLinkState(task.robot.id, link, physicsClientId=task.sim_id)[0])
    rng = np.random.default_rng(0)
    for direction in rng.normal(size=(200, 3)):
        # put the probe 10 cm away, then move it onto the closest point of the link's surface (1 mm overlap)
        far = center + 0.1 * direction / np.linalg.norm(direction)
        p.resetBasePositionAndOrientation(probe, far, [0, 0, 0, 1], physicsClientId=task.sim_id)
        pts = p.getClosestPoints(task.robot.id, probe, 1.0, linkIndexA=link, physicsClientId=task.sim_id)
        on_link, on_probe = np.array(pts[0][5]), np.array(pts[0][6])
        normal = (on_probe - on_link) / np.linalg.norm(on_probe - on_link)
        p.resetBasePositionAndOrientation(probe, on_link + 0.004 * normal, [0, 0, 0, 1], physicsClientId=task.sim_id)
        if touching_links(task, probe) == {link_name}:
            return
    raise RuntimeError(f"no probe position touches only {link_name}")


def press_into_table(env):
    task = env.unwrapped.task
    q = task.robot.ik(np.array([0.5, 0.0, -0.02]))
    task.robot.set_joint_angles(np.concatenate([q, [0.0, 0.0]]))
    assert touching_links(task, task.bodies["table"]) - {"panda_link1"}


def zero_step(env):
    return env.step(np.zeros(env.action_space.shape, dtype=np.float32))


def collided_now(env):
    """The collision definition on the current pose, without stepping (contacts computed in place)."""
    task = env.unwrapped.task
    p.performCollisionDetection(physicsClientId=task.sim_id)
    return task.check_collided()


def test_thesis_ignored_links_have_no_collision_geometry():
    """The ReachAO (thesis) URDF models the fingers by the panda_ee cylinder; the finger and grasp-target links
    have no collision shapes. Ignoring them therefore changes nothing: the thesis link set equals HEAD's."""
    env = make(ReachAOConfig())
    task = env.unwrapped.task
    for name in THESIS_IGNORE_LINKS:
        link = task.robot.link_names.index(name)
        assert p.getCollisionShapeData(task.robot.id, link, physicsClientId=task.sim_id) == ()
    env.close()


@pytest.mark.parametrize("ignore,expected", [((), True), (("panda_link4",), False), (THESIS_IGNORE_LINKS, True)])
def test_ignored_link_contact(ignore, expected):
    env = make(ReachAOConfig(collision_ignore_links=ignore))
    add_probe_touching(env, "panda_link4")
    assert collided_now(env) is expected
    if expected:
        assert env.unwrapped.task.collision_link == ("panda_link4", "probe")
    env.close()


@pytest.mark.parametrize("name", ["strict", "thesis"])
def test_distance_and_contact_checks_agree_on_a_static_pose(name):
    env = make(ReachAOConfig.from_collision_def(name))
    add_probe_touching(env, "panda_link4")
    assert collided_now(env)
    env.close()


@pytest.mark.parametrize("name,expected", [("strict", True), ("strict_notable", False), ("thesis", False),
                                           ("thesis_table", True), ("thesis_substep", False)])
def test_table_contact(name, expected):
    env = make(ReachAOConfig.from_collision_def(name))
    press_into_table(env)
    assert collided_now(env) is expected
    env.close()


def test_step_end_check_misses_penetration_resolved_within_the_step():
    """The thesis check runs once after the 20 substeps; the solver has pushed the hand out of the table by then."""
    outcomes = {}
    for name in ("strict", "thesis_table"):
        env = make(ReachAOConfig.from_collision_def(name))
        press_into_table(env)
        outcomes[name] = zero_step(env)[4]["is_collided"]
        env.close()
    assert outcomes == {"strict": True, "thesis_table": False}


@pytest.mark.parametrize("config,terminated,truncated", [(ReachAOConfig.strict(), True, False),
                                                         (ReachAOConfig.thesis(table_collides=True, collision_check="contact"), False, True)])
def test_collision_semantics(config, terminated, truncated):
    env = make(config)
    press_into_table(env)
    _, _, term, trunc, info = zero_step(env)
    assert info["is_collided"] and (term, trunc) == (terminated, truncated)
    env.close()


def test_defaults_are_head_behavior():
    assert ReachAOConfig() == ReachAOConfig.strict()
    for preset in COLLISION_DEFS.values():
        ReachAOConfig(**preset)  # every named definition is a valid config


def test_unknown_ignore_link_is_rejected():
    with pytest.raises(ValueError):
        make(ReachAOConfig(collision_ignore_links=("panda_lefthand",)))


def test_live_her_penalty_ignores_stored_flags():
    """her_penalty_source="live" reproduces the thesis bug: the stored per-transition flags are ignored."""
    env = make(ReachAOConfig.thesis())
    task = env.unwrapped.task
    goal = np.zeros((2, 3))
    info = np.array([{"is_collided": True}, {"is_collided": False}])
    task.is_collided = False
    assert np.all(task.compute_reward(goal, goal, info) == 0.0)
    task.is_collided = True
    assert np.all(task.compute_reward(goal, goal, info) == -100.0)
    env.close()
