import gymnasium as gym
import pytest
from sb3_contrib import TQC

import evaluate as E
import panda_gym  # noqa: F401


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "untrained.zip"
    env = gym.make("PandaReachAO-v3", scenario="reachao3")
    TQC("MultiInputPolicy", env, buffer_size=100, learning_starts=0, seed=0).save(path)
    env.close()
    return str(path)


def key(r):
    return r["episode_seed"], r["hash"], r["outcome"], r["steps"], round(r["path_length"], 9)


@pytest.mark.parametrize("scene", ["reachao3", "library2"])
def test_episodes_do_not_depend_on_order_or_chunking(model_path, scene):
    spec = ("single", (model_path,), None)
    seeds = [12000, 12001, 12002, 12003]

    def run(chunk):
        return E._work((spec, "PandaReachAO-v3", scene, 30, "strict", chunk, True))

    whole = run(seeds)
    chunked = run(seeds[:2]) + run(seeds[2:])
    reversed_ = [r for s in reversed(seeds) for r in run([s])][::-1]
    assert [key(r) for r in whole] == [key(r) for r in chunked] == [key(r) for r in reversed_]


def test_records_have_the_design_fields(model_path):
    r = E._work((("single", (model_path,), None), "PandaReachAO-v3", "reachao3", 20, "thesis", [12000], True))[0]
    for field in ("episode_seed", "hash", "outcome", "steps", "path_length", "mean_jerk", "max_jerk",
                  "mean_ee_speed", "latency_ms_mean", "latency_ms_p99", "collision_link"):
        assert field in r
    assert r["outcome"] in ("success", "collision", "timeout")


@pytest.mark.parametrize("scene", ["narrow_tunnel", "industrial"])
def test_random_start_scenes_are_shared_across_collision_definitions(model_path, scene):
    """These scenes sample the start pose by IK from the current joint state (seeded constructor reset)."""
    spec = ("single", (model_path,), None)
    hashes = [[r["hash"] for r in E._work((spec, "PandaReachAO-v3", scene, 5, d, [25000, 25001], True))]
              for d in ("strict", "thesis")]
    assert hashes[0] == hashes[1]
