import time

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from sb3_contrib import TQC, CrossQ
from sb3_contrib.crossq.policies import CrossQPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor

import panda_gym  # noqa: F401
import train  # noqa: F401  (patches CrossQ for dict observations)
from panda_gym.ensemble import STRATEGIES, PolicyEnsemble


@pytest.fixture(scope="module")
def env():
    env = gym.make("PandaReachAO-v3", scenario="reachao3")
    yield env
    env.close()


def make(kind, env, seed):
    common = dict(buffer_size=100, learning_starts=0, seed=seed, device="cpu")
    if kind == "crossq":
        return CrossQ(CrossQPolicy, env, policy_kwargs=dict(features_extractor_class=CombinedExtractor,
                                                            net_arch=dict(pi=[256, 256], qf=[256, 256])), **common)
    sde = dict(use_sde=True) if kind == "tqc_sde" else {}
    kw = dict(log_std_init=-3) if kind == "tqc_sde" else {}
    return TQC("MultiInputPolicy", env, policy_kwargs=dict(net_arch=[256, 256], **kw), **sde, **common)


def observations(env, n):
    obs = [env.reset(seed=s)[0] for s in range(n)]
    return {k: np.stack([o[k] for o in obs]) for k in obs[0]}


@pytest.mark.parametrize("kind", ["tqc", "tqc_sde", "crossq"])
@pytest.mark.parametrize("strategy", ["min_variance", "mean", "precision_weighted", "critic_argmax"])
def test_single_member_reproduces_predict(env, kind, strategy):
    model = make(kind, env, 0)
    ens = PolicyEnsemble([model], strategy=strategy)
    obs = observations(env, 4)
    action, member = ens.predict(obs)
    expected, _ = model.predict(obs, deterministic=True)
    np.testing.assert_allclose(action, expected, atol=1e-6)
    assert member.shape == (4,)


@pytest.mark.parametrize("space", ["pre_tanh", "action"])
def test_precision_weighting_of_identical_members_is_the_member(env, space):
    model = make("tqc_sde", env, 0)
    obs = observations(env, 3)
    action, member = PolicyEnsemble([model, model, model], "precision_weighted", space=space).predict(obs)
    np.testing.assert_allclose(action, model.predict(obs, deterministic=True)[0], atol=1e-5)
    assert (member == -1).all()


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_output_shapes(env, strategy):
    ens = PolicyEnsemble([make("tqc", env, s) for s in range(3)], strategy, rng=np.random.default_rng(0))
    action, member = ens.predict(observations(env, 5))
    assert action.shape == (5, 7) and member.shape == (5,)
    assert np.all(np.abs(action) <= 1)
    single_action, single_member = ens.predict(env.reset(seed=0)[0])  # unbatched in, unbatched out
    assert single_action.shape == (7,) and isinstance(single_member, int)
    if strategy in ("min_variance", "critic_argmax", "random"):
        assert ((member >= 0) & (member < 3)).all()


def test_vmapped_and_looped_members_agree(env):
    models = [make("tqc_sde", env, s) for s in range(3)]
    ens = PolicyEnsemble(models, "min_variance")
    obs_t, _ = ens.policy.obs_to_tensor(observations(env, 4))
    mu, sigma = ens.member_stats(obs_t)
    ens._actor_stack = None  # force the per-member loop
    mu2, sigma2 = ens.member_stats(obs_t)
    th.testing.assert_close(mu, mu2)
    th.testing.assert_close(sigma, sigma2)


def test_gsde_sigma_depends_on_the_state(env):
    """With gSDE sigma comes from the latent features, not from log_std alone."""
    ens = PolicyEnsemble([make("tqc_sde", env, 0)], "min_variance")
    obs_t, _ = ens.policy.obs_to_tensor(observations(env, 4))
    _, sigma = ens.member_stats(obs_t)
    assert not th.allclose(sigma[0, 0], sigma[0, 1])


def test_latency_k20_cpu(env):
    th.set_num_threads(1)
    ens = PolicyEnsemble([make("tqc_sde", env, s) for s in range(20)], "min_variance")
    obs = env.reset(seed=0)[0]
    for _ in range(10):
        ens.predict(obs)
    t0 = time.perf_counter()
    for _ in range(100):
        ens.predict(obs)
    ms = (time.perf_counter() - t0) * 10
    print(f"K=20 min_variance: {ms:.2f} ms/step")
    assert ms < 5.0
