"""Uncertainty-arbitrated policy ensembles of SAC-family agents (TQC, CrossQ; stable-baselines3 / sb3-contrib).

    ens = PolicyEnsemble([TQC.load(p) for p in paths], strategy="min_variance")
    action, member = ens.predict(obs)          # member: index of the acting policy, -1 if the action is fused

Member statistics: each actor's Gaussian N(mu_k, sigma_k) in pre-tanh space, built through the actor's own action
distribution (get_action_dist_params -> proba_distribution), so sigma is right under gSDE, where it depends on the
latent features and not on log_std alone. The executed action of a member is tanh(mu_k) (deterministic).

Strategies (formulas restored from 9e12871^:evaluation/ensemble_utils/action_selection.py, vectorized):
  mean               average of tanh(mu_k)
  precision_weighted precision-weighted mean in `space`; "pre_tanh": tanh(sum w mu / sum w), w = 1/sigma^2;
                     "action": sigma_a = (1 - tanh(mu)^2) sigma (tanh Jacobian), mean of tanh(mu) weighted by 1/sigma_a^2
  bayes_sample       sample from the fused Gaussian in `space` (then tanh, or clip in action space); always samples
  min_variance       the member with the smallest reduced sigma in `space` acts; reduce = "sum_log_std" (log-det of
                     the diagonal covariance, up to a factor 2) or "sum_std"
  critic_argmax      every member's action is scored by all members' critics (mean over critics and quantiles,
                     critics in eval mode, so CrossQ's BatchRenorm uses its running statistics); the best one acts
  random             a uniformly random member acts at each step
The "oracle" (per-episode best member) belongs to the analysis of single-member runs, not here.
"""
import numpy as np
import torch as th
from torch import nn
from torch.func import functional_call, stack_module_state

STRATEGIES = ("mean", "precision_weighted", "bayes_sample", "min_variance", "critic_argmax", "random")
FUSED = -1


class _DistParams(nn.Module):
    """Actor -> (mean_actions, log_std, latent_sde) as a plain forward, so that members can be vmapped."""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs):
        mean, log_std, kwargs = self.actor.get_action_dist_params(obs)
        latent = kwargs.get("latent_sde")
        return mean, log_std, mean if latent is None else latent


class _MeanQ(nn.Module):
    """Critic -> Q averaged over critics (and TQC's quantiles), shape [batch]."""

    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, obs, actions):
        q = self.critic(obs, actions)
        q = th.cat(q, dim=1) if isinstance(q, tuple) else q  # CrossQ: tuple of [B, 1]; TQC: [B, critics, quantiles]
        return q.reshape(q.shape[0], -1).mean(dim=1)


def _stack(modules):
    """vmappable (params, buffers, base module) if all members share one architecture, else None."""
    shapes = [tuple((k, v.shape) for k, v in m.state_dict().items()) for m in modules]
    if len(modules) < 2 or any(s != shapes[0] for s in shapes):
        return None
    params, buffers = stack_module_state(modules)
    return params, buffers, modules[0]


class PolicyEnsemble:
    def __init__(self, models, strategy="min_variance", space="pre_tanh", reduce="sum_log_std", rng=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if space not in ("pre_tanh", "action") or reduce not in ("sum_log_std", "sum_std"):
            raise ValueError("space: pre_tanh|action, reduce: sum_log_std|sum_std")
        self.models, self.strategy, self.space, self.reduce = list(models), strategy, space, reduce
        self.rng = rng if rng is not None else np.random.default_rng()
        for m in self.models:
            m.policy.set_training_mode(False)  # eval mode: BatchRenorm running statistics, no dropout
            if m.observation_space != self.models[0].observation_space:
                raise ValueError("members must share the observation space")
        self.policy = self.models[0].policy  # observation preprocessing
        self._actors = [_DistParams(m.actor) for m in self.models]
        self._actor_stack = _stack(self._actors)
        self._critics, self._critic_stack = None, None

    # --- member statistics ---------------------------------------------------------------------------------------
    @th.no_grad()
    def member_stats(self, obs_t):
        """mu, sigma in pre-tanh space, each [K, batch, act_dim]."""
        if self._actor_stack is not None:
            params, buffers, base = self._actor_stack
            call = lambda p, b, o: functional_call(base, (p, b), (o,))
            means, log_stds, latents = th.vmap(call, in_dims=(0, 0, None), randomness="same")(params, buffers, obs_t)
            # same formulas as the members' (identical) distributions, batched over members
            dist = self.models[0].actor.action_dist
            if self.models[0].actor.use_sde:  # StateDependentNoiseDistribution.proba_distribution
                std = th.stack([dist.get_std(log_std) for log_std in log_stds])
                return means, th.sqrt(th.bmm(latents ** 2, std ** 2) + dist.epsilon)
            return means, th.ones_like(means) * log_stds.exp()  # DiagGaussianDistribution.proba_distribution
        mus, sigmas = [], []  # heterogeneous members: each through its own distribution object
        for model, actor in zip(self.models, self._actors):
            mean, log_std, latent = actor(obs_t)
            kwargs = {"latent_sde": latent} if model.actor.use_sde else {}
            dist = model.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
            mus.append(dist.distribution.mean)
            sigmas.append(dist.distribution.stddev)
        return th.stack(mus), th.stack(sigmas)

    @th.no_grad()
    def mean_q(self, obs_t, actions):
        """Q of candidate actions [K, B, A] under every member's critic, averaged: [K, B]."""
        if self._critics is None:
            self._critics = [_MeanQ(m.critic) for m in self.models]
            self._critic_stack = _stack(self._critics)
        k, b, a = actions.shape
        obs_rep = {key: v.repeat(k, *([1] * (v.dim() - 1))) for key, v in obs_t.items()} if isinstance(obs_t, dict) \
            else obs_t.repeat(k, *([1] * (obs_t.dim() - 1)))
        flat = actions.reshape(k * b, a)
        if self._critic_stack is not None:
            params, buffers, base = self._critic_stack
            call = lambda p, bu, o, x: functional_call(base, (p, bu), (o, x))
            q = th.vmap(call, in_dims=(0, 0, None, None))(params, buffers, obs_rep, flat)  # [critics, K*B]
        else:
            q = th.stack([critic(obs_rep, flat) for critic in self._critics])
        return q.mean(dim=0).reshape(k, b)

    # --- action selection ----------------------------------------------------------------------------------------
    def _sigma_in_space(self, mu, sigma):
        return sigma if self.space == "pre_tanh" else (1 - th.tanh(mu) ** 2) * sigma

    def _member_action(self, mu, sigma, deterministic):
        if deterministic:
            return th.tanh(mu)
        return th.tanh(mu + sigma * th.as_tensor(self.rng.standard_normal(mu.shape), dtype=mu.dtype, device=mu.device))

    def _fused(self, mu, sigma):
        if self.space == "pre_tanh":
            w = 1.0 / (sigma ** 2 + 1e-8)
            return (w * mu).sum(0) / w.sum(0), th.sqrt(1.0 / w.sum(0))
        m, s = th.tanh(mu), self._sigma_in_space(mu, sigma)
        w = 1.0 / (s ** 2 + 1e-8)
        return (w * m).sum(0) / w.sum(0), th.sqrt(1.0 / w.sum(0))

    @th.no_grad()
    def predict(self, obs, deterministic=True):
        """actions [n_env, act_dim] and acting member [n_env] (-1 if fused); unbatched in, unbatched out."""
        obs_t, vectorized = self.policy.obs_to_tensor(obs)
        mu, sigma = self.member_stats(obs_t)
        k, b, _ = mu.shape
        members = th.full((b,), FUSED, dtype=th.long, device=mu.device)
        if self.strategy == "mean":
            action = th.tanh(mu).mean(0)
        elif self.strategy == "precision_weighted":
            fused, _ = self._fused(mu, sigma)
            action = th.tanh(fused) if self.space == "pre_tanh" else fused
        elif self.strategy == "bayes_sample":
            fused, fused_sigma = self._fused(mu, sigma)
            noise = th.as_tensor(self.rng.standard_normal(fused.shape), dtype=fused.dtype, device=fused.device)
            sample = fused + fused_sigma * noise
            action = th.tanh(sample) if self.space == "pre_tanh" else sample.clamp(-1, 1)
        else:
            candidates = self._member_action(mu, sigma, deterministic)  # [K, B, A]
            if self.strategy == "min_variance":
                s = self._sigma_in_space(mu, sigma)
                score = th.log(s).sum(-1) if self.reduce == "sum_log_std" else s.sum(-1)
                members = score.argmin(0)
            elif self.strategy == "critic_argmax":
                members = self.mean_q(obs_t, candidates).argmax(0)
            else:  # random
                members = th.as_tensor(self.rng.integers(k, size=b), device=mu.device)
            action = candidates[members, th.arange(b)]
        action = action.cpu().numpy().clip(-1, 1)
        members = members.cpu().numpy()
        if not vectorized:
            return action[0], int(members[0])
        return action, members
