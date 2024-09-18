import numpy as np


def bayesian_fusion(agent_means, agent_variances):
    """
    Perform Bayesian fusion of actions from multiple agents.

    :param agent_means: Array of shape (num_agents, action_dim) containing mean actions for each agent
    :param agent_variances: Array of shape (num_agents, action_dim) containing action variances for each agent
    :return: Tuple containing fused mean and variance of shape (action_dim,)
    """
    # Compute precision (inverse of variance)
    precisions = 1.0 / (agent_variances + 1e-8)  # Add small epsilon to avoid division by zero

    # Compute fused precision
    fused_precision = np.sum(precisions, axis=0)

    # Compute fused mean
    fused_mean = np.sum(agent_means * precisions, axis=0) / fused_precision

    # Compute fused variance
    fused_variance = 1.0 / fused_precision

    # sample from the fused distribution
    return np.random.normal(fused_mean, np.sqrt(fused_variance))


def weighted_aggregation(agent_variances, agent_means):
    # convert variances to weights
    weights = 1.0 / (np.array(agent_variances) + 1e-8)  # add small epsilon to avoid division by zero
    # normalize weights
    norm_weights = weights / np.sum(weights)
    # perform weighted aggregation
    return np.average(agent_means, axis=0, weights=norm_weights)


def mean(agent_means):
    return np.mean(agent_means, axis=0)


def confidence(agent_means, agent_variances):
    min_variance = min(agent_variances)
    action_sovereignty = agent_variances.index(min_variance)
    return agent_means[action_sovereignty], action_sovereignty
