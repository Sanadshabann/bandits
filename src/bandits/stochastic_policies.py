"""Explore-Then-Commit (ETC) and UCB for the (non-contextual) stochastic bandit.

These are the two baseline policies used to motivate KNN-UCB / UCBogram: UCB
is shown to be competitive with ETC even when ETC's exploration length m is
tuned with oracle knowledge of the suboptimality gaps, which ETC needs and
UCB does not.
"""

import math

import numpy as np


def etc(n, m, bandit):
    """Explore each arm m times, then commit to the best sample mean.

    Returns the final pseudo-regret, matching how ``ucb`` below is scored.
    """
    k = bandit.k
    rewards = np.zeros(n)
    best_mean_index = 0
    for i in range(n):
        if i <= m * k - 1:
            arm = i % k
            rewards[i] = bandit.pull(arm)
        else:
            if i == m * k:
                exploration_means = np.mean(rewards[: m * k].reshape(m, k), axis=0)
                best_mean_index = int(np.argmax(exploration_means))
            rewards[i] = bandit.pull(best_mean_index)
    return bandit.regret


def ucb(delta, bandit, n):
    """Run UCB(delta) for n rounds and return the final pseudo-regret."""
    k = bandit.k
    upper_bounds = np.full(k, np.inf)
    means = np.zeros((k, 2))  # column 0: sample mean, column 1: play count
    for _ in range(n):
        arm = int(np.argmax(upper_bounds))
        reward = bandit.pull(arm)
        means[arm, 0] = (means[arm, 0] * means[arm, 1] + reward) / (means[arm, 1] + 1)
        means[arm, 1] += 1
        upper_bounds[arm] = means[arm, 0] + math.sqrt(2 * math.log(1 / delta) / means[arm, 1])
    return bandit.regret
