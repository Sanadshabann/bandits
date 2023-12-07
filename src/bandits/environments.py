"""Bandit environments used throughout the experiments.

Both environments track cumulative pseudo-regret internally, so a policy only
needs to call ``pull`` and read ``bandit.regret`` at the end of each round.
"""

import random


class GaussianBandit:
    """A k-armed stochastic bandit with Gaussian rewards of unit variance."""

    def __init__(self, means):
        self.means = means
        self.k = len(means)
        self.best_mean = max(means)
        self.regret = 0.0

    def pull(self, arm):
        self.regret += self.best_mean - self.means[arm]
        return random.gauss(self.means[arm], 1)


class ContextBandit:
    """A k-armed contextual bandit.

    Parameters
    ----------
    reward_fns : list of callables
        ``reward_fns[a](x)`` is the expected reward of arm ``a`` given context ``x``.
    noise : callable
        ``noise(x)`` draws a zero-mean noise sample, which may itself depend on ``x``.
    """

    def __init__(self, reward_fns, noise):
        self.k = len(reward_fns)
        self.reward_fns = reward_fns
        self.noise = noise
        self.regret = 0.0

    def pull(self, arm, x):
        rewards = [reward_fn(x) for reward_fn in self.reward_fns]
        self.regret += max(rewards) - rewards[arm]
        return rewards[arm] + self.noise(x)
