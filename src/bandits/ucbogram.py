"""UCBogram: a piecewise-constant contextual bandit policy (Rigollet & Zeevi, 2010).

UCBogram partitions the context space into a regular grid of hypercubes and
runs an independent instance of UCB inside each one, using only the history
of rounds whose context landed in the same cell.
"""

import math
from collections import defaultdict

import numpy as np


def _bin_index(x, m, lo, hi):
    """Map a context vector to the index of the hypercube it falls in."""
    return tuple(math.floor(m * (xi - lo) / (hi - lo)) for xi in x)


def ucbogram(bandit, contexts, m, lo=0, hi=1, h=lambda x: x):
    """Run UCBogram for len(contexts) rounds and return the pseudo-regret trace.

    Parameters
    ----------
    bandit : ContextBandit
    contexts : array-like, shape (n, ambient_dim)
    m : int
        Number of bins per dimension; the context space is split into m**d cells.
        Bins are created lazily on first visit rather than all m**d up front,
        since m**d is intractable to enumerate once d reaches double digits.
    lo, hi : float
        Bounds of the (per-dimension) context space, used to place the grid.
    h : callable, optional
        Feature map applied to each context before binning.
    """
    features = [h(x) for x in contexts]
    n = len(features)
    n_arms = bandit.k
    bins = defaultdict(lambda: np.zeros((n_arms, 2)))
    regrets = []

    for t in range(n):
        cell = _bin_index(features[t], m, lo, hi)
        counts, totals = bins[cell][:, 0], bins[cell][:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(counts != 0, totals / counts, math.inf)
            bonus = np.where(counts != 0, np.sqrt(2 * math.log(max(t, 1)) / counts), 0)
        index = np.where(counts != 0, means + bonus, math.inf)

        arm = int(np.argmax(index))
        reward = bandit.pull(arm, contexts[t])
        bins[cell][arm, 0] += 1
        bins[cell][arm, 1] += reward
        regrets.append(bandit.regret)

    return np.array(regrets)
