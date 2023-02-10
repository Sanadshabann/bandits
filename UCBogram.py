import math
import numpy as np


def bin_index(x, m, lo=0, hi=1):
    bin_i = []
    for i in range(len(x)):
        bin_i.append(math.floor(m * x[i]))
    return tuple(bin_i)


def ucbogram(bandit, X, m, lo=0, hi=1):
    n = len(X)
    bins = [bin_index(x, m) for x in X]
    actions = []
    rewards = []
    regrets = []
    n_arms = bandit.k
    for t in range(n):
        U = np.zeros(n_arms)
        f = np.zeros(n_arms)
        for a in range(n_arms):
            U[a] =

