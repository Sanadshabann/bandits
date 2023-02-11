import math
import numpy as np


def bin_index(x, m, lo=0, hi=1):
    bin_i = []
    for i in range(len(x)):
        bin_i.append(math.floor((m * (x[i] - lo) / (hi - lo))))
    return tuple(bin_i)


# initialize a dictionary with keys as indices for the hypercubes
# and values are matrices that store number of times each arm was played and total reward
# in that hypercube
def initial_dict(m, d, n_arms):
    dic = dict()
    for i in range(m ** d):
        index = np.zeros(d)
        for k in range(d):
            index[d - k - 1] = math.floor(i / (m ** k)) % m
        dic[tuple(index)] = np.zeros((n_arms, 2))
    return dic


def ucbogram(bandit, Z, m, lo=0, hi=1, h=lambda x: x):
    X = [h(z) for z in Z]
    n = len(X)
    n_arms = bandit.k
    dic = initial_dict(m, np.shape(X)[1], n_arms)
    regrets = []
    n_arms = bandit.k
    for t in range(n):
        cube = bin_index(X[t], m, lo, hi)
        indices = np.zeros(n_arms)
        for a in range(n_arms):
            N = dic[cube][(a, 0)]
            f = dic[cube][(a, 1)] / N if (N != 0) else math.inf
            indices[a] = f + math.sqrt(2 * math.log(t) / N) if (N != 0) else math.inf
        action = np.argmax(indices)
        reward = bandit.pull(action, Z[t])
        dic[cube][(action, 0)] += 1
        dic[cube][(action, 1)] += reward
        regrets.append(bandit.regret)
    return regrets
