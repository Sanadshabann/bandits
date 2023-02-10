import numpy as np
from scipy.spatial.distance import cdist
import math as m
from Ternary_Search import k_ternary_search


def knn_ucb(bandit, Z, theta, phi, h=lambda x: x):
    X = [h(z) for z in Z]
    n = len(Z)
    regrets = []
    n_arms = bandit.k  # number of arms
    actions = []
    rewards = []
    assert (n >= n_arms)
    for a in range(n_arms):  ##play each arm once regardless of context
        reward = bandit.pull(a, Z[a])
        rewards.append(reward)
        actions.append(a)
        regrets.append(bandit.regret)
    for t in range(n_arms, n, 1):
        distances = cdist(X[:t], [X[t]], 'Euclidean')
        distances = [dist for sublist in distances for dist in sublist]
        k_a = np.zeros(n_arms)
        index = np.zeros(n_arms)
        for a in range(n_arms):
            k_a[a] = k_ternary_search(actions, rewards, distances, theta, phi, a)
            k_least = np.argpartition(distances, int(k_a[a] - 1))[:int(k_a[a])]
            N = sum(np.array(actions)[k_least] == a)
            S = sum(np.array(rewards)[k_least] * (np.array(actions)[k_least] == a))
            f_hat = S / N if (N != 0) else 0
            index[a] = f_hat + m.sqrt(theta * m.log(t) / N) + phi(t) * distances[k_least[-1]] if (N != 0) else m.inf
        arm = np.argmax(index)
        reward = bandit.pull(arm, Z[t])
        regrets.append(bandit.regret)
        actions.append(arm)
        rewards.append(reward)
    return regrets
