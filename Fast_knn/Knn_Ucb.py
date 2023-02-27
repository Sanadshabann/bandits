from scipy.spatial.distance import cdist
import numpy as np
from Ternary_Search import k_ternary_search
import math as m
from ContextBandit import ContextBandit
from pathos.multiprocessing import ProcessingPool as Pool


def knn_ucb(bandit, Z, theta, phi, h, k_max):  ##X:context vector
    X = np.array([h(z) for z in Z])
    n = np.shape(Z)[0]
    regrets = np.array([])
    n_arms = bandit.k  # number of arms
    actions = np.array([])
    rewards = np.array([])
    assert (n >= n_arms)
    for a in range(n_arms):  ##play each arm once regardless of context
        reward = bandit.pull(a, Z[a])
        rewards = np.append(rewards, reward)
        actions = np.append(actions, a)
        regrets = np.append(regrets, bandit.regret)
    for t in range(n_arms, n, 1):
        upper = min(t, k_max)
        distances = cdist(X[:t], [X[t]], 'Euclidean').ravel()
        indices = np.argpartition(distances, upper - 1)[
                  :k_max]  # indices of nearest n neighbours, where k is searched \in [n]
        k_nearest = np.stack((indices, distances[indices]), axis=1)  # distances at these distances
        k_nearest_sorted = k_nearest[k_nearest[:, 1].argsort()]
        k_a = np.zeros(n_arms)
        index = np.zeros(n_arms)  ## to store I_t,k
        for a in range(n_arms):  ##evaluate k_a for all a
            k_a[a], u_a = k_ternary_search(actions, k_nearest_sorted, theta, phi, a, upper)
            k_least = k_nearest_sorted[:int(k_a[a]), 0].astype(int)
            N = sum(actions[k_least] == a)
            S = sum(rewards[k_least] * (actions[k_least] == a))
            f_hat = S / N if (N != 0) else 0
            index[a] = f_hat + u_a
        arm = np.argmax(index)
        reward = bandit.pull(arm, Z[t])
        rewards = np.append(rewards, reward)
        actions = np.append(actions, arm)
        regrets = np.append(regrets, bandit.regret)
    with open("data_C/knn_regrets.csv", "ab") as f:
        np.savetxt(f, np.asarray(regrets), delimiter=",")
    return regrets


