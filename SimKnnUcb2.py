import numpy as np
from scipy.spatial.distance import cdist
import math as m
from pathos.multiprocessing import ProcessingPool as Pool
import Ternary_Search
from ContextBandit import ContextBandit
from scipy.stats import special_ortho_group as ortho


def sim_knn_amb(bandit, Z, theta, phi, h):
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
            k_a[a] = Ternary_Search.k_ternary_search(actions, rewards, distances, theta, phi, a)
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


def case2(n_trials, n_cores, d, D):
    Z = np.random.uniform(0, 1, size=(15000, d))
    U = ortho.rvs(D)
    h = lambda z: np.dot(z, U[:d])
    lambdas = [lambda y: np.prod([m.sin(3 * xs * m.pi) for xs in y]),
               lambda y: np.prod([m.cos(5 * xs * m.pi) for xs in y])]
    noise = lambda y: np.random.normal(0, 0.5)
    pool = Pool(n_cores)
    multiple_results = []
    bandits = []
    for i in range(n_trials):
        bandits.append(ContextBandit(lambdas, noise))
        multiple_results.append(pool.apipe(sim_knn_amb, bandits[i], Z, 1, lambda x: 16, h))
    data = [res.get() for res in multiple_results]
    np.savetxt('regrets2.csv', np.asarray(data), delimiter=',')


if __name__ == '__main__':
    case2(10, 6, 2, 3)
