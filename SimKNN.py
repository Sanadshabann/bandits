from pathos.multiprocessing import ProcessingPool as Pool
import numpy
import math as m
from scipy.spatial.distance import cdist

import ContextBandit
import Ternary_Search


def sim_knn(bandit, X, theta, phi):
    n = len(X)
    regrets = []
    n_arms = bandit.k  # number of arms
    actions = []
    rewards = []
    assert (n >= n_arms)
    for a in range(n_arms):  ##play each arm once regardless of context
        reward = bandit.pull(a, X[a])
        rewards.append(reward)
        actions.append(a)
        regrets.append(bandit.regret)
    for t in range(n_arms, n, 1):
        distances = cdist(X[:t], [X[t]], 'Euclidean')
        distances = [dist for sublist in distances for dist in sublist]
        k_a = numpy.zeros(n_arms)
        index = numpy.zeros(n_arms)  ## to store I_t,k
        for a in range(n_arms):  ##evaluate k_a for all a
            k_a[a] = Ternary_Search.k_ternary_search(actions, rewards, distances, theta, phi, a)
            k_least = numpy.argpartition(distances, int(k_a[a] - 1))[:int(k_a[a])]
            N = sum(numpy.array(actions)[k_least] == a)
            S = sum(numpy.array(rewards)[k_least] * (numpy.array(actions)[k_least] == a))
            f_hat = S / N if (N != 0) else 0
            index[a] = f_hat + m.sqrt(theta * m.log(t) / N) + phi(t) * distances[k_least[-1]] if (N != 0) else m.inf
        arm = numpy.argmax(index)
        reward = bandit.pull(arm, X[t])
        regrets.append(bandit.regret)
        actions.append(arm)
        rewards.append(reward)
    return regrets




def case1(n_trials, n_cores):
    X = numpy.random.uniform(0, 1, size=(10000, 2))
    lambdas = [lambda x: numpy.prod([m.sin(4 * xs * m.pi) for xs in x]),
               lambda x: numpy.prod([m.cos(7 * xs * m.pi) for xs in x])]
    noise = lambda x: numpy.random.normal(0, 0.5)
    pool = Pool(n_cores)
    multiple_results = []
    bandits = []
    for i in range(n_trials):
        bandits.append(ContextBandit.ContextBandit(lambdas, noise))
        multiple_results.append(pool.apipe(sim_knn, bandits[i], X, 1, lambda x: 23))
    data = [res.get() for res in multiple_results]
    numpy.savetxt('regrets1.csv', numpy.asarray(data), delimiter=',')


if (__name__ == "__main__"):
    case1(50, 12)
