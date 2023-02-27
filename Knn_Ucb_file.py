import numpy as np
import math as m
from Ternary_Search import k_ternary_search
from scipy.spatial.distance import cdist
import ContextBandit as cb


def Knn_Ucb_file(bandit, context_file, theta, phi, data_file, h=lambda x: x):
    Z = np.loadtxt(context_file, delimiter=",")
    X = [h(z) for z in Z]
    data = np.loadtxt(data_file, delimiter=",")
    actions = list(data[0, :]) if (len(data) != 0) else []
    rewards = list(data[1, :]) if (len(data) != 0) else []
    regrets = list(data[2, :]) if (len(data) != 0) else []
    bandit.regret = regrets[-1] if (len(regrets) != 0) else 0
    n_arms = bandit.k  # number of arms
    t0 = len(actions)
    n = np.shape(Z)[0]
    if (t0 == 0):
        for a in range(n_arms):  ##play each arm once regardless of context
            reward = bandit.pull(a, Z[a])
            rewards.append(reward)
            actions.append(a)
            regrets.append(bandit.regret)
            t0 += 1
    for t in range(t0, n, 1):
        distances = cdist(X[:t], [X[t]], 'Euclidean')
        distances = [dist for sublist in distances for dist in sublist]
        k_a = np.zeros(n_arms)
        index = np.zeros(n_arms)  ## to store I_t,k
        for a in range(n_arms):  ##evaluate k_a for all a
            k_a[a] = k_ternary_search(actions, distances, theta, phi, a)
            k_least = np.argpartition(distances, int(k_a[a] - 1))[:int(k_a[a])]
            N = sum(np.array(actions)[k_least] == a)
            S = sum(np.array(rewards)[k_least] * (np.array(actions)[k_least] == a))
            f_hat = S / N if (N != 0) else 0
            index[a] = f_hat + m.sqrt(theta * m.log(t) / N) + phi(t) * distances[k_least[-1]] if (N != 0) else m.inf
        # print(k_a)
        arm = np.argmax(index)
        reward = bandit.pull(arm, Z[t])
        regrets.append(bandit.regret)
        actions.append(arm)
        rewards.append(reward)
        if t % 1000 == 999:
            data = np.stack((actions, rewards, regrets))
            np.savetxt(data_file, np.asarray(data), delimiter=",")
    return actions, rewards, regrets


if __name__ == "__main__":
    n = 100000
    # Z = np.random.uniform(0, 1, size=(n, 2))
    # np.savetxt("data_D/context.csv", np.asarray(Z), delimiter=",")
    h = lambda z: [z[0], z[1], np.prod(z), m.sin(sum(z) * m.pi)]
    lambdas = [lambda x: np.random.normal(0, 0.3),
               lambda x: np.prod([m.sin(4 * xs * m.pi) for xs in x])]
    noise = lambda x: np.random.normal(0, 0.2)
    bandit = cb.ContextBandit(lambdas, noise)
    regrets = Knn_Ucb_file(bandit, "data_D/context.csv", 1, lambda x: 15, "data_D/knn_regrets.csv", h=h)
    np.savetxt("data_D/knn_regrets.csv", np.asarray(regrets), delimiter=",")
