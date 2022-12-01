import numpy as np
import random
import math


class GaussianBandit:
    def __init__(self, means):
        self.means = means
        self.k = len(means)
        self.best_mean = max(means)
        self.regret = 0

    def pull(self, a):
        self.regret += (self.best_mean - self.means[a])
        return random.gauss(self.means[a], 1)


def etc(n, m, bandit):  ##Explore then commit algorithm
    k = bandit.k
    rewards = np.zeros(n)
    total_reward = 0
    actions = np.zeros(n)
    for i in range(n):
        if (i <= m * k - 1):
            rewards[i] = bandit.pull(i % k)
            total_reward += rewards[i]
            actions[i] = i % k
        else:
            if (i == m * k):  ##optimization
                best_mean_index = np.argmax(np.mean(np.transpose(np.reshape(rewards[0:m * k], (m, k))), 1))
            actions[i] = best_mean_index
            rewards[i] = bandit.pull(best_mean_index)
            total_reward += rewards[i]
    random_regret = n * bandit.best_mean - total_reward
    return random_regret  ## returns a list of a_t, x_t pairs and the observed random regret


def ucb(delta, bandit, n):
    k = bandit.k  # number of arms
    upper_bounds = float('inf') * np.ones(k)  # set upper bounds to infinity
    # result = []  # to store pairs of a_t, x_t
    means = np.zeros((k, 2))  # to store means and T_i
    for i in range(n):
        arm = np.argmax(upper_bounds)
        reward = bandit.pull(arm)
        # result.append((arm, reward))
        means[arm, 0] = (means[arm, 0] * means[arm, 1] + reward) / (means[arm, 1] + 1)
        means[arm, 1] += 1
        upper_bounds[arm] = means[arm, 0] + np.sqrt(2 * np.log(1 / delta) / means[arm, 1])
    return bandit.regret


def ucbi(bandit, delta, n):
    k = bandit.k  # number of arms
    upper_bounds = float('inf') * np.ones(k)  # set upper bounds to infinity
    # result = []  # to store pairs of a_t, x_t
    means = np.zeros((k, 2))  # to store means and T_i
    for i in range(k):  # pull each arm once
        reward = bandit.pull(i)
        # result.append((i, reward))
        means[i, 0] = reward
        means[i, 1] = 1
        upper_bounds[i] = means[i, 0] + np.sqrt(2 * np.log(1 / delta))
    for i in range(n - k):
        if (math.log(i + 2, 2) - math.floor(
                math.log((i + 2), 2)) == 0):  ##only updates the arm at the beginning of each phase
            arm = np.argmax(upper_bounds)
        reward = bandit.pull(arm)
        # result.append((arm, reward))
        means[arm, 0] = (means[arm, 0] * means[arm, 1] + reward) / (means[arm, 1] + 1)
        means[arm, 1] += 1
        upper_bounds[arm] = means[arm, 0] + np.sqrt(2 * np.log(1 / delta) / means[arm, 1])

    return bandit.regret


def ucbii(bandit, delta, n, alpha):
    k = bandit.k  # number of arms
    upper_bounds = float('inf') * np.ones(k)  # set upper bounds to infinity
    # result = []  # to store pairs of a_t, x_t
    means = np.zeros((k, 2))  # to store means and T_i

    for i in range(k):  # pull each arm once
        reward = bandit.pull(i)
        # result.append((i, reward))
        means[i, 0] = reward
        means[i, 1] = 1
        upper_bounds[i] = means[i, 0] + np.sqrt(2 * np.log(1 / delta))

    arm = np.argmax(upper_bounds)
    t_l = np.array(
        means[:, 1])  # T_i(t_l - 1), array of number of times each arm is pulled before the beginning of current phase
    for i in range(n - k):
        if means[arm, 1] >= alpha * t_l[arm]:  ##only updates the arm when T_i(t) >= alpha * T_i(t_l-1)
            t_l[arm] = means[arm, 1]
            arm = np.argmax(upper_bounds)
        reward = bandit.pull(arm)
        # result.append((arm, reward))
        means[arm, 0] = (means[arm, 0] * means[arm, 1] + reward) / (means[arm, 1] + 1)
        means[arm, 1] += 1
        upper_bounds[arm] = means[arm, 0] + np.sqrt(2 * np.log(1 / delta) / means[arm, 1])

    return bandit.regret


def sim_ucb(trials, horizon, deltas):
    mean_regrets = np.zeros(len(deltas))
    for k, delta in enumerate(deltas):
        regrets = np.zeros(trials)
        for i in range(trials):
            bandit = GaussianBandit([0, -0.1, -0.15])
            regret = ucb(delta, bandit, horizon)
            regrets[i] = regret
        mean_regrets[k] = np.mean(regrets)
    return mean_regrets


def sim_ucbii(trials, horizon, alphas):
    delta = 0.1
    mean_regrets = np.zeros(len(alphas))
    for k, alpha in enumerate(alphas):
        regrets = np.zeros(trials)
        for i in range(trials):
            bandit = GaussianBandit([0, -0.1, -0.15])
            regret = ucbii(bandit, delta, horizon, alpha)
            regrets[i] = regret
        mean_regrets[k] = np.mean(regrets)
    return mean_regrets


def sim_ucbi(trials, horizon, deltas):
    mean_regrets = np.zeros(len(deltas))
    for k, delta in enumerate(deltas):
        regrets = np.zeros(trials)
        for i in range(trials):
            bandit = GaussianBandit([0, -0.1, -0.15])
            regret = ucbi(bandit, delta, horizon)
            regrets[i] = regret
        mean_regrets[k] = np.mean(regrets)
    return mean_regrets


def sim_etc(trials, horizon, m, deltas):
    mean_regrets = np.zeros(len(deltas))
    for k, delta in enumerate(deltas):
        regrets = np.zeros(trials)
        for i in range(trials):
            bandit = GaussianBandit([0, -delta])
            regret = etc(horizon, m, bandit)
            regrets[i] = regret
        mean_regrets[k] = np.mean(regrets)
    return mean_regrets


def sim_ucb_Delta(trials, horizon, Deltas, delta):  # for different suboptimality gaps (Q8)
    mean_regrets = np.zeros(len(Deltas))
    for k, Delta in enumerate(Deltas):
        regrets = np.zeros(trials)
        for i in range(trials):
            bandit = GaussianBandit([0, -Delta])
            regret = ucb(delta, bandit, horizon)
            regrets[i] = regret
        mean_regrets[k] = np.mean(regrets)
    return mean_regrets
