import numpy as np
import random


class GaussianBandit:
    def __init__(self, means):
        self.means = means
        self.k = len(means)
        self.best_mean = max(means)
        self.regret = 0

    def pull(self, a):
        self.regret += (self.best_mean - self.means[a])
        return random.gauss(self.means[a], 1)


class LinContextGauss:
    def __init__(self, coeff):
        self.coeff = coeff
        self.k = len(coeff)
        self.regret = 0

    def pull(self, a, cov):
        self.regret += np.max(np.dot(self.coeff, cov)) - np.dot(self.coeff[a], cov)
        return random.gauss(np.dot(self.coeff[a], cov)[0], 1)
