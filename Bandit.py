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
