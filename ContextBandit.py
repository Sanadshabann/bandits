import numpy as np
import random



class ContextBandit:
    def __init__(self, lambdas, noise):  # lambdas is a list of functions that take context vector x and return a reward
        self.k = len(lambdas)
        self.lambdas = lambdas
        self.noise = noise
        self.regret = 0

    def pull(self, a, x):
        rewards = [reward(x) for reward in self.lambdas]
        self.regret += max(rewards) - rewards[a]
        return rewards[a] + self.noise(x)
