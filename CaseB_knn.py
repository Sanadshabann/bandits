import numpy as np
import math as m
from pathos.multiprocessing import ProcessingPool as Pool
from ContextBandit import ContextBandit
from scipy.stats import special_ortho_group as ortho
from Knn_Ucb import knn_ucb


def caseB_knn(n_trials, n_cores, d, D):
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
        multiple_results.append(pool.apipe(knn_ucb, bandits[i], Z, 1, lambda x: 16, h))
    data = [res.get() for res in multiple_results]
    ##np.savetxt('regretsB_knn.csv', np.asarray(data), delimiter=',')


if __name__ == '__main__':
    caseB_knn(10, 6, 2, 3)
