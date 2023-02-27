import numpy as np
import math as m
from ContextBandit import ContextBandit
from pathos.multiprocessing import ProcessingPool as Pool
from Knn_Ucb import knn_ucb

if __name__ == "__main__":
    n = 80000
    n_cores = 5
    pool = Pool(n_cores)
    Z = np.random.uniform(0, 1, size=(n, 2))
    h = lambda z: [z[0], z[1], np.prod(z), m.sin(sum(z) * m.pi)]
    lambdas = [lambda x: np.random.normal(0, 0.3),
               lambda x: np.prod([m.sin(4 * xs * m.pi) for xs in x])]
    noise = lambda x: np.random.normal(0, 2)
    multiple_results = []
    bandits = []
    pool = Pool(n_cores)
    for i in range(10):
        bandits.append(ContextBandit(lambdas, noise))
        multiple_results.append(pool.apipe(knn_ucb, bandits[i], Z, 1, phi=lambda x: 20, h=h, k_max=500))
    data = [res.get() for res in multiple_results]
    np.savetxt('data_C/knn_regrets.csv', np.asarray(data), delimiter=',')
