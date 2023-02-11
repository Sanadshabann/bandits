from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import math as m
import ContextBandit
from Knn_Ucb import knn_ucb



def caseA_knn(n_trials, n_cores):
    X = np.random.uniform(0, 1, size=(10000, 2))
    lambdas = [lambda x: np.prod([m.sin(4 * xs * m.pi) for xs in x]),
               lambda x: np.prod([m.cos(7 * xs * m.pi) for xs in x])]
    noise = lambda x: np.random.normal(0, 0.5)
    pool = Pool(n_cores)
    multiple_results = []
    bandits = []
    for i in range(n_trials):
        bandits.append(ContextBandit.ContextBandit(lambdas, noise))
        multiple_results.append(pool.apipe(knn_ucb, bandits[i], X, 1, lambda x: 23))
    data = [res.get() for res in multiple_results]
    ##np.savetxt('regretsA_knn.csv', np.asarray(data), delimiter=',')


if __name__ == "__main__":
    caseA_knn(50, 12)
