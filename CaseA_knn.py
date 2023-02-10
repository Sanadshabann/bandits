from pathos.multiprocessing import ProcessingPool as Pool
import numpy
import math as m
import ContextBandit
from Knn_Ucb import knn_ucb



def caseA_knn(n_trials, n_cores):
    X = numpy.random.uniform(0, 1, size=(10000, 2))
    lambdas = [lambda x: numpy.prod([m.sin(4 * xs * m.pi) for xs in x]),
               lambda x: numpy.prod([m.cos(7 * xs * m.pi) for xs in x])]
    noise = lambda x: numpy.random.normal(0, 0.5)
    pool = Pool(n_cores)
    multiple_results = []
    bandits = []
    for i in range(n_trials):
        bandits.append(ContextBandit.ContextBandit(lambdas, noise))
        multiple_results.append(pool.apipe(knn_ucb, bandits[i], X, 1, lambda x: 23))
    data = [res.get() for res in multiple_results]
    numpy.savetxt('regrets1.csv', numpy.asarray(data), delimiter=',')


if (__name__ == "__main__"):
    caseA_knn(50, 12)
