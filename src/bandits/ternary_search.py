"""Ternary search over the neighbourhood size k for KNN-UCB.

For a fixed round and arm, the KNN-UCB uncertainty index U(k) (see
``knn_ucb.py``) is unimodal in k: the confidence-radius term shrinks as k
grows while the distance-radius term grows, so U is U-shaped. This lets us
find the minimising k in O(log k_max) evaluations instead of scanning every
k in [1, k_max], which is the main optimisation this project contributes on
top of the original KNN-UCB policy (Reeve et al., 2018).
"""

import math

import numpy as np


def uncertainty(k, actions, k_nearest_sorted, theta, phi, arm):
    """KNN-UCB uncertainty index U^a_{t,k}(x) for a candidate neighbourhood size k."""
    t = actions.shape[0]
    k_nearest = k_nearest_sorted[:k, 0].astype(int)
    n_a = np.sum(actions[k_nearest] == arm)
    if n_a == 0:
        return math.inf
    return math.sqrt(theta * math.log(t) / n_a) + phi(t) * k_nearest_sorted[k - 1, 1]


def k_ternary_search(actions, k_nearest_sorted, theta, phi, arm, k_max):
    """Return (k*, U(k*)) minimising the uncertainty index over k in [1, k_max - 1].

    ``k_nearest_sorted`` is an (n, 2) array of (original_index, distance) pairs
    for the k_max nearest neighbours of the current context, sorted by distance.
    """
    f = lambda k: uncertainty(k, actions, k_nearest_sorted, theta, phi, arm)
    lo, hi = 1, k_max - 1
    while True:
        c1 = lo + math.floor((hi - lo) / 3)
        c2 = lo + 2 * math.floor((hi - lo) / 3)
        u1, u2 = f(c1), f(c2)
        if u1 > u2:
            lo = c1
        elif u1 < u2:
            hi = c2
        else:
            if c1 == c2:
                return c1, u1
            lo, hi = c1, c2
        if abs(lo - hi) <= 1:
            break
    k_star = math.floor((lo + hi) / 2)
    return k_star, f(k_star)
