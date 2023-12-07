"""KNN-UCB: the K-Nearest-Neighbour UCB policy (Reeve, Mellor & Brown, 2018).

On each round, the policy looks at the k nearest previously-seen contexts
(for a k chosen per-arm to balance sample size against relevance) and runs a
UCB-style index computed only from those neighbours. See the accompanying
report for the formal definition and regret discussion; ``ternary_search.py``
implements the search over k that makes this practical to run.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .ternary_search import k_ternary_search


def knn_ucb(bandit, contexts, theta, phi, k_max, h=lambda x: x):
    """Run KNN-UCB for len(contexts) rounds and return the pseudo-regret trace.

    Parameters
    ----------
    bandit : ContextBandit
        Environment to play against; ``bandit.regret`` is read after each pull.
    contexts : array-like, shape (n, ambient_dim)
        The context vector observed on each round (already in the space the
        learner sees, e.g. after mapping a latent variable into a higher
        dimension -- see ``experiments/knn_ucb_vs_ucbogram.py``).
    theta : float
        Exploration scale in the confidence-radius term of the index.
    phi : callable
        Non-decreasing function phi(t) weighting the distance-radius term.
    k_max : int
        Upper bound on the neighbourhood size considered each round. Capping
        k avoids the O(n^2) style blow-up discussed in the report: for large
        t the optimal k is typically far smaller than t.
    h : callable, optional
        Feature map applied to each context before computing distances.
        Defaults to the identity.

    Returns
    -------
    np.ndarray
        Cumulative pseudo-regret after each round.
    """
    contexts = np.asarray(contexts)
    features = np.array([h(x) for x in contexts])
    n = contexts.shape[0]
    n_arms = bandit.k
    assert n >= n_arms

    actions = np.array([])
    rewards = np.array([])
    regrets = np.array([])

    for arm in range(n_arms):  # play each arm once regardless of context
        reward = bandit.pull(arm, contexts[arm])
        actions = np.append(actions, arm)
        rewards = np.append(rewards, reward)
        regrets = np.append(regrets, bandit.regret)

    for t in range(n_arms, n):
        upper = min(t, k_max)
        distances = cdist(features[:t], [features[t]], "euclidean").ravel()
        nearest = np.argpartition(distances, upper - 1)[:upper]
        k_nearest_sorted = np.stack((nearest, distances[nearest]), axis=1)
        k_nearest_sorted = k_nearest_sorted[k_nearest_sorted[:, 1].argsort()]

        index = np.zeros(n_arms)
        for arm in range(n_arms):
            k_star, u_star = k_ternary_search(actions, k_nearest_sorted, theta, phi, arm, upper)
            k_nearest = k_nearest_sorted[: int(k_star), 0].astype(int)
            in_arm = actions[k_nearest] == arm
            n_a = np.sum(in_arm)
            mean_hat = np.sum(rewards[k_nearest] * in_arm) / n_a if n_a != 0 else 0.0
            index[arm] = mean_hat + u_star

        arm = int(np.argmax(index))
        reward = bandit.pull(arm, contexts[t])
        actions = np.append(actions, arm)
        rewards = np.append(rewards, reward)
        regrets = np.append(regrets, bandit.regret)

    return regrets
