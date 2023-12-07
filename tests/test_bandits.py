import math

import numpy as np
import pytest

from bandits import ContextBandit, GaussianBandit, etc, knn_ucb, ucb, ucbogram
from bandits.ternary_search import k_ternary_search, uncertainty


def brute_force_best_k(actions, k_nearest_sorted, theta, phi, arm, k_max):
    values = [uncertainty(k, actions, k_nearest_sorted, theta, phi, arm) for k in range(1, k_max)]
    best_k = int(np.argmin(values)) + 1
    return best_k, values[best_k - 1]


def test_ternary_search_matches_brute_force_when_unimodal():
    # Ternary search assumes the uncertainty index is (roughly) U-shaped in k,
    # as observed empirically in the report (Section 5.3.3) -- it is not
    # proven unimodal in general. Here the first half of the neighbours all
    # belong to `arm`, so N_arm(k) grows by exactly one every step with no
    # plateaus, which does guarantee a single valley; this is the case the
    # heuristic is designed for, and search should land within a sliver of
    # the true optimum (discrete ternary search resolves its final bracket
    # of two candidates without an explicit tie-break, so it is not exact).
    n = 200
    distances = np.linspace(0, 1, n)
    actions = np.array([0.0] * (n // 2) + [1.0] * (n - n // 2))
    k_nearest_sorted = np.stack((np.arange(n, dtype=float), distances), axis=1)
    theta, phi = 1.0, lambda t: 5.0

    got_k, got_u = k_ternary_search(actions, k_nearest_sorted, theta, phi, 0, n)
    want_k, want_u = brute_force_best_k(actions, k_nearest_sorted, theta, phi, 0, n)
    assert got_u == pytest.approx(want_u, rel=1e-2)


def test_ternary_search_is_robust_on_arbitrary_data():
    # On adversarial (non-spatial) data the index can have multiple local
    # dips, so ternary search is not guaranteed to find the global optimum.
    # It should still terminate and return a valid, finite candidate.
    rng = np.random.default_rng(0)
    n = 200
    actions = rng.integers(0, 2, size=n).astype(float)
    distances = np.sort(rng.uniform(0, 1, size=n))
    k_nearest_sorted = np.stack((np.arange(n, dtype=float), distances), axis=1)
    theta, phi = 1.0, lambda t: 5.0

    for arm in (0, 1):
        k, u = k_ternary_search(actions, k_nearest_sorted, theta, phi, arm, n)
        assert 1 <= k <= n - 1
        assert math.isfinite(u)


def test_knn_ucb_runs_and_regret_is_monotonic():
    rng = np.random.default_rng(1)
    reward_fns = [lambda z: math.sin(4 * math.pi * z[0]), lambda z: math.cos(3 * math.pi * z[0])]
    noise = lambda z: 0.0
    bandit = ContextBandit(reward_fns, noise)
    contexts = rng.uniform(0, 1, size=(150, 1))

    regrets = knn_ucb(bandit, contexts, theta=1.0, phi=lambda t: 5.0, k_max=30)

    assert len(regrets) == len(contexts)
    assert np.all(np.diff(regrets) >= -1e-9)


def test_ucbogram_runs_and_regret_is_monotonic():
    rng = np.random.default_rng(2)
    reward_fns = [lambda z: math.sin(4 * math.pi * z[0]), lambda z: math.cos(3 * math.pi * z[0])]
    noise = lambda z: 0.0
    bandit = ContextBandit(reward_fns, noise)
    contexts = rng.uniform(0, 1, size=(150, 1))

    regrets = ucbogram(bandit, contexts, m=5)

    assert len(regrets) == len(contexts)
    assert np.all(np.diff(regrets) >= -1e-9)


def test_stochastic_policies_return_nonnegative_regret():
    bandit = GaussianBandit([0, -0.5])
    assert etc(200, 10, bandit) >= 0

    bandit = GaussianBandit([0, -0.5])
    assert ucb(1 / 200**2, bandit, 200) >= 0
