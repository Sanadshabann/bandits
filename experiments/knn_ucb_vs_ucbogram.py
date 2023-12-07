"""Reproduce the report's headline result (Section 5.4, Figure 8):
KNN-UCB vs. UCBogram as the ambient context dimension grows.

A 2-dimensional latent variable Z drives the expected rewards, but the
learner only observes it mapped into a higher-dimensional ambient space
(via a random combination of basis functions), so the *intrinsic*
dimensionality of the problem stays low even as the *ambient* dimension D
grows. KNN-UCB's regret is roughly stable across D; UCBogram's degrades,
since it bins the ambient space directly and its bin count grows as m**D.

The defaults below (small horizon, few trials) run in well under a minute
so the comparison is easy to reproduce and sanity-check. To approach the
report's actual figure (n=10**5, 10 trials, D up to 20) pass
`--horizon 100000 --trials 10 --dims 2 5 15 20`, but expect that to take
hours single-threaded -- use `--workers` to parallelise across trials.

Usage:
    python experiments/knn_ucb_vs_ucbogram.py
    python experiments/knn_ucb_vs_ucbogram.py --horizon 20000 --trials 8 --workers 4
"""

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bandits import ContextBandit, knn_ucb, ucbogram

LATENT_DIM = 2
BASIS_FUNCTIONS = [
    lambda x: np.ones_like(x),
    lambda x: x,
    lambda x: x**2,
    lambda x: x**3,
    np.sin,
    np.cos,
    np.exp,
]


def random_ambient_map(ambient_dim, rng):
    """A random map h: [0, 1]^2 -> R^ambient_dim built from paired basis
    functions, as described in Section 5.4 of the report."""
    basis_idx = rng.integers(0, len(BASIS_FUNCTIONS), size=2 * ambient_dim)

    def h(z):
        out = np.empty(ambient_dim)
        for ell in range(ambient_dim):
            f1 = BASIS_FUNCTIONS[basis_idx[2 * ell]]
            f2 = BASIS_FUNCTIONS[basis_idx[2 * ell + 1]]
            out[ell] = f1(z[0]) * f2(z[1])
        return out

    return h


def make_bandit(noise_std=math.sqrt(0.5)):
    reward_fns = [
        lambda z: math.sin(4 * math.pi * z[0]) * math.sin(4 * math.pi * z[1]),
        lambda z: math.cos(3 * math.pi * z[0]) * math.cos(3 * math.pi * z[1]),
    ]
    noise = lambda z: np.random.normal(0, noise_std)
    return ContextBandit(reward_fns, noise)


def run_trial(args):
    ambient_dim, horizon, seed, knn_kwargs, ucbogram_m = args
    rng = np.random.default_rng(seed)
    h = random_ambient_map(ambient_dim, rng)
    # Rewards are a function of the latent Z; only h(Z) is ever handed to a
    # policy for decision-making. Basis functions are non-negative on
    # [0, 1], so h(Z) components live in [0, e] * [0, e] ~= [0, 7.4].
    latent_contexts = rng.uniform(0, 1, size=(horizon, LATENT_DIM))

    knn_regret = knn_ucb(make_bandit(), latent_contexts, h=h, **knn_kwargs)
    ucbogram_regret = ucbogram(
        make_bandit(), latent_contexts, m=ucbogram_m, lo=0, hi=8, h=h
    )
    return knn_regret, ucbogram_regret


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=4000, help="rounds per trial")
    parser.add_argument("--trials", type=int, default=6, help="trials averaged per dimension")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 15], help="ambient dimensions to sweep")
    parser.add_argument("--theta", type=float, default=1.0, help="KNN-UCB exploration scale")
    parser.add_argument("--phi", type=float, default=6.0, help="KNN-UCB distance-radius weight (constant phi(t))")
    parser.add_argument("--k-max", type=int, default=200, help="KNN-UCB neighbourhood size cap")
    parser.add_argument("--bins", type=int, default=4, help="UCBogram bins per ambient dimension")
    parser.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/knn_ucb_vs_ucbogram.png")
    args = parser.parse_args()

    knn_kwargs = dict(theta=args.theta, phi=lambda t: args.phi, k_max=args.k_max)

    fig, axes = plt.subplots(len(args.dims), 1, figsize=(7, 3.2 * len(args.dims)), sharex=True)
    axes = np.atleast_1d(axes)
    rounds = np.arange(1, args.horizon + 1)

    for ax, ambient_dim in zip(axes, args.dims):
        print(f"Ambient dimension D={ambient_dim} ({args.trials} trials)")
        jobs = [
            (ambient_dim, args.horizon, args.seed + t, knn_kwargs, args.bins)
            for t in range(args.trials)
        ]

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                results = list(pool.map(run_trial, jobs))
        else:
            results = [run_trial(job) for job in jobs]

        knn_regrets = np.stack([r[0] for r in results])
        ucbogram_regrets = np.stack([r[1] for r in results])

        for label, regrets, colour in [
            ("KNN-UCB", knn_regrets, "tab:green"),
            ("UCBogram", ucbogram_regrets, "tab:blue"),
        ]:
            mean = regrets.mean(axis=0)
            std = regrets.std(axis=0)
            ax.plot(rounds, mean, label=label, color=colour)
            ax.fill_between(rounds, mean - std, mean + std, color=colour, alpha=0.2)

        ax.set_ylabel(r"$\bar{R}_n$")
        ax.set_title(f"Ambient dimension D = {ambient_dim} (intrinsic dimension = {LATENT_DIM})")
        ax.legend()

    axes[-1].set_xlabel("n")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
