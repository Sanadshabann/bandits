"""Reproduce the report's ETC vs. UCB comparison (Section 4.3, Figure 3).

For a 2-armed Gaussian bandit N(0, 1) vs. N(-Delta, 1), this sweeps the
suboptimality gap Delta and compares UCB against ETC at a few choices of the
exploration length m, including the m that minimises ETC's own regret bound
(Theorem 4.1) -- which requires oracle knowledge of Delta that UCB does not
need. UCB tracks the oracle-tuned ETC closely, and beats every fixed m away
from its optimum.

Usage:
    python experiments/etc_vs_ucb.py
    python experiments/etc_vs_ucb.py --horizon 1000 --trials 1000 --n-gaps 40
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bandits import GaussianBandit, etc, ucb


def etc_regret_bound(m, k, n, gaps, sigma=1.0):
    """Theorem 4.1's upper bound on ETC's regret, for picking m*."""
    exploit_term = m * np.sum(gaps)
    explore_term = (n - m * k) * np.sum(gaps * np.exp(-m * gaps / (4 * sigma**2)))
    return exploit_term + explore_term


def best_m(k, n, gaps):
    candidates = np.arange(1, n // k + 1)
    bounds = [etc_regret_bound(m, k, n, gaps) for m in candidates]
    return int(candidates[np.argmin(bounds)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--horizon", type=int, default=1000, help="rounds per game")
    parser.add_argument("--trials", type=int, default=200, help="repeats averaged per gap")
    parser.add_argument("--n-gaps", type=int, default=25, help="number of suboptimality gaps to sweep")
    parser.add_argument("--out", type=str, default="results/etc_vs_ucb.png")
    args = parser.parse_args()

    n = args.horizon
    delta = 1 / n**2  # UCB confidence parameter, as in Theorem 4.2
    gaps = np.linspace(0.02, 1.0, args.n_gaps)
    fixed_ms = [50, 100]

    curves = {f"ETC (m={m})": [] for m in fixed_ms}
    curves["ETC (m optimal)"] = []
    curves["UCB"] = []

    for gap_idx, gap in enumerate(gaps):
        print(f"[{gap_idx + 1}/{len(gaps)}] Delta = {gap:.3f}")
        m_star = best_m(k=2, n=n, gaps=np.array([0.0, gap]))

        per_policy_regrets = {name: [] for name in curves}
        for _ in range(args.trials):
            for m in fixed_ms:
                bandit = GaussianBandit([0, -gap])
                per_policy_regrets[f"ETC (m={m})"].append(etc(n, m, bandit))
            bandit = GaussianBandit([0, -gap])
            per_policy_regrets["ETC (m optimal)"].append(etc(n, m_star, bandit))
            bandit = GaussianBandit([0, -gap])
            per_policy_regrets["UCB"].append(ucb(delta, bandit, n))

        for name, regrets in per_policy_regrets.items():
            curves[name].append(np.mean(regrets))

    plt.figure(figsize=(7, 5))
    for name, values in curves.items():
        plt.plot(gaps, values, label=name)
    plt.xlabel("Delta")
    plt.ylabel(r"$\bar{R}_n$")
    plt.title(f"Expected regret vs. suboptimality gap (n={n}, {args.trials} trials/point)")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
