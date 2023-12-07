# Contextual Multi-Armed Bandits: KNN-UCB vs. UCBogram

A from-scratch implementation and empirical study of contextual bandit
policies, written as an undergraduate research project supervised by
Dr. Henry Reeve. The full write-up — definitions, regret proofs, and the
experiment design — is in [`docs/report.pdf`](docs/report.pdf); this README
covers the code and how to reproduce the experiments.

**Headline result:** [KNN-UCB](https://arxiv.org/abs/1803.00316) (Reeve,
Mellor & Brown, 2018) outperforms
[UCBogram](https://www.jmlr.org/proceedings/papers/v9/rigollet10a/rigollet10a.pdf)
(Rigollet & Zeevi, 2010) in high ambient dimension when the *intrinsic*
dimensionality of the context is low. UCBogram bins the observed context
space directly, so its bin count grows as `m^D` in the ambient dimension `D`
and its sample efficiency collapses; KNN-UCB adapts its neighbourhood size to
local data density and is largely insensitive to `D`. `docs/report.pdf`
(Section 5) derives a regret bound for UCBogram and discusses why KNN-UCB is
harder to bound but empirically more robust.

The report's KNN-UCB implementation also uses **ternary search** to pick its
neighbourhood size `k` each round: the exploration-bonus term is
non-increasing in `k` while the distance-radius term is non-decreasing, so
their sum is (empirically) U-shaped and its minimiser can be found in
`O(log k_max)` evaluations instead of scanning every candidate `k`
(`src/bandits/ternary_search.py`).

## Repository layout

```
src/bandits/            core implementations (import as `bandits`)
    environments.py     GaussianBandit, ContextBandit
    stochastic_policies.py   ETC, UCB (non-contextual)
    ucbogram.py          UCBogram
    knn_ucb.py           KNN-UCB
    ternary_search.py    the k-selection speedup used by KNN-UCB
experiments/            scripts that reproduce the report's figures
notebooks/quickstart.ipynb   minimal runnable demo
tests/                  unit tests (pytest)
docs/report.pdf         the full report
```

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # installs numpy/scipy/matplotlib + pytest
```

Verify the install:

```bash
pytest
```

## Reproducing the experiments

Both scripts save a plot under `results/` and print progress as they run.
Defaults are sized to finish in well under a minute; pass larger
`--horizon`/`--trials` to approach the report's actual figures (see each
script's `--help` and its docstring for the full-scale settings and expected
runtime).

**KNN-UCB vs. UCBogram across ambient dimension** (report Figure 8):

```bash
python experiments/knn_ucb_vs_ucbogram.py
```

**ETC vs. UCB on a stochastic bandit** (report Figure 3):

```bash
python experiments/etc_vs_ucb.py
```

Or open `notebooks/quickstart.ipynb` for an interactive, single-trial version
of the same comparison.

## Using the library directly

```python
import numpy as np
from bandits import ContextBandit, knn_ucb, ucbogram

reward_fns = [
    lambda z: np.sin(4 * np.pi * z[0]),
    lambda z: np.cos(3 * np.pi * z[0]),
]
noise = lambda z: np.random.normal(0, 0.5)

contexts = np.random.uniform(0, 1, size=(5000, 1))

bandit = ContextBandit(reward_fns, noise)
regret = knn_ucb(bandit, contexts, theta=1.0, phi=lambda t: 6.0, k_max=200)

bandit = ContextBandit(reward_fns, noise)
regret = ucbogram(bandit, contexts, m=10)
```

Both policies return the cumulative pseudo-regret after each round.

## References

1. Bajwa, Agarwal & Manchanda (2015). *Ternary search algorithm: improvement of binary search.*
2. Lattimore & Szepesvári (2020). *Bandit Algorithms.* Cambridge University Press.
3. Reeve, Mellor & Brown (2018). *The K-Nearest Neighbour UCB Algorithm for Multi-Armed Bandits with Covariates.* [arXiv:1803.00316](https://arxiv.org/abs/1803.00316)
4. Rigollet & Zeevi (2010). *Nonparametric Bandits with Covariates.* COLT.
