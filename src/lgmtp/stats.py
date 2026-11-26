"""Bootstrap statistics over sentences.

Speed-up on a resample = sum(emitted) / sum(cost) over the resampled sentences (ratio of sums, the
same estimator as the point value). Paired comparisons use the same resampled indices for both
policies, which must have been simulated on the same sentence list.
"""
from __future__ import annotations

import numpy as np


def _idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(emitted, cost, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    e, c = np.asarray(emitted, float), np.asarray(cost, float)
    I = _idx(len(e), n_boot, seed)
    s = e[I].sum(1) / c[I].sum(1)
    a = (1 - level) / 2
    return dict(point=float(e.sum() / c.sum()), lo=float(np.quantile(s, a)), hi=float(np.quantile(s, 1 - a)),
                n_boot=n_boot, seed=seed)


def paired_bootstrap(emitted_a, cost_a, emitted_b, cost_b, n_boot: int = 1000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """Difference speedup(a) - speedup(b) with a paired bootstrap over sentences.
    p_better = fraction of resamples where the difference is > 0."""
    ea, ca = np.asarray(emitted_a, float), np.asarray(cost_a, float)
    eb, cb = np.asarray(emitted_b, float), np.asarray(cost_b, float)
    if not (len(ea) == len(ca) == len(eb) == len(cb)):
        raise ValueError("paired bootstrap needs per-sentence arrays of equal length")
    I = _idx(len(ea), n_boot, seed)
    d = ea[I].sum(1) / ca[I].sum(1) - eb[I].sum(1) / cb[I].sum(1)
    a = (1 - level) / 2
    return dict(diff=float(ea.sum() / ca.sum() - eb.sum() / cb.sum()), lo=float(np.quantile(d, a)),
                hi=float(np.quantile(d, 1 - a)), p_better=float((d > 0).mean()), n_boot=n_boot, seed=seed)


def bootstrap_stat_by_sentence(groups: list, fn, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """Generic sentence-level bootstrap: ``groups`` is a list of per-sentence objects, ``fn`` maps a
    list of them to a scalar."""
    I = _idx(len(groups), n_boot, seed)
    vals = np.array([fn([groups[i] for i in row]) for row in I], float)
    n_valid = int(np.isfinite(vals).sum())
    vals = vals[np.isfinite(vals)]
    a = (1 - level) / 2
    lo = float(np.quantile(vals, a)) if n_valid else float("nan")      # e.g. constant variable
    hi = float(np.quantile(vals, 1 - a)) if n_valid else float("nan")
    return dict(point=float(fn(groups)), lo=lo, hi=hi, n_boot=n_boot, n_valid=n_valid, seed=seed)
