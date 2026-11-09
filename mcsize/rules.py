"""Stopping rules for estimating a probability p, and an experiment that measures how well each one works.

Goal in every case: estimate p to within +/-10% of its own value (relative error 0.1) with 95% confidence.
Each simulated run is a coin flip that comes up "event" with probability p, so counts are drawn directly from the
binomial and negative binomial distributions (exactly equivalent to running the flips one by one, and much faster).

Rules:
- fixed:      N = 10,000 runs, the rule of thumb for an absolute error of 0.01 (x = 100, N = x^2); Wald interval.
- pilot:      1,000 pilot runs give a first estimate; if it is 0 we stop and report 0, otherwise we compute the N that the
              formula asks for and make N fresh runs; Wald interval.
- naive:      run in batches of 100 and stop as soon as the Wald half-width is at most 10% of the current estimate.
              With 0 events the half-width is 0, so the rule stops immediately: the trap.
- inverse:    keep running until 385 events have been seen (385 = 1.96^2 / 0.1^2, rounded up), then estimate
              (K - 1) / (n - 1) with a Wilson interval. The number of runs adapts to p by itself.
Every rule is capped at `max_n` runs.
"""
from __future__ import annotations

from math import ceil

import numpy as np

from .stats import n_for_prob_rel, zval

REL, CONF = 0.1, 0.95


def _wald_half(k, n, z):
    p = k / n
    return z * np.sqrt(p * (1 - p) / n)


def _wilson(k, n, z):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre, half


def run_rule(rule: str, p: float, reps: int, rng: np.random.Generator, max_n: int = 5_000_000,
             rel: float = REL, conf: float = CONF) -> dict[str, np.ndarray]:
    z = zval(conf)
    if rule == "fixed":
        n = np.full(reps, 10_000)
        k = rng.binomial(n, p)
        est, half = k / n, _wald_half(k, n, z)
    elif rule == "pilot":
        n0 = 1_000
        k0 = rng.binomial(n0, p, size=reps)
        n = np.array([min(max_n, n_for_prob_rel(k / n0, rel, conf)) if k > 0 else 0 for k in k0])
        k = rng.binomial(np.maximum(n, 1), p) * (n > 0)
        est = np.where(n > 0, k / np.maximum(n, 1), 0.0)
        half = np.where(n > 0, _wald_half(k, np.maximum(n, 1), z), 0.0)
        n = n + n0
    elif rule == "naive":
        batch = 100
        k = np.zeros(reps, dtype=np.int64)
        n = np.zeros(reps, dtype=np.int64)
        active = np.ones(reps, dtype=bool)
        while active.any():
            idx = np.flatnonzero(active)
            k[idx] += rng.binomial(batch, p, size=len(idx))
            n[idx] += batch
            ph = k[idx] / n[idx]
            stop = (_wald_half(k[idx], n[idx], z) <= rel * ph) | (n[idx] >= max_n)
            active[idx[stop]] = False
        est, half = k / n, _wald_half(k, n, z)
    elif rule == "inverse":
        K = ceil(z * z / rel ** 2)
        n = K + rng.negative_binomial(K, p, size=reps)
        n = np.minimum(n, max_n)
        est = (K - 1) / (n - 1)
        _, half = _wilson(np.full(reps, K), n, z)
        # Wilson interval centred on its own centre; coverage is judged on that interval
        centre, _ = _wilson(np.full(reps, K), n, z)
        return {"n": n, "est": est, "lo": centre - half, "hi": centre + half}
    else:
        raise ValueError(rule)
    return {"n": n, "est": est, "lo": est - half, "hi": est + half}


def summarize(res: dict[str, np.ndarray], p: float, rel: float = REL) -> dict[str, float]:
    covered = (res["lo"] <= p) & (p <= res["hi"])
    # small tolerance: in floating point |k/n - p| / p can come out as 0.10000000000000005 exactly at the boundary
    within = np.abs(res["est"] - p) <= rel * p * (1 + 1e-9)
    return {"coverage": float(covered.mean()), "within_target": float(within.mean()),
            "mean_n": float(res["n"].mean()), "median_n": float(np.median(res["n"])),
            "share_zero": float((res["est"] == 0).mean()),
            "mean_est_over_p": float(res["est"].mean() / p)}
