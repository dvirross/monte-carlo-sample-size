"""Idealized daily-reset leveraged fund, the model of the site's volatility simulator.

Underlying daily simple return r = mu/252 + sigma/sqrt(252) * z with z standard normal (or a standardized real QQQ day
for the bootstrap variant); the fund's daily factor is 1 + L * r (floored at 0); 252 trading days a year. Defaults are the
simulator's: L = 3, mu = 10%, sigma = 25%. No fees or financing costs.

The rare event studied is "the fund ends the year below `level` of its starting value" (level = 0.2: it loses 80% or more).

Variance-reduction estimators of p = P(event), each returning one number per simulated year (their mean estimates p):
- plain:       the 0/1 indicator;
- antithetic:  the average of the indicator on the path z and on its mirror -z ("mirror pairs"; one value per pair);
- control:     indicator - b * (control indicator - its exact probability), where the control is "the sum of the 252
               shocks z is below c". That sum is exactly normal with variance 252, so its probability is known; b is
               estimated from the same run (the usual regression coefficient);
- importance:  shocks drawn with mean -theta instead of 0, so crashes are common, and each path weighted by the
               likelihood ratio exp(theta * sum z + 252 * theta^2 / 2) (sum over the tilted shocks).
"""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np

DAYS = 252
L, MU, SIGMA = 3.0, 0.10, 0.25


def fund_terminal(z: np.ndarray, lev: float = L, mu: float = MU, sigma: float = SIGMA) -> np.ndarray:
    r = mu / DAYS + sigma / sqrt(DAYS) * z
    f = np.maximum(1.0 + lev * r, 0.0)
    return np.prod(f, axis=1)


def underlying_terminal(z: np.ndarray, mu: float = MU, sigma: float = SIGMA) -> np.ndarray:
    return np.prod(1.0 + mu / DAYS + sigma / sqrt(DAYS) * z, axis=1)


def standardized_qqq(qqq: np.ndarray) -> np.ndarray:
    return (qqq - qqq.mean()) / qqq.std(ddof=0)


def draw_z(n: int, rng: np.random.Generator, dist: str = "normal", pool: np.ndarray | None = None,
           df: float = 3.0) -> np.ndarray:
    if dist == "normal":
        return rng.standard_normal((n, DAYS))
    if dist == "t":
        return rng.standard_t(df, (n, DAYS)) / sqrt(df / (df - 2.0))
    if dist == "bootstrap":
        return pool[rng.integers(0, len(pool), size=(n, DAYS))]
    raise ValueError(dist)


def control_threshold(level: float = 0.2, lev: float = L, mu: float = MU, sigma: float = SIGMA) -> float:
    """Shock-sum threshold c at which the fund, under the log approximation, ends at `level`.

    log fund ~ lev*mu - lev^2 sigma^2 / 2 + lev*sigma/sqrt(252) * sum z, so the event is roughly sum z < c.
    """
    drift = lev * mu - 0.5 * lev ** 2 * sigma ** 2
    return (np.log(level) - drift) / (lev * sigma / sqrt(DAYS))


def control_prob(c: float) -> float:
    return NormalDist().cdf(c / sqrt(DAYS))


def estimator_values(method: str, n: int, rng: np.random.Generator, level: float = 0.2, theta: float | None = None,
                     **kw) -> np.ndarray:
    """Per-sample values whose mean estimates p. For 'antithetic', n is the number of mirror pairs."""
    c = control_threshold(level, **kw)
    if method == "plain":
        return (fund_terminal(draw_z(n, rng), **kw) < level).astype(float)
    if method == "antithetic":
        z = draw_z(n, rng)
        return 0.5 * ((fund_terminal(z, **kw) < level).astype(float) + (fund_terminal(-z, **kw) < level))
    if method == "control":
        z = draw_z(n, rng)
        y = (fund_terminal(z, **kw) < level).astype(float)
        x = (z.sum(axis=1) < c).astype(float)
        px = control_prob(c)
        cov = np.cov(y, x)
        b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        return y - b * (x - px)
    if method == "importance":
        th = -c / DAYS if theta is None else theta  # centre the tilted shock sum on the threshold
        z = draw_z(n, rng) - th
        w = np.exp(th * z.sum(axis=1) + DAYS * th * th / 2.0)
        return (fund_terminal(z, **kw) < level) * w
    raise ValueError(method)
