"""Monte Carlo error formulas: confidence intervals and the number of runs needed.

Notation: N runs, confidence level `conf` (0.95 by default), z the matching two-sided normal quantile (1.96 for 95%).
"""
from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist

import numpy as np


def zval(conf: float = 0.95) -> float:
    return NormalDist().inv_cdf(0.5 + conf / 2.0)


# ---- confidence intervals -------------------------------------------------------------------------------------------

def mean_ci(x: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """Estimate and half-width of the CLT interval for a mean."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), zval(conf) * float(x.std(ddof=1)) / sqrt(len(x))


def wald(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    p = k / n
    return p, zval(conf) * sqrt(p * (1 - p) / n)


def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval (centre, half-width): stays sensible when k is 0 or very small."""
    z = zval(conf)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre, half


def quantile_ci(x: np.ndarray, q: float, conf: float = 0.95) -> tuple[float, float, float]:
    """Order-statistic interval for the q-quantile: (estimate, lower, upper). Uses no density estimate."""
    xs = np.sort(np.asarray(x, dtype=float))
    n = len(xs)
    z = zval(conf)
    half = z * sqrt(n * q * (1 - q))
    lo = int(max(0, np.floor(n * q - half)))
    hi = int(min(n - 1, np.ceil(n * q + half)))
    return float(np.quantile(xs, q)), float(xs[lo]), float(xs[hi])


def expected_shortfall(x: np.ndarray, alpha: float, conf: float = 0.95) -> tuple[float, float]:
    """Mean of the worst `alpha` share of outcomes (lower tail) and the half-width of its CLT interval.

    ES = v + E[(X - v) 1{X <= v}] / alpha with v the alpha-quantile; the variance of the estimator is approximately
    Var((X - v) 1{X <= v}) / (alpha^2 N).
    """
    x = np.asarray(x, dtype=float)
    v = np.quantile(x, alpha)
    h = np.where(x <= v, x - v, 0.0)
    es = v + h.mean() / alpha
    return float(es), zval(conf) * float(h.std(ddof=1)) / (alpha * sqrt(len(x)))


# ---- how many runs ----------------------------------------------------------------------------------------------------

def n_rule_of_thumb(eps: float) -> int:
    """The LinkedIn rule: for an error below 1/x, run at least x^2 times."""
    return ceil((1.0 / eps) ** 2)


def n_for_mean(sigma: float, eps: float, conf: float = 0.95) -> int:
    return ceil((zval(conf) * sigma / eps) ** 2)


def n_for_prob_abs(p: float, eps: float, conf: float = 0.95) -> int:
    return ceil(zval(conf) ** 2 * p * (1 - p) / eps ** 2)


def n_for_prob_rel(p: float, rel: float, conf: float = 0.95) -> int:
    return ceil(zval(conf) ** 2 * (1 - p) / (p * rel ** 2))


def n_for_difference(pa: float, pb: float, delta: float, conf: float = 0.95, power: float | None = None,
                     paired_var: float | None = None) -> int:
    """Runs needed so that a difference of size `delta` between two probabilities is resolved.

    Without `power`: the half-width of the interval for pa - pb is at most delta (the difference is not noise).
    With `power`: a true difference delta is detected with that probability.
    Variance per run of the difference estimator: for two outcomes of one multinomial run, pa + pb - (pa - pb)^2;
    for two independent simulations with N runs each, pa(1-pa) + pb(1-pb); for two plans on the same scenarios pass the
    measured per-run variance of (indicator a - indicator b) as `paired_var`.
    """
    var = pa + pb - (pa - pb) ** 2 if paired_var is None else paired_var
    z = zval(conf)
    if power is not None:
        z += NormalDist().inv_cdf(power)
    return ceil(z * z * var / delta ** 2)
