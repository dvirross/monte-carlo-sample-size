"""Retirement withdrawal plan: does a portfolio last 30 years?

Start with 1 (one unit of real money). At the start of each year withdraw a fixed real amount `rate` (0.04 = 4% of the
starting balance, kept constant in real terms), then the rest earns that year's real portfolio return. The plan fails
if a withdrawal cannot be paid in full. The portfolio is rebalanced every year to `stock_share` stocks and the rest
10-year bonds.

Return scenarios (all built from the same 152 real annual returns, 1871-2022):
- "iid":        each year drawn independently from the 152 historical years (simple bootstrap);
- "block":      circular block bootstrap, consecutive runs of `block` historical years, so bad years that came in
                clusters stay together;
- "lognormal":  log(1 + portfolio return) drawn from a normal distribution with the historical mean and variance;
- "historical": the 123 actual 30-year windows that start in 1871, ..., 1993 (overlapping, so far fewer than 123
                independent pieces of evidence).
A scenario matrix has one row per simulated retirement and one column per year. Evaluating several plans on the same
matrix is what "common random numbers" means below.
"""
from __future__ import annotations

import numpy as np

from .data import load_annual

HORIZON = 30


def portfolio_returns(stock_share: float = 0.6, annual=None) -> np.ndarray:
    a = load_annual() if annual is None else annual
    return stock_share * a["stock_real"].to_numpy() + (1 - stock_share) * a["bond_real"].to_numpy()


def draw_scenarios(n: int, rng: np.random.Generator, method: str = "iid", horizon: int = HORIZON,
                   block: int = 10, stock_share: float = 0.6, hist: np.ndarray | None = None) -> np.ndarray:
    r = portfolio_returns(stock_share) if hist is None else hist
    T = len(r)
    if method == "iid":
        return r[rng.integers(0, T, size=(n, horizon))]
    if method == "block":
        nblocks = -(-horizon // block)
        starts = rng.integers(0, T, size=(n, nblocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T
        return r[idx.reshape(n, -1)[:, :horizon]]
    if method == "lognormal":
        lr = np.log1p(r)
        return np.expm1(rng.normal(lr.mean(), lr.std(ddof=1), size=(n, horizon)))
    if method == "historical":
        windows = np.array([r[s:s + horizon] for s in range(T - horizon + 1)])
        return windows  # deterministic: all windows, n and rng are ignored
    raise ValueError(method)


def evaluate(scen: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (terminal real balance, failed flag) for each row; a failed plan ends at 0."""
    w = np.ones(scen.shape[0])
    failed = np.zeros(scen.shape[0], dtype=bool)
    for t in range(scen.shape[1]):
        failed |= w < rate - 1e-12
        w = np.where(failed, 0.0, (w - rate) * (1.0 + scen[:, t]))
    return w, failed
