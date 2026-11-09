"""Real annual returns from Robert Shiller's monthly US data, and real QQQ daily returns.

Shiller data (via the public-domain mirror github.com/datasets/s-and-p-500, ODC-PDDL-1.0) gives, for each month since
January 1871: the S&P Composite price (monthly average of daily closes), the annualized dividend, the CPI and the
10-year US government bond yield ("Long Interest Rate", percent). The full Shiller series in that mirror ends in June 2023.

Conventions (documented in METHODOLOGY.md):
- Stock total return for month t: (P[t+1] + D[t] / 12) / P[t].
- Bond total return for month t: buy a 10-year par bond at yield y[t] (coupon = y[t], paid monthly), sell it one month
  later as a bond with 119 months left at yield y[t+1], keeping the coupon. This is the standard way to turn a yield
  series into an approximate constant-maturity bond return.
- Calendar-year return: the product of the 12 monthly gross returns from January to December, deflated by the CPI
  ratio January of the next year over January of the year (so year y uses prices from Jan y to Jan y+1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_SHILLER = ROOT / "data" / "raw" / "shiller_sp500_monthly.csv"
ANNUAL_CSV = ROOT / "data" / "annual_real_returns.csv"
QQQ_CSV = ROOT / "data" / "qqq_daily_returns.csv"


def _bond_price(coupon: float, yld: float, months: int) -> float:
    """Price per 1 face value of a bond paying coupon/12 monthly for `months` months, at annual yield `yld`."""
    i = yld / 12.0
    c = coupon / 12.0
    if abs(i) < 1e-12:
        return c * months + 1.0
    disc = (1.0 + i) ** (-months)
    return c * (1.0 - disc) / i + disc


def load_monthly(path: Path = RAW_SHILLER) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"SP500": "price", "Dividend": "dividend", "Consumer Price Index": "cpi",
                            "Long Interest Rate": "gs10"})
    # Rows after the end of the Shiller series carry zeros in every column except the price.
    df = df[(df["cpi"] > 0) & (df["gs10"] > 0)].reset_index(drop=True)
    return df[["Date", "price", "dividend", "cpi", "gs10"]]


def annual_real_returns(monthly: pd.DataFrame | None = None) -> pd.DataFrame:
    m = load_monthly() if monthly is None else monthly
    p, d, y = m["price"].to_numpy(), m["dividend"].to_numpy(), m["gs10"].to_numpy() / 100.0
    stock = (p[1:] + d[:-1] / 12.0) / p[:-1]
    bond = np.array([_bond_price(y[t], y[t + 1], 119) + y[t] / 12.0 for t in range(len(y) - 1)])
    months = m.iloc[:-1].copy()
    months["stock_gross"], months["bond_gross"] = stock, bond
    months["year"] = months["Date"].dt.year
    rows = []
    for year, g in months.groupby("year"):
        if len(g) != 12:
            continue
        jan = m.index[m["Date"] == pd.Timestamp(year=year, month=1, day=1)]
        nxt = m.index[m["Date"] == pd.Timestamp(year=year + 1, month=1, day=1)]
        if len(jan) == 0 or len(nxt) == 0:
            continue
        infl = m.loc[nxt[0], "cpi"] / m.loc[jan[0], "cpi"]
        rows.append({"year": int(year),
                     "stock_real": g["stock_gross"].prod() / infl - 1.0,
                     "bond_real": g["bond_gross"].prod() / infl - 1.0,
                     "inflation": infl - 1.0})
    return pd.DataFrame(rows)


def load_annual(path: Path = ANNUAL_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


def load_qqq(path: Path = QQQ_CSV) -> np.ndarray:
    return pd.read_csv(path)["daily_simple_return"].to_numpy()
