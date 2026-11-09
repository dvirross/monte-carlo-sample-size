# How many simulations are enough?

Reproducible code, data and figures for the article
[How Many Simulations Are Enough?](https://dvirross.com/quant-research/notes/how-many-simulations-are-enough) by Dvir Ross
(Hebrew version: [כמה סימולציות מספיקות?](https://dvirross.com/he/quant-research/notes/how-many-simulations-are-enough)).

The article starts from a rule of thumb (for an error below 1/x, run at least x² simulations) and asks when it is
right, when it wastes runs, and when it fails. Every number and figure in the article comes from this repository.

## What is here

| Path | Content |
|---|---|
| `mcsize/stats.py` | Confidence intervals (mean, Wald, Wilson, quantile, Expected Shortfall) and run-count formulas |
| `mcsize/withdrawal.py` | The main example: does a 4% withdrawal plan last 30 years? Four ways to generate return scenarios |
| `mcsize/leverage.py` | The rare event: a daily-reset 3× fund losing 90% in a year; plain simulation and three variance-reduction estimators |
| `mcsize/rules.py` | Four stopping rules and the coverage experiment |
| `mcsize/data.py` | Real annual returns from Robert Shiller's monthly data |
| `scripts/prepare_data.py` | Rebuilds `data/annual_real_returns.csv` from the raw file (checks its SHA-256) |
| `scripts/run_experiments.py` | Runs all experiments (about 3 minutes on 4 cores) and writes `results/` |
| `scripts/make_figures.py` | Builds the interactive Plotly figures (`figures/*.json`) |
| `scripts/render_png.mjs` | Renders the static PNG fallbacks with plotly.js in headless Chromium |
| `notebooks/walkthrough.ipynb` | A short tour: the rule of thumb, one simulation, its interval |
| `tests/` | 17 unit tests (`python -m pytest`) |
| `METHODOLOGY.md` | Data construction, models, estimators, stopping rules and limitations |

## Reproduce

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/run_experiments.py
python scripts/make_figures.py
python -m pytest
# optional PNGs (needs Node, Playwright and a plotly.js bundle):
node scripts/render_png.mjs path/to/plotly.min.js
```

All randomness uses NumPy's PCG64 generator with fixed seeds derived from 20250627 (the date of the original post),
so a rerun reproduces the published numbers exactly on the same library versions.

## Data

- `data/raw/shiller_sp500_monthly.csv`: monthly S&P Composite price, dividend, CPI and 10-year Treasury yield,
  January 1871 to June 2023, prepared by Robert Shiller, from the public-domain mirror
  [datasets/s-and-p-500](https://github.com/datasets/s-and-p-500) (ODC-PDDL-1.0) at commit
  `87a2284276f79d75041c71c8f6d9f2aa1c965e84`. Please credit Robert Shiller when you reuse it.
- `data/annual_real_returns.csv`: derived real (inflation-adjusted) annual returns of US stocks and 10-year bonds,
  1871 to 2022 (see `METHODOLOGY.md`).
- `data/qqq_daily_returns.csv`: 1,252 daily simple returns of QQQ (Interactive Brokers closes, 11 October 2021 to
  7 October 2026, not adjusted for distributions), from
  [dvirross/leveraged-etf-compounding](https://github.com/dvirross/leveraged-etf-compounding). Used only for the
  model-error comparison of the leveraged fund.

## Not investment advice

The withdrawal plan and the leveraged fund are statistical examples about how many simulation runs a question needs.
Nothing here is a recommendation about retirement plans, withdrawal rates or leveraged products.

## License

Code: MIT (see `LICENSE`). Data files keep the terms of their sources listed above.
