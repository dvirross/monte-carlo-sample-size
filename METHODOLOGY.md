# Methodology

## 1. Data

**Source.** Robert Shiller's monthly US series (S&P Composite price as a monthly average of daily closes, annualized
dividend, CPI, 10-year Treasury yield), January 1871 to June 2023, via the public-domain mirror
`datasets/s-and-p-500` (commit `87a2284`). Rows after June 2023 carry only the price and are dropped.

**Stocks.** Monthly gross return `(P[t+1] + D[t]/12) / P[t]`. Because `P` is a monthly average, these returns are
slightly smoothed compared with month-end prices; this matters little for annual returns and is a known property of
the Shiller series.

**Bonds.** A 10-year par bond is bought at yield `y[t]` (coupon `y[t]`, paid monthly) and sold a month later as a
bond with 119 months left, priced at `y[t+1]`, plus one month of coupon. This approximates a constant-maturity
10-year Treasury return.

**Annual real returns.** For calendar year `y`, the product of the 12 monthly gross returns January to December,
divided by `CPI[Jan y+1] / CPI[Jan y]`. Result: 152 years, 1871 to 2022. Stocks: mean 8.4%, standard deviation 17.9%;
bonds: mean 2.9%, standard deviation 9.0% (real, arithmetic).

## 2. Main example: a 30-year withdrawal plan

Start with 1 unit. At the start of each year withdraw `rate` (fixed in real terms), then apply the year's real return
of a portfolio rebalanced annually to 60% stocks and 40% bonds. The plan **fails** if a withdrawal cannot be paid in
full. Estimands: the probability of failure, the mean and 10th percentile of the final real balance, and the average
of the worst 10% of final balances (Expected Shortfall at 10%).

**Scenario generators** (the "models" of the model-error comparison):
1. `iid`: each year drawn uniformly with replacement from the 152 years;
2. `block`: circular block bootstrap with 10-year blocks (Künsch 1989; Politis and Romano 1992), keeping runs of
   consecutive years;
3. `lognormal`: `log(1 + R)` normal with the historical mean and standard deviation of the 60/40 portfolio;
4. `historical`: the 123 actual 30-year windows starting 1871 to 1993. They overlap, so they hold the information of
   only about five non-overlapping 30-year periods; their Wilson intervals, which assume independence, are therefore too
   narrow and are shown only for orientation.

**Reference values** come from 4,000,000 simulated retirements per rate (`iid`), for example 6.33% failure at 4%.

## 3. Rare event: a daily-reset 3× fund

The model of the site's volatility simulator: underlying daily simple return `mu/252 + sigma/sqrt(252) z`, `z`
standard normal, `L = 3`, `mu = 10%`, `sigma = 25%`, 252 days, no costs; the fund's daily factor is `max(1 + L r, 0)`.
Event: the fund ends the year below 10% of its start (loses 90% or more). Reference probability 0.1046% from 10,000,000
importance-sampling draws (95% half-width 0.0001 percentage points).

**Estimators** (all unbiased for `p`; their per-sample variances are compared at equal computing time):
- plain indicator;
- antithetic pairs ("mirror pairs"): the average of the indicator on `z` and on `-z`;
- control variate: the control is `1{sum z < c}` with `c` the shock sum at which the log-approximate fund ends at 10%;
  its probability is exact (`sum z ~ N(0, 252)`); the coefficient is estimated from the same run, which adds a bias
  of order `1/N`, negligible here;
- importance sampling: shocks drawn from `N(-theta, 1)` with `theta = -c / 252`, weighted by
  `exp(theta * sum z + 252 theta^2 / 2)`.

**Efficiency** = (plain variance × plain time per sample) / (method variance × method time per sample). Timings are
wall-clock seconds of NumPy code on a 4-core cloud container; ratios are more portable than absolute times.

## 4. Stopping rules (estimating a probability to ±10% with 95% confidence)

Counts are drawn from the binomial and negative binomial distributions, exactly equivalent to running Bernoulli trials
one by one. 4,000 repetitions per rule and probability (0.5, 0.1, 0.01, 0.001); coverage has a Monte Carlo standard
error of about 0.35 percentage points.

1. **Fixed**: 10,000 runs (the rule of thumb for an absolute error of 0.01); Wald interval.
2. **Pilot**: 1,000 pilot runs; if no event, stop and report 0; otherwise `N = z^2 (1-p) / (p r^2)` fresh runs; Wald.
3. **Naive sequential**: batches of 100; stop when the Wald half-width is at most 10% of the current estimate.
4. **Inverse sampling** (Haldane 1945): run until `K = ceil(1.96^2 / 0.1^2) = 385` events; estimate
   `(K - 1) / (n - 1)` (unbiased); Wilson interval.

## 5. Two close plans

Plans withdrawing 4% and 4.25% (failure 6.33% and 8.90%). With two separate simulations the per-run variance of the
difference is `pa(1-pa) + pb(1-pb)`; on the same scenarios (common random numbers) it is the measured variance of
`1{b fails} - 1{a fails}` over 1,000,000 shared scenarios. Every scenario on which the 4% plan fails also fails at 4.25%,
which is why sharing scenarios removes most of the noise.

## 6. Limitations

- Historical returns are one path of one country (the United States, a historical winner); bootstrapping them cannot
  create scenarios worse than history in kind, only in combination.
- The `iid` and `lognormal` generators ignore the dependence between years (for example mean reversion), which the
  block bootstrap only partly keeps. Section 2 shows how much these choices move the answer.
- No fees, taxes or flexible spending rules; the 60/40 portfolio is rebalanced without cost.
- The leveraged-fund model has normal daily returns with constant volatility; real returns have fat tails and
  volatility clusters, which make crashes more likely than this model says.
