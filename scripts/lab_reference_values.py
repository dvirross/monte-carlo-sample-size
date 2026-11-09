"""Long-run reference values shown by the site's convergence lab (src/lib/mc-lab.ts in dvirross/dvirross.com).

Withdrawal plans 3% to 5% in steps of 0.25: failure probability from 2,000,000 simulated retirements each ("iid" years).
3x fund: probability of ending the year below 50%, 20% and 10% of its start, from 4,000,000 importance-sampling draws each,
and the control-variate coefficient b = Cov(Y, X) / Var(X) from 2,000,000 plain paths each. The lab uses this fixed b instead of
estimating it from its own runs: a coefficient estimated from the same few runs makes the estimator biased in small samples.
Usage: python scripts/lab_reference_values.py   ->  results/lab_references.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcsize import leverage as lev  # noqa: E402
from mcsize import withdrawal as wd  # noqa: E402

out = {"withdrawal": {}, "fund": {}, "control_b": {}}
for i, rate in enumerate(np.arange(3.0, 5.01, 0.25)):
    g = np.random.default_rng([20250627, 99, i])
    k = n = 0
    for _ in range(10):
        _, f = wd.evaluate(wd.draw_scenarios(200_000, g, "iid"), rate / 100)
        k += int(f.sum())
        n += len(f)
    out["withdrawal"][f"{rate:.2f}"] = k / n
for j, level in enumerate([0.5, 0.2, 0.1]):
    v = np.concatenate([lev.estimator_values("importance", 500_000, np.random.default_rng([20250627, 98, j, b]), level=level)
                        for b in range(8)])
    out["fund"][str(level)] = float(v.mean())
    c = lev.control_threshold(level)
    sy = sx = sxy = sxx = 0.0
    m = 0
    for b in range(10):
        z = lev.draw_z(200_000, np.random.default_rng([20250627, 97, j, b]))
        y = (lev.fund_terminal(z) < level).astype(float)
        x = (z.sum(axis=1) < c).astype(float)
        sy += y.sum(); sx += x.sum(); sxy += (x * y).sum(); sxx += (x * x).sum(); m += len(y)
    out["control_b"][str(level)] = float((sxy / m - sx / m * sy / m) / (sxx / m - (sx / m) ** 2))
(ROOT / "results" / "lab_references.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
