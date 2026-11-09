"""Run every experiment of the article and write the results to results/.

Usage: python scripts/run_experiments.py            (about 5 minutes on 4 CPU cores)
All random numbers come from numpy's PCG64 generator with the fixed seeds below, so a rerun gives the same numbers.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcsize import leverage as lev  # noqa: E402
from mcsize import rules, stats  # noqa: E402
from mcsize import withdrawal as wd  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 20250627  # the date of the LinkedIn post
summary: dict = {}


def rng(*key: int) -> np.random.Generator:
    return np.random.default_rng([SEED, *key])


def fail_prob(n: int, g: np.random.Generator, rate: float = 0.04, method: str = "iid", chunk: int = 200_000):
    """Number of failed plans out of n simulated retirements, computed in chunks to bound memory."""
    k = 0
    done = 0
    while done < n:
        m = min(chunk, n - done)
        _, f = wd.evaluate(wd.draw_scenarios(m, g, method), rate)
        k += int(f.sum())
        done += m
    return k


# ---- reference values -------------------------------------------------------------------------------------------------
t0 = time.time()
REF_N = 4_000_000
ref = {}
for i, rate in enumerate([0.03, 0.04, 0.0425, 0.045]):
    k = fail_prob(REF_N, rng(1, i), rate)
    ref[rate] = k / REF_N
summary["withdrawal_reference"] = {"n": REF_N, "p_fail": {str(r): v for r, v in ref.items()}}
p4 = ref[0.04]

# reference for terminal-wealth estimands (4% plan), from 2,000,000 retirements
w_ref, f_ref = wd.evaluate(wd.draw_scenarios(2_000_000, rng(2), "iid"), 0.04)
q10_ref = float(np.quantile(w_ref, 0.10))
es10_ref, _ = stats.expected_shortfall(w_ref, 0.10)
summary["terminal_reference"] = {"mean": float(w_ref.mean()), "sd": float(w_ref.std()), "q10": q10_ref,
                                 "es10": es10_ref, "median": float(np.median(w_ref))}
print("references", round(time.time() - t0), "s", summary["withdrawal_reference"], summary["terminal_reference"])

# ---- E1: convergence of the failure probability -----------------------------------------------------------------------
grid = np.unique(np.round(np.logspace(1, 5, 120)).astype(int))
traces = {}
for s in range(5):
    _, f = wd.evaluate(wd.draw_scenarios(100_000, rng(3, s), "iid"), 0.04)
    run = np.cumsum(f) / np.arange(1, len(f) + 1)
    traces[f"seed_{s + 1}"] = run[grid - 1]
e1 = pd.DataFrame({"n": grid, **traces})
z = stats.zval()
e1["funnel_half"] = z * np.sqrt(p4 * (1 - p4) / grid)
e1["reference"] = p4
e1.to_csv(RES / "e1_convergence.csv", index=False, float_format="%.8g")
# the post's 1,000-run example and the x^2 rule at this p
summary["e1"] = {"p_at_1000_by_seed": [float(traces[k][np.searchsorted(grid, 1000)]) for k in traces],
                 "half_width_1000": float(z * np.sqrt(p4 * (1 - p4) / 1000)),
                 "half_width_10000": float(z * np.sqrt(p4 * (1 - p4) / 10000))}
print("E1 done", round(time.time() - t0))

# ---- E2: the estimate is itself random -------------------------------------------------------------------------------
REPS2 = 2000
rows = []
for j, n in enumerate([100, 1_000, 10_000]):
    g = rng(4, j)
    for r in range(REPS2):
        _, f = wd.evaluate(wd.draw_scenarios(n, g, "iid"), 0.04)
        rows.append({"n": n, "rep": r, "estimate": f.mean()})
e2 = pd.DataFrame(rows)
e2.to_csv(RES / "e2_sampling_distribution.csv", index=False, float_format="%.8g")
summary["e2"] = {str(n): {"sd": float(g_["estimate"].std()), "q025": float(g_["estimate"].quantile(0.025)),
                          "q975": float(g_["estimate"].quantile(0.975)), "share_zero": float((g_["estimate"] == 0).mean())}
                 for n, g_ in e2.groupby("n")}
print("E2 done", round(time.time() - t0))

# ---- E3: error versus N for different questions -----------------------------------------------------------------------
REPS3 = 200
ns = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
truth = {"mean": summary["terminal_reference"]["mean"], "p_fail_4": p4, "p_fail_3": ref[0.03], "q10": q10_ref,
         "es10": es10_ref}
rows = []
for j, n in enumerate(ns):
    g = rng(5, j)
    for r in range(REPS3):
        scen = wd.draw_scenarios(n, g, "iid")
        w4, f4 = wd.evaluate(scen, 0.04)
        _, f3 = wd.evaluate(scen, 0.03)
        rows.append({"n": n, "rep": r, "mean": w4.mean(), "p_fail_4": f4.mean(), "p_fail_3": f3.mean(),
                     "q10": np.quantile(w4, 0.10), "es10": stats.expected_shortfall(w4, 0.10)[0]})
e3raw = pd.DataFrame(rows)
e3raw.to_csv(RES / "e3_replicates.csv", index=False, float_format="%.8g")
e3 = []
for n, g_ in e3raw.groupby("n"):
    row = {"n": n}
    for k, tv in truth.items():
        row[f"{k}_rel95"] = float(z * g_[k].std(ddof=1) / tv)  # relative 95% half-width
        row[f"{k}_bias"] = float(g_[k].mean() / tv - 1)
    e3.append(row)
e3 = pd.DataFrame(e3)
e3.to_csv(RES / "e3_error_vs_n.csv", index=False, float_format="%.6g")
summary["e3_truth"] = truth
print("E3 done", round(time.time() - t0))

# ---- E4: runs needed versus rarity (formula) -------------------------------------------------------------------------
ps = np.logspace(-4, np.log10(0.5), 80)
e4 = pd.DataFrame({"p": ps, **{f"rel_{int(r * 100)}": [stats.n_for_prob_rel(p, r) for p in ps] for r in (0.05, 0.1, 0.2)},
                   "rule_abs_001": 10_000})
e4.to_csv(RES / "e4_required_n.csv", index=False, float_format="%.6g")

# ---- E5: stopping rules ----------------------------------------------------------------------------------------------
rows = []
for i, p in enumerate([0.5, 0.1, 0.01, 0.001]):
    for j, rule in enumerate(["fixed", "pilot", "naive", "inverse"]):
        res = rules.run_rule(rule, p, 4000, rng(6, i, j))
        rows.append({"p": p, "rule": rule, **rules.summarize(res, p)})
e5 = pd.DataFrame(rows)
e5.to_csv(RES / "e5_stopping_rules.csv", index=False, float_format="%.6g")
print("E5 done", round(time.time() - t0))

# ---- E6: model error versus simulation error -------------------------------------------------------------------------
rows = []
for i, method in enumerate(["iid", "block", "lognormal"]):
    for rate in (0.035, 0.04, 0.045):
        for n in (1_000, 10_000, 1_000_000):
            k = fail_prob(n, rng(7, i, int(rate * 1e4), n), rate, method)
            c, h = stats.wilson(k, n)
            rows.append({"model": method, "rate": rate, "n": n, "p": k / n, "lo": c - h, "hi": c + h})
hist = wd.draw_scenarios(0, rng(0), "historical")
for rate in (0.035, 0.04, 0.045):
    _, f = wd.evaluate(hist, rate)
    k, n = int(f.sum()), len(f)
    c, h = stats.wilson(k, n)
    rows.append({"model": "historical", "rate": rate, "n": n, "p": k / n, "lo": c - h, "hi": c + h})
e6 = pd.DataFrame(rows)
e6.to_csv(RES / "e6_model_error.csv", index=False, float_format="%.6g")
summary["historical_windows"] = {"n": int(len(hist)), "first_start": 1871, "last_start": 1871 + len(hist) - 1,
                                 "failures_4pct": int(wd.evaluate(hist, 0.04)[1].sum())}
print("E6 done", round(time.time() - t0))

# ---- E7: cost versus accuracy on the rare leveraged-fund event -------------------------------------------------------
LEVEL = 0.1
ref_vals = np.concatenate([lev.estimator_values("importance", 500_000, rng(8, i), level=LEVEL) for i in range(20)])
p_lev = float(ref_vals.mean())
summary["leverage_rare"] = {"level": LEVEL, "p_ref": p_lev, "p_ref_half": float(z * ref_vals.std(ddof=1) / np.sqrt(len(ref_vals))),
                            "n_ref_importance": int(len(ref_vals)), "L": lev.L, "mu": lev.MU, "sigma": lev.SIGMA}
rows, budget_rows = [], []
for i, m in enumerate(["plain", "antithetic", "control", "importance"]):
    vals, secs = [], 0.0
    for b in range(10):
        t = time.perf_counter()
        v = lev.estimator_values(m, 100_000, rng(9, i, b), level=LEVEL)
        secs += time.perf_counter() - t
        vals.append(v)
    v = np.concatenate(vals)
    paths_per_value = 2 if m == "antithetic" else 1
    rows.append({"method": m, "estimate": float(v.mean()), "var_per_value": float(v.var(ddof=1)),
                 "sec_per_value": secs / len(v), "paths_per_value": paths_per_value})
    # empirical points: independent runs at several budgets
    for j, n in enumerate([1_000, 3_000, 10_000, 30_000, 100_000, 300_000]):
        for r in range(5):
            t = time.perf_counter()
            vv = lev.estimator_values(m, n, rng(10, i, j, r), level=LEVEL)
            dt = time.perf_counter() - t
            budget_rows.append({"method": m, "n_values": n, "seconds": dt, "estimate": float(vv.mean()),
                                "rel_err": float(abs(vv.mean() / p_lev - 1))})
e7 = pd.DataFrame(rows)
base = e7.loc[e7.method == "plain"].iloc[0]
e7["variance_ratio"] = base.var_per_value / e7.var_per_value
e7["efficiency"] = base.var_per_value * base.sec_per_value / (e7.var_per_value * e7.sec_per_value)
e7["values_for_10pct"] = np.ceil((z / 0.1) ** 2 * e7.var_per_value / p_lev ** 2)
e7["seconds_for_10pct"] = e7.values_for_10pct * e7.sec_per_value
e7.to_csv(RES / "e7_variance_reduction.csv", index=False, float_format="%.6g")
pd.DataFrame(budget_rows).to_csv(RES / "e7_budget_runs.csv", index=False, float_format="%.6g")
print("E7 done", round(time.time() - t0))

# ---- E8: comparing two close plans, independently or on the same scenarios -------------------------------------------
ra, rb = 0.04, 0.0425
pa, pb = ref[ra], ref[rb]
scen = wd.draw_scenarios(1_000_000, rng(11), "iid")
_, fa = wd.evaluate(scen, ra)
_, fb = wd.evaluate(scen, rb)
paired_var = float((fb.astype(float) - fa).var(ddof=1))
indep_var = pa * (1 - pa) + pb * (1 - pb)
ns8 = np.unique(np.round(np.logspace(2, 6, 60)).astype(int))
e8 = pd.DataFrame({"n": ns8, "half_independent": z * np.sqrt(indep_var / ns8), "half_same": z * np.sqrt(paired_var / ns8)})
e8.to_csv(RES / "e8_close_plans.csv", index=False, float_format="%.6g")
# one realized comparison at N = 1,000 and N = 10,000 each way
real = []
for j, n in enumerate([1_000, 10_000]):
    g = rng(12, j)
    s1, s2 = wd.draw_scenarios(n, g, "iid"), wd.draw_scenarios(n, g, "iid")
    da = wd.evaluate(s2, rb)[1].mean() - wd.evaluate(s1, ra)[1].mean()
    _, f1 = wd.evaluate(s1, ra)
    _, f2 = wd.evaluate(s1, rb)
    dp = f2.mean() - f1.mean()
    real.append({"n": n, "diff_independent": float(da), "half_independent": float(z * np.sqrt(indep_var / n)),
                 "diff_same": float(dp), "half_same": float(z * np.sqrt(paired_var / n))})
summary["e8"] = {"rate_a": ra, "rate_b": rb, "p_a": pa, "p_b": pb, "indep_var": indep_var, "paired_var": paired_var,
                 "ratio": indep_var / paired_var,
                 "n_for_half_1pp_independent": int(np.ceil((z / 0.01) ** 2 * indep_var)),
                 "n_for_half_1pp_same": int(np.ceil((z / 0.01) ** 2 * paired_var)), "realized": real}

# ---- the post's example ---------------------------------------------------------------------------------------------
summary["post_example"] = {
    "counts": [497, 157, 346], "n": 1000,
    "half_widths": [stats.wald(k, 1000)[1] for k in (497, 157, 346)],
    "rule_worst_case_half_at_x2": float(z * 0.5),
    "n_close_035_034_half": stats.n_for_difference(0.35, 0.34, 0.01),
    "n_close_035_034_power80": stats.n_for_difference(0.35, 0.34, 0.01, power=0.8),
    "n_rare_0001_rel10": stats.n_for_prob_rel(0.001, 0.1),
    "p_zero_hits_1000_at_0001": 0.999 ** 1000,
}
summary["runtime_seconds"] = round(time.time() - t0)
(RES / "summary.json").write_text(json.dumps(summary, indent=2))
print("all done", summary["runtime_seconds"], "s")
