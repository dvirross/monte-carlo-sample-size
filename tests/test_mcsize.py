import numpy as np
import pytest

from mcsize import leverage as lev
from mcsize import rules, stats
from mcsize import withdrawal as wd
from mcsize.data import annual_real_returns, load_annual


def test_rule_of_thumb_is_the_worst_case_95_percent_interval():
    # 1.96 * sqrt(1/4 / x^2) = 0.98 / x < 1 / x
    for x in (10, 100, 1000):
        n = stats.n_rule_of_thumb(1 / x)
        assert n == x * x
        assert stats.zval() * np.sqrt(0.25 / n) < 1 / x


def test_post_example_half_widths():
    assert stats.wald(157, 1000)[1] == pytest.approx(0.02255, abs=1e-5)
    assert stats.wald(497, 1000)[1] == pytest.approx(0.03099, abs=1e-5)


def test_required_runs():
    assert stats.n_for_prob_rel(0.001, 0.1) == 383_762
    assert stats.n_for_difference(0.35, 0.34, 0.01) == 26_503
    assert 54_000 < stats.n_for_difference(0.35, 0.34, 0.01, power=0.8) < 54_300


def test_wilson_with_zero_events_has_positive_upper_bound():
    c, h = stats.wilson(0, 1000)
    assert c - h == pytest.approx(0, abs=1e-12)
    assert 0.003 < c + h < 0.004  # close to the "rule of three", 3 / n


def test_quantile_interval_brackets_estimate():
    x = np.random.default_rng(0).standard_normal(10_000)
    est, lo, hi = stats.quantile_ci(x, 0.05)
    assert lo <= est <= hi
    assert abs(est + 1.645) < 0.08


def test_expected_shortfall_normal():
    x = np.random.default_rng(1).standard_normal(400_000)
    es, half = stats.expected_shortfall(x, 0.05)
    assert es == pytest.approx(-2.0627, abs=0.02)  # -phi(1.645) / 0.05
    assert half < 0.02


def test_annual_returns_cover_1871_to_2022():
    a = load_annual()
    assert a["year"].iloc[0] == 1871 and a["year"].iloc[-1] == 2022 and len(a) == 152
    rebuilt = annual_real_returns()
    assert np.allclose(rebuilt["stock_real"], a["stock_real"], atol=1e-9)


def test_withdrawal_evaluate_by_hand():
    scen = np.array([[0.0] * 30, [0.10] * 30, [-0.5] + [0.0] * 29])
    w, failed = wd.evaluate(scen, 0.03)
    assert not failed[0] and w[0] == pytest.approx(1 - 30 * 0.03)
    w4, failed4 = wd.evaluate(scen[:1], 0.04)
    assert failed4[0] and w4[0] == 0.0  # 30 withdrawals of 4% need 120% of the start
    assert not failed[1] and w[1] > 1
    assert failed[2] and w[2] == 0.0


def test_common_random_numbers_nest_failures():
    scen = wd.draw_scenarios(20_000, np.random.default_rng(2))
    _, f4 = wd.evaluate(scen, 0.04)
    _, f45 = wd.evaluate(scen, 0.045)
    assert np.all(f45[f4])  # a plan that fails at 4% also fails at 4.5% on the same scenario


def test_block_bootstrap_keeps_runs_of_years():
    hist = np.arange(152, dtype=float)
    s = wd.draw_scenarios(50, np.random.default_rng(3), "block", hist=hist, block=10)
    steps = np.diff(s[:, :10], axis=1)
    assert np.all((steps == 1) | (steps == -151))


@pytest.mark.parametrize("method", ["plain", "antithetic", "control", "importance"])
def test_variance_reduction_estimators_agree(method):
    v = lev.estimator_values(method, 60_000, np.random.default_rng(4), level=0.2)
    assert v.mean() == pytest.approx(0.0153, rel=0.08)


def test_control_probability_is_exact_normal():
    z = np.random.default_rng(5).standard_normal((200_000, lev.DAYS))
    c = lev.control_threshold(0.2)
    assert (z.sum(axis=1) < c).mean() == pytest.approx(lev.control_prob(c), rel=0.05)


def test_inverse_sampling_rule_has_nominal_coverage():
    res = rules.run_rule("inverse", 0.001, 4000, np.random.default_rng(6))
    s = rules.summarize(res, 0.001)
    assert 0.935 < s["coverage"] < 0.965


def test_naive_sequential_rule_stops_at_zero_for_rare_events():
    res = rules.run_rule("naive", 0.001, 2000, np.random.default_rng(7))
    s = rules.summarize(res, 0.001)
    assert s["share_zero"] > 0.8 and s["coverage"] < 0.2


def test_within_target_includes_the_boundary_counts():
    # p = 0.001, n = 10,000: counts 9, 10 and 11 are all within 10% of the expected 10
    res = {"n": np.array([10_000] * 3), "est": np.array([9, 10, 11]) / 10_000, "lo": np.zeros(3), "hi": np.ones(3)}
    assert rules.summarize(res, 0.001)["within_target"] == 1.0
