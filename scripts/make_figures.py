"""Build the article's interactive figures (Plotly JSON) from results/.

Usage: python scripts/make_figures.py   ->  figures/*.json
Static PNG fallbacks are rendered from the same JSON by scripts/render_png.mjs (Playwright + plotly.js).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcsize import stats  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
S = json.loads((RES / "summary.json").read_text())

NAVY, ORANGE, TEAL, BLUE, PURPLE, GOLD, RED = "#142033", "#c96f33", "#1f8a8a", "#3758d7", "#7a4fb3", "#b8860b", "#c0392b"
GRID, LINE, MUTED = "#e3e6ee", "#b9bfcc", "#5d6678"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"


def axis(**kw):
    return {"gridcolor": GRID, "linecolor": LINE, "zeroline": False, **kw}


def base_layout(height: int, **kw):
    lay = {"font": {"family": FONT, "size": 13, "color": NAVY}, "margin": {"l": 64, "r": 16, "t": 16, "b": 56},
           "paper_bgcolor": "white", "plot_bgcolor": "white", "height": height, "hovermode": "closest",
           "legend": {"font": {"size": 11}, "orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"}}
    lay.update(kw)
    return lay


def save(fig: go.Figure, name: str):
    (FIG / f"{name}.json").write_text(fig.to_json())
    print("wrote", name)


pct = lambda v: 100 * np.asarray(v)  # noqa: E731

# ---- F1 convergence --------------------------------------------------------------------------------------------------
e1 = pd.read_csv(RES / "e1_convergence.csv")
p = e1["reference"].iloc[0]
fig = go.Figure()
fig.add_trace(go.Scatter(x=e1.n, y=pct(p + e1.funnel_half), mode="lines", line={"width": 0}, hoverinfo="skip",
                         showlegend=False))
fig.add_trace(go.Scatter(x=e1.n, y=pct(np.maximum(p - e1.funnel_half, 0)), mode="lines", line={"width": 0},
                         fill="tonexty", fillcolor="rgba(55,88,215,0.13)", name="Where 95% of estimates land",
                         hovertemplate="N = %{x:,}<br>95%% range: ±%{customdata:.2f} points<extra></extra>",
                         customdata=pct(e1.funnel_half)))
for i, (col, c) in enumerate(zip([f"seed_{k}" for k in range(1, 6)], [ORANGE, TEAL, PURPLE, GOLD, RED])):
    fig.add_trace(go.Scatter(x=e1.n, y=pct(e1[col]), mode="lines", line={"width": 1.6, "color": c},
                             name=f"Run {i + 1}", legendgroup="runs", showlegend=i == 0 or False,
                             hovertemplate="Run " + str(i + 1) + "<br>after %{x:,} retirements: %{y:.2f}%<extra></extra>"))
fig.data[2].name = "Five independent runs"
fig.add_trace(go.Scatter(x=[10, 100_000], y=pct([p, p]), mode="lines", line={"color": NAVY, "dash": "dash", "width": 1.5},
                         name=f"Long-run value ({100 * p:.1f}%)", hoverinfo="skip"))
fig.add_vline(x=1000, line={"color": LINE, "width": 1, "dash": "dot"})
fig.add_annotation(x=np.log10(1000), y=19, xref="x", yref="y", text="1,000 runs", showarrow=False,
                   font={"size": 11, "color": MUTED}, xanchor="left", xshift=4)
fig.update_layout(base_layout(470, xaxis=axis(type="log", title={"text": "Number of simulated retirements (log scale)"},
                                                tickvals=[10, 100, 1000, 10000, 100000],
                                                ticktext=["10", "100", "1,000", "10,000", "100,000"]),
                              yaxis=axis(title={"text": "Estimated chance the money runs out"}, ticksuffix="%",
                                         range=[0, 20])))
save(fig, "01-convergence")

# ---- F2 the estimate is random ---------------------------------------------------------------------------------------
e2 = pd.read_csv(RES / "e2_sampling_distribution.csv")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                    subplot_titles=[f"{n:,} retirements per run" for n in (100, 1000, 10000)])
for i, (n, c) in enumerate(zip([100, 1000, 10000], [ORANGE, TEAL, BLUE])):
    v = pct(e2.loc[e2.n == n, "estimate"])
    fig.add_trace(go.Histogram(x=v, xbins={"start": -0.25, "end": 20.25, "size": 0.5 if n < 10000 else 0.1},
                               histnorm="percent", marker={"color": c, "line": {"width": 0}}, showlegend=False,
                               hovertemplate="estimate %{x}%<br>%{y:.1f}% of 2,000 runs<extra></extra>"),
                  row=i + 1, col=1)
    fig.add_vline(x=100 * p, line={"color": NAVY, "dash": "dash", "width": 1.2}, row=i + 1, col=1)
for a in fig.layout.annotations:
    a.font = {"size": 12, "color": MUTED}
    a.x, a.xanchor = 0, "left"
fig.update_layout(base_layout(560, bargap=0.05, margin={"l": 64, "r": 16, "t": 28, "b": 56}))
for r in (1, 2, 3):
    fig.update_yaxes(axis(ticksuffix="%"), row=r, col=1)
    fig.update_xaxes(axis(ticksuffix="%", range=[0, 16]), row=r, col=1)
fig.update_xaxes(title={"text": "Estimated chance the money runs out, in each of 2,000 repeated runs"}, row=3, col=1)
fig.update_yaxes(title={"text": "Share of runs"}, row=2, col=1)
save(fig, "02-sampling-distribution")

# ---- F3 error versus N for different questions -----------------------------------------------------------------------
e3 = pd.read_csv(RES / "e3_error_vs_n.csv")
T = S["e3_truth"]
series = [("mean", f"Average final balance ({T['mean']:.2f})", NAVY),
          ("p_fail_4", f"Chance of running out at 4% ({100 * T['p_fail_4']:.1f}%)", BLUE),
          ("q10", f"10th-percentile final balance ({T['q10']:.2f})", TEAL),
          ("p_fail_3", f"Chance of running out at 3% ({100 * T['p_fail_3']:.2f}%)", ORANGE),
          ("es10", f"Average of the worst 10% ({T['es10']:.3f})", PURPLE)]
fig = go.Figure()
for key, name, c in series:
    fig.add_trace(go.Scatter(x=e3.n, y=pct(e3[f"{key}_rel95"]), mode="lines+markers", name=name,
                             line={"color": c, "width": 2}, marker={"size": 6},
                             hovertemplate=name + "<br>N = %{x:,}: ±%{y:.1f}% of the true value<extra></extra>"))
fig.add_trace(go.Scatter(x=[100, 100000], y=[10, 10], mode="lines", line={"color": LINE, "dash": "dot"},
                         name="±10% target", hoverinfo="skip"))
fig.update_layout(base_layout(520, legend={"font": {"size": 11}, "orientation": "v", "x": 1, "xanchor": "right", "y": 1,
                                           "bgcolor": "rgba(255,255,255,0.85)"},
                              xaxis=axis(type="log", title={"text": "Number of simulated retirements (log scale)"},
                                         tickvals=[100, 1000, 10000, 100000], ticktext=["100", "1,000", "10,000", "100,000"]),
                              yaxis=axis(type="log", title={"text": "95% error, as a share of the true value"},
                                         tickvals=[1, 3, 10, 30, 100, 300], ticktext=["1%", "3%", "10%", "30%", "100%", "300%"])))
save(fig, "03-error-by-question")

# ---- F4 runs needed versus rarity ------------------------------------------------------------------------------------
e4 = pd.read_csv(RES / "e4_required_n.csv")
fig = go.Figure()
for col, name, c in [("rel_5", "within ±5% of p", PURPLE), ("rel_10", "within ±10% of p", BLUE), ("rel_20", "within ±20% of p", TEAL)]:
    fig.add_trace(go.Scatter(x=pct(e4.p), y=e4[col], mode="lines", name=name, line={"color": c, "width": 2},
                             hovertemplate="p = %{x:.2g}%<br>" + name + ": %{y:,.0f} runs<extra></extra>"))
fig.add_trace(go.Scatter(x=pct(e4.p), y=e4.rule_abs_001, mode="lines", line={"color": ORANGE, "dash": "dash"},
                         name="Rule of thumb for ±0.01 (10,000 runs)", hoverinfo="skip"))
marks = [(S["withdrawal_reference"]["p_fail"]["0.04"], "4% plan"), (S["withdrawal_reference"]["p_fail"]["0.03"], "3% plan"),
         (S["leverage_rare"]["p_ref"], "3× fund loses 90%")]

fig.add_trace(go.Scatter(x=[100 * m for m, _ in marks], y=[stats.n_for_prob_rel(m, 0.1) for m, _ in marks], mode="markers+text",
                         text=[t for _, t in marks], textposition="top right", marker={"color": NAVY, "size": 8},
                         textfont={"size": 11, "color": NAVY}, showlegend=False,
                         hovertemplate="%{text}: p = %{x:.2f}%<br>±10% needs %{y:,.0f} runs<extra></extra>"))
fig.update_layout(base_layout(500, legend={"font": {"size": 11}, "orientation": "v", "x": 1, "xanchor": "right", "y": 1,
                                           "bgcolor": "rgba(255,255,255,0.85)"},
                              xaxis=axis(type="log", title={"text": "How likely the event is, p (log scale)"},
                                         tickvals=[0.01, 0.1, 1, 10, 50], ticktext=["0.01%", "0.1%", "1%", "10%", "50%"]),
                              yaxis=axis(type="log", title={"text": "Runs needed for 95% confidence"},
                                         tickvals=[100, 1e3, 1e4, 1e5, 1e6, 1e7],
                                         ticktext=["100", "1,000", "10,000", "100,000", "1 million", "10 million"])))
save(fig, "04-runs-needed")

# ---- F5 stopping rules -----------------------------------------------------------------------------------------------
e5 = pd.read_csv(RES / "e5_stopping_rules.csv")
labels = {"fixed": "Fixed 10,000 runs", "pilot": "Pilot run, then formula", "naive": "Stop when the interval looks narrow",
          "inverse": "Stop after 385 events"}
colors = {"fixed": ORANGE, "pilot": GOLD, "naive": RED, "inverse": TEAL}
cats = ["50%", "10%", "1%", "0.1%"]
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                    subplot_titles=["Coverage: how often the 95% interval contains the true value",
                                    "Share of estimates within ±10% of the true value"])
for rule in ["fixed", "pilot", "naive", "inverse"]:
    d = e5[e5.rule == rule].sort_values("p", ascending=False)
    custom = np.c_[d.mean_n, pct(d.share_zero)]
    fig.add_trace(go.Bar(x=cats, y=pct(d.coverage), name=labels[rule], marker={"color": colors[rule]}, customdata=custom,
                         legendgroup=rule,
                         hovertemplate=labels[rule] + "<br>p = %{x}: coverage %{y:.1f}%<br>average runs %{customdata[0]:,.0f}"
                                       "<br>estimates equal to 0: %{customdata[1]:.0f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=cats, y=pct(d.within_target), name=labels[rule], marker={"color": colors[rule]},
                         customdata=custom, legendgroup=rule, showlegend=False,
                         hovertemplate=labels[rule] + "<br>p = %{x}: %{y:.1f}% within ±10%<br>average runs %{customdata[0]:,.0f}"
                                       "<extra></extra>"), row=2, col=1)
fig.add_hline(y=95, line={"color": NAVY, "dash": "dash", "width": 1}, row=1, col=1)
for a in fig.layout.annotations:
    a.font = {"size": 12, "color": MUTED}
    a.x, a.xanchor = 0, "left"
fig.update_layout(base_layout(620, barmode="group", bargap=0.2, margin={"l": 64, "r": 16, "t": 80, "b": 56},
                              legend={"font": {"size": 11}, "orientation": "h", "x": 0, "y": 1.08, "yanchor": "bottom"}))
fig.update_yaxes(axis(ticksuffix="%", range=[0, 102]), row=1, col=1)
fig.update_yaxes(axis(ticksuffix="%", range=[0, 102]), row=2, col=1)
fig.update_xaxes(axis(), row=1, col=1)
fig.update_xaxes(axis(title={"text": "True probability of the event"}), row=2, col=1)
save(fig, "05-stopping-rules")

# ---- F6 model error versus simulation error --------------------------------------------------------------------------
e6 = pd.read_csv(RES / "e6_model_error.csv")
mlabel = {"iid": "Years drawn independently", "block": "10-year blocks of history", "lognormal": "Normal model of log returns",
          "historical": "The 123 actual 30-year windows"}
mcolor = {"iid": BLUE, "block": TEAL, "lognormal": PURPLE, "historical": ORANGE}
rates = [0.035, 0.04, 0.045]
fig = go.Figure()
for j, m in enumerate(["iid", "block", "lognormal", "historical"]):
    d = e6[(e6.model == m) & ((e6.n == 1_000_000) | (m == "historical"))].sort_values("rate")
    x = [rates.index(r) + (j - 1.5) * 0.14 for r in d.rate]
    fig.add_trace(go.Scatter(x=x, y=pct(d.p), mode="markers", name=mlabel[m], marker={"color": mcolor[m], "size": 9},
                             error_y={"type": "data", "symmetric": False, "array": pct(d.hi - d.p), "arrayminus": pct(d.p - d.lo),
                                      "color": mcolor[m], "thickness": 1.6, "width": 4},
                             customdata=np.c_[pct(d.lo), pct(d.hi), d.n],
                             hovertemplate=mlabel[m] + "<br>%{y:.2f}% (95% interval %{customdata[0]:.2f}% to %{customdata[1]:.2f}%)"
                                           "<br>based on %{customdata[2]:,} retirements<extra></extra>"))
fig.update_layout(base_layout(480, legend={"font": {"size": 11}, "orientation": "v", "x": 0, "y": 1, "bgcolor": "rgba(255,255,255,0.85)"},
                              xaxis=axis(tickvals=[0, 1, 2], ticktext=["Withdraw 3.5% a year", "4%", "4.5%"], range=[-0.5, 2.5]),
                              yaxis=axis(title={"text": "Chance the money runs out within 30 years"}, ticksuffix="%", rangemode="tozero")))
save(fig, "06-model-error")

# ---- F7 cost versus accuracy -----------------------------------------------------------------------------------------
e7 = pd.read_csv(RES / "e7_variance_reduction.csv")
runs = pd.read_csv(RES / "e7_budget_runs.csv")
pl = S["leverage_rare"]["p_ref"]
vlabel = {"plain": "Plain simulation", "antithetic": "Mirror pairs", "control": "Control variate", "importance": "Importance sampling"}
vcolor = {"plain": NAVY, "antithetic": GOLD, "control": TEAL, "importance": BLUE}
tgrid = np.logspace(-4, 2, 60)
fig = go.Figure()
z = 1.959963984540054
for _, r in e7.iterrows():
    m = r["method"]
    hw = 100 * z * np.sqrt(r.var_per_value * r.sec_per_value / tgrid) / pl
    fig.add_trace(go.Scatter(x=tgrid, y=hw, mode="lines", name=vlabel[m], line={"color": vcolor[m], "width": 2}, legendgroup=m,
                             hovertemplate=vlabel[m] + "<br>%{x:.3g} s of computing: ±%{y:.1f}%<extra></extra>"))
    d = runs[runs.method == m]
    fig.add_trace(go.Scatter(x=d.seconds, y=100 * d.rel_err, mode="markers", legendgroup=m, showlegend=False,
                             marker={"color": vcolor[m], "size": 5, "opacity": 0.55},
                             hovertemplate=vlabel[m] + "<br>one run, %{x:.3g} s: off by %{y:.1f}%<extra></extra>"))
fig.add_trace(go.Scatter(x=[1e-4, 1e2], y=[10, 10], mode="lines", line={"color": LINE, "dash": "dot"}, name="±10% target",
                         hoverinfo="skip"))
fig.update_layout(base_layout(500, legend={"font": {"size": 11}, "orientation": "v", "x": 1, "xanchor": "right", "y": 1,
                                           "bgcolor": "rgba(255,255,255,0.85)"},
                              xaxis=axis(type="log", title={"text": "Computing time in seconds (log scale)"}, range=[-4, 2],
                                         tickvals=[1e-4, 1e-3, 0.01, 0.1, 1, 10, 100],
                                         ticktext=["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"]),
                              yaxis=axis(type="log", title={"text": "Error, as a share of the true probability"},
                                         tickvals=[0.3, 1, 3, 10, 30, 100, 300], ticktext=["0.3%", "1%", "3%", "10%", "30%", "100%", "300%"],
                                         range=[np.log10(0.2), np.log10(400)])))
save(fig, "07-cost-vs-accuracy")

# ---- F8 two close plans ----------------------------------------------------------------------------------------------
e8 = pd.read_csv(RES / "e8_close_plans.csv")
d8 = S["e8"]
diff = 100 * (d8["p_b"] - d8["p_a"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=e8.n, y=pct(e8.half_independent), mode="lines", line={"color": ORANGE, "width": 2},
                         name="Two separate simulations", hovertemplate="N = %{x:,} each: ±%{y:.2f} points<extra></extra>"))
fig.add_trace(go.Scatter(x=e8.n, y=pct(e8.half_same), mode="lines", line={"color": TEAL, "width": 2},
                         name="Both plans on the same scenarios", hovertemplate="N = %{x:,}: ±%{y:.2f} points<extra></extra>"))
fig.add_trace(go.Scatter(x=[100, 1e6], y=[diff, diff], mode="lines", line={"color": NAVY, "dash": "dash", "width": 1.3},
                         name=f"True difference ({diff:.1f} points)", hoverinfo="skip"))
fig.update_layout(base_layout(460, legend={"font": {"size": 11}, "orientation": "v", "x": 1, "xanchor": "right", "y": 1,
                                           "bgcolor": "rgba(255,255,255,0.85)"},
                              xaxis=axis(type="log", title={"text": "Simulated retirements per plan (log scale)"},
                                         tickvals=[100, 1e3, 1e4, 1e5, 1e6], ticktext=["100", "1,000", "10,000", "100,000", "1 million"]),
                              yaxis=axis(type="log", title={"text": "95% error of the difference (percentage points)"},
                                         tickvals=[0.03, 0.1, 0.3, 1, 3, 10], ticktext=["0.03", "0.1", "0.3", "1", "3", "10"])))
save(fig, "08-close-plans")
