"""Build the article's interactive figures (Plotly JSON) from results/.

Usage: python scripts/make_figures.py   ->  figures/*.json
Static PNG fallbacks are rendered from the same JSON by scripts/render_png.mjs (Playwright + plotly.js).

Design (one system for all eight figures):
- a headline that states the finding and a muted subtitle that says what is plotted;
- one categorical order, validated for colour-vision deficiency (dataviz validator, light surface):
  blue, gold, teal, purple, orange; context marks in grey; text in ink and muted ink, never in series colours;
- hairline grids one step off white, 2 px lines, >= 8 px markers with a white ring, bars with rounded ends;
- direct labels where they fit, a legend always present for two or more series;
- `narrow`: a phone layout (smaller headline, no direct labels that would collide, legend instead), read by the
  site's src/scripts/plotly-figures.ts when the figure is narrower than 560 px.
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

INK, MUTED, GRID, AXIS, CONTEXT = "#142033", "#5d6678", "#eceef3", "#c9ced8", "#b9bfca"
BLUE, GOLD, TEAL, PURPLE, ORANGE = "#3758d7", "#b8860b", "#0e9488", "#7a4fb3", "#c96f33"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
Z = stats.zval()


def headline(title: str, sub: str, size: int = 16) -> dict:
    return {"text": f"<b>{title}</b><br><span style='font-size:{size - 3}px;color:{MUTED}'>{sub}</span>",
            "x": 0, "xref": "paper", "xanchor": "left", "y": 1, "yref": "container", "yanchor": "top",
            "pad": {"t": 14}, "font": {"size": size, "color": INK}}


def axis(**kw) -> dict:
    return {"gridcolor": GRID, "gridwidth": 1, "linecolor": AXIS, "zeroline": False, "ticks": "", "showline": True,
            "tickfont": {"color": MUTED, "size": 12}, "title": {"font": {"color": INK, "size": 13}}, **kw}


def legend(y: float = 1.0) -> dict:
    return {"font": {"size": 12, "color": INK}, "orientation": "h", "x": 0, "xanchor": "left", "y": y, "yanchor": "bottom",
            "bgcolor": "rgba(255,255,255,0)", "itemwidth": 30}


def base(height: int, title: dict, **kw) -> dict:
    lay = {"font": {"family": FONT, "size": 13, "color": INK}, "title": title,
           "margin": {"l": 64, "r": 24, "t": 104, "b": 60}, "paper_bgcolor": "white", "plot_bgcolor": "white",
           "height": height, "hovermode": "closest", "legend": legend(),
           "hoverlabel": {"bgcolor": "white", "bordercolor": AXIS, "font": {"family": FONT, "color": INK, "size": 12}}}
    lay.update(kw)
    return lay


def note(x, y, text, **kw) -> dict:
    """A direct label or annotation in muted ink (never in the series colour)."""
    d = {"x": x, "y": y, "text": text, "showarrow": False, "font": {"size": 12, "color": kw.pop("color", MUTED)},
         "xanchor": "left", "yanchor": "middle", "align": "left"}
    d.update(kw)
    return d


def swatch_label(color: str, text: str) -> str:
    return f"<span style='color:{color}'>●</span> {text}"


def save(fig: go.Figure, name: str, narrow_layout: dict | None = None):
    spec = json.loads(fig.to_json())
    if narrow_layout is not None:
        narrow_layout = dict(narrow_layout)
        # on a phone the headline and subtitle already say what the y-axis measures, so the rotated y-axis title is
        # dropped (it collides with the tick labels at that width); the rest of each axis is kept as it is
        for key, ax in spec["layout"].items():
            if key.startswith("yaxis") and key not in narrow_layout:
                ax = json.loads(json.dumps(ax))
                ax["title"] = {"text": ""}
                narrow_layout[key] = ax
        spec["narrow"] = {"maxWidth": 560, "layout": narrow_layout}
    (FIG / f"{name}.json").write_text(json.dumps(spec, separators=(",", ":")))
    print("wrote", name)


def narrow_common(title: str, sub: str, **kw) -> dict:
    d = {"title": headline(title, sub, size=14), "margin": {"l": 52, "r": 12, "t": 128, "b": 56}, "annotations": [],
         "showlegend": True, "legend": legend()}
    d.update(kw)
    return d


pct = lambda v: 100 * np.asarray(v)  # noqa: E731
LOGN = dict(type="log", range=[0.97, 5.03], tickvals=[10, 100, 1000, 10000, 100000], ticktext=["10", "100", "1,000", "10,000", "100,000"])

# ---- F1 convergence --------------------------------------------------------------------------------------------------
e1 = pd.read_csv(RES / "e1_convergence.csv")
p = float(e1["reference"].iloc[0])
at1000 = S["e1"]["p_at_1000_by_seed"]
fig = go.Figure()
fig.add_trace(go.Scatter(x=e1.n, y=pct(p + e1.funnel_half), mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=e1.n, y=pct(np.maximum(p - e1.funnel_half, 0)), mode="lines", line={"width": 0}, fill="tonexty",
                         fillcolor="rgba(55,88,215,0.10)", name="Where 95% of estimates land", customdata=pct(e1.funnel_half),
                         hovertemplate="N = %{x:,}<br>95% of estimates within ±%{customdata:.2f} points<extra></extra>"))
highlight = "seed_4"
for i in range(1, 6):
    col = f"seed_{i}"
    hi = col == highlight
    fig.add_trace(go.Scatter(x=e1.n, y=pct(e1[col]), mode="lines",
                             line={"width": 2 if hi else 1.2, "color": BLUE if hi else CONTEXT},
                             name="One run, followed as runs are added" if hi else "Four other runs",
                             legendgroup="hi" if hi else "ctx", showlegend=hi or i == 1,
                             hovertemplate=f"Run {i}<br>after %{{x:,}} retirements: %{{y:.2f}}%<extra></extra>"))
fig.add_trace(go.Scatter(x=[10, 100_000], y=pct([p, p]), mode="lines", line={"color": INK, "width": 1.2, "dash": "dash"},
                         name="Long-run value", hoverinfo="skip"))
lo, hi_ = min(at1000), max(at1000)
fig.add_trace(go.Scatter(x=[1000, 1000], y=pct([lo, hi_]), mode="lines+markers", line={"color": INK, "width": 1.5},
                         marker={"size": 8, "color": INK, "line": {"color": "white", "width": 2}}, showlegend=False,
                         hovertemplate="After 1,000 runs the five estimates range from " + f"{100*lo:.1f}% to {100*hi_:.1f}%<extra></extra>"))
ann = [note(np.log10(1000), 100 * hi_ + 0.4, f"After 1,000 runs: {100*lo:.1f}% to {100*hi_:.1f}%", xanchor="left", yanchor="bottom",
            color=INK, xshift=6),
       note(np.log10(100000), 100 * p, f"{100*p:.2f}%", xanchor="left", xshift=6, color=INK)]
fig.update_layout(base(500, headline("Every run settles near 6.3%, but slowly",
                                     "Estimated chance that a 4% withdrawal plan runs out within 30 years, five independent simulations"),
                       margin={"l": 64, "r": 64, "t": 104, "b": 60}, annotations=ann,
                       xaxis=axis(title={"text": "Simulated retirements (log scale)"}, **LOGN),
                       yaxis=axis(title={"text": "Estimated chance of running out"}, ticksuffix="%", range=[0, 20])))
save(fig, "01-convergence", narrow_common("Every run settles near 6.3%, slowly", "Chance a 4% plan runs out, five simulations",
                                           annotations=[ann[0] | {"font": {"size": 11, "color": INK}, "xanchor": "center", "xshift": 0, "y": 100 * hi_ + 2.2,
                                                                  "text": f"1,000 runs: {100*lo:.1f}%–{100*hi_:.1f}%"}]))

# ---- F2 the estimate is random ---------------------------------------------------------------------------------------
e2 = pd.read_csv(RES / "e2_sampling_distribution.csv")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09)
ann = []
for i, n in enumerate([100, 1000, 10000]):
    v = pct(e2.loc[e2.n == n, "estimate"])
    q = S["e2"][str(n)]
    fig.add_trace(go.Histogram(x=v, xbins={"start": -0.25, "end": 20.25, "size": 0.5 if n < 10000 else 0.1}, histnorm="percent",
                               marker={"color": BLUE, "cornerradius": 3}, showlegend=False,
                               hovertemplate="estimate %{x}%<br>%{y:.1f}% of 2,000 runs<extra></extra>"), row=i + 1, col=1)
    fig.add_vline(x=100 * p, line={"color": INK, "dash": "dash", "width": 1.2}, row=i + 1, col=1)
    yref = "y domain" if i == 0 else f"y{i + 1} domain"
    xref = "x" if i == 0 else f"x{i + 1}"
    ann.append({"x": 0, "y": 1.02, "xref": "paper", "yref": yref, "showarrow": False, "xanchor": "left", "yanchor": "bottom",
                "text": f"<b>{n:,} retirements per run</b>  ·  middle 95%: {100*q['q025']:.1f}% to {100*q['q975']:.1f}%",
                "font": {"size": 12, "color": INK}})
fig.update_layout(base(600, headline("The estimate itself is random",
                                     "Each panel: the answers of 2,000 repeated simulations of the 4% plan; dashed line: the long-run value 6.33%"),
                       bargap=0.08, annotations=ann, margin={"l": 64, "r": 24, "t": 112, "b": 60}))
for r in (1, 2, 3):
    fig.update_yaxes(axis(ticksuffix="%", nticks=4), row=r, col=1)
    fig.update_xaxes(axis(ticksuffix="%", range=[0, 16]), row=r, col=1)
fig.update_xaxes(title={"text": "Estimated chance of running out"}, row=3, col=1)
fig.update_yaxes(title={"text": "Share of runs"}, row=2, col=1)
narrow_ann = [a | {"text": f"<b>{t}</b>", "font": {"size": 11, "color": INK}} for a, t in
              zip(ann, ["100 per run: 2.0% to 11.0%", "1,000 per run: 4.9% to 7.9%", "10,000 per run: 5.9% to 6.8%"])]
save(fig, "02-sampling-distribution", narrow_common("The estimate itself is random", "2,000 repeated simulations per panel",
                                                     annotations=narrow_ann, showlegend=False))

# ---- F3 error versus N for different questions -----------------------------------------------------------------------
e3 = pd.read_csv(RES / "e3_error_vs_n.csv")
T = S["e3_truth"]
series = [("mean", "Average final balance", BLUE), ("p_fail_4", "Chance of running out at 4%", GOLD),
          ("q10", "10th-percentile final balance", TEAL), ("p_fail_3", "Chance of running out at 3%", PURPLE),
          ("es10", "Average of the worst 10%", ORANGE)]
true_txt = {"mean": f"{T['mean']:.2f}", "p_fail_4": f"{100*T['p_fail_4']:.1f}%", "q10": f"{T['q10']:.2f}",
            "p_fail_3": f"{100*T['p_fail_3']:.2f}%", "es10": f"{T['es10']:.3f}"}
fig = go.Figure()
fig.add_trace(go.Scatter(x=[100, 100000], y=[10, 10], mode="lines", line={"color": INK, "width": 1, "dash": "dot"},
                         name="±10% of the true value", hoverinfo="skip"))
ann = []
cross = {}
ends = []
for key, name, c in series:
    y = pct(e3[f"{key}_rel95"])
    fig.add_trace(go.Scatter(x=e3.n, y=y, mode="lines+markers", name=name, line={"color": c, "width": 2},
                             marker={"size": 8, "color": c, "line": {"color": "white", "width": 2}},
                             hovertemplate=f"{name} (true value {true_txt[key]})<br>N = %{{x:,}}: ±%{{y:.1f}}% of the true value<extra></extra>"))
    ends.append([np.log10(y[-1]), name])
    cross[key] = 10000 * (e3.loc[e3.n == 10000, f"{key}_rel95"].iloc[0] / 0.1) ** 2
# spread the end labels to at least 0.085 decades apart (about one text line), keeping their order
ends.sort()
pos = [e[0] for e in ends]
for _ in range(50):
    for i in range(1, len(pos)):
        if pos[i] - pos[i - 1] < 0.085:
            mid = (pos[i] + pos[i - 1]) / 2
            pos[i - 1], pos[i] = mid - 0.0425, mid + 0.0425
for (yv, name), yp in zip(ends, pos):
    ann.append({"x": 5, "y": yv, "ax": 5.12, "ay": yp, "axref": "x", "ayref": "y", "showarrow": True, "arrowhead": 0,
                "arrowwidth": 1, "arrowcolor": AXIS, "standoff": 6, "text": name, "xanchor": "left", "font": {"size": 12, "color": INK}})
for key, txt in [("mean", "≈ 460 runs for ±10%"), ("es10", "≈ 85,000 runs for ±10%")]:
    fig.add_trace(go.Scatter(x=[cross[key]], y=[10], mode="markers", marker={"size": 9, "color": INK, "line": {"color": "white", "width": 2}},
                             showlegend=False, hovertemplate=f"±10% reached at about {cross[key]:,.0f} runs<extra></extra>"))
    ann.append({"x": np.log10(cross[key]), "y": 1, "ax": np.log10(cross[key]), "ay": np.log10(1.5), "axref": "x", "ayref": "y",
                "showarrow": True, "arrowhead": 0, "arrowwidth": 1, "arrowcolor": INK, "standoff": 6, "startstandoff": 0,
                "text": txt, "yanchor": "top", "xanchor": "center", "font": {"size": 12, "color": INK}})
fig.update_layout(base(540, headline("Same simulation, five questions: same slope, very different heights",
                                     "95% error of each estimate as a share of its true value, from 200 repeated simulations at each size"),
                       showlegend=True, annotations=ann, margin={"l": 64, "r": 210, "t": 112, "b": 60},
                       xaxis=axis(title={"text": "Simulated retirements (log scale)"}, type="log", range=[1.85, 5.06],
                                  tickvals=[100, 1000, 10000, 100000], ticktext=["100", "1,000", "10,000", "100,000"]),
                       yaxis=axis(type="log", range=[-0.35, 2.62], title={"text": "95% error, share of true value"},
                                  tickvals=[1, 3, 10, 30, 100, 300], ticktext=["1%", "3%", "10%", "30%", "100%", "300%"])))
save(fig, "03-error-by-question", narrow_common("Five questions, very different heights", "95% error as a share of the true value",
                                                 annotations=[a | {"font": {"size": 11, "color": INK}, "text": t, "xanchor": xa} for a, t, xa in
                                                              zip(ann[-2:], ["≈ 460", "≈ 85,000"], ["center", "right"])]))

# ---- F4 runs needed versus rarity ------------------------------------------------------------------------------------
e4 = pd.read_csv(RES / "e4_required_n.csv")
fig = go.Figure()
ann = []
for col, name, c in [("rel_5", "within ±5% of p", BLUE), ("rel_10", "within ±10% of p", GOLD), ("rel_20", "within ±20% of p", TEAL)]:
    fig.add_trace(go.Scatter(x=pct(e4.p), y=e4[col], mode="lines", name=name, line={"color": c, "width": 2},
                             hovertemplate="p = %{x:.2g}%<br>" + name + ": %{y:,.0f} runs<extra></extra>"))
    ann.append(note(np.log10(100 * e4.p.iloc[-1]), np.log10(e4[col].iloc[-1]), name.replace("within ", "").replace(" of p", ""),
                    xshift=8, color=INK))
fig.add_trace(go.Scatter(x=pct(e4.p), y=e4.rule_abs_001, mode="lines", line={"color": INK, "width": 1.2, "dash": "dash"},
                         name="Rule of thumb for ±0.01: 10,000 runs", hoverinfo="skip"))
ann.append(note(np.log10(50), 4, "rule of thumb, ±0.01", xanchor="right", yanchor="bottom", yshift=4, color=INK))
marks = [(S["withdrawal_reference"]["p_fail"]["0.04"], "4% plan"), (S["withdrawal_reference"]["p_fail"]["0.03"], "3% plan"),
         (S["leverage_rare"]["p_ref"], "3× fund ends year down 90%")]
fig.add_trace(go.Scatter(x=[100 * m for m, _ in marks], y=[stats.n_for_prob_rel(m, 0.1) for m, _ in marks], mode="markers",
                         marker={"color": INK, "size": 9, "line": {"color": "white", "width": 2}}, showlegend=False,
                         customdata=[t for _, t in marks], hovertemplate="%{customdata}: p = %{x:.2f}%<br>±10% needs %{y:,.0f} runs<extra></extra>"))
for m, t in marks:
    up = t != "4% plan"  # the 4% plan sits just under the rule-of-thumb line, so its label goes below the curve
    ann.append(note(np.log10(100 * m), np.log10(stats.n_for_prob_rel(m, 0.1)), t, xshift=-8 if not up else 8,
                    xanchor="left" if up else "right", yshift=8 if up else -8, yanchor="bottom" if up else "top", color=INK))
fig.update_layout(base(520, headline("Rare events need many more runs",
                                     "Runs needed for a 95% interval within a given share of the probability p"),
                       annotations=ann, showlegend=True, margin={"l": 64, "r": 70, "t": 104, "b": 60},
                       xaxis=axis(type="log", title={"text": "Probability of the event, p (log scale)"},
                                  tickvals=[0.01, 0.1, 1, 10, 50], ticktext=["0.01%", "0.1%", "1%", "10%", "50%"]),
                       yaxis=axis(type="log", title={"text": "Runs needed"}, tickvals=[100, 1e3, 1e4, 1e5, 1e6, 1e7],
                                  ticktext=["100", "1,000", "10,000", "100,000", "1M", "10M"])))
save(fig, "04-runs-needed", narrow_common("Rare events need many more runs", "Runs for a 95% interval within ±5%, ±10%, ±20% of p"))

# ---- F5 two close plans ----------------------------------------------------------------------------------------------
e8 = pd.read_csv(RES / "e8_close_plans.csv")
d8 = S["e8"]
diff = 100 * (d8["p_b"] - d8["p_a"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=e8.n, y=pct(e8.half_independent), mode="lines", line={"color": BLUE, "width": 2},
                         name="Two separate simulations", hovertemplate="N = %{x:,} each: ±%{y:.2f} points<extra></extra>"))
fig.add_trace(go.Scatter(x=e8.n, y=pct(e8.half_same), mode="lines", line={"color": GOLD, "width": 2},
                         name="Both plans on the same scenarios", hovertemplate="N = %{x:,}: ±%{y:.2f} points<extra></extra>"))
fig.add_trace(go.Scatter(x=[100, 1e6], y=[1, 1], mode="lines", line={"color": INK, "width": 1, "dash": "dot"},
                         name="±1 point target", hoverinfo="skip"))
n_ind, n_same = d8["n_for_half_1pp_independent"], d8["n_for_half_1pp_same"]
fig.add_trace(go.Scatter(x=[n_same, n_ind], y=[1, 1], mode="markers", marker={"size": 9, "color": INK, "line": {"color": "white", "width": 2}},
                         showlegend=False, customdata=[n_same, n_ind], hovertemplate="±1 point at %{customdata:,} runs per plan<extra></extra>"))
ann = [note(np.log10(n_same), 0, f"{n_same:,} runs", xanchor="center", yanchor="top", yshift=-10, color=INK),
       note(np.log10(n_ind), 0, f"{n_ind:,} runs", xanchor="center", yanchor="top", yshift=-10, color=INK),
       note(np.log10(1e6), np.log10(pct(e8.half_independent.iloc[-1])), "separate", xshift=8, color=INK),
       note(np.log10(1e6), np.log10(pct(e8.half_same.iloc[-1])), "shared", xshift=8, color=INK)]
fig.update_layout(base(480, headline(f"Shared scenarios: the same precision with {d8['ratio']:.1f}× fewer runs",
                                     f"95% error of the difference between a 4% and a 4.25% plan (true difference {diff:.1f} points)"),
                       annotations=ann, margin={"l": 64, "r": 84, "t": 112, "b": 60},
                       xaxis=axis(type="log", title={"text": "Simulated retirements per plan (log scale)"},
                                  tickvals=[100, 1e3, 1e4, 1e5, 1e6], ticktext=["100", "1,000", "10,000", "100,000", "1M"]),
                       yaxis=axis(type="log", title={"text": "95% error of the difference (points)"},
                                  tickvals=[0.03, 0.1, 0.3, 1, 3, 10], ticktext=["0.03", "0.1", "0.3", "1", "3", "10"])))
save(fig, "05-close-plans", narrow_common(f"Shared scenarios: {d8['ratio']:.1f}× fewer runs", "Error of the difference, 4% vs 4.25% plan",
                                           annotations=ann[:2]))

# ---- F6 stopping rules -----------------------------------------------------------------------------------------------
e5 = pd.read_csv(RES / "e5_stopping_rules.csv")
labels = {"fixed": "Fixed 10,000 runs", "pilot": "Pilot run, then formula", "naive": "Stop when the interval looks narrow",
          "inverse": "Stop after 385 events"}
colors = {"fixed": BLUE, "pilot": GOLD, "naive": TEAL, "inverse": PURPLE}
cats = ["p = 50%", "10%", "1%", "0.1%"]
fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16)
for rule in ["fixed", "pilot", "naive", "inverse"]:
    d = e5[e5.rule == rule].sort_values("p", ascending=False)
    custom = np.c_[d.mean_n, pct(d.share_zero)]
    common = dict(x=cats, marker={"color": colors[rule], "cornerradius": 4}, customdata=custom, legendgroup=rule)
    fig.add_trace(go.Bar(y=pct(d.coverage), name=labels[rule], **common,
                         hovertemplate=labels[rule] + "<br>%{x}: interval contains the true value in %{y:.1f}% of runs"
                                       "<br>average runs %{customdata[0]:,.0f}; estimates equal to 0: %{customdata[1]:.0f}%<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(y=pct(d.within_target), name=labels[rule], showlegend=False, **common,
                         hovertemplate=labels[rule] + "<br>%{x}: %{y:.1f}% of estimates within ±10%<br>average runs %{customdata[0]:,.0f}<extra></extra>"),
                  row=2, col=1)
fig.add_hline(y=95, line={"color": INK, "dash": "dash", "width": 1}, row=1, col=1)
ann = [{"x": 0, "y": 1.04, "xref": "paper", "yref": "y domain", "showarrow": False, "xanchor": "left", "yanchor": "bottom",
        "text": "<b>Coverage</b>: how often the reported 95% interval contains the true value (dashed: the promised 95%)",
        "font": {"size": 12, "color": INK}},
       {"x": 0, "y": 1.04, "xref": "paper", "yref": "y2 domain", "showarrow": False, "xanchor": "left", "yanchor": "bottom",
        "text": "<b>Precision</b>: share of estimates within ±10% of the true value", "font": {"size": 12, "color": INK}}]
fig.update_layout(base(660, headline("Stopping when the interval looks narrow fails for rare events",
                                     "Four stopping rules, each repeated 4,000 times per probability; goal: ±10% with 95% confidence"),
                       barmode="group", bargap=0.42, bargroupgap=0.14, annotations=ann,
                       margin={"l": 56, "r": 24, "t": 150, "b": 56}, legend=legend(1.13)))
for r in (1, 2):
    fig.update_yaxes(axis(ticksuffix="%", range=[0, 104], tickvals=[0, 25, 50, 75, 100]), row=r, col=1)
    fig.update_xaxes(axis(showline=True), row=r, col=1)
save(fig, "06-stopping-rules", narrow_common("Narrow-interval stopping fails for rare events", "Four rules, 4,000 repetitions each",
                                              margin={"l": 44, "r": 8, "t": 96, "b": 150},
                                              legend=legend(-0.08) | {"yanchor": "top", "orientation": "v"},
                                              annotations=[a | {"text": t, "font": {"size": 11, "color": INK}} for a, t in
                                                           zip(ann, ["<b>Coverage</b> (dashed: 95%)", "<b>Within ±10%</b>"])]))

# ---- F7 cost versus accuracy -----------------------------------------------------------------------------------------
e7 = pd.read_csv(RES / "e7_variance_reduction.csv")
runs = pd.read_csv(RES / "e7_budget_runs.csv")
pl = S["leverage_rare"]["p_ref"]
vlabel = {"plain": "Plain simulation", "antithetic": "Mirror pairs", "control": "Control variate", "importance": "Importance sampling"}
vcolor = {"plain": BLUE, "antithetic": GOLD, "control": TEAL, "importance": PURPLE}
tgrid = np.logspace(-4, 2, 60)
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1e-4, 1e2], y=[10, 10], mode="lines", line={"color": INK, "width": 1, "dash": "dot"},
                         name="±10% target", hoverinfo="skip"))
ann = []
for _, r in e7.iterrows():
    m = r["method"]
    d = runs[runs.method == m]
    fig.add_trace(go.Scatter(x=d.seconds, y=100 * d.rel_err, mode="markers", legendgroup=m, showlegend=False,
                             marker={"color": vcolor[m], "size": 6, "opacity": 0.35},
                             hovertemplate=vlabel[m] + "<br>one run, %{x:.3g} s: off by %{y:.1f}%<extra></extra>"))
    hw = 100 * Z * np.sqrt(r.var_per_value * r.sec_per_value / tgrid) / pl
    fig.add_trace(go.Scatter(x=tgrid, y=hw, mode="lines", name=vlabel[m], line={"color": vcolor[m], "width": 2}, legendgroup=m,
                             hovertemplate=vlabel[m] + "<br>%{x:.3g} s of computing: ±%{y:.1f}%<extra></extra>"))
    t10 = r.seconds_for_10pct
    fig.add_trace(go.Scatter(x=[t10], y=[10], mode="markers", marker={"size": 9, "color": vcolor[m], "line": {"color": "white", "width": 2}},
                             legendgroup=m, showlegend=False, hovertemplate=vlabel[m] + f": ±10% after about {t10:.3g} s<extra></extra>"))
    lbl = f"{t10:.2g} s" if t10 >= 0.01 else f"{1000 * t10:.0f} ms"
    below = m == "antithetic"  # its crossing sits next to plain simulation's, so its label goes under the line
    ann.append(note(np.log10(t10), np.log10(10), lbl, xanchor="center", yanchor="top" if below else "bottom",
                    yshift=-8 if below else 8, color=INK))
fig.update_layout(base(520, headline("Variance reduction buys accuracy without more computing",
                                     "Chance that a 3× fund ends the year down 90% or more (0.105%): error against computing time; dots are single runs"),
                       annotations=ann,
                       xaxis=axis(type="log", title={"text": "Computing time, seconds (log scale)"}, range=[-4, 2],
                                  tickvals=[1e-4, 1e-3, 0.01, 0.1, 1, 10, 100], ticktext=["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"]),
                       yaxis=axis(type="log", title={"text": "Error, share of true probability"},
                                  tickvals=[0.3, 1, 3, 10, 30, 100, 300], ticktext=["0.3%", "1%", "3%", "10%", "30%", "100%", "300%"],
                                  range=[np.log10(0.2), np.log10(400)])))
save(fig, "07-cost-vs-accuracy", narrow_common("Variance reduction buys accuracy", "Error against computing time, rare 3× fund loss",
                                                annotations=[]))

# ---- F8 model error versus simulation error --------------------------------------------------------------------------
e6 = pd.read_csv(RES / "e6_model_error.csv")
mlabel = {"iid": "Years drawn independently", "block": "10-year blocks of history", "lognormal": "Normal model of log returns",
          "historical": "The 123 actual 30-year windows"}
mcolor = {"iid": BLUE, "block": GOLD, "lognormal": TEAL, "historical": PURPLE}
rates = [0.035, 0.04, 0.045]
fig = go.Figure()
for j, m in enumerate(["iid", "block", "lognormal", "historical"]):
    d = e6[(e6.model == m) & ((e6.n == 1_000_000) | (m == "historical"))].sort_values("rate")
    x = [rates.index(r) + (j - 1.5) * 0.16 for r in d.rate]
    fig.add_trace(go.Scatter(x=x, y=pct(d.p), mode="markers", name=mlabel[m],
                             marker={"color": mcolor[m], "size": 10, "line": {"color": "white", "width": 2}},
                             error_y={"type": "data", "symmetric": False, "array": pct(d.hi - d.p), "arrayminus": pct(d.p - d.lo),
                                      "color": mcolor[m], "thickness": 2, "width": 0},
                             customdata=np.c_[pct(d.lo), pct(d.hi), d.n],
                             hovertemplate=mlabel[m] + "<br>%{y:.2f}% (95% interval %{customdata[0]:.2f}% to %{customdata[1]:.2f}%)"
                                                       "<br>based on %{customdata[2]:,} retirements<extra></extra>"))
fig.update_layout(base(500, headline("More runs cannot fix the model",
                                     "Chance the money runs out within 30 years under four models; simulated models use a million runs (error ±0.05 points)"),
                       xaxis=axis(tickvals=[0, 1, 2], ticktext=["Withdraw 3.5% a year", "4%", "4.5%"], range=[-0.5, 2.5], showgrid=False),
                       yaxis=axis(title={"text": "Chance of running out"}, ticksuffix="%", rangemode="tozero")))
save(fig, "08-model-error", narrow_common("More runs cannot fix the model", "Chance of running out under four models"))
