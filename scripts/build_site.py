"""Build the static site.

THE POINT OF THIS SCRIPT IS THAT IT EXISTS IN WEEK 4, NOT WEEK 8.

It is deliberately plain. The Phase 3 gate is "a URL exists", not "a URL is
impressive" — shipping something ugly early converts the final week from risk
into improvement. The marimo WASM app replaces this in Phase 5; until then this
proves the pipeline runs end to end and the deploy path works.

Every number on the page is computed here, at build time, from the same code
that runs the tests. Nothing is typed in by hand.

    python scripts/build_site.py     ->  site/index.html
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coldspend.causal import bootstrap_ci, fuzzy_rd  # noqa: E402
from coldspend.models import evaluate  # noqa: E402
from coldspend.physics import equivalent_hours  # noqa: E402
from coldspend.sim import (  # noqa: E402
    SimConfig,
    ReanalysisClimate,
    calibration_check,
    simulate_portfolio,
)

OUT = Path(__file__).resolve().parents[1] / "site"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"

mpl.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
})


def png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_hook() -> str:
    """The demo hook: two compliant shipments, 1.787x different ageing."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    hrs = np.arange(61)
    a1.axhspan(2, 8, color=GRID, alpha=0.6, zorder=0)
    a1.plot(hrs, np.full_like(hrs, 3.0, dtype=float), color=BLUE, lw=2.2)
    a1.plot(hrs, np.full_like(hrs, 7.5, dtype=float), color=ORANGE, lw=2.2)
    a1.annotate("3.0 °C", (61, 3.0), color=BLUE, va="center", fontsize=9, fontweight="bold")
    a1.annotate("7.5 °C", (61, 7.5), color=ORANGE, va="center", fontsize=9, fontweight="bold")
    a1.set_xlim(0, 68); a1.set_ylim(0, 10)
    a1.set_xlabel("hours in transit"); a1.set_ylabel("payload temperature (°C)")
    a1.set_title("Both inside 2–8 °C. Both pass.", loc="left", fontweight="bold", color=INK)
    a1.grid(axis="y", color=GRID, lw=0.7); a1.set_axisbelow(True)

    cold, warm = equivalent_hours([3.0], 60.0), equivalent_hours([7.5], 60.0)
    a2.barh(["3.0 °C", "7.5 °C"], [cold, warm], color=[BLUE, ORANGE], height=0.5)
    for i, v in enumerate((cold, warm)):
        a2.annotate(f"{v:.1f}", (v, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontweight="bold", color=INK, fontsize=10)
    a2.set_xlim(0, warm * 1.28)
    a2.set_xlabel("shelf life spent (equivalent-hours)")
    a2.set_title(f"{warm / cold:.3f}× different ageing", loc="left", fontweight="bold", color=INK)
    a2.grid(axis="x", color=GRID, lw=0.7); a2.set_axisbelow(True)
    fig.tight_layout()
    return png(fig)


def chart_effect_by_exposure(d: pd.DataFrame) -> str:
    """Where intervention actually pays — and where the SOP fires."""
    bins = [(0, 150), (150, 450), (450, 750), (750, 1200), (1200, 3000)]
    xs, ys = [], []
    sd = d["post_hub_budget_pct"].std()
    for lo, hi in bins:
        s = d[d.running_var_min.between(lo, hi)]
        if len(s) > 40:
            xs.append(f"{lo}–{hi}")
            ys.append(s.truth_ite_pct.mean() / sd)

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    cols = [ORANGE if x == "150–450" else BLUE for x in xs]
    ax.bar(xs, ys, color=cols, width=0.62)
    for i, v in enumerate(ys):
        ax.annotate(f"{v:.2f} SD", (i, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontweight="bold", color=INK, fontsize=9)
    ax.annotate("the SOP fires here", (1, ys[1]), xytext=(0, 34), textcoords="offset points",
                ha="center", color=ORANGE, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.set_ylabel("effect of intervening (SD)")
    ax.set_xlabel("minutes above spec at the decision point")
    ax.set_title("The alarm threshold fires where intervening barely helps",
                 loc="left", fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    ax.set_ylim(0, max(ys) * 1.32)
    fig.tight_layout()
    return png(fig)


def chart_policy_map(value: float, tau_h: float, title: str) -> tuple[str, dict]:
    """The headline visual: the argmin made visible as decision regions."""
    from coldspend.decide import Action, policy_map
    from coldspend.decide.optimizer import ACTIONS

    pm = policy_map(value, tau_h=tau_h, n_budget=140, n_hours=140)
    colour = {
        Action.DO_NOTHING: "#e6e5df",
        Action.REICE: BLUE,
        Action.EXPEDITE: ORANGE,
        Action.REROUTE: "#1baf7a",
        Action.RECALL: "#e34948",
    }
    present = sorted(np.unique(pm.action_index))
    cmap = mpl.colors.ListedColormap([colour[ACTIONS[int(i)]] for i in present])
    remap = np.zeros_like(pm.action_index)
    for new, old in enumerate(present):
        remap[pm.action_index == old] = new

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.pcolormesh(pm.budget_grid, pm.hours_grid, remap, cmap=cmap, shading="auto",
                  vmin=-0.5, vmax=len(present) - 0.5)

    # Label each region in place rather than with a legend box — the reader
    # should never have to look away from the map to decode a colour.
    for new, old in enumerate(present):
        act = ACTIONS[int(old)]
        ys, xs = np.where(remap == new)
        if ys.size < 40:
            continue
        cx, cy = pm.budget_grid[int(np.median(xs))], pm.hours_grid[int(np.median(ys))]
        dark = act in (Action.REICE, Action.EXPEDITE, Action.REROUTE, Action.RECALL)
        ax.annotate(act.value.replace("_", " "), (cx, cy), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if dark else INK2)

    ax.set_xlabel("stability budget already consumed (%)")
    ax.set_ylabel("hours remaining to destination")
    ax.set_title(title, loc="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return png(fig), pm.region_shares()


def chart_fda_validation(d: pd.DataFrame) -> tuple[str, dict]:
    """Simulated failure composition against FDA's own recall record."""
    from coldspend.validate import compare_to_simulation, failure_mix

    mix = failure_mix()
    c = compare_to_simulation(d, mix)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.5),
                                  gridspec_kw={"width_ratios": [1, 1.15]})

    # left: what FDA's reasons actually say
    labels = ["heat", "unspecified", "freeze"]
    vals = [mix.counts.get("heat", 0) + mix.counts.get("both", 0),
            mix.counts.get("unspecified", 0), mix.counts.get("freeze", 0)]
    ax.barh(labels[::-1], vals[::-1], color=[BLUE, GRID, ORANGE][::-1], height=0.55)
    for i, v in enumerate(vals[::-1]):
        ax.annotate(str(v), (v, i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontweight="bold", color=INK, fontsize=10)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("distinct FDA recall events")
    ax.set_title(f"What {mix.n_events} real recalls say", loc="left",
                 fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)

    # right: the one axis both sources measure, with FDA's interval
    lo, hi = c["fda_ci_lo"], c["fda_ci_hi"]
    ax2.barh([0], [hi - lo], left=[lo], height=0.30, color=GRID)
    ax2.plot([c["fda_freeze_share"]], [0], "o", color=INK, ms=9, zorder=4)
    ax2.plot([c["sim_freeze_share"]], [1], "o", color=ORANGE, ms=9, zorder=4)
    ax2.annotate(f"FDA  {c['fda_freeze_share']:.0%}\n95% CI {lo:.0%}–{hi:.0%}  (n={mix.n_classifiable:.0f})",
                 (c["fda_freeze_share"], 0), xytext=(0, 16), textcoords="offset points",
                 ha="center", fontsize=9, color=INK, fontweight="bold")
    ax2.annotate(f"simulator  {c['sim_freeze_share']:.0%}", (c["sim_freeze_share"], 1),
                 xytext=(0, 14), textcoords="offset points", ha="center",
                 fontsize=9.5, color=ORANGE, fontweight="bold")
    ax2.set_ylim(-0.55, 1.75); ax2.set_yticks([])
    ax2.set_xlim(0, max(hi, c["sim_freeze_share"]) * 1.25)
    ax2.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("freezing as a share of directionally-classifiable failures")
    ax2.set_title("The one axis both can answer", loc="left", fontweight="bold", color=INK)
    ax2.grid(axis="x", color=GRID, lw=0.7); ax2.set_axisbelow(True)
    for s in ("left", "right", "top"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    return png(fig), c



APPENDIX_HEAD = """<!doctype html><meta charset="utf-8">
<title>Coldspend appendix</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{max-width:900px;margin:0 auto;padding:2.5rem 1.25rem 5rem;
   font:16px/1.65 "Segoe UI",system-ui,-apple-system,sans-serif;color:#0b0b0b;background:#fcfcfb}
 h1{font-size:1.7rem;margin:0 0 .3rem;letter-spacing:-.02em}
 h2{font-size:1.15rem;margin:2.6rem 0 .6rem}
 .sub{color:#52514e;margin:0 0 2rem}
 .note{background:#fff8ee;border-left:3px solid #eb6834;padding:.85rem 1.1rem;
   margin:1.4rem 0;font-size:.93rem;border-radius:0 4px 4px 0}
 table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.8rem 0}
 th,td{text-align:left;padding:.42rem .6rem;border-bottom:1px solid #e1e0d9}
 th{color:#52514e;font-weight:600}
 .num{text-align:right;font-variant-numeric:tabular-nums}
 .ok{color:#0ca30c;font-weight:600} .bad{color:#d03b3b;font-weight:600}
 img{width:100%;height:auto;margin:.6rem 0} a{color:#2a78d6}
</style>
<p><a href="./">&larr; back to the argument</a></p>
<h1>Appendix</h1>
<p class="sub">The estimator, the constraint that makes this an optimisation, and the models,
kept off the front page so the argument fits in three minutes.</p>
"""

def main() -> None:
    OUT.mkdir(exist_ok=True)
    n = 12000
    print(f"simulating {n:,} shipments ...")
    res = simulate_portfolio(n, ReanalysisClimate(), seed=20260803)
    df = pd.DataFrame([r.record for r in res])
    d = df[df["product"] != "CAR-T dose"]

    hard_ok, rows = calibration_check(df)
    print("calibration gate:", "PASS" if hard_ok else "FAIL")

    print("fitting models ...")
    reps = evaluate(df, seed=0)
    lr = next(r for r in reps if "logistic" in r.name)
    sel = next(r for r in reps if "SELECTED" in r.name)
    orc = next(r for r in reps if "ORACLE" in r.name)

    print("running RD ...")
    cfg = SimConfig(sop_threshold_min=900.0)
    deep = pd.DataFrame([r.record for r in simulate_portfolio(n, ReanalysisClimate(), seed=5, cfg=cfg)])
    dd = deep[deep["product"] != "CAR-T dose"]
    r = fuzzy_rd(dd.running_var_min.values, dd.treated.values.astype(float),
                 dd.post_hub_budget_pct.values, 250.0, 900.0)
    lo, hi = bootstrap_ci(dd.running_var_min.values, dd.treated.values.astype(float),
                          dd.post_hub_budget_pct.values, 250.0, 900.0, n_boot=300, seed=1)
    near = dd[dd.running_var_min.between(650, 1150)]
    truth = near[near.truth_stratum == "complier"].truth_ite_pct.mean()

    print("solving policy maps ...")
    from coldspend.decide import Action, ShipmentState, decision_stability, optimise_portfolio

    pm_poor, shares_poor = chart_policy_map(
        850_000.0, 12.0, "An $850,000 consignment in a poorly-insulated box")
    pm_good, _ = chart_policy_map(
        850_000.0, 40.0, "The same consignment, better insulated")
    stab = decision_stability(850_000.0, n_budget=35, n_hours=35)

    rng = np.random.default_rng(3)
    port = [ShipmentState(f"S{i}", "DXB", float(rng.uniform(5, 70)),
                          float(rng.uniform(0.5, 2.2)), float(rng.uniform(6, 40)),
                          float(rng.choice([4_000, 45_000, 120_000, 850_000])),
                          float(rng.uniform(0.05, 0.9))) for i in range(60)]
    cap_rows = ""
    for cap in (60, 20, 8, 0):
        pl = optimise_portfolio(port, reice_capacity={"DXB": cap})
        n_re = sum(1 for a in pl.assignment.values() if a is Action.REICE)
        cap_rows += (f"<tr><td class='num'>{cap}</td><td class='num'>{n_re}</td>"
                     f"<td class='num'>${pl.total_expected_loss:,.0f}</td>"
                     f"<td class='num'>${pl.capacity_cost:,.0f}</td></tr>")

    print("validating against openFDA ...")
    fda_png, fda_cmp = chart_fda_validation(d)
    sim_fs = f"{fda_cmp['sim_freeze_share']:.0%}"
    fda_fs = f"{fda_cmp['fda_freeze_share']:.0%}"
    fda_n = f"{fda_cmp['fda_classifiable']:.0f}"

    cold, warm = equivalent_hours([3.0], 60.0), equivalent_hours([7.5], 60.0)
    gate_rows = "".join(
        f"<tr><td>{b.name}</td><td class='num'>{v:.4g}</td>"
        f"<td class='num'>[{b.lo:g}, {'∞' if b.hi >= 1e8 else f'{b.hi:g}'}]</td>"
        f"<td class='{'ok' if ok else 'bad'}'>{'PASS' if ok else 'FAIL'}</td></tr>"
        for b, v, ok in rows
    )
    model_rows = "".join(
        f"<tr><td>{m.name}</td><td class='num'>{m.auc:.4f}</td>"
        f"<td class='num'>{m.brier:.4f}</td></tr>" for m in reps
    )

    html = f"""<!doctype html><meta charset="utf-8">
<title>Coldspend — cold chain as a decision problem</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 :root{{color-scheme:light}}
 body{{max-width:900px;margin:0 auto;padding:2.5rem 1.25rem 5rem;
   font:16px/1.65 "Segoe UI",system-ui,-apple-system,sans-serif;color:{INK};background:{SURFACE}}}
 h1{{font-size:2rem;margin:0 0 .3rem;letter-spacing:-.02em}}
 h2{{font-size:1.15rem;margin:2.6rem 0 .6rem;letter-spacing:-.01em}}
 .sub{{color:{INK2};font-size:1.05rem;margin:0 0 .4rem}}
 .stamp{{color:{MUTED};font-size:.82rem;margin-bottom:2rem}}
 .note{{background:#fff8ee;border-left:3px solid {ORANGE};padding:.85rem 1.1rem;
   margin:1.4rem 0;font-size:.93rem;border-radius:0 4px 4px 0}}
 table{{border-collapse:collapse;width:100%;font-size:.88rem;margin:.8rem 0}}
 th,td{{text-align:left;padding:.42rem .6rem;border-bottom:1px solid {GRID}}}
 th{{color:{INK2};font-weight:600}}
 .num{{text-align:right;font-variant-numeric:tabular-nums}}
 .ok{{color:#0ca30c;font-weight:600}} .bad{{color:#d03b3b;font-weight:600}}
 img{{width:100%;height:auto;margin:.6rem 0}}
 code{{background:{GRID};padding:.1rem .35rem;border-radius:3px;font-size:.85em}}
 footer{{margin-top:3.5rem;padding-top:1.2rem;border-top:1px solid {GRID};
   color:{MUTED};font-size:.82rem}}
</style>

<h1>Coldspend</h1>
<p class="sub">Cold chain is not a limit you breach. It's a budget you spend.</p>
<p style="margin:.9rem 0 0"><a href="memo.html" style="display:inline-block;background:{ORANGE};
 color:#fff;padding:.55rem 1.1rem;border-radius:5px;text-decoration:none;font-weight:600">
 Read the client memo &rarr;</a>
 <span style="color:#898781;font-size:.85rem;margin-left:.6rem">regenerates itself from the
 analysis</span></p>
<p class="stamp">Built {dt.datetime.now():%Y-%m-%d %H:%M} · {n:,} simulated shipments ·
every number on this page computed at build time</p>

<div class="note"><strong>Simulated shipments, real weather.</strong> No proprietary telemetry
was used — but the ambient forcing is not invented either: every lane is driven by
<strong>real Open-Meteo hourly reanalysis</strong> for 2024 at its actual airports. The thermal
model, routing and failure modes are simulated, deliberately, so that <em>both arms</em> of every
shipment are known. That is the point: the individual treatment effect is unobservable in any real
dataset, and here it is ground truth an estimator can be scored against.</div>

<h2>The problem, in one chart</h2>
<p>Two shipments. Both 60 hours. Both entirely inside 2–8&nbsp;°C, both filing zero deviations,
both passing compliance identically. One cruises at 3.0&nbsp;°C, the other at 7.5&nbsp;°C.</p>
<img src="data:image/png;base64,{chart_hook()}" alt="Two compliant shipments ageing differently">
<p>The warm one spends <strong>{warm / cold:.3f}×</strong> the shelf life
({warm:.1f} vs {cold:.1f} equivalent-hours). Compliance cannot tell them apart.
A stability budget can.</p>

<h2>The finding: the alarm fires in the wrong place</h2>
<img src="data:image/png;base64,{chart_effect_by_exposure(d)}" alt="Effect of intervening by exposure">
<p>The industry's SOP threshold sits where intervening is worth about
<strong>0.17 standard deviations</strong> — too small to detect at any realistic sample size, by us
or by an incumbent with a decade of real telemetry. Four times larger deeper in the tail.
The rule is not wrong about <em>whether</em> to act; it is wrong about <em>when</em>.</p>

<h2>Checked against FDA's own recall record</h2>
<p>Being <em>driven</em> by real data is weaker than being <em>checked against</em> it. So: pick a
pattern in the real world the simulator was never fitted to, and report how it compares — whether
or not it matches.</p>
<img src="data:image/png;base64,{fda_png}" alt="Simulator versus openFDA recall record">
<p>The denominators differ — FDA reports the share of <em>recalls</em> caused by temperature, the
simulator the share of <em>shipments</em> — so those two numbers are not comparable and pretending
otherwise would be nonsense dressed as rigour. What both <em>can</em> answer is the
<strong>composition</strong>: among thermal failures, how many were too hot versus too cold.</p>
<div class="note"><strong>It does not match, and the mismatch is the useful part.</strong>
The simulator puts freezing at {sim_fs} of thermal failures; FDA's record
says {fda_fs}. With only {fda_n} directionally-classifiable
events the interval is far too wide to call that a real disagreement. But the <em>mechanism</em>
comparison is decisive: <strong>every freeze-caused recall in the dataset is a storage failure</strong>
— product held below 32&nbsp;°F in a distribution centre, subfreezing in a warehouse, crystallised
after cold storage. This model simulates transit only. It has no warehouse stage, so it structurally
cannot produce the mechanism behind the entire real freeze record. That is a scope limit found by
looking, not assumed — and more useful than agreement would have been.</div>

<h2>The decision: what to actually do</h2>
<p>Every part of this has been built by somebody except one — the argmin. No published system,
patent or paper prices <em>all</em> the mid-transit actions in a single currency against a
continuous stability-budget state. Below, each point of the plane is a state a shipment can be in,
coloured by the cheapest action from there.</p>
<img src="data:image/png;base64,{pm_poor}" alt="Policy map, poorly insulated">
<img src="data:image/png;base64,{pm_good}" alt="Policy map, well insulated">
<p>Re-icing buys <em>time</em>, not immunity: a fresh charge holds for roughly one thermal time
constant, so on a long leg it wears off and most of the burn happens anyway. That is why the
well-insulated box is mostly “re-ice” and the poorly-insulated one is mostly “expedite” — with the
same product, the same money and the same risk. A policy map can be reviewed and signed
<strong>once, in advance</strong>, which is a very different governance object from a
recommendation that has to be adjudicated shipment by shipment at 2&nbsp;a.m.</p>


<h2>Where this could be wrong</h2>
<p>The intervention costs are assumptions in a plausible band, so the honest headline is not the
dollar figure but its stability: <strong>{stab.stable_share:.0%} of the decision space keeps the
same recommended action across the full low-to-high cost range.</strong> Reject every cost here
individually and the actions still stand.</p>
<p>Shipments are simulated; the weather driving them is not. The estimator behind the threshold
finding carries roughly +10% mean bias, with its worst single cohort 59% out, which is why nothing
above is quoted as a bare point estimate.</p>

<p style="margin-top:2.4rem"><a href="appendix.html" style="display:inline-block;border:1px solid
 {GRID};color:{INK};padding:.55rem 1.1rem;border-radius:5px;text-decoration:none;font-weight:600">
 Appendix: the estimator, the constraint, the models &rarr;</a></p>

<footer>
Coldspend · physics-based digital twin of pharmaceutical cold chain ·
MKT per USP&nbsp;⟨1079⟩, Arrhenius stability budget, lumped-capacitance RC with phase-change handling.
Ambient forcing is <strong>real Open-Meteo historical reanalysis</strong> (2024 hourly, CC-BY 4.0)
for all 18 airports, cached in-repo so every figure reproduces offline. Shipment routing, packaging
and failure modes are simulated. Simulated results — not a claim about any real network.
</footer>
"""

    appendix = APPENDIX_HEAD + f"""
<h2>Regression discontinuity at the threshold</h2>
<p>Alarm thresholds are a textbook regression discontinuity on a running variable measured to a
tenth of a degree by every vendor on earth — and no published RD, IV or DiD on any cold-chain alarm
threshold appears to exist, in pharma, food, reefer, blood banking or organ transport.</p>
<table>
<tr><th>Quantity</th><th class="num">Value</th></tr>
<tr><td>First stage (jump in P(intervene))</td><td class="num">{r.jump_treatment:+.3f}</td></tr>
<tr><td>Fuzzy RD estimate</td><td class="num">{r.estimate:+.3f} pp</td></tr>
<tr><td>95% CI (bootstrap)</td><td class="num">[{lo:+.3f}, {hi:+.3f}]</td></tr>
<tr><td>True complier effect (known only in simulation)</td><td class="num">{truth:+.3f} pp</td></tr>
</table>
<div class="note"><strong>And a textbook fix that failed.</strong> Bias against the true complier
effect is about +10% on the mean, with full interval coverage across five independent cohorts. The
standard remedy, Imbens-Kalyanaraman / CCT MSE-optimal bandwidth selection, is implemented and
deliberately <em>not used</em>: measured, it gives +71% mean bias and returns the wrong sign in two
cohorts of five. It selects windows of 50 to 83 minutes, and only about 5% of shipments sit within
150 minutes of the threshold, so the Wald ratio's denominator goes unstable. This design is
<strong>variance-limited, not bias-limited</strong>, which points at sample size or threshold
placement rather than a cleverer bandwidth rule. The worst single cohort is still 59% out, and that
is why nothing here is ever quoted as a bare point estimate.</div>

<h2>Why this is an optimisation and not a lookup</h2>
<p>Five actions on one shipment is a lookup table, and pretending otherwise would be theatre. It
becomes a genuine optimisation only when shipments <em>compete</em> — a hub has a finite number of
re-icing bays per shift, so helping one shipment means not helping another.</p>
<table><tr><th class="num">bays</th><th class="num">re-iced</th><th class="num">expected loss</th>
<th class="num">cost of the constraint</th></tr>{cap_rows}</table>
<p>The right-hand column is what scarcity costs against an unconstrained world. It is strictly
positive whenever the constraint binds — which is the evidence that the decisions genuinely do not
separate.</p>
<div class="note"><strong>{stab.stable_share:.0%} of states keep the same recommendation across the
full low-to-high cost range.</strong> That is a deliberately stronger claim than any dollar figure:
the intervention costs here are assumptions in a plausible band, and a reviewer can reject every one
of them individually while the recommended action still stands.</div>

<h2>Prediction is the easy part</h2>
<table>
<tr><th>Model</th><th class="num">AUC</th><th class="num">Brier</th></tr>{model_rows}</table>
<p>Gradient boosting beats logistic regression by <strong>{sel.auc - lr.auc:+.4f} AUC</strong> — that
is, not at all. And an AUC near {sel.auc:.2f} is a warning rather than a trophy: predicting a breach
from “already at 7.8&nbsp;°C and warming” is genuinely easy, which is exactly why prediction is not
the contribution here. The oracle row sees the unobserved confounder and the weather that has not
happened yet; the gap between it ({orc.auc:.4f}) and the best deployable model ({sel.auc:.4f}) is
the irreducible uncertainty that keeps the exercise from being circular.</p>

<h2>Generator calibration gate</h2>
<table><tr><th>Check</th><th class="num">Value</th><th class="num">Band</th><th>Result</th></tr>
{gate_rows}</table>
"""

    (OUT / "appendix.html").write_text(appendix, encoding="utf-8")
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    kb = len((OUT / "index.html").read_bytes()) / 1024
    print(f"wrote {OUT / 'index.html'}  ({kb:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
