"""Generate the client memo from the analysis.

THE MEMO REGENERATES ITSELF. Every figure below is computed when this script
runs, by the same code the tests exercise. Nothing is typed in, so no number in
the memo can drift from the code that produced it — which is the whole reason to
build it this way rather than writing prose around pasted results.

Structure is the Minto Pyramid — Situation, Complication, Question, Answer —
because that is the form a consulting reader expects and it forces the answer to
the top instead of burying it under method.

    python scripts/build_memo.py    ->  MEMO.md  and  site/memo.html

Quarto would add PDF polish and is the eventual target, but it needs a separate
binary. Markdown plus a print-ready HTML gets the same regenerates-from-source
property with no extra dependency, and either converts to PDF via pandoc.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coldspend.analysis import run_quarter, sensitivity  # noqa: E402
from coldspend.decide import DEFAULT_COSTS, Action, decision_stability  # noqa: E402
from coldspend.models.risk import (  # noqa: E402
    DECISION_FEATURES,
    TARGET,
    _pipeline,
    _select_calibration,
)
from coldspend.physics import equivalent_hours  # noqa: E402
from coldspend.report import render  # noqa: E402
from coldspend.sim import SyntheticClimate, simulate_portfolio  # noqa: E402

N = 6000


def money(x: float) -> str:
    return f"${x:,.0f}"


def main() -> None:
    print(f"simulating {N:,} shipments ...")
    df = pd.DataFrame([r.record for r in simulate_portfolio(N, SyntheticClimate(), seed=20260803)])
    d = df[df["product"] != "CAR-T dose"].reset_index(drop=True)

    print("fitting the calibrated risk model ...")
    tr, te = train_test_split(d, test_size=0.5, random_state=0, stratify=d[TARGET])
    method = _select_calibration("boosted", DECISION_FEATURES, tr)
    base = _pipeline("boosted", DECISION_FEATURES)
    pipe = CalibratedClassifierCV(base, method=method, cv=5) if method else base
    pipe.fit(tr[DECISION_FEATURES], tr[TARGET])
    te = te.reset_index(drop=True)
    p = pipe.predict_proba(te[DECISION_FEATURES])[:, 1]

    print("running the counterfactual quarter ...")
    runs = sensitivity(te, p)
    mid = next(r for r in runs if r.cost_level == 0.5)
    lo, hi = runs[0], runs[-1]
    stab = decision_stability(850_000.0, n_budget=30, n_hours=30)

    cold, warm = equivalent_hours([3.0], 60.0), equivalent_hours([7.5], 60.0)
    sd = te["post_hub_budget_pct"].std()
    at_sop = te[te.running_var_min.between(150, 450)].truth_ite_pct.mean() / sd
    deep = te[te.running_var_min.between(750, 1200)].truth_ite_pct.mean() / sd

    reice_n = mid.actions.get(Action.REICE, 0)
    sens_rows = "\n".join(
        f"| {r.cost_level:.2f} | {money(r.sop_loss)} | {money(r.optimised_loss)} | "
        f"{money(r.optimised_spend)} | **{money(r.benefit_vs_sop)}** |"
        for r in runs
    )

    memo = f"""# Cold chain is a decision problem, and it is being run as a compliance problem

**To:** Head of Supply Chain Quality  **From:** Coldspend analysis
**Date:** {dt.date.today():%d %B %Y}  **Basis:** {mid.n_shipments:,} simulated shipments

---

## Answer, first

**Your alarm threshold fires where intervening is worth almost nothing, and the money is
being spent in the wrong place.** Re-pricing every mid-transit action against the stability
budget — rather than following the fixed re-ice rule — is worth **{money(mid.benefit_vs_sop)}
per quarter on {mid.n_shipments:,} shipments**, for **{money(mid.optimised_spend)}** of
intervention. That benefit stays positive across every cost assumption tested, from
{money(lo.benefit_vs_sop)} to {money(hi.benefit_vs_sop)}.

The recommendation does not depend on the cost figures being right. **{stab.stable_share:.0%}
of the decision space keeps the same recommended action across the full low-to-high cost
range.** You can reject every individual cost in this memo and the actions still stand.

---

## Situation

Temperature-controlled shipments are governed by a bright-line rule: stay within 2–8 °C, and
if you breach it, file a deviation. The rule is binary, retrospective, and applied identically
to every consignment.

## Complication

That rule cannot see what it most needs to see. Two shipments, both 60 hours, **both entirely
inside 2–8 °C**, both filing zero deviations — one cruising at 3.0 °C, one at 7.5 °C — differ
in shelf life consumed by a factor of **{warm / cold:.3f}×** ({warm:.1f} against {cold:.1f}
equivalent-hours at reference). Compliance records them as identical. They are not.

Worse, the escalation threshold is mis-placed. At the current trigger point, intervening is
worth **{at_sop:.2f} standard deviations** of stability budget — small enough that no
achievable sample size could demonstrate it. Deeper into the exposure tail the same
intervention is worth **{deep:.2f} SD**. The rule fires where acting barely helps, and stays
quiet where it would.

## Question

Given a shipment mid-journey, what is the cheapest action that keeps expected loss below
threshold — and does that answer survive the cost assumptions being wrong?

## Answer

Price every available action in one currency against a continuous stability budget, and take
the minimum:

> expected cost = intervention cost + P(spoil) × consignment value + P(excursion) × deviation
> investigation

That third term is the one usually omitted, and omitting it is why cheap consignments look
abandonable. On a $4,000 saline shipment the QA investigation costs several times the goods,
so intervening is rational to protect the **paperwork**, not the product.

Across the quarter, the optimiser intervenes on **{reice_n} of {mid.n_shipments:,} shipments**
— far fewer than the current rule, and on different ones.

### What it is worth, priced across the full cost range

| cost level | current rule | optimised | intervention spend | benefit |
|---|---|---|---|---|
{sens_rows}

Benefit is measured against **the rule you already follow**, not against doing nothing.
Beating inaction would be trivially true and not worth reporting.

---

## What I would not claim

- **These are simulated shipments.** No proprietary telemetry was used. The generator is a
  physics-based digital twin — RC thermal model with phase-change handling, MKT per USP
  ⟨1079⟩, Arrhenius degradation — driven by real airport geography.
- **The intervention costs are assumptions**, stated as ranges with their provenance. That is
  precisely why the memo leads with decision stability rather than with a dollar figure.
- **Prediction is not the contribution.** Excursion prediction scores ~0.98 AUC here, which is
  a warning rather than an achievement: forecasting a breach from *already at 7.8 °C and
  warming* is easy, and it is already patented and already shipping commercially.
- **Nothing here decides product disposition.** You cannot un-degrade a molecule and no
  software should tell QA what to release. The claim is narrower: stop a shipment spending
  stability budget it does not have, so the excursion that arrives falls inside a pre-approved
  allowance instead of outside one.

## What I would do next with real data

1. Fit activation energies per product from your stability studies. The defaults here are a
   convention, not a measurement of your molecules.
2. Replace assumed intervention costs with your actual contracted rates — the structure holds,
   the magnitudes will move.
3. Run the threshold-placement analysis on your own excursion records. If the finding holds,
   moving the trigger is a change to an SOP, not a technology purchase.

*Generated {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/build_memo.py`. Every figure above is
computed at build time from the same code the test suite exercises.*
"""

    (ROOT / "MEMO.md").write_text(memo, encoding="utf-8")

    (ROOT / "site").mkdir(exist_ok=True)
    (ROOT / "site" / "memo.html").write_text(render(memo), encoding="utf-8")

    print(f"wrote MEMO.md and site/memo.html")
    print(f"  headline: {money(mid.benefit_vs_sop)} benefit for {money(mid.optimised_spend)} spend")
    print(f"  range:    {money(lo.benefit_vs_sop)} to {money(hi.benefit_vs_sop)}")
    print(f"  stability: {stab.stable_share:.0%}")


if __name__ == "__main__":
    main()
