# Study guide — own this before you present it

Claude wrote most of this repository. That is not a problem with the project; it is a problem with
the project **as evidence about you**, which is the only thing an interviewer is buying.

Every score in `ROADMAP-TO-10.md` collapses to whatever you can defend under questioning. This file
exists to close that gap. It is not a script to memorise — reciting is worse than not knowing,
because it invites the follow-up you cannot answer.

**Test yourself this way:** cover the answer, say yours out loud, then open the file named beside
it. If you cannot get within a sentence of the reasoning, you do not own it yet.

---

## Tier 1 — you will be asked these

### 1. Why can MKT never exceed the highest temperature in the trace?
`src/coldspend/physics/mkt.py`, `tests/test_invariants.py::test_mkt_always_lies_within_the_trace`

Because it is a weighted mean of a monotone transform of temperature, then transformed back. A
weighted average of values cannot leave their range, and `exp(-ΔH/RT)` is monotone in `T`, so the
inverse maps back inside. Hence *"it never breached 8 °C but its MKT is 9.4 °C"* is arithmetically
impossible — which is why the original demo hook had to be thrown away.

**Be able to derive it**, not just assert it. And know the corollary: MKT is bounded but *not* equal
to the mean — it sits above, by Jensen, because the transform is convex over this range. That
convexity is the entire reason the metric carries information.

### 2. Why does re-icing "buy time, not immunity"?
`src/coldspend/decide/optimizer.py::_reice_burn_multiplier`

A fresh coolant charge holds the payload down for roughly one thermal time constant τ; after that
the box is driven by ambient again. So the benefit is the share of the *remaining* journey the
charge actually covers — `min(1, τ / hours_remaining)`.

**Why it matters:** modelled as a flat multiplier instead, re-icing was optimal across 95% of the
state space. Physically wrong, and a useless policy map. With the decay in place a well-insulated
box is mostly *re-ice* and the same consignment poorly insulated is mostly *expedite* — same
product, same money, same risk, different physics. That boundary is the whole argument for pricing
actions against each other rather than following a playbook.

### 3. Your AUC is 0.95+. Isn't the model just rediscovering your own simulator?
`src/coldspend/models/risk.py`

Partly yes, and that is why it is not the headline. Predicting a breach from *"already at 7.8 °C and
warming"* is genuinely easy — and it is the part that is already patented (US 11,769,103 B2) and
already shipping commercially.

Two things make it non-circular. There is a **Bayes oracle** that sees the unobserved confounder and
the weather that has not happened yet; the gap between it and the best deployable model is genuine
irreducible uncertainty. And the simulator carries **post-decision randomness** — missed connections,
forecast error, handling exposure — drawn *after* the treatment choice and applied to both arms, so
the outcome cannot be reconstructed from decision-time inputs.

Know that the first version had **no** such noise, scored 0.988 on plain logistic regression, and had
an oracle only 0.002 above it. That was the circularity attack landing, and it got fixed by changing
the generator rather than the metric.

### 4. Where did $239,000 come from, and why should I believe it?
`src/coldspend/analysis/quarter.py`, `scripts/build_memo.py`

Computed end to end — simulator → calibrated risk model → optimiser → expected loss — and measured
against **the SOP the industry already follows**, not against doing nothing. Beating inaction is
trivially true and would be a dishonest headline.

**Then take the number away yourself.** The costs are assumptions in a plausible band, so the claim
that matters is not the dollar figure but that **86% of the state space keeps the same recommended
action across the full low-to-high cost range.** A reviewer can reject every individual cost and the
actions still stand. Decision stability beats point precision.

### 5. Isn't five actions on one shipment just a lookup table?
`src/coldspend/decide/optimizer.py::optimise_portfolio`

Yes — and saying so first is the right move. It becomes a genuine optimisation only when shipments
**compete**: a hub has finite re-icing bays per shift, so helping one means not helping another. The
plan reports the cost of that scarcity explicitly — zero when the constraint is slack, hundreds of
thousands when it binds. That number is the evidence the decisions do not separate.

---

## Tier 2 — the three findings, and why each is worth telling

These are the strongest material in the project because in each one **the project caught me**. Tell
them as stories, not as bullet points.

### A. Real weather corrected an invented assumption
Freeze events fell from 6.0% to 1.4% the moment real Open-Meteo reanalysis replaced my synthetic
climatology — which had been manufacturing cold exposure at the Gulf hubs that does not exist.

*The point:* an assumption I had no reason to doubt was wrong, and only real data showed it.

### B. FDA's record revealed a missing mechanism
`src/coldspend/validate/openfda.py`

Storage appears in **51%** of real temperature-related drug recalls; transit in **14%**. And *every*
freeze-caused recall is a warehouse event — held below 32 °F in a distribution centre, subfreezing
in storage, crystallised after cold storage. The model was transit-only. It structurally could not
produce the mechanism behind the entire real freeze record.

Adding a storage stage moved the simulated freeze share from 3.5% to 11.3%, inside FDA's interval.

**Know the two traps**, because they are the substance:
- **Denominators.** FDA reports the share of *recalls* caused by temperature; the simulator reports
  the share of *shipments* destroyed. Comparing them directly would look quantitative and be
  nonsense. Only the *composition* — hot versus cold — is comparable.
- **Deduplication.** openFDA returns one row per *product*: 619 records collapse to 78 events, an
  8× inflation. One distributor recalling many SKUs would otherwise dominate everything.

And the classification bug: **"Temperature Abuse" is not a heat indicator.** FDA uses it for both
directions — one recall reads *"Temperature Abuse: product samples were stored at temperatures below
32 °F"*. Treating it as heat drove the measured freeze share to zero.

### C. The textbook fix failed — the best story you have
`src/coldspend/causal/rd.py::mse_optimal_bandwidth`

The RD carried a residual bias. The standard remedy is Imbens–Kalyanaraman / CCT MSE-optimal
bandwidth selection. I implemented it, measured it against ground truth, and it was **seven times
worse**: +71% mean bias against +10% for a fixed bandwidth, wrong sign in two cohorts of five.

The reason is density, not algebra. The MSE criterion trades squared bias against variance *assuming
enough mass at the cutoff*. Only ~5% of shipments sit within 150 minutes of the threshold, so the
selector picks a 55-minute window, leaving a few hundred units split across two sides, and the Wald
ratio's denominator goes unstable.

**So the design is variance-limited, not bias-limited** — which points at sample size or threshold
placement, not a cleverer rule. The code is kept and exported precisely *because* it failed: without
it, choosing h=200 looks arbitrary.

*Why this beats a clean success:* running a method shows competence. Knowing when its assumptions
do not hold shows judgment.

---

## Tier 3 — the traps you should raise before they do

| Trap | The honest line |
|---|---|
| It is all simulated | "The weather is real reanalysis and I checked the failure composition against FDA's own recall record. It disagreed, and the disagreement told me my model was missing a storage stage." |
| The costs are made up | "They are. That is why the headline is decision stability, not the dollar figure." |
| Prediction is not novel | "Correct — it is patented and shipping. The argmin is the contribution." |
| The RD is still imprecise | "Worst single cohort is 59% out. That is why nothing is quoted as a bare point estimate." |
| Nobody has used it | "True. No user, no client, no feedback loop. It is a closed system that validates itself against external records where it can." |

---

## What to run before you present

```bash
uv run pytest -v
```

Read the test **names**. They are written as claims, and they are the fastest route into what this
project actually asserts. `tests/test_invariants.py` is the file to understand first — it tests
properties that must hold for *every* input rather than a handful of cases, and *"I don't test cases,
I test invariants"* is a materially stronger answer than a coverage number.

Then read, in order: `DECISIONS.md` (why each choice was made, and the abort triggers written before
the pressure arrived), `NOVELTY-POSITION.md` (what is genuinely new and what is not), and
`ROADMAP-TO-10.md` (the honest audit).

## The last thing

**Make some commits yourself.** Even small ones. A repository whose entire history is one
contributor invites a question you would rather not be asked, and the fix is cheap.
