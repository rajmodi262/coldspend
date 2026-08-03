# Coldspend — the pitch

Every number here is computed by this repository. Where a figure came from outside, its source
is named. Where something is unresolved, it says so — an interviewer who finds a hole you
didn't flag stops evaluating your work and starts evaluating your judgement.

**Regenerate the figures:** `python scripts/build_memo.py` and `python scripts/build_site.py`.

---

## The one sentence

> The pharmaceutical industry treats cold chain as a compliance problem — did we breach 2–8 °C?
> — when it is a decision problem: given what we know right now, what is the cheapest action
> that keeps expected loss below threshold?

## Sixty seconds

> "Two shipments, both sixty hours, both entirely inside two to eight degrees, both filing zero
> deviations. One cruises at 3.0, the other at 7.5. Compliance records them as identical. Under
> Arrhenius kinetics the warm one has spent 1.787 times the shelf life. That gap is invisible to
> every alarm in the industry.
>
> So I built the decision layer that gap implies. It computes Mean Kinetic Temperature and an
> Arrhenius stability budget, predicts breach risk with calibrated probabilities, and prices
> every mid-transit action — re-ice, expedite, re-route, recall — in one currency, then takes the
> argmin.
>
> In my simulated network, on about two thousand shipments, that is worth $444,000 a quarter
> against the rule they already follow, for $240,000 of intervention. And the benefit stays
> positive across every cost assumption I tested.
>
> But the finding I'd actually lead with is smaller and more uncomfortable: **their alarm
> threshold fires where intervening is worth 0.17 standard deviations.** Four hundred minutes
> later the same action is worth 0.68. The rule isn't wrong about whether to act. It's wrong
> about when."

## The demo, four beats

1. **The hook.** Two flat temperature traces, both inside spec. *"Which one passes?"* Both do.
   Then the second panel: 82.7 versus 46.2 equivalent-hours. **1.787×.** No gotcha, no trick
   metric — just kinetics doing what thresholds can't.
2. **The threshold finding.** The bar chart of effect size by exposure. The SOP fires on the
   0.17 SD bar. *"They are spending money in the place it does the least good."*
3. **The policy map.** Drag τ from 12 to 40 hours and watch *expedite* give way to *re-ice*.
   Same product, same money, same risk — different physics. *"Re-icing buys time, not immunity."*
4. **The zoom-out.** The capacity table. Bays 60 → 8 and the cost of scarcity goes $0 → $451k.
   *"That column is why this is an optimisation and not a lookup table."*

**If the live demo dies**, beat 1 is pure argument and needs no laptop. Open with it.

## Twelve slides

| # | Assertion | What's on it |
|---|---|---|
| 1 | Two shipments that pass identically differ by 1.787× in shelf life | the hook chart |
| 2 | Compliance is binary, retrospective, and value-blind | the three consequences |
| 3 | The industry's own data says excursions are economic, not safety, events | openFDA Class III skew, p = 0.015 |
| 4 | A stability budget replaces the alarm with a continuous state | MKT + Arrhenius, the burn-down |
| 5 | The alarm fires where intervening is worth 0.17 SD | effect-by-exposure chart |
| 6 | Prediction is the easy part, and it is already patented | model board + US 11,769,103 B2 |
| 7 | The missing piece is the argmin | the policy map |
| 8 | Re-icing buys time, not immunity | the two policy maps, τ=12 vs τ=40 |
| 9 | It is an optimisation because shipments compete | the capacity table |
| 10 | $240k of intervention, $444k better than the current rule | the quarter, priced as a range |
| 11 | The recommendation survives the costs being wrong | 86% decision stability |
| 12 | What I'd do differently with real data | the three next steps |

Slides 3, 6 and 11 are the ones that win the room, and none of them is a victory lap — one cites
the regulator against the industry, one concedes the modelling layer is commodity, and one
argues the dollar figure matters less than its stability.

## Resume bullets

**Decision Analytics flavour**

> **Coldspend — pharmaceutical cold-chain decision intelligence.** Built a physics-based digital
> twin (RC thermal model with phase-change handling, Arrhenius kinetics, Mean Kinetic Temperature
> per USP ⟨1079⟩) generating calibrated shipment data with both potential outcomes retained;
> trained calibrated risk models scored against a Bayes-oracle ceiling; developed a portfolio MILP
> that prices six mid-transit interventions in one currency under shared hub capacity. Simulated
> quarter: **$444k expected loss avoided for $240k intervention spend, stable across the full cost
> range.** Identified via regression discontinuity that the industry's standard alarm threshold
> fires where intervention is worth 0.17 SD versus 0.68 SD deeper in the exposure tail.
> *Python, scikit-learn, scipy/HiGHS, marimo/WASM.*

**Engineering flavour**

> **Coldspend** — physics simulator, calibrated ML and a MILP optimiser, shipped as a static site
> with the solver running client-side in WebAssembly (Pyodide + scipy/HiGHS); zero hosting cost,
> no backend. 89 tests including property-based invariants over the thermal model; CI runs the
> acceptance gates before it deploys. *Python 3.12, uv, scikit-learn, scipy, marimo, GitHub Actions.*

**One line**

> Built a cold-chain decision engine — physics simulator → calibrated risk → MILP optimiser →
> browser-side app — showing $444k/quarter against the incumbent rule, and that the industry's
> alarm threshold is set where intervening barely helps.

## The five questions, and what to say

**"Your AUC is 0.98. Isn't your model just rediscovering your own simulator?"**
> "Partly, yes — and that's why I don't lead with it. Predicting a breach from *already at 7.8 °C
> and warming* is easy, and it's the part that's already patented and already shipping. I built a
> Bayes oracle that sees the unobserved confounder and the realised weather; it scores 0.994
> against my 0.983, so there is genuine irreducible uncertainty, but the honest reading is that
> prediction isn't the contribution. The argmin is."

**"Where did $444,000 come from?"**
> "From the counterfactual quarter, computed — not assumed. And I'd rather you didn't trust it:
> the intervention costs are assumptions in a plausible band. That's why the memo leads with
> decision stability instead. 86% of the state space keeps the same recommended action across the
> full low-to-high cost range, so you can reject every cost figure I've given you and the actions
> still stand."

**"Five actions on one shipment is a lookup table, not an optimisation."**
> "Correct, and I say so in the code. It becomes an optimisation when shipments compete — a hub
> has finite re-icing bays per shift, so helping one means not helping another. I report the cost
> of that scarcity explicitly; it's $0 when the constraint is slack and $451k when it binds. That
> number is the evidence the decisions genuinely don't separate."

**"What's the hardest thing you got wrong?"**
> "The regression discontinuity gave a biased estimate that got *worse* as I narrowed the
> bandwidth — backwards from how RD bias behaves. I ruled out curvature and complier composition
> before finding it: counting whole timesteps above spec had quantised my running variable, so
> there was a mass point sitting exactly on the threshold, anchoring one side of the fit while the
> other extrapolated. Interpolating the crossing times fixed it. There's still a residual ~50%
> overestimate I haven't closed — it needs MSE-optimal bandwidth selection — so that point
> estimate stays out of my deck."

**"Why ZS?"**
> "Because the deliverable here was never the code. It was a memo a client could act on Monday,
> and the analysis behind it had to survive someone checking. That arc — domain physics, a model,
> an optimiser, and a recommendation with its own limitations stated — is what ZS does for life
> sciences clients. And cold chain specifically sits in the cell-and-gene-therapy logistics space
> that ZS's own capability materials name."

## Things to never say

- ~~"25% of vaccines arrive degraded"~~ — untraceable folklore. Use the openFDA base rate.
- ~~"A CAR-T shipment is worth $2M"~~ — that's episode cost. Product WAC is ~$443,600 (Milliman).
- ~~"It never breached 8 °C but its MKT was 9.4"~~ — arithmetically impossible; MKT is bounded by
  the trace.
- ~~"This saves product QA would otherwise destroy"~~ — you cannot un-degrade a molecule.
- ~~"There is no server"~~ — no *application* server; Pyodide still comes from a CDN.
