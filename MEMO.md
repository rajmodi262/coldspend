# Cold chain is a decision problem, and it is being run as a compliance problem

**To:** Head of Supply Chain Quality  **From:** Coldspend analysis
**Date:** 03 August 2026  **Basis:** 2,434 simulated shipments

---

## Answer, first

**Your alarm threshold fires where intervening is worth almost nothing, and the money is
being spent in the wrong place.** Re-pricing every mid-transit action against the stability
budget — rather than following the fixed re-ice rule — is worth **$239,395
per quarter on 2,434 shipments**, for **$198,000** of
intervention. That benefit stays positive across every cost assumption tested, from
$89,100 to $677,113.

The recommendation does not depend on the cost figures being right. **86%
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
in shelf life consumed by a factor of **1.787×** (82.6 against 46.2
equivalent-hours at reference). Compliance records them as identical. They are not.

Worse, the escalation threshold is mis-placed. At the current trigger point, intervening is
worth **0.19 standard deviations** of stability budget — small enough that no
achievable sample size could demonstrate it. Deeper into the exposure tail the same
intervention is worth **0.50 SD**. The rule fires where acting barely helps, and stays
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

Across the quarter, the optimiser intervenes on **165 of 2,434 shipments**
— far fewer than the current rule, and on different ones.

### What it is worth, priced across the full cost range

| cost level | current rule | optimised | intervention spend | benefit |
|---|---|---|---|---|
| 0.00 | $3,718,000 | $3,628,900 | $33,000 | **$89,100** |
| 0.25 | $8,931,040 | $8,774,786 | $130,500 | **$156,255** |
| 0.50 | $14,144,081 | $13,904,686 | $198,000 | **$239,395** |
| 0.75 | $27,442,963 | $26,986,954 | $349,650 | **$456,009** |
| 1.00 | $40,741,845 | $40,064,731 | $490,000 | **$677,113** |

Benefit is measured against **the rule you already follow**, not against doing nothing.
Beating inaction would be trivially true and not worth reporting.

---

## What I would not claim

- **These are simulated shipments, driven by real weather.** No proprietary telemetry was used,
  but the ambient forcing is not invented: every lane runs on real Open-Meteo hourly reanalysis
  for 2024 at its actual airports. The thermal model — RC with phase-change handling, MKT per
  USP ⟨1079⟩, Arrhenius degradation — and the routing and failure modes are simulated.
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

*Generated 2026-08-03 22:33 by `scripts/build_memo.py`. Every figure above is
computed at build time from the same code the test suite exercises.*
