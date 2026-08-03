# Coldspend — build phases

Seven phases, ordered by **dependency**, not by how interesting they are. The single most important
ordering rule is the one that isn't obvious: **the deploy happens in the middle, not at the end.**

---

## The ordering rule

Two things go early that instinct says should go late:

1. **Acceptance tests before the thing they test.** Done in Phase 0. A simulator that runs and
   produces plausible curves but is mis-calibrated is invisible until week 6, by which time every
   model, cost figure and chart downstream is worthless. The declared bands catch it in week 2.
2. **Deploy something ugly in Phase 3, not Phase 6.** The week-4 gate. Shipping a bad page early
   converts the final week from *risk* into *improvement*. Almost every student project dies here.

And one thing goes earlier than it looks like it should: **compliance friction and the unobserved
confounder belong in Phase 1**, inside the simulator's first version. They are not realism polish —
without them the regression discontinuity identifies nothing (see `DECISIONS.md` D10). Retrofitting
them in Phase 4 means regenerating all data and re-running everything downstream.

---

## Phase 0 — Foundation ✅ COMPLETE

Repo skeleton, Python 3.12 via uv, and the pharmacopoeial physics with its acceptance tests.

| Delivered | |
|---|---|
| `physics/mkt.py` | MKT, USP ⟨1079⟩, `EA_OVER_R` exact |
| `physics/stability.py` | Arrhenius budget, equivalent-hours, freeze damage |
| `physics/thermal.py` | closed-form RC + PCM latent reservoir, hold-time inversion |
| `tests/test_physics.py` | 20 acceptance tests |
| `tests/test_invariants.py` | 12 Hypothesis property tests |
| `DECISIONS.md` | ADR + pre-committed abort triggers |

**Gate passed:** 32/32 green; the 1.787× ratio reproduces from the package.

## Phase 1 — The Generator ⬅ **NEXT**

The simulator. Everything downstream is built on this, so it is the highest-risk phase in the project.

- Product catalogue with real stability profiles and real consignment values
- Lane network from real airport coordinates and real pharma corridors
- Open-Meteo ingest with a **committed on-disk cache** — same seed must give same weather
- Shipment simulation: route legs, ground events, packaging, failure modes (tarmac hold, customs,
  reefer failure, gel-pack pre-conditioning error, sensor dropout and drift)
- **Compliance friction** — SOP obeyed imperfectly (identification prerequisite)
- **Unobserved confounder** — operator judgment the logger never records (identification prerequisite)
- **Both potential outcomes retained** — treated and untreated, under common random numbers
- Calibration harness with declared acceptance bands
- Data contracts over the output schema

**Gate:** excursion rate and freeze rate inside declared bands; Crēdo hold time within 10% of the
published figure; the whole dataset rebuilds from one command and one seed, reproducibly.

## Phase 2 — The Measurement Layer

Turning traces into numbers a decision can use.

- Stability-budget burn-down as a running state per shipment
- Logistic-regression baseline **first**, then HistGradientBoosting — and report the gap honestly
- **Probability calibration** (isotonic / Platt, reliability curves, Brier). Non-negotiable: the
  optimizer multiplies these by money. Uncalibrated probabilities silently corrupt every dollar figure.
- Weibull AFT survival with interval censoring; Cox time-varying for inference only
- Bayes oracle — the simulator's own irreducible noise ceiling, so model skill is read against what
  is *achievable*, not against 1.0
- SHAP

**Gate:** calibration curve within tolerance; model skill reported against the oracle ceiling.

## Phase 3 — Deploy Ugly + The Causal Layer

Two tracks, run together. The deploy is a **hard gate**, not an aspiration.

- **Static page live on GitHub Pages by end of this phase.** Ugly is fine. Live is the requirement.
- `prototypes/rd_validation.py` → `causal/rd.py`, with tests
- Diagnostics: McCrary density, one-sided placebo cutoffs, bandwidth sweep, covariate balance,
  first-stage strength
- Validation against known ITE — the thing real data cannot do
- Uplift model: *which shipments benefit*, not which are at risk
- Lane-adjusted carrier scorecard via propensity matching

**Gate:** a URL exists. RD recovers the known LATE within tolerance with valid coverage.

## Phase 4 — The Decision Layer

The argmin. This is the project's novelty claim, so it gets the most care.

- Cost model in **ranges, not points**, with every figure sourced
- MILP via `scipy.optimize.milp`, with `threads=1` and a lexicographic tie-break so the
  recommendation is stable across machines
- **Shared hub re-icing capacity and a portfolio-wide intervention budget** — this is what makes the
  optimization non-trivial rather than an enumeration in a costume
- **Policy map** by backward induction over the state grid — the headline visual
- CVaR sweep: the priced menu of caution
- **Decision stability** — show the recommendation is invariant across the plausible cost range.
  A far stronger claim than any single dollar figure, and immune to "you made the costs up".

**Gate:** the policy map renders; the recommendation is stable under the cost sweep.

## Phase 5 — The Application

- marimo notebook as the app; WASM export replacing the Phase 3 ugly page
- The six charts: policy map, RD figure, counterfactual pairs, burn-down with uncertainty fan,
  carrier bump chart, decision curve
- Client-side 60× replay

**Gate:** a stranger can open the URL and reach the recommendation unaided.

## Phase 6 — The Deliverables

The code is not the deliverable. The decision is.

- Quarto memo that **regenerates from the analysis**, so every number traces to a computed cell
- 12-slide deck, assertion titles
- Counterfactual quarter
- README, resume bullets, interview answers
- GitHub Actions running acceptance tests and redeploying on push

**Gate:** the memo rebuilds end-to-end from a clean checkout.

---

## Against the 8 weeks

| Week | Phase |
|---|---|
| 1 | 0 ✅ + start 1 |
| 2 | 1 — generator + calibration gate |
| 3 | 2 — models + calibration |
| 4 | 3 — **deploy ugly** + RD |
| 5 | 4 — optimizer + policy map |
| 6 | 4→5 — charts |
| 7 | 5 — the app |
| 8 | 6 — memo, deck, counterfactual |

Realistic capacity is 200–240 hours and the full scope wants more. The abort triggers in
`DECISIONS.md` decide what goes, and they were written before the pressure arrived — which is the
only time such decisions are made honestly.
