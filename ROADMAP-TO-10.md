# Roadmap: 6.5/10 → 9-10/10 as a ZS portfolio piece

Honest starting point, from a full audit on 2026-08-03.

| Dimension | Now | Target | Gap |
|---|---|---|---|
| Technical depth | 9 | 9 | — already strong |
| Engineering quality | 8 | 9 | one deployed component is broken |
| Consulting framing | 9 | 9 | — the strongest part |
| Intellectual honesty | 10 | 10 | — keep it |
| **Evidence quality** | **3** | **7** | **every number describes an invented world** |
| **Novelty** | **4** | **8** | **the RD that supports it is 50% biased** |
| **Presentability** | **5** | **9** | **seven sections; you get fifteen minutes** |
| **Ownership** | **?** | **10** | **Claude wrote most of this** |

**Artifact today: 8/10. Evidence about the candidate: 6.5/10.** The gap is comprehension,
not code.

---

## The finding that drives Phase 1

The project's defence against *"you made it all up"* was supposed to be "grounded in real
external data." An audit of the code says otherwise:

- `OpenMeteoArchive` — written, tested, **never called**. Every figure runs on `SyntheticClimate`.
- openFDA, US BTS — **zero references** anywhere in `src/`, `scripts/` or `app/`.
- The only real input is 18 hardcoded airport coordinates.

The research documented six free, verified data sources. The code uses one of them, barely.
That is the single largest and most fixable weakness in the project.

---

## Phase 1 — Make "grounded in real data" true

**Moves evidence quality 3 → 7. Roughly 2 days. The biggest lever available.**

### 1.1 Real ambient forcing
Replace `SyntheticClimate` with `OpenMeteoArchive` in the site and memo builds. Naive
per-shipment fetching would issue tens of thousands of requests, so fetch **one year of hourly
reanalysis per airport (18 requests)**, cache it on disk, and slice per shipment. The cache is
committed — same seed must give the same weather or nothing reproduces.

### 1.2 External validation — the part that actually answers the criticism
Being *driven* by real data is weaker than being *checked against* it. Pattern-oriented
modelling: pick external patterns the simulator never saw, and report how it compares —
**whether or not it matches.**

| Simulated quantity | External reference |
|---|---|
| excursion / write-off rate | openFDA drug enforcement: temperature-caused recalls |
| ground-hold duration | US BTS `TaxiOut` for the actual hub airports |
| ambient distribution by lane | Open-Meteo vs Meteostat station observations |
| failure-mode taxonomy | openFDA recall reason text, verbatim |

A chart showing "my simulated rate vs FDA's recall base rate" turns *"it's all made up"* from a
concession into a conversation about calibration.

### 1.3 Real failure modes
Replace the invented failure list with the taxonomy recovered from openFDA recall text.

**Gate:** the site and memo run on real reanalysis; at least two external validation
comparisons render; the cache makes it reproducible offline.

---

## Phase 2 — Close the RD bias

**Moves novelty 4 → 8. About 1 day. Do this one yourself.**

The estimator overstates the complier LATE by ~50%. Diagnosed to the numerator: `E[D|R]` is flat
within each side so its local-linear fit is near-exact, but `E[Y|R]` is steeply sloped and curved
so its boundary intercepts are biased. Ruled out: complier composition, and a mass point at the
cutoff (fixed separately by interpolating crossing times).

Remaining fix: **MSE-optimal bandwidth selection** — the half of Calonico–Cattaneo–Titiunik not
yet implemented. Only bias-reduction via local quadratic is in place.

**Do it yourself.** It is self-contained, fully diagnosed, and *"I closed a 50% bias by
implementing MSE-optimal bandwidth selection"* is worth more coming from you than the fix is
worth on its own.

**Gate:** the estimate covers the complier LATE across bandwidths with < 15% bias, and the
caveat comes out of the README.

---

## Phase 3 — Cut it down

**Moves presentability 5 → 9. Half a day.**

Seven site sections is four too many. Build one three-minute path:

1. the hook — two compliant shipments, 1.787×
2. the finding — the alarm fires where intervening is worth 0.17 SD
3. the answer — the policy map
4. the number — $240k spend, $444k better than the current rule, stable across costs

Everything else moves behind an **Appendix** link: model board, calibration gate, RD detail,
capacity table. Nothing is deleted; it stops competing for the first ninety seconds.

Also: either fix the WASM app or remove it from the repo. A broken deployed component in the
commit history reads as honest; one still sitting there reads as unfinished.

**Gate:** a stranger reaches the recommendation in under three minutes without scrolling past
anything they don't need.

---

## Phase 4 — Own it

**Decides whether everything above reads as 9 or as 5. Cannot be delegated.**

Claude wrote most of this repository. That is not a problem with the project; it is a problem
with the project *as evidence about you*, which is the only thing an interviewer is buying.

- Read `PITCH.md`, `DECISIONS.md` and every test name until each claim is yours.
- Be able to explain, without notes: why MKT is bounded by its trace; why re-icing buys time not
  immunity; why the RD bias grew as the bandwidth narrowed; why Platt beat isotonic; why $444k
  matters less than 86% decision stability.
- **Make the last commits yourself.** Phase 2 is the ideal candidate.

---

## Honest ceiling

**10/10 for a student portfolio is reachable. 10/10 in absolute terms is not** — that needs real
client data and measured production impact, which no student has. Don't chase it. Stating that
limit plainly is itself a strength, and this project's credibility rests on exactly that habit.

Phases 1–3 get to a genuine 9. Phase 4 decides how it reads.
