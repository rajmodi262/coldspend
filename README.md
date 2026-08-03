# Coldspend

**Cold chain is not a limit you breach. It's a budget you spend.**

The pharmaceutical industry treats cold chain as a *compliance* problem — did we breach 2–8 °C? File
a deviation, quarantine the batch. Coldspend treats it as a *decision* problem: given what we know
right now, what is the cheapest action that keeps expected loss below threshold?

Two shipments, both 60 hours, **both entirely inside 2–8 °C**, both filing zero deviations, both
passing compliance identically. One cruises at 3.0 °C, the other at 7.5 °C. The warm one ages the
product **1.787× faster**. Compliance cannot tell them apart. A stability budget can.

**Live: https://rajmodi262.github.io/coldspend/** — every figure recomputed from source on each push.

## Status

Phases 0–6 complete (Phase 6 bar Quarto PDF polish) (see [PHASES.md](PHASES.md)). 89 tests; CI runs them *before* it deploys,
so a broken simulator cannot ship a working-looking page.

| Component | State |
|---|---|
| MKT (USP ⟨1079⟩) | ✅ |
| Arrhenius stability budget + freeze damage | ✅ |
| RC + PCM thermal model | ✅ |
| Shipment generator, both potential outcomes | ✅ |
| Calibration gate (5 hard identification checks) | ✅ |
| Risk models + probability calibration + Bayes oracle | ✅ |
| Fuzzy RD with placebo/density/balance diagnostics | ✅ ~10% mean bias, full coverage; variance-limited |
| Cost model, portfolio MILP, policy map | ✅ |
| marimo WASM app | ⚠️ deployed but **not working** — see below |
| Counterfactual quarter + self-regenerating memo | ✅ |
| Deck outline, resume bullets, interview answers | ✅ [PITCH.md](PITCH.md) |
| Uplift model, carrier scorecard | ⬜ |

## Results worth knowing before reading further

- **Gradient boosting beats logistic regression by −0.0001 AUC.** Not a typo. Reported rather than buried.
- **AUC ≈ 0.98 is a warning, not a trophy.** Predicting a breach from "already at 7.8 °C and warming"
  is genuinely easy — which is exactly why prediction is not the contribution, and it is the part
  that is already patented.
- **The industry's alarm threshold fires where intervening barely helps** — 0.17 SD at the SOP
  cutoff, rising to 0.68 SD deeper in the tail.
- **87% of the state space keeps the same recommendation across the full cost range.** A reviewer
  can reject every individual cost assumption and the recommended action still stands.
- **On a $4,000 saline shipment the QA deviation investigation costs several times the goods**, so
  intervening protects the paperwork, not the product.
- **$198k of intervention beats the incumbent rule by $239k per simulated quarter** — computed, not
  asserted, and positive across every cost assumption tested.

## The RD bias, and a textbook fix that failed

The estimator's bias is now **+10.4% on the mean with 5/5 interval coverage**, measured over five
independent cohorts of 15,000 shipments against the true complier LATE (was ~50%).

Most of that came from the model, not the estimator: adding controlled storage gave the design more
signal. The part that did *not* work is worth more than the part that did — **Imbens–Kalyanaraman /
CCT MSE-optimal bandwidth selection is implemented, tested, and not used**, because measuring it
showed it makes things worse:

| bandwidth | mean bias | worst cohort | coverage |
|---|---|---|---|
| MSE-optimal (h* ≈ 50–83) | **+71%** | 132% | 5/5 |
| fixed h = 200 | **+10.4%** | 59% | 5/5 |

The reason is density, not algebra. The MSE criterion assumes enough mass at the cutoff for its
variance term to behave asymptotically; only ~5% of shipments sit within ±150 min of the threshold,
so a 55-minute window leaves the Wald ratio's denominator unstable — two cohorts came back with the
*wrong sign*. **This design is variance-limited, not bias-limited**, which points at sample size or
threshold placement rather than a cleverer bandwidth rule.

The worst single cohort is still 59% out, which is why the project reports the interval and the
bandwidth sweep and never a bare point estimate.

## Known broken: the WASM app

`/app/` loads Pyodide, installs the package, and renders its sliders — but the three cells that
draw with matplotlib (policy map, recommendation, stability callout) produce no output and raise
no error. Two fixes got it this far (resolving the wheel via `mo.notebook_location()`, and loading
scipy explicitly since marimo's scanner cannot see imports inside an installed package); selecting
the Agg backend did not finish the job. **It is not linked from the landing page** until it works.

The static site does not depend on it — the policy map there is rendered server-side at build time.

## If you are Raj

Read [STUDY.md](STUDY.md) before presenting this. Claude wrote most of this repository, which is
fine for the artifact and fatal for the interview if you cannot defend it. That file is the gap.

## Quickstart

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
```

## Design notes worth knowing before reading the code

- **`EA_OVER_R` is hard-coded to 10000.0.** The conventional ΔH = 83.144 kJ/mol exists precisely so
  that ΔH/R is exactly 10⁴ K. Deriving it from a rounded R gives 10000.481 and drifts every
  hand-computed test in the third decimal.
- **MKT is bounded by the trace.** A shipment that never exceeds 8 °C cannot have an MKT above 8 °C.
  There is a test enforcing this, because the inverse claim is an attractive and wrong demo hook.
- **The thermal model uses the closed-form RC solution, never an ODE solver.** Ambient from a weather
  API is piecewise-constant, so the closed form is exact and ~100× faster.
- **PCM is a plateau, not an exponential.** A pure RC fit through a phase-change shipper's data fails,
  and fails *optimistically*. Latent heat is modelled as a reservoir in kelvin of equivalent
  sensible capacity.
- **Freeze damage is tracked separately and is not Arrhenius.** It exists so the optimizer's `re-ice`
  action can be penalised for over-cooling — an intervention model that cannot harm is not a model.

## Honest limits

Activation energy is fitted and product-specific; the default is a convention, not a measurement of
any particular molecule. Protein aggregation is known non-Arrhenius, so for mAbs and cell therapy
this is an approximation whose error grows with excursion severity. No real telemetry is used
anywhere — the data comes from a physics simulator grounded in real weather, real airport networks
and published excursion base rates.

## Licence

MIT.
