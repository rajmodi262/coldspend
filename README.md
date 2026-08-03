# Coldspend

**Cold chain is not a limit you breach. It's a budget you spend.**

The pharmaceutical industry treats cold chain as a *compliance* problem — did we breach 2–8 °C? File
a deviation, quarantine the batch. Coldspend treats it as a *decision* problem: given what we know
right now, what is the cheapest action that keeps expected loss below threshold?

Two shipments, both 60 hours, **both entirely inside 2–8 °C**, both filing zero deviations, both
passing compliance identically. One cruises at 3.0 °C, the other at 7.5 °C. The warm one ages the
product **1.787× faster**. Compliance cannot tell them apart. A stability budget can.

## Status

Early build. The physics core and its acceptance tests are in place; the simulator, optimizer and
app are not yet.

| Component | State |
|---|---|
| MKT (USP ⟨1079⟩) | ✅ implemented + tested |
| Arrhenius stability budget | ✅ implemented + tested |
| RC + PCM thermal model | ✅ implemented + tested |
| Regression-discontinuity estimator | ✅ prototyped and validated (`../prototypes/`) |
| Shipment simulator | ⬜ next |
| MILP intervention optimizer | ⬜ |
| Policy map | ⬜ |
| marimo WASM app | ⬜ |

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
