# Decision record

Architecture decisions, and the abort decisions **pre-committed while calm**. The point of writing
the abort conditions down now is that week 7 is exactly when you will not want to make them.

---

## D1 — Python 3.12, pinned

`shap` requires ≥3.12; `polars` and `mapie` cap at 3.13. Intersection is 3.12–3.13; 3.12 chosen
because every wheel on PyPI exists for cp312. Installed via `uv python install 3.12`.

## D2 — `scipy.optimize.milp`, not PuLP + highspy

Verified: scipy's `milp` **is** HiGHS (docstring names it; toy intervention MILP solves correctly).
It needs no extra dependency, and decisively — it runs under Pyodide, where `highspy` has no
emscripten wheel. This supersedes `research/02-free-toolchain.md`, which recommended PuLP + highspy
before browser deployment was on the table.

## D3 — scikit-learn `HistGradientBoosting`, not LightGBM

LightGBM has no Pyodide wheel. Same algorithm family; the difference is within noise at n≈5,000.
Revisit only if browser deployment is abandoned.

## D4 — `lifelines`, not scikit-survival

scikit-survival is GPL-3.0 — viral copyleft inside a repo intended as a portfolio piece. lifelines is
MIT, pure Python, and is the only one with a real interval-censoring API, which this project needs:
loggers sample every 5–10 min, so the breach instant is never observed, only bracketed.

## D5 — `EA_OVER_R` hard-coded to 10000.0

ΔH = 83.144 kJ/mol exists precisely so ΔH/R = 10⁴ K exactly. Deriving it from a rounded R (8.314)
gives 10000.481 and drifts every hand-computed test in the third decimal.

## D6 — Closed-form RC, never an ODE solver

Ambient from a weather API is piecewise-constant, so `T(t+Δt) = T_amb + (T−T_amb)·exp(−Δt/τ)` is the
*exact* solution, not an approximation, and roughly 100× faster than `solve_ivp`.

## D7 — PCM as a latent reservoir, not an RC fit

Phase-change material produces a **plateau**, not exponential decay. A pure RC fit through PCM data
fails, and fails optimistically. Latent heat is carried in kelvin of equivalent sensible capacity so
the arithmetic stays dimensionally clean without needing R and C separately.

## D8 — Freeze damage tracked separately, and not Arrhenius

Cold damage is a distinct, often irreversible failure mode, and openFDA recall reasons include it.
It exists so the optimizer's `re-ice` action can be penalised for over-cooling. **An intervention
model that cannot harm is not a model.**

## D9 — No backend

The app is a marimo notebook exported to WASM on GitHub Pages; everything runs in the viewer's
browser via Pyodide. Chosen so that the thing which could fail during a demo does not exist. Costs:
5–15 s first load, no persistence, no auth, and the replay is a client-side timer rather than real
streaming — say "replay", never "live telemetry".

## D10 — The simulator must model compliance friction and an unobserved confounder

Not realism decoration — both are **identification prerequisites**. With perfect SOP compliance,
P(treat) goes 0→1 at the threshold: a positivity failure no sample size repairs, and no RD to run.
And with treatment assigned purely on the observed running variable, plain covariate adjustment beats
RD, which collapses the causal argument. See `../design/RD-MODULE.md` §3–4.

---

# Abort decisions — decided now, executed without renegotiation

| Trigger | Action |
|---|---|
| **End of week 4** and no page is deployed | Ship the ugly static page immediately. The week-4 deploy gate is schedule insurance; skipping it converts week 7 from *improvement* into *risk*. |
| **End of week 6** and the marimo app is not interactive | Ship `--mode run` with static charts and no replay animation. The argument survives intact; only theatre is lost. |
| **Week 5** and the MDP/policy map is not converging | Ship the one-shot MILP and present the policy map as a grid of solved MILPs. Same picture, less elegance, no schedule risk. |
| **Any week**, simulator excursion rate outside the declared band | Stop all downstream work and recalibrate. Everything built on a wrong generator is worthless, and this is the failure that hides longest. |
| **Week 7** and the network/lane view is unfinished | Cut it. The RD figure and the policy map carry the argument alone. |
| `lifelines` fails under Pyodide | Precompute survival curves at build time, ship as Parquet. Test this in **week 4**, not week 7. |
