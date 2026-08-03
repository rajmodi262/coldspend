"""The calibration gate — Phase 1's exit criterion.

WHY THIS FILE IS THE MOST IMPORTANT ONE IN THE PHASE
----------------------------------------------------
A simulator that runs, produces plausible-looking curves, and is quietly
mis-calibrated is the single worst failure mode available to this project. It is
invisible until something downstream looks strange, by which time every model,
dollar figure and chart has been built on it. This gate makes that failure loud
and early.

TWO KINDS OF BAND, AND THE DIFFERENCE MATTERS
---------------------------------------------
* DESIGN TARGETS — plausibility ranges. Deliberately wide. They are not claims
  about the real world, because the honest position is that published
  excursion-rate figures are largely untraceable (see research/06). They exist to
  catch DRIFT: if a change moves the excursion rate from 34% to 4%, something
  broke, and this says so on the next run rather than in week 6.

* IDENTIFICATION REQUIREMENTS — hard gates. These are not preferences. If the
  first stage collapses or the running variable has no density at the cutoff,
  the regression discontinuity identifies NOTHING and the project's causal claim
  is void. Failing one of these means stop and fix, not tune and continue.

Do not quietly widen a band to make a run pass. Widening a band is a decision
with a reason; record it in DECISIONS.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Band", "DESIGN_TARGETS", "IDENTIFICATION_GATES", "check", "report"]


@dataclass(frozen=True)
class Band:
    name: str
    lo: float
    hi: float
    why: str
    hard: bool = False

    def holds(self, v: float) -> bool:
        return bool(self.lo <= v <= self.hi)


DESIGN_TARGETS: tuple[Band, ...] = (
    Band("excursion_rate", 0.15, 0.50,
         "Plausibility only. Published excursion rates are largely untraceable — the "
         "widely-quoted '25% of vaccines arrive degraded' could not be sourced to a "
         "primary document. Wide on purpose; this catches drift, not truth."),
    Band("destroyed_rate", 0.002, 0.06,
         "Total write-offs should be rare. openFDA gives ~3% of drug RECALLS as "
         "temperature-caused, which is a different quantity, so this is not "
         "calibrated to it — only sanity-bounded by it."),
    Band("freeze_rate", 0.01, 0.15,
         "WHO/PATH field studies find freeze exposure is common, not marginal. A "
         "rate near zero would mean the over-cooling failure mode is not firing, "
         "and the optimizer would never learn that re-icing can harm."),
    Band("treated_rate", 0.10, 0.55,
         "Interventions should be a minority of shipments but not vanishing."),
    Band("median_transit_h", 20.0, 90.0,
         "Two-leg intercontinental air freight with ground dwell."),
)

IDENTIFICATION_GATES: tuple[Band, ...] = (
    Band("first_stage_jump", 0.35, 1.0,
         "P(treat) must jump materially at the SOP threshold. This IS the RD's "
         "first stage; a weak one makes the Wald ratio unstable and its CI a lie.", hard=True),
    Band("treated_above_cutoff", 0.50, 0.97,
         "Must be below 1.0. Perfect compliance means no untreated units above the "
         "threshold — a positivity failure no sample size repairs.", hard=True),
    Band("treated_below_cutoff", 0.02, 0.35,
         "Must be above 0. Some discretionary treatment below the cutoff is what "
         "makes the design FUZZY rather than sharp.", hard=True),
    Band("density_near_cutoff", 0.02, 1.0,
         "Share of units within +/-150 min of the cutoff. RD fits locally; with no "
         "density at the boundary there is nothing to fit.", hard=True),
    Band("distinct_rv_values_near_cutoff", 8.0, 1e9,
         "The running variable must be near-continuous. At hourly resolution it is "
         "quantized to 60-min steps and RD on a discrete running variable "
         "undercovers. See sim.shipment.STEPS_PER_HOUR.", hard=True),
)


def metrics(df: pd.DataFrame, cutoff_min: float = 300.0) -> dict[str, float]:
    """Compute every gated quantity from a portfolio frame."""
    # Cryogenic shipments run a different thermal regime and a different
    # intervention; they are excluded from running-variable diagnostics, which
    # are about the 2-8 degC decision rule.
    rv = df[df["product"] != "CAR-T dose"]
    near = rv["running_var_min"].between(cutoff_min - 150, cutoff_min + 150)
    win = rv["running_var_min"].between(cutoff_min - 100, cutoff_min + 100)

    below = df[df["running_var_min"] < cutoff_min]["treated"]
    above = df[df["running_var_min"] >= cutoff_min]["treated"]

    return {
        "excursion_rate": float(df["excursion"].mean()),
        "destroyed_rate": float(df["destroyed"].mean()),
        "freeze_rate": float((df["freeze_degree_h"] > 0).mean()),
        "treated_rate": float(df["treated"].mean()),
        "median_transit_h": float(df["transit_h"].median()),
        "treated_below_cutoff": float(below.mean()) if len(below) else np.nan,
        "treated_above_cutoff": float(above.mean()) if len(above) else np.nan,
        "first_stage_jump": float(above.mean() - below.mean()) if len(below) and len(above) else np.nan,
        "density_near_cutoff": float(near.mean()),
        "distinct_rv_values_near_cutoff": float(rv.loc[win, "running_var_min"].nunique()),
    }


def check(df: pd.DataFrame, cutoff_min: float = 300.0) -> tuple[bool, list[tuple[Band, float, bool]]]:
    """Returns (all_hard_gates_pass, [(band, value, passed), ...])."""
    m = metrics(df, cutoff_min)
    rows = [(b, m[b.name], b.holds(m[b.name])) for b in (*IDENTIFICATION_GATES, *DESIGN_TARGETS)]
    hard_ok = all(ok for b, _, ok in rows if b.hard)
    return hard_ok, rows


def report(df: pd.DataFrame, cutoff_min: float = 300.0) -> str:
    hard_ok, rows = check(df, cutoff_min)
    lines = [f"CALIBRATION GATE  —  n = {len(df):,} shipments", "=" * 72]

    for title, hard in (("IDENTIFICATION REQUIREMENTS (hard)", True), ("DESIGN TARGETS", False)):
        lines += ["", title, "-" * 72]
        for b, v, ok in rows:
            if b.hard is not hard:
                continue
            hi = "inf" if b.hi >= 1e8 else f"{b.hi:g}"
            lines.append(f"  {'PASS' if ok else 'FAIL'}  {b.name:<32} {v:>10.4g}   "
                         f"want [{b.lo:g}, {hi}]")
            if not ok:
                lines.append(f"        -> {b.why}")

    lines += ["", "=" * 72]
    lines.append("GATE PASSED — Phase 1 may close." if hard_ok else
                 "GATE FAILED — a hard identification requirement is not met. "
                 "Stop and fix; do not widen the band.")
    return "\n".join(lines)
