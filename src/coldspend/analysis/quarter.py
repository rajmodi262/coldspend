"""The counterfactual quarter — what the decision layer would have been worth.

THE NUMBER THIS PRODUCES IS THE ONE AN INTERVIEWER WILL PUSH HARDEST ON, so it
is computed end to end rather than asserted, and it is reported as a RANGE.

The pipeline is deliberately the honest one:

    simulator -> calibrated risk model -> optimiser -> expected loss

The risk model's *calibrated* probability is what the optimiser multiplies by
money. Feeding it a raw score would inflate or deflate every dollar figure while
leaving the ROC curve untouched, which is why calibration was treated as
load-bearing rather than cosmetic in Phase 2.

WHAT IS COMPARED
----------------
* SOP policy      — what the industry does now: re-ice when the alarm fires,
                    subject to the same hub capacity.
* Optimised       — the argmin under identical capacity constraints.
* No intervention — the do-nothing floor, for scale.

Comparing against the SOP rather than against doing nothing is the harder and
fairer test. "We beat doing nothing" is trivially true and would be a
dishonest headline; "we beat the rule you already follow" is the claim worth
making.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..decide import Action, ShipmentState, expected_loss, optimise_portfolio
from ..decide.costs import DEFAULT_COSTS, CostModel

__all__ = ["QuarterResult", "build_states", "run_quarter", "sensitivity"]


@dataclass
class QuarterResult:
    n_shipments: int
    cost_level: float
    sop_loss: float
    sop_spend: float
    optimised_loss: float
    optimised_spend: float
    nothing_loss: float
    actions: dict[Action, int]
    capacity_binding: bool

    @property
    def benefit_vs_sop(self) -> float:
        return self.sop_loss - self.optimised_loss

    @property
    def benefit_vs_nothing(self) -> float:
        return self.nothing_loss - self.optimised_loss

    def __str__(self) -> str:
        return (f"n={self.n_shipments}  spend ${self.optimised_spend:,.0f}  "
                f"loss ${self.optimised_loss:,.0f}  "
                f"vs SOP ${self.benefit_vs_sop:+,.0f}")


def build_states(df: pd.DataFrame, p_excursion: np.ndarray) -> list[ShipmentState]:
    """Turn simulated shipments plus calibrated risk into optimiser inputs.

    Only decision-time columns are read. `hub_budget_pct` and `hub_elapsed_h` are
    both measured strictly before the hub, so the implied burn rate is knowable
    at the moment the decision is made.
    """
    if len(df) != len(p_excursion):
        raise ValueError("risk vector length does not match the frame")

    states = []
    for (_, r), p in zip(df.iterrows(), p_excursion, strict=True):
        elapsed = max(float(r["hub_elapsed_h"]), 1e-6)
        states.append(ShipmentState(
            shipment_id=str(r["shipment_id"]),
            hub=str(r["hub"]),
            budget_pct=float(r["hub_budget_pct"]),
            burn_rate_pct_per_h=float(r["hub_budget_pct"]) / elapsed,
            hours_remaining=float(r["remaining_h"]),
            consignment_value_usd=float(r["consignment_value_usd"]),
            p_excursion=float(np.clip(p, 0.0, 1.0)),
            tau_h=float(r["tau_h"]),
        ))
    return states


def _sop_policy(states: list[ShipmentState], df: pd.DataFrame,
                threshold_min: float, capacity: dict[str, int] | None,
                costs: CostModel, t: float) -> tuple[float, float]:
    """What the current rule costs: re-ice when the alarm fires, capacity permitting.

    Where a hub is oversubscribed the SOP has no principled tie-break, so it is
    served in arrival order — which is what actually happens on a ramp, and is
    precisely the arbitrariness the optimiser replaces.
    """
    rv = df["running_var_min"].to_numpy(float)
    fires = rv >= threshold_min

    used: dict[str, int] = {}
    loss = spend = 0.0
    for s, wants in zip(states, fires, strict=True):
        act = Action.DO_NOTHING
        if wants:
            cap = None if capacity is None else capacity.get(s.hub)
            if cap is None or used.get(s.hub, 0) < cap:
                act = Action.REICE
                used[s.hub] = used.get(s.hub, 0) + 1
        loss += expected_loss(s, act, costs, t)
        spend += costs.action_cost(act, t)
    return loss, spend


def run_quarter(
    df: pd.DataFrame,
    p_excursion: np.ndarray,
    cost_level: float = 0.5,
    costs: CostModel = DEFAULT_COSTS,
    sop_threshold_min: float = 300.0,
    reice_capacity_per_hub: int | None = 40,
) -> QuarterResult:
    """Score SOP against the optimiser over one quarter of shipments."""
    states = build_states(df, p_excursion)
    hubs = sorted({s.hub for s in states})
    capacity = ({h: reice_capacity_per_hub for h in hubs}
                if reice_capacity_per_hub is not None else None)

    sop_loss, sop_spend = _sop_policy(states, df, sop_threshold_min, capacity, costs, cost_level)

    plan = optimise_portfolio(states, costs=costs, t=cost_level, reice_capacity=capacity)
    nothing = float(sum(expected_loss(s, Action.DO_NOTHING, costs, cost_level) for s in states))

    counts: dict[Action, int] = {}
    for a in plan.assignment.values():
        counts[a] = counts.get(a, 0) + 1

    return QuarterResult(
        n_shipments=len(states),
        cost_level=cost_level,
        sop_loss=sop_loss,
        sop_spend=sop_spend,
        optimised_loss=plan.total_expected_loss,
        optimised_spend=plan.total_intervention_spend,
        nothing_loss=nothing,
        actions=counts,
        capacity_binding=plan.capacity_cost > 1.0,
    )


def sensitivity(
    df: pd.DataFrame,
    p_excursion: np.ndarray,
    levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    **kw,
) -> list[QuarterResult]:
    """The same quarter priced across the full cost range.

    Reporting the range rather than the midpoint is the point. A single figure
    invites "where did that come from?" and loses; a range that stays positive
    throughout survives the question — the claim becomes "this is worth money
    under every cost assumption I considered", which is far harder to dismiss.
    """
    return [run_quarter(df, p_excursion, cost_level=t, **kw) for t in levels]
