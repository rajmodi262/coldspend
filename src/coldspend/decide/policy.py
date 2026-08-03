"""The policy map — the headline visual, and the argmin made visible.

WHAT IT IS
----------
A plane whose axes are the state a shipment can be in — stability budget already
spent, and hours left to run — coloured by the cheapest action from that state.
Boundaries are exactly where the optimal action changes.

WHY IT MATTERS MORE THAN A RECOMMENDATION
-----------------------------------------
A per-shipment recommendation has to be adjudicated shipment by shipment, at
whatever hour it arrives, by whoever is on shift. A policy MAP can be reviewed
and signed ONCE, in advance, by the Qualified Person — which dissolves the
"who owns the decision" objection that kills most attempts to put optimisation
near a GxP process. The output stops being a prediction and becomes a policy
somebody can be accountable for.

It is also, simply, rare. Decision-region maps come from control theory and
dynamic programming and almost never appear in business analytics.

IMPLEMENTATION NOTE, STATED PLAINLY
-----------------------------------
This is a grid of independently solved argmins, not a backward-induction MDP.
The MDP is the more correct formulation — intervening mid-transit is genuinely a
sequential decision under uncertainty, and a one-shot argmin cannot price the
option of WAITING, so it over-intervenes early. That bias is named rather than
hidden, and the fallback was pre-committed in DECISIONS.md before the schedule
came under pressure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .costs import DEFAULT_COSTS, Action, CostModel
from .optimizer import ACTIONS, ShipmentState, expected_loss, recommend

__all__ = ["PolicyMap", "policy_map", "decision_stability", "StabilityReport"]


@dataclass
class PolicyMap:
    budget_grid: np.ndarray
    hours_grid: np.ndarray
    action_index: np.ndarray
    """(len(hours), len(budget)) grid of indices into ACTIONS."""
    expected_loss: np.ndarray
    consignment_value_usd: float
    burn_rate_pct_per_h: float
    tau_h: float = 12.0

    def action_at(self, budget_pct: float, hours_remaining: float) -> Action:
        i = int(np.argmin(np.abs(self.hours_grid - hours_remaining)))
        j = int(np.argmin(np.abs(self.budget_grid - budget_pct)))
        return ACTIONS[int(self.action_index[i, j])]

    def region_shares(self) -> dict[Action, float]:
        vals, counts = np.unique(self.action_index, return_counts=True)
        total = self.action_index.size
        return {ACTIONS[int(v)]: c / total for v, c in zip(vals, counts, strict=True)}


def policy_map(
    consignment_value_usd: float,
    burn_rate_pct_per_h: float = 1.2,
    tau_h: float = 12.0,
    p_excursion_at_full_burn: float = 0.85,
    budget_max_pct: float = 100.0,
    hours_max: float = 60.0,
    n_budget: int = 90,
    n_hours: int = 90,
    costs: CostModel = DEFAULT_COSTS,
    t: float = 0.5,
) -> PolicyMap:
    """Solve the argmin over a grid of states."""
    budget = np.linspace(0.0, budget_max_pct, n_budget)
    hours = np.linspace(0.5, hours_max, n_hours)

    idx = np.zeros((n_hours, n_budget), dtype=int)
    loss = np.zeros((n_hours, n_budget))

    for i, h in enumerate(hours):
        for j, b in enumerate(budget):
            # Excursion risk rises with how much burn is still to come.
            projected_extra = burn_rate_pct_per_h * h
            p_exc = float(np.clip(
                p_excursion_at_full_burn * (projected_extra / max(budget_max_pct, 1e-9)), 0.0, 1.0
            ))
            s = ShipmentState(
                shipment_id="grid", hub="grid",
                budget_pct=float(b), burn_rate_pct_per_h=burn_rate_pct_per_h,
                hours_remaining=float(h), consignment_value_usd=consignment_value_usd,
                p_excursion=p_exc, tau_h=tau_h,
            )
            best, losses = recommend(s, costs, t)
            idx[i, j] = ACTIONS.index(best)
            loss[i, j] = losses[best]

    return PolicyMap(budget, hours, idx, loss, consignment_value_usd,
                     burn_rate_pct_per_h, tau_h)


@dataclass
class StabilityReport:
    n_states: int
    n_stable: int
    flips: dict[tuple[Action, Action], int]

    @property
    def stable_share(self) -> float:
        return self.n_stable / max(self.n_states, 1)

    def __str__(self) -> str:
        return (f"{self.stable_share:.1%} of states keep the same recommendation "
                f"across the full low-to-high cost range ({self.n_stable}/{self.n_states})")


def decision_stability(
    consignment_value_usd: float,
    burn_rate_pct_per_h: float = 1.2,
    costs: CostModel = DEFAULT_COSTS,
    n_budget: int = 45,
    n_hours: int = 45,
    ts: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> StabilityReport:
    """Does the RECOMMENDATION survive the cost assumptions being wrong?

    This is the strongest claim the project can make about money, and it is much
    stronger than any point estimate: not "intervening saves $X" — which depends
    on costs nobody can verify — but "the action I recommend does not change
    anywhere across the plausible cost range". A reviewer can reject every
    individual cost figure and the recommendation still stands.
    """
    maps = [policy_map(consignment_value_usd, burn_rate_pct_per_h,
                       n_budget=n_budget, n_hours=n_hours, costs=costs, t=t) for t in ts]
    stacked = np.stack([m.action_index for m in maps])

    stable = (stacked == stacked[0]).all(axis=0)
    flips: dict[tuple[Action, Action], int] = {}
    for i, j in zip(*np.where(~stable), strict=True):
        seen = [ACTIONS[int(v)] for v in stacked[:, i, j]]
        key = (seen[0], seen[-1])
        flips[key] = flips.get(key, 0) + 1

    return StabilityReport(int(stable.size), int(stable.sum()), flips)
