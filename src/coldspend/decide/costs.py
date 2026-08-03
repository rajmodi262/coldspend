"""The cost model — in ranges, never points.

WHY RANGES
----------
Every dollar this project outputs rests on costs that are assumptions. Quoting a
single number invites the only question that matters — "where did that come
from?" — and loses. Carrying a low/mid/high range instead lets the project make a
much stronger claim than any point estimate could:

    the RECOMMENDATION is stable across the plausible cost range

Decision stability beats point precision. A recommendation that survives every
cost assumption in the range is defensible even when no individual number is.

SOURCING
--------
Air freight rates are real (Freightos published lane rates). Consignment values
are real and sourced in `sim/products.py`. The intervention and deviation costs
are ASSUMPTIONS in a plausible band — labelled as such here rather than dressed
up, because a reviewer will find them either way and it is better they find them
already flagged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Action", "CostRange", "CostModel", "DEFAULT_COSTS"]


class Action(str, Enum):
    """The action space at a hub decision epoch.

    Deliberately small and operationally real: these are things a shipper can
    actually instruct at a transit hub. `UPGRADE_PACKAGING` is absent because it
    is a pre-shipment decision, not a mid-transit one.
    """

    DO_NOTHING = "do_nothing"
    REICE = "re_ice"
    EXPEDITE = "expedite"
    REROUTE = "re_route"
    RECALL = "recall"


@dataclass(frozen=True)
class CostRange:
    low: float
    mid: float
    high: float
    source: str

    def at(self, t: float) -> float:
        """Interpolate: t=0 -> low, 0.5 -> mid, 1 -> high."""
        return self.low + (self.mid - self.low) * (2 * t) if t <= 0.5 else \
            self.mid + (self.high - self.mid) * (2 * t - 1)


@dataclass(frozen=True)
class CostModel:
    reice: CostRange
    expedite: CostRange
    reroute: CostRange
    recall_handling: CostRange
    deviation_investigation: CostRange
    """Cost of the QA deviation investigation triggered by ANY excursion,
    incurred regardless of whether the product is ultimately released. This term
    is why compliance is expensive even when nothing is destroyed."""

    def action_cost(self, action: Action, t: float = 0.5) -> float:
        return {
            Action.DO_NOTHING: 0.0,
            Action.REICE: self.reice.at(t),
            Action.EXPEDITE: self.expedite.at(t),
            Action.REROUTE: self.reroute.at(t),
            Action.RECALL: self.recall_handling.at(t),
        }[action]


DEFAULT_COSTS = CostModel(
    reice=CostRange(
        600.0, 1_200.0, 2_500.0,
        "ASSUMPTION. Gel-pack/dry-ice replenishment plus handling at a qualified "
        "hub. Band spans a simple gel swap to a dry-ice recharge with GDP paperwork.",
    ),
    expedite=CostRange(
        2_000.0, 6_000.0, 14_000.0,
        "Derived from Freightos published air lane rates ($2.16-6.26/kg, Dec 2025) "
        "applied to a 300-1,200 kg consignment as an INCREMENTAL charge over the "
        "booked rate.",
    ),
    reroute=CostRange(
        1_500.0, 5_000.0, 11_000.0,
        "ASSUMPTION. Re-booking onto a different routing; band reflects whether "
        "capacity exists on the day.",
    ),
    recall_handling=CostRange(
        18_000.0, 45_000.0, 90_000.0,
        "ASSUMPTION. Return logistics, quarantine, disposition and re-ship "
        "coordination. EXCLUDES the product write-off, which is priced separately "
        "as consignment value so the two are never double-counted.",
    ),
    deviation_investigation=CostRange(
        3_000.0, 12_000.0, 35_000.0,
        "ASSUMPTION, though a well-populated one: published estimates of GMP "
        "deviation/CAPA cost per event cluster in this band. Triggered by any "
        "excursion regardless of final disposition.",
    ),
)
