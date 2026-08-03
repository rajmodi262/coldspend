"""Phase 4 gate tests — the decision layer.

The optimiser is the project's novelty claim, so these guard the ways it could
be quietly wrong or quietly trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from coldspend.decide import (
    DEFAULT_COSTS,
    Action,
    ShipmentState,
    decision_stability,
    expected_loss,
    optimise_portfolio,
    policy_map,
    project_budget,
    recommend,
)


def state(**kw) -> ShipmentState:
    base = dict(shipment_id="S", hub="DXB", budget_pct=30.0, burn_rate_pct_per_h=1.4,
                hours_remaining=20.0, consignment_value_usd=850_000.0,
                p_excursion=0.6, tau_h=12.0)
    return ShipmentState(**{**base, **kw})


# ------------------------------------------------------------ economic sanity


def test_value_at_risk_changes_the_recommendation():
    """The second consequence of the thesis: compliance treats every breach
    equally, decisions must not. Identical physics, different money."""
    cheap_losses = recommend(state(consignment_value_usd=4_000.0))[1]
    dear_losses = recommend(state(consignment_value_usd=2_000_000.0))[1]
    # The gap between acting and not acting must widen with what is at stake.
    cheap_gap = cheap_losses[Action.DO_NOTHING] - min(cheap_losses.values())
    dear_gap = dear_losses[Action.DO_NOTHING] - min(dear_losses.values())
    assert dear_gap > cheap_gap * 5


def test_low_value_low_risk_shipments_are_left_alone():
    s = state(consignment_value_usd=4_000.0, p_excursion=0.05,
              budget_pct=5.0, burn_rate_pct_per_h=0.3, hours_remaining=8.0)
    assert recommend(s)[0] is Action.DO_NOTHING


def test_for_cheap_products_the_investigation_outweighs_the_goods():
    """A finding, not a bug, and a good line for the deck: on a $4,000 saline
    shipment the QA deviation investigation costs several times the consignment,
    so intervening is rational to protect the PAPERWORK, not the product. This is
    the compliance-cost term most models omit entirely — and omitting it makes
    cheap shipments look like they should always be abandoned."""
    s = state(consignment_value_usd=4_000.0, p_excursion=0.85)
    losses = recommend(s)[1]
    investigation = 0.85 * DEFAULT_COSTS.deviation_investigation.at(0.5)
    assert investigation > s.consignment_value_usd
    assert losses[Action.REICE] < losses[Action.DO_NOTHING]


def test_doing_nothing_is_not_free():
    """An excursion costs a QA investigation whether or not product is released.
    A model where inaction is costless would systematically under-intervene."""
    s = state(p_excursion=0.9)
    assert expected_loss(s, Action.DO_NOTHING) > 0.0


def test_recall_is_the_expensive_certainty():
    """It removes spoilage risk and should therefore almost never win — if it
    did, the cost model would be broken."""
    s = state()
    assert expected_loss(s, Action.RECALL) > s.consignment_value_usd
    assert recommend(s)[0] is not Action.RECALL


def test_reice_buys_time_not_immunity():
    """THE physics that shapes the policy map. A fresh charge covers about one
    time constant; on a long leg it wears off and most of the burn happens
    anyway. A flat multiplier made re-icing dominate 95% of the state space."""
    short = state(hours_remaining=6.0, tau_h=12.0)
    long = state(hours_remaining=60.0, tau_h=12.0)
    short_gain = project_budget(short, Action.DO_NOTHING) - project_budget(short, Action.REICE)
    long_gain = project_budget(long, Action.DO_NOTHING) - project_budget(long, Action.REICE)
    # Absolute gain can be larger on a long leg, but the FRACTION saved must fall.
    assert (short_gain / project_budget(short, Action.DO_NOTHING)) > \
           (long_gain / project_budget(long, Action.DO_NOTHING))


def test_better_insulation_makes_reicing_more_attractive():
    poor = policy_map(850_000.0, tau_h=8.0, n_budget=30, n_hours=30).region_shares()
    good = policy_map(850_000.0, tau_h=40.0, n_budget=30, n_hours=30).region_shares()
    assert good.get(Action.REICE, 0) > poor.get(Action.REICE, 0)


def test_projection_is_monotone_in_time_and_burn():
    assert project_budget(state(hours_remaining=30.0), Action.DO_NOTHING) > \
           project_budget(state(hours_remaining=10.0), Action.DO_NOTHING)
    assert project_budget(state(burn_rate_pct_per_h=3.0), Action.DO_NOTHING) > \
           project_budget(state(burn_rate_pct_per_h=0.5), Action.DO_NOTHING)


# ------------------------------------------------------- the MILP earns its place


def test_shared_capacity_actually_binds():
    """The honest answer to 'isn't this just an enumeration?'. With a scarce
    resource the per-shipment decisions stop separating, and the constraint costs
    real money. If capacity_cost were always zero, the solver would be theatre."""
    rng = np.random.default_rng(3)
    states = [
        state(shipment_id=f"S{i}", budget_pct=float(rng.uniform(5, 70)),
              burn_rate_pct_per_h=float(rng.uniform(0.5, 2.2)),
              hours_remaining=float(rng.uniform(6, 40)),
              consignment_value_usd=float(rng.choice([4_000, 45_000, 850_000])),
              p_excursion=float(rng.uniform(0.05, 0.9)))
        for i in range(30)
    ]
    loose = optimise_portfolio(states, reice_capacity={"DXB": 30})
    tight = optimise_portfolio(states, reice_capacity={"DXB": 3})

    assert loose.capacity_cost == pytest.approx(0.0, abs=1.0)
    assert tight.capacity_cost > 0.0
    assert tight.total_expected_loss > loose.total_expected_loss
    assert sum(1 for a in tight.assignment.values() if a is Action.REICE) <= 3


def test_every_shipment_gets_exactly_one_action():
    states = [state(shipment_id=f"S{i}") for i in range(12)]
    plan = optimise_portfolio(states, reice_capacity={"DXB": 4})
    assert len(plan.assignment) == len(states)
    assert all(isinstance(a, Action) for a in plan.assignment.values())


def test_budget_constraint_is_respected():
    states = [state(shipment_id=f"S{i}") for i in range(15)]
    cap = 8_000.0
    plan = optimise_portfolio(states, budget_usd=cap)
    assert plan.total_intervention_spend <= cap + 1e-6


def test_optimiser_never_loses_to_doing_nothing():
    states = [state(shipment_id=f"S{i}") for i in range(10)]
    plan = optimise_portfolio(states)
    assert plan.total_expected_loss <= plan.baseline_loss


def test_empty_portfolio_is_handled():
    plan = optimise_portfolio([])
    assert plan.assignment == {}


# ------------------------------------------------------------- decision stability


def test_recommendation_survives_the_cost_assumptions():
    """The strongest claim available about money, and much stronger than any
    point estimate: a reviewer can reject every individual cost figure and the
    recommendation still stands over most of the state space."""
    rep = decision_stability(850_000.0, n_budget=25, n_hours=25)
    assert rep.stable_share > 0.7, f"only {rep.stable_share:.0%} of states are cost-stable"


def test_policy_map_is_not_degenerate():
    """A map that is 95% one action is a lookup table, not a policy — and it was
    exactly what a non-physical re-ice model produced."""
    shares = policy_map(850_000.0, n_budget=40, n_hours=40).region_shares()
    assert len(shares) >= 2
    assert max(shares.values()) < 0.92


def test_policy_map_lookup_matches_the_grid():
    pm = policy_map(850_000.0, n_budget=30, n_hours=30)
    assert pm.action_at(pm.budget_grid[5], pm.hours_grid[7]) is \
        Action(list(Action)[pm.action_index[7, 5]])


# --------------------------------------------------------------- cost model


def test_cost_ranges_are_ordered_and_interpolate():
    for cr in (DEFAULT_COSTS.reice, DEFAULT_COSTS.expedite, DEFAULT_COSTS.reroute,
               DEFAULT_COSTS.recall_handling, DEFAULT_COSTS.deviation_investigation):
        assert cr.low <= cr.mid <= cr.high
        assert cr.at(0.0) == pytest.approx(cr.low)
        assert cr.at(0.5) == pytest.approx(cr.mid)
        assert cr.at(1.0) == pytest.approx(cr.high)
        assert cr.source, "every cost must carry its provenance, assumption or not"


def test_doing_nothing_costs_nothing_to_do():
    assert DEFAULT_COSTS.action_cost(Action.DO_NOTHING) == 0.0
