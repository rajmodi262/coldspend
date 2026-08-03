"""Phase 6 gate tests — the counterfactual quarter and the memo renderer.

The quarter produces the one number an interviewer will push hardest on, so
these guard the ways it could flatter itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coldspend.analysis import build_states, run_quarter, sensitivity
from coldspend.decide import Action
from coldspend.report import render
from coldspend.sim import SyntheticClimate, simulate_portfolio


@pytest.fixture(scope="module")
def quarter():
    df = pd.DataFrame([r.record for r in simulate_portfolio(1200, SyntheticClimate(), seed=42)])
    d = df[df["product"] != "CAR-T dose"].reset_index(drop=True)
    # A deliberately crude risk proxy: the point of these tests is the accounting,
    # not the model. The real pipeline feeds a calibrated probability in.
    p = np.clip(d["running_var_min"] / 1500.0, 0.02, 0.95).to_numpy()
    return d, p


# ------------------------------------------------------------ the accounting


def test_optimiser_beats_the_rule_it_replaces(quarter):
    """Measured against the SOP, not against doing nothing. Beating inaction is
    trivially true and would be a dishonest headline."""
    d, p = quarter
    r = run_quarter(d, p)
    assert r.optimised_loss <= r.sop_loss
    assert r.benefit_vs_sop >= 0.0


def test_doing_nothing_is_the_worst_of_the_three(quarter):
    d, p = quarter
    r = run_quarter(d, p)
    assert r.nothing_loss >= r.sop_loss or r.nothing_loss >= r.optimised_loss


def test_benefit_stays_positive_across_the_whole_cost_range(quarter):
    """THE claim the memo leads with. If the benefit flipped sign anywhere in the
    plausible cost band, the honest headline would have to be 'it depends' —
    and that is worth knowing before it is printed."""
    d, p = quarter
    runs = sensitivity(d, p)
    assert len(runs) == 5
    assert all(r.benefit_vs_sop > 0 for r in runs), \
        [f"{r.cost_level}: {r.benefit_vs_sop:,.0f}" for r in runs]


def test_costs_scale_the_stakes_monotonically(quarter):
    d, p = quarter
    runs = sensitivity(d, p)
    losses = [r.optimised_loss for r in runs]
    assert losses == sorted(losses), "higher assumed costs must not lower total loss"


def test_capacity_limits_are_respected(quarter):
    d, p = quarter
    r = run_quarter(d, p, reice_capacity_per_hub=3)
    n_hubs = d["hub"].nunique()
    assert r.actions.get(Action.REICE, 0) <= 3 * n_hubs


def test_optimiser_intervenes_selectively(quarter):
    """A plan that treats everything is not a decision layer, it is a policy of
    panic — and one that treats nothing has no product."""
    d, p = quarter
    r = run_quarter(d, p)
    n_touched = sum(v for a, v in r.actions.items() if a is not Action.DO_NOTHING)
    assert 0 < n_touched < r.n_shipments


# ------------------------------------------------------------------ plumbing


def test_states_are_built_only_from_decision_time_columns(quarter):
    """`hub_budget_pct` and `hub_elapsed_h` are both measured strictly before the
    hub, so the implied burn rate is knowable when the decision is made."""
    d, p = quarter
    states = build_states(d.head(20), p[:20])
    assert len(states) == 20
    for s in states:
        assert s.burn_rate_pct_per_h >= 0
        assert s.hours_remaining > 0
        assert 0.0 <= s.p_excursion <= 1.0


def test_mismatched_risk_vector_is_rejected(quarter):
    d, p = quarter
    with pytest.raises(ValueError):
        build_states(d.head(10), p[:5])


# ------------------------------------------------------------------ renderer


def test_renderer_handles_the_memo_subset():
    md = "# H1\n\n## H2\n\n**b** *i* `c`\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n> q\n\n- item\n"
    h = render(md)
    for frag in ("<h1>H1</h1>", "<h2>H2</h2>", "<strong>b</strong>", "<em>i</em>",
                 "<code>c</code>", "<th>a</th>", "<td>1</td>", "<blockquote>", "&bull;"):
        assert frag in h, frag
    assert "|---|" not in h, "the table separator row must not be rendered"


def test_renderer_escapes_html():
    assert "&lt;script&gt;" in render("<script>alert(1)</script>")


def test_renderer_closes_tables():
    h = render("| a |\n|---|\n| 1 |\n\ntext after")
    assert h.count("<table>") == h.count("</table>") == 1
    assert "<p>text after</p>" in h
