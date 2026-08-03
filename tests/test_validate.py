"""External-validation tests.

These guard the two traps that would make the openFDA comparison look
quantitative while being nonsense, plus the honesty of the reported uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from coldspend.validate import classify, compare_to_simulation, failure_mix

CACHE = Path("data/cache/openfda/enforcement_temperature.json")
needs_cache = pytest.mark.skipif(
    not CACHE.exists(), reason="openFDA cache absent — run the validation build"
)


@needs_cache
def test_events_are_deduplicated():
    """THE trap. openFDA returns one row per PRODUCT: 619 records collapse to 78
    events, an inflation near 8x. A single distributor recall of many SKUs would
    otherwise dominate every statistic computed from this."""
    events = json.loads(CACHE.read_text(encoding="utf-8"))
    ids = [e.get("event_id") for e in events if e.get("event_id")]
    assert len(ids) == len(set(ids)), "duplicate event_id — deduplication failed"


def test_temperature_abuse_is_not_a_heat_indicator():
    """A real classification bug this caught. FDA uses 'Temperature Abuse' for
    BOTH directions — one recall reads 'Temperature Abuse: product samples were
    stored at temperatures below 32 F'. Treating 'abuse' as heat drove the
    measured freeze share to zero."""
    below = ("Temperature Abuse: product samples were stored at temperatures "
             "below 32* F which is not in accordance with storage requirements")
    assert classify(below) == "freeze"


def test_undirected_reasons_stay_unspecified():
    """Roughly 60% of real reasons say only 'held outside labeled storage
    conditions'. Forcing them into a direction would manufacture a result."""
    assert classify("CGMP Deviations: Products were exposed to temperatures "
                    "outside of the products labeled storage conditions.") == "unspecified"
    assert classify("") == "unspecified"


def test_clear_heat_and_freeze_are_separated():
    assert classify("exposed to subfreezing temperatures") == "freeze"
    assert classify("product exposed to elevated temperature during shipping") == "heat"


@needs_cache
def test_uncertainty_is_reported_and_wide():
    """The point estimate rests on four events. Quoting it without the interval
    would imply precision the evidence does not have."""
    mix = failure_mix(json.loads(CACHE.read_text(encoding="utf-8")))
    lo, hi = mix.freeze_share_ci
    assert lo < mix.freeze_share < hi
    assert hi - lo > 0.15, "an interval this narrow would be suspicious at n=4"


@needs_cache
def test_comparison_reports_both_sides_and_the_interval():
    """A comparison that returned only a ratio would hide whether the difference
    is real or an artefact of four data points."""
    from coldspend.sim import SyntheticClimate, simulate_portfolio

    df = pd.DataFrame([r.record for r in simulate_portfolio(400, SyntheticClimate(), seed=3)])
    c = compare_to_simulation(df)
    for k in ("fda_freeze_share", "sim_freeze_share", "fda_ci_lo", "fda_ci_hi",
              "fda_classifiable", "inside_interval"):
        assert k in c
    assert 0.0 <= c["sim_freeze_share"] <= 1.0
    assert c["fda_ci_lo"] <= c["fda_freeze_share"] <= c["fda_ci_hi"]
