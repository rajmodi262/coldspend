"""Phase 2 gate tests.

These guard the three failure modes that would leave the metrics looking
excellent while every downstream dollar figure is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coldspend.models.risk import (
    DECISION_FEATURES,
    ORACLE_EXTRA,
    TARGET,
    assert_no_leakage,
    evaluate,
    reliability_curve,
)
from coldspend.sim import SyntheticClimate, simulate_portfolio


@pytest.fixture(scope="module")
def portfolio() -> pd.DataFrame:
    return pd.DataFrame([r.record for r in simulate_portfolio(3000, SyntheticClimate(), seed=11)])


@pytest.fixture(scope="module")
def reports(portfolio):
    return evaluate(portfolio, seed=0)


# --------------------------------------------------------------- leakage


def test_decision_features_contain_no_outcome_columns():
    """The one mistake that would invalidate everything while making the metrics
    look better. Cheap to check, so check it."""
    assert_no_leakage(DECISION_FEATURES)


def test_leakage_guard_actually_catches_leaks():
    """A guard that never fires is not a guard."""
    for bad in ("budget_consumed_pct", "mkt_c", "excursion", "truth_confounder_u"):
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage([*DECISION_FEATURES, bad])


def test_no_hub_feature_is_measured_after_the_hub():
    """Every decision-time feature must be derivable from the pre-hub trace."""
    assert all(f.startswith(("hub_", "remaining_", "running_", "depart_", "tau_"))
               or f in {"tarmac_hold", "consignment_value_usd", "lane", "product", "packaging", "hub"}
               for f in DECISION_FEATURES)


def test_target_is_strictly_post_hub():
    """Whole-journey excursion is partly observable AT the hub, so predicting it
    would score the model on something it can read off its own inputs."""
    assert TARGET == "post_hub_excursion"


# ----------------------------------------------------------- calibration


def test_calibration_improves_brier_without_moving_auc(reports):
    """THE point of this phase. The optimizer multiplies these probabilities by
    money, so being right about ranking is not enough — the level must be right.
    AUC is invariant to any monotone distortion and therefore cannot detect
    miscalibration at all. Brier can."""
    raw = next(r for r in reports if r.name == "HistGradientBoosting")
    sel = next(r for r in reports if "SELECTED" in r.name)
    # The SELECTED model is chosen by CV Brier among {isotonic, Platt, none}, so
    # it can never be worse than raw by more than selection noise. Asserting that
    # isotonic specifically always wins is FALSE -- it overfits at this n, which
    # is exactly why the selection step exists.
    assert sel.brier <= raw.brier * 1.02
    assert abs(sel.auc - raw.auc) < 0.02, "calibration should not materially change ranking"


def test_calibrated_probabilities_track_observed_frequency(reports):
    cal = next(r for r in reports if "SELECTED" in r.name)
    rc = reliability_curve(cal.probs, cal.truth)
    assert len(rc) >= 4
    assert np.abs(rc.predicted - rc.observed).max() < 0.15


def test_probabilities_are_in_range(reports):
    for r in reports:
        assert r.probs.min() >= 0.0 and r.probs.max() <= 1.0


# ---------------------------------------------------------------- oracle


def test_oracle_beats_every_deployable_model(reports):
    """The oracle sees the unobserved confounder and the weather that has not
    happened. If a deployable model matched it, the simulator would contain no
    genuine uncertainty and every accuracy number would be circular."""
    oracle = next(r for r in reports if "ORACLE" in r.name)
    deployable = [r for r in reports if "ORACLE" not in r.name]
    assert all(oracle.auc >= r.auc for r in deployable)
    assert all(oracle.brier <= r.brier for r in deployable)


def test_there_is_genuine_irreducible_uncertainty(reports):
    """A meaningful gap between the best deployable model and the oracle is what
    makes the exercise non-circular. If it closed, the model would simply be
    re-deriving the simulator's own equations."""
    oracle = next(r for r in reports if "ORACLE" in r.name)
    best = max((r for r in reports if "ORACLE" not in r.name), key=lambda r: r.auc)
    assert oracle.auc - best.auc > 0.005, "no irreducible noise — the generator is too easy"
    assert oracle.brier < best.brier * 0.9


def test_oracle_features_are_never_available_to_deployable_models():
    assert all(f.startswith("truth_") for f in ORACLE_EXTRA)
    assert not set(ORACLE_EXTRA) & set(DECISION_FEATURES)


# ------------------------------------------------------------- generalisation


def test_unseen_lanes_cost_something(reports):
    """Random row splits let a model memorise lane quirks and flatter itself.
    Held-out corridors are the question a client actually asks. A drop is
    expected and healthy; no drop at all would mean lane carries no information,
    which would itself be a finding."""
    lane = next(r for r in reports if "UNSEEN LANES" in r.name)
    cal = next(r for r in reports if "SELECTED" in r.name)
    assert lane.auc < cal.auc


def test_baseline_is_always_reported(reports):
    """Never ship a boosted model without the simple one beside it."""
    assert any("logistic" in r.name for r in reports)
