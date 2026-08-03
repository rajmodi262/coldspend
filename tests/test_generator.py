"""Phase 1 gate tests.

These do not check that the generator produces pretty data. They check that it
produces IDENTIFIABLE data — that the regression discontinuity downstream has
something to estimate. Every test here corresponds to a way the causal claim can
silently die.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coldspend.sim import SyntheticClimate, simulate_portfolio
from coldspend.sim.calibrate import IDENTIFICATION_GATES, check, metrics


@pytest.fixture(scope="module")
def portfolio() -> pd.DataFrame:
    res = simulate_portfolio(1500, SyntheticClimate(), seed=1)
    return pd.DataFrame([r.record for r in res])


# ------------------------------------------------- identification prerequisites


def test_all_hard_identification_gates_pass(portfolio):
    hard_ok, rows = check(portfolio)
    failures = [f"{b.name}={v:.4g} want [{b.lo:g},{b.hi:g}]" for b, v, ok in rows if b.hard and not ok]
    assert hard_ok, "hard identification gates failed: " + "; ".join(failures)


def test_compliance_is_imperfect(portfolio):
    """The fuzzy-design requirement. If everyone above the threshold were treated
    there would be no untreated units up there — a positivity failure, and no
    Wald ratio to compute."""
    m = metrics(portfolio)
    assert 0.0 < m["treated_below_cutoff"] < 1.0
    assert 0.0 < m["treated_above_cutoff"] < 1.0


def test_the_running_variable_is_near_continuous(portfolio):
    """Regression against a real bug: at hourly resolution the running variable
    could only be 0, 60, 120 ... leaving an empty band between 300 and 360 where
    the cutoff sits. RD on a discrete running variable undercovers."""
    rv = portfolio[portfolio["product"] != "CAR-T dose"]["running_var_min"]
    near = rv[rv.between(200, 400)]
    assert near.nunique() >= 8
    gaps = np.diff(np.sort(near.unique()))
    assert gaps.max() < 60.0, "a 60-minute gap means the hourly quantization is back"


def test_confounder_is_unobserved_but_real(portfolio):
    """U must move BOTH the treatment decision and the outcome — that is what
    makes this confounding rather than noise. And it must not appear in any
    non-truth column, or the whole exercise is circular."""
    assert portfolio["truth_confounder_u"].std() > 0.5
    treated_u = portfolio.loc[portfolio.treated == 1, "truth_confounder_u"].mean()
    untreated_u = portfolio.loc[portfolio.treated == 0, "truth_confounder_u"].mean()
    assert treated_u > untreated_u, "U does not push the treatment decision"

    observable = [c for c in portfolio.columns if not c.startswith("truth_")]
    assert not any("confounder" in c or c == "u" for c in observable)


def test_confounder_distribution_is_continuous_at_the_cutoff(portfolio):
    """THE RD assumption. U may confound treatment, but its distribution must not
    jump at the threshold — otherwise the discontinuity picks up U's jump instead
    of the treatment effect and the estimate is garbage."""
    p = portfolio
    lo = p.loc[p.running_var_min.between(150, 300), "truth_confounder_u"]
    hi = p.loc[p.running_var_min.between(300, 450), "truth_confounder_u"]
    assert abs(lo.mean() - hi.mean()) < 0.45


def test_both_potential_outcomes_are_retained(portfolio):
    """The instrument argument. Real data can never contain both arms."""
    assert {"truth_y0_pct", "truth_y1_pct", "truth_ite_pct"} <= set(portfolio.columns)
    assert np.allclose(
        portfolio.truth_ite_pct, portfolio.truth_y0_pct - portfolio.truth_y1_pct, atol=1e-9
    )


def test_observed_outcome_equals_the_arm_that_happened(portfolio):
    """Consistency / SUTVA bookkeeping. If this fails, the dataset is incoherent
    and every downstream estimate is meaningless."""
    p = portfolio
    expected = np.where(p.treated == 1, p.truth_y1_pct, p.truth_y0_pct)
    assert np.allclose(p.budget_consumed_pct, expected, atol=1e-9)


# ------------------------------------------------------------- physical sanity


def test_intervention_helps_warm_shipments_and_does_nothing_to_cold_ones(portfolio):
    """Re-icing an already-cold box must have no effect — anything else would be
    the model inventing benefit. And the effect must GROW with exposure."""
    p = portfolio[portfolio["product"] != "CAR-T dose"]
    cold = p[p.running_var_min < 30].truth_ite_pct
    hot = p[p.running_var_min > 600].truth_ite_pct
    assert cold.median() == pytest.approx(0.0, abs=1e-6)
    assert hot.median() > cold.median()


def test_cryogenic_products_get_cryogenic_packaging(portfolio):
    """A -135 degC consignment in a passive 2-8 degC cooler is not a risky shipment,
    it is an impossible one — and it makes the model Arrhenius-extrapolate across
    ~165 K, which produced multipliers around 1e19 before this was fixed."""
    cryo = portfolio[portfolio["product"] == "CAR-T dose"]
    assert (cryo.packaging == "cryo dry shipper").all()
    assert not (portfolio[portfolio["product"] != "CAR-T dose"].packaging == "cryo dry shipper").any()


def test_no_budget_figure_is_absurd(portfolio):
    """Regression against the 1e17% blow-up. The Arrhenius ceiling should mean no
    refrigerated product ever reports a nonsense number."""
    fridge = portfolio[portfolio["product"].isin(["mAb pallet", "Vaccine pallet", "Insulin shipment"])]
    assert fridge.budget_consumed_pct.max() < 1000.0


def test_seasonality_exists(portfolio):
    """Network diagnostics depend on it, and a single-month draw cannot show it."""
    p = portfolio.copy()
    p["month"] = pd.to_datetime(p.depart).dt.month
    by_month = p.groupby("month").excursion.mean()
    assert len(by_month) >= 10
    assert by_month.max() - by_month.min() > 0.05


# ------------------------------------------------------------ reproducibility


def test_same_seed_gives_identical_data():
    """Non-negotiable. Without this nothing in the project reproduces, and the
    weather cache exists precisely to hold this property across runs."""
    a = pd.DataFrame([r.record for r in simulate_portfolio(60, SyntheticClimate(), seed=7)])
    b = pd.DataFrame([r.record for r in simulate_portfolio(60, SyntheticClimate(), seed=7)])
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = pd.DataFrame([r.record for r in simulate_portfolio(60, SyntheticClimate(), seed=7)])
    b = pd.DataFrame([r.record for r in simulate_portfolio(60, SyntheticClimate(), seed=8)])
    assert not a.budget_consumed_pct.equals(b.budget_consumed_pct)


def test_synthetic_climate_is_deterministic_across_calls():
    import datetime as dt

    c = SyntheticClimate()
    x = c.hourly(19.0887, 72.8679, dt.datetime(2024, 7, 1), 48)
    y = c.hourly(19.0887, 72.8679, dt.datetime(2024, 7, 1), 48)
    assert np.array_equal(x, y)


def test_climate_has_latitude_and_season_structure():
    import datetime as dt

    c = SyntheticClimate(noise_c=0.0)
    mumbai = c.hourly(19.09, 72.87, dt.datetime(2024, 7, 1), 24).mean()
    frankfurt = c.hourly(50.04, 8.56, dt.datetime(2024, 1, 15), 24).mean()
    assert mumbai > frankfurt, "tropical July must beat European January"
    jan = c.hourly(50.04, 8.56, dt.datetime(2024, 1, 15), 24).mean()
    jul = c.hourly(50.04, 8.56, dt.datetime(2024, 7, 15), 24).mean()
    assert jul > jan, "northern hemisphere seasons are inverted"


def test_gate_report_renders(portfolio):
    from coldspend.sim.calibrate import report

    txt = report(portfolio)
    assert "CALIBRATION GATE" in txt
    assert len(IDENTIFICATION_GATES) == 5
