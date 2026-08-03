"""Acceptance tests for the physics core.

Written BEFORE the simulator, deliberately. These are the tests that catch a
mis-calibrated model in week 2 rather than week 6, when everything downstream
has already been built on top of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from coldspend.physics.mkt import EA_OVER_R, mkt, mkt_from_series
from coldspend.physics.stability import (
    budget_consumed,
    equivalent_hours,
    freeze_damage,
    rate_multiplier,
)
from coldspend.physics.thermal import (
    ACTIVE_REEFER,
    EPS,
    VIP_PCM,
    PackagingClass,
    simulate,
    tau_from_hold_time,
)

# ---------------------------------------------------------------- MKT


def test_the_constant_is_exact():
    """83.144 kJ/mol exists so that dH/R is exactly 10000 K. Deriving it from a
    rounded R gives 10000.481 and drifts every hand-computed case."""
    assert EA_OVER_R == 10000.0


def test_constant_trace_returns_that_constant():
    for c in (-70.0, -20.0, 2.0, 5.0, 7.99, 25.0, 40.0):
        assert mkt([c] * 10) == pytest.approx(c, abs=1e-9)


def test_exceeds_arithmetic_mean_for_varying_trace():
    """The classic worked case: equal time at 20 and 30 degC. MKT sits ABOVE the
    arithmetic mean because exp(-EA/T) is convex — that convexity is the whole
    reason MKT exists as a metric."""
    assert mkt([20.0, 30.0]) == pytest.approx(26.2603, abs=1e-3)
    assert mkt([20.0, 30.0]) > 25.0


def test_mkt_is_bounded_by_the_trace():
    """THE structural fact. A trace that never exceeds 8 degC cannot have an MKT
    above 8 degC. The pitch claim 'never breached 8 but its MKT is 9.4' is
    arithmetically impossible, and this test is here so nobody reintroduces it."""
    rng = np.random.default_rng(0)
    trace = rng.uniform(2.0, 8.0, 500)
    m = mkt(trace)
    assert trace.min() <= m <= trace.max()
    assert m <= 8.0


def test_a_short_hot_spike_needs_20c_not_15c():
    """The plan claimed '30 min at 15 degC hurts far more than 4 h at 9 degC'.
    It does not. Duration beats the temperature non-linearity at those values;
    the spike has to reach about 20 degC to break even."""
    def _mkt(segments):
        t, w = zip(*segments, strict=True)
        return mkt(t, w)

    four_h_9c = _mkt([(5.0, 44.0), (9.0, 4.0)])

    assert _mkt([(5.0, 47.5), (15.0, 0.5)]) < four_h_9c   # 15 degC spike is NOT worse
    assert _mkt([(5.0, 47.5), (25.0, 0.5)]) > four_h_9c   # 25 degC clearly is

    # The break-even spike temperature is 20.03 degC. Pinning it stops the
    # rounded "about 20" from drifting back in as an assertion, which is exactly
    # the mistake that produced this test's first version.
    lo, hi = 9.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _mkt([(5.0, 47.5), (mid, 0.5)]) < four_h_9c:
            lo = mid
        else:
            hi = mid
    assert (lo + hi) / 2 == pytest.approx(20.033, abs=0.01)


def test_irregular_sampling_gives_endpoints_half_weight():
    """Trapezoid semantics: a reading represents the interval around it, so the
    first and last readings each cover only half an interval. This is NOT the
    same as equal weighting, and conflating the two is a real (small) bias in
    every trace-summarising implementation that gets it wrong."""
    temps = [4.0, 6.0, 9.0, 5.0, 7.0]
    hours = [0.0, 1.0, 2.0, 3.0, 4.0]
    half_weighted = mkt(temps, [0.5, 1.0, 1.0, 1.0, 0.5])
    assert mkt_from_series(temps, hours) == pytest.approx(half_weighted, abs=1e-12)
    assert mkt_from_series(temps, hours) != pytest.approx(mkt(temps), abs=1e-6)


def test_irregular_sampling_of_a_constant_trace_is_that_constant():
    assert mkt_from_series([6.0] * 5, [0.0, 0.7, 3.1, 3.2, 9.0]) == pytest.approx(6.0, abs=1e-9)


def test_mkt_rejects_bad_input():
    with pytest.raises(ValueError):
        mkt([])
    with pytest.raises(ValueError):
        mkt([5.0, 6.0], [1.0])
    with pytest.raises(ValueError):
        mkt([-300.0])


# ---------------------------------------------------------- stability budget


def test_reference_temperature_has_unit_rate():
    assert rate_multiplier([5.0], t_ref_c=5.0)[0] == pytest.approx(1.0, abs=1e-12)


def test_the_1787_ratio():
    """THE headline acceptance test, and the demo hook.

    Two shipments, both 60 h, BOTH entirely inside 2-8 degC, both filing zero
    deviations, both 'PASS' under compliance. One cruises at 3.0 degC, the other
    at 7.5 degC. The warm one ages the product 1.787x faster.

    Compliance cannot tell them apart. The stability budget can."""
    cold = equivalent_hours([3.0], 60.0)
    warm = equivalent_hours([7.5], 60.0)
    assert warm / cold == pytest.approx(1.787, abs=0.002)


def test_budget_is_a_fraction_and_may_exceed_one():
    """Exceeding 1.0 is the model reporting the product is out of budget.
    Clamping would hide precisely the case that matters."""
    assert budget_consumed([5.0], 24.0, shelf_life_hours=240.0) == pytest.approx(0.1)
    assert budget_consumed([5.0], 480.0, shelf_life_hours=240.0) > 1.0


def test_warmer_always_consumes_more():
    prev = -np.inf
    for c in (-20.0, 0.0, 5.0, 8.0, 15.0, 25.0, 40.0):
        cur = equivalent_hours([c], 10.0)
        assert cur > prev
        prev = cur


def test_freezing_is_tracked_separately_from_arrhenius():
    """Cold damage is not Arrhenius. It exists so the optimizer's re-ice action
    can be penalised for over-cooling — an intervention model that cannot harm
    is not a model."""
    assert freeze_damage([4.0, 5.0], 1.0) == 0.0
    assert freeze_damage([-3.0], 2.0) == pytest.approx(5.0)   # (-0.5 - -3.0) * 2


# ------------------------------------------------------------- thermal model


def test_rc_decays_toward_ambient_and_never_overshoots():
    out = simulate(np.full(400, 25.0), dt_h=1.0, pkg=EPS, t_start_c=5.0)
    # Non-decreasing, not strictly increasing: once it converges to ambient the
    # float difference is exactly 0. Demanding strict monotonicity here asserts
    # something about float resolution, not about the physics.
    assert np.all(np.diff(out) >= 0)
    assert np.all(np.diff(out[:50]) > 0)     # strictly increasing while it still has a gap to close
    assert np.all(out <= 25.0)               # never overshoots ambient
    assert out[-1] == pytest.approx(25.0, abs=0.01)


def test_one_tau_reaches_63_percent():
    """After exactly one time constant, an RC system closes 1 - 1/e of the gap."""
    out = simulate(np.full(8, 25.0), dt_h=1.0, pkg=EPS, t_start_c=5.0)
    assert out[7] == pytest.approx(5.0 + (25.0 - 5.0) * (1 - np.exp(-1.0)), abs=1e-9)


def test_hold_time_inversion_round_trips():
    """A vendor advertising 96 h to 8 degC in 30 degC ambient from 5 degC."""
    tau = tau_from_hold_time(hold_time_h=96.0, t_amb_c=30.0, t_start_c=5.0, t_limit_c=8.0)
    pkg = PackagingClass("vendor EPS", tau_h=tau)
    out = simulate(np.full(200, 30.0), dt_h=1.0, pkg=pkg, t_start_c=5.0)
    crossing = int(np.argmax(out >= 8.0))
    assert crossing == pytest.approx(96, abs=1)


def test_pcm_holds_a_plateau_not_an_exponential():
    """The subtlety a pure RC model gets wrong. While latent heat remains, the
    payload sits AT the phase-change temperature; only afterwards does it decay.
    Fitting an exponential through PCM data fails, and fails optimistically."""
    out = simulate(np.full(120, 30.0), dt_h=1.0, pkg=VIP_PCM, t_start_c=5.0)
    plateau = out[:40]
    assert np.allclose(plateau, VIP_PCM.pcm_temp_c, atol=1e-9)   # flat, not curved
    assert out[-1] > VIP_PCM.pcm_temp_c + 1.0                    # then it warms


def test_pcm_latent_capacity_sets_plateau_length():
    """Plateau length must scale with latent capacity — the parameter has to do
    what it claims, or calibrating it against a vendor datasheet is meaningless."""
    small = simulate(np.full(300, 30.0), 1.0, VIP_PCM, 5.0)
    big = simulate(
        np.full(300, 30.0), 1.0,
        PackagingClass("more PCM", tau_h=72.0, pcm_temp_c=5.0, pcm_latent_k=36.0), 5.0,
    )
    assert (big <= 5.0 + 1e-9).sum() > (small <= 5.0 + 1e-9).sum()


def test_active_unit_holds_setpoint_until_it_fails():
    never_fails = PackagingClass("reefer", tau_h=200.0, setpoint_c=5.0, failure_rate_per_h=0.0)
    out = simulate(np.full(200, 35.0), 1.0, never_fails, 5.0)
    assert np.allclose(out, 5.0)

    rng = np.random.default_rng(3)
    failed = simulate(np.full(2000, 35.0), 1.0, ACTIVE_REEFER, 5.0, rng=rng)
    assert failed[-1] > 5.0, "an active unit that never fails is not a risk model"


def test_simulate_rejects_bad_input():
    with pytest.raises(ValueError):
        simulate([], 1.0, EPS)
    with pytest.raises(ValueError):
        simulate([20.0], 0.0, EPS)
    with pytest.raises(ValueError):
        PackagingClass("bad", tau_h=-1.0)
