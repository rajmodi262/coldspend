"""Property-based invariants over the physics, via Hypothesis.

WHY THIS FILE IS THE INTERESTING ONE
------------------------------------
A physics model has properties that must hold for EVERY input, not just the
handful someone thought to write down. Hypothesis generates thousands of traces
and tries to break them, then shrinks any failure to a minimal counterexample.

When an interviewer asks "how do you know your simulator is right?", the answer
"I don't test cases, I test invariants — and these are the ones that must hold"
is a materially stronger answer than a coverage number.

Each invariant below is a physical or mathematical statement, not an
implementation detail. If one of these ever fails, the model is wrong — not the
test.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from coldspend.physics.mkt import mkt
from coldspend.physics.stability import budget_consumed, equivalent_hours, freeze_damage
from coldspend.physics.thermal import PackagingClass, simulate

# Cold-chain-plausible temperatures: dry ice through a hot tarmac.
temps = st.lists(
    st.floats(min_value=-80.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=200,
)
positive_hours = st.floats(min_value=1e-3, max_value=500.0, allow_nan=False, allow_infinity=False)


# ------------------------------------------------------------------ MKT


@given(temps)
@settings(max_examples=400)
def test_mkt_always_lies_within_the_trace(trace):
    """THE invariant. MKT is a weighted mean of a monotone transform, so it can
    never leave the range of its inputs. This is what makes 'never breached
    8 degC but MKT is 9.4' impossible for any trace whatsoever."""
    m = mkt(trace)
    assert min(trace) - 1e-9 <= m <= max(trace) + 1e-9


@given(temps)
@settings(max_examples=400)
def test_mkt_never_below_the_arithmetic_mean(trace):
    """Jensen's inequality on the convexity of exp(-EA/T). Equality only for a
    constant trace. This is precisely why MKT exists rather than a plain mean —
    if this ever failed, the metric would carry no information."""
    assert mkt(trace) >= np.mean(trace) - 1e-9


@given(temps)
@settings(max_examples=200)
def test_mkt_is_order_invariant(trace):
    """With equal weights, MKT depends on the multiset of temperatures, not the
    order they arrived in. (Note the physical consequence: MKT alone cannot
    distinguish a slow warm drift from a late spike. That is a real limitation
    of the metric and an argument for carrying the budget as a running state.)"""
    shuffled = list(reversed(trace))
    assert mkt(trace) == np.float64(mkt(shuffled)).item() or abs(mkt(trace) - mkt(shuffled)) < 1e-9


@given(trace=temps, shift=st.floats(min_value=0.1, max_value=20.0))
@settings(max_examples=200)
def test_warming_every_reading_raises_mkt(trace, shift):
    """Monotonicity. Uniformly warming a trace must raise its MKT."""
    assert mkt([t + shift for t in trace]) > mkt(trace)


# -------------------------------------------------------- stability budget


@given(trace=temps, hours=positive_hours, shift=st.floats(min_value=0.1, max_value=20.0))
@settings(max_examples=300)
def test_warmer_never_consumes_less_budget(trace, hours, shift):
    """The core claim of the whole project, as an invariant: a uniformly warmer
    journey always spends more stability budget. If this can fail, the burn-down
    chart is meaningless."""
    warm = equivalent_hours([t + shift for t in trace], hours)
    cool = equivalent_hours(trace, hours)
    assert warm > cool


@given(trace=temps, h1=positive_hours, extra=positive_hours)
@settings(max_examples=200)
def test_budget_is_monotone_in_duration(trace, h1, extra):
    """Time only ever runs one way. A longer journey at the same temperatures
    cannot consume less."""
    assert equivalent_hours(trace, h1 + extra) > equivalent_hours(trace, h1)


@given(trace=temps, hours=positive_hours, shelf=st.floats(min_value=1.0, max_value=10_000.0))
@settings(max_examples=200)
def test_budget_consumed_is_non_negative(trace, hours, shelf):
    assert budget_consumed(trace, hours, shelf) >= 0.0


@given(trace=temps, hours=positive_hours)
@settings(max_examples=200)
def test_freeze_damage_is_non_negative_and_zero_when_warm(trace, hours):
    assert freeze_damage(trace, hours) >= 0.0
    if min(trace) >= 0.0:
        assert freeze_damage(trace, hours) == 0.0


# ------------------------------------------------------------ thermal model


@given(
    ambient=st.floats(min_value=-40.0, max_value=55.0),
    start=st.floats(min_value=-40.0, max_value=55.0),
    tau=st.floats(min_value=0.5, max_value=300.0),
    n=st.integers(min_value=1, max_value=300),
)
@settings(max_examples=300)
def test_interior_never_escapes_the_bracket(ambient, start, tau, n):
    """Second law, in effect: with a constant ambient, the payload can only move
    toward it and can never pass it. An interior temperature outside
    [min(start, ambient), max(start, ambient)] means energy appeared from
    nowhere."""
    out = simulate(np.full(n, ambient), 1.0, PackagingClass("p", tau_h=tau), start)
    lo, hi = min(start, ambient), max(start, ambient)
    assert np.all(out >= lo - 1e-9)
    assert np.all(out <= hi + 1e-9)


@given(
    ambient=st.floats(min_value=-40.0, max_value=55.0),
    start=st.floats(min_value=-40.0, max_value=55.0),
    tau=st.floats(min_value=0.5, max_value=300.0),
)
@settings(max_examples=200)
def test_approach_to_ambient_is_monotone(ambient, start, tau):
    """The gap to ambient shrinks every step and never reverses."""
    out = simulate(np.full(120, ambient), 1.0, PackagingClass("p", tau_h=tau), start)
    gaps = np.abs(out - ambient)
    assert np.all(np.diff(gaps) <= 1e-9)


@given(
    ambient=st.floats(min_value=20.0, max_value=55.0),
    tau=st.floats(min_value=5.0, max_value=400.0),
)
@settings(max_examples=200)
def test_a_bigger_tau_always_holds_colder(ambient, tau):
    """tau IS the insulation quality. If a better-insulated box did not stay
    colder in a warm ambient, the parameter would mean nothing — and calibrating
    it against vendor hold times would be meaningless."""
    assume(ambient > 6.0)
    worse = simulate(np.full(48, ambient), 1.0, PackagingClass("a", tau_h=tau), 5.0)
    better = simulate(np.full(48, ambient), 1.0, PackagingClass("b", tau_h=tau * 2), 5.0)
    assert np.all(better <= worse + 1e-9)


@given(
    ambient=st.floats(min_value=15.0, max_value=50.0),
    latent=st.floats(min_value=1.0, max_value=60.0),
)
@settings(max_examples=200)
def test_pcm_is_never_worse_than_no_pcm(ambient, latent):
    """Adding latent heat capacity cannot make a shipper perform worse in a warm
    ambient. A PCM implementation that violates this has its energy bookkeeping
    backwards — and would silently overstate performance."""
    plain = simulate(np.full(96, ambient), 1.0, PackagingClass("plain", tau_h=48.0), 5.0)
    pcm = simulate(
        np.full(96, ambient), 1.0,
        PackagingClass("pcm", tau_h=48.0, pcm_temp_c=5.0, pcm_latent_k=latent), 5.0,
    )
    assert np.all(pcm <= plain + 1e-9)
