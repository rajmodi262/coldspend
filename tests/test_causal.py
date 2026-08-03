"""RD estimator tests, including the bandwidth experiment that failed."""

from __future__ import annotations

import numpy as np
import pytest

from coldspend.causal import RECOMMENDED_BANDWIDTH, fuzzy_rd, mse_optimal_bandwidth


def _toy(n=6000, cutoff=300.0, seed=0):
    """Sharp-ish fuzzy design with a known effect, enough density to be fair."""
    rng = np.random.default_rng(seed)
    R = rng.uniform(cutoff - 500, cutoff + 500, n)
    U = rng.normal(0, 1, n)
    p = np.where(R >= cutoff, 0.85, 0.12) + 0.15 * U
    D = (rng.random(n) < np.clip(p, 0.02, 0.98)).astype(float)
    Y0 = 10 + 0.01 * R + 2.0 * U + rng.normal(0, 1.5, n)
    Y = Y0 - 3.0 * D
    return R, D, Y


def test_recommended_bandwidth_recovers_a_known_effect():
    R, D, Y = _toy()
    est = fuzzy_rd(R, D, Y, RECOMMENDED_BANDWIDTH, 300.0).estimate
    assert est == pytest.approx(3.0, rel=0.25)


def test_mse_optimal_bandwidth_is_kept_but_not_used():
    """It is exported and tested precisely because it FAILED on the real
    generator — 71% mean bias against 10% for a fixed h=200, because this design
    is variance-limited rather than bias-limited. Deleting it would erase the
    evidence for a decision that looks arbitrary without it."""
    R, D, Y = _toy()
    h = mse_optimal_bandwidth(R, Y, 300.0)
    assert np.isfinite(h) and h > 0
    assert h < np.ptp(R)


def test_bandwidth_selector_is_regularised():
    """Without the regularisation term, near-equal curvature on the two sides
    sends the optimum toward the whole sample."""
    rng = np.random.default_rng(1)
    R = rng.uniform(-500, 500, 4000)
    Y = 5 + 0.02 * R + rng.normal(0, 1, 4000)      # perfectly linear: no curvature
    h = mse_optimal_bandwidth(R, Y, 0.0)
    assert h <= np.ptp(R) * 0.5


def test_wald_ratio_is_sign_correct_on_a_clean_design():
    R, D, Y = _toy(seed=7)
    r = fuzzy_rd(R, D, Y, RECOMMENDED_BANDWIDTH, 300.0)
    assert r.estimate > 0
    assert r.jump_treatment > 0.4
