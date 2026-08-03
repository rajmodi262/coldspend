"""Real-reanalysis tests.

The rest of the suite runs on SyntheticClimate deliberately — offline, fast, and
with no network dependency. These tests guard the REAL source that the headline
figures are actually built from, and skip cleanly when the cache is absent.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from coldspend.sim import AIRPORTS, ReanalysisClimate, SyntheticClimate

CACHE = Path("data/cache/weather")
needs_cache = pytest.mark.skipif(
    not CACHE.exists() or not any(CACHE.glob("*.npz")),
    reason="reanalysis cache absent — run scripts/fetch_weather.py",
)


@needs_cache
def test_every_airport_is_cached():
    """The build must not silently fall back to synthetic weather for a lane."""
    clim = ReanalysisClimate()
    missing = [c for c, a in AIRPORTS.items() if clim._load(a.lat, a.lon) is None]
    assert not missing, f"no cached reanalysis for {missing}"


@needs_cache
def test_slices_are_deterministic():
    """Reproducibility rests on this: the same request must always return the
    same series, which is why the cache is committed rather than refetched."""
    clim = ReanalysisClimate()
    a = AIRPORTS["BOM"]
    when = dt.datetime(2024, 7, 15, 6)
    assert np.array_equal(
        clim.hourly(a.lat, a.lon, when, 48), clim.hourly(a.lat, a.lon, when, 48)
    )


@needs_cache
def test_real_data_beats_synthetic_on_known_climate_facts():
    """The point of using reanalysis at all. These are facts about the world that
    a latitude-and-season approximation gets roughly, and real data gets right."""
    clim = ReanalysisClimate()
    sin, ord_ = AIRPORTS["SIN"], AIRPORTS["ORD"]
    jan = dt.datetime(2024, 1, 15)

    # Singapore is equatorial: almost no seasonal swing, and never cold.
    sin_year = clim._load(sin.lat, sin.lon)
    assert sin_year.min() > 18.0, "Singapore does not get cold"
    assert sin_year.max() - sin_year.min() < 20.0, "Singapore has little annual range"

    # Chicago in January genuinely does go far below freezing.
    ord_year = clim._load(ord_.lat, ord_.lon)
    assert ord_year.min() < -15.0, "a Chicago winter reaches well below -15 C"
    assert clim.hourly(ord_.lat, ord_.lon, jan, 24).mean() < 5.0


@needs_cache
def test_gulf_hubs_are_genuinely_hostile():
    """DXB and DOH are the transit hubs on most lanes — if the model understates
    their ambient, it understates the whole network's thermal load."""
    clim = ReanalysisClimate()
    for code in ("DXB", "DOH"):
        a = AIRPORTS[code]
        year = clim._load(a.lat, a.lon)
        assert year.max() > 44.0, f"{code} summer peak should exceed 44 C"
        assert year.mean() > 25.0


@needs_cache
def test_synthetic_and_real_disagree_enough_to_matter():
    """If the stand-in matched reality closely there would be no reason to fetch
    anything. It does not: the synthetic model manufactured cold exposure at the
    Gulf hubs that the reanalysis shows is simply not there."""
    real, syn = ReanalysisClimate(), SyntheticClimate(noise_c=0.0)
    a = AIRPORTS["DXB"]
    when = dt.datetime(2024, 7, 1)
    r = real.hourly(a.lat, a.lon, when, 24 * 14).mean()
    s = syn.hourly(a.lat, a.lon, when, 24 * 14).mean()
    assert abs(r - s) > 2.0, f"real {r:.1f} vs synthetic {s:.1f} — suspiciously close"


@needs_cache
def test_missing_location_fails_loudly():
    """A silent fallback to synthetic weather would be the worst possible
    failure: the figures would still render and would no longer be real."""
    clim = ReanalysisClimate()
    with pytest.raises(FileNotFoundError, match="fetch_weather"):
        clim.hourly(0.0, 0.0, dt.datetime(2024, 1, 1), 24)


@needs_cache
def test_window_wraps_rather_than_running_off_the_end():
    clim = ReanalysisClimate()
    a = AIRPORTS["FRA"]
    tail = clim.hourly(a.lat, a.lon, dt.datetime(2024, 12, 30, 12), 72)
    assert tail.size == 72
    assert np.isfinite(tail).all()
