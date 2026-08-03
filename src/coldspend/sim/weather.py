"""Ambient temperature — the forcing function T_amb(t) for the thermal model.

TWO SOURCES, ONE INTERFACE.

`OpenMeteoArchive` is the real one: Open-Meteo's historical reanalysis, free, no
API key, CC-BY 4.0, hourly, 1940-01-01 to yesterday. It is the reason this
project can claim real grounding rather than invented weather.

`SyntheticClimate` is a deterministic physical stand-in used by the tests and by
anyone running offline. It is NOT presented as real data anywhere and it is not
what the headline results run on.

CACHING IS NOT AN OPTIMISATION, IT IS A CORRECTNESS REQUIREMENT.
Two reasons, and the second is the one that bites:
  1. Thousands of lane-hours would otherwise hammer a free public API.
  2. **The same seed must produce the same weather**, or nothing reproduces. A
     re-run that silently pulls a revised reanalysis changes every downstream
     number. The cache is committed to the repo for exactly this reason.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

__all__ = ["AmbientSource", "SyntheticClimate", "ReanalysisClimate", "OpenMeteoArchive"]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class AmbientSource(Protocol):
    """Hourly ambient temperature in degrees Celsius at a point."""

    def hourly(self, lat: float, lon: float, start: dt.datetime, hours: int) -> np.ndarray: ...


class SyntheticClimate:
    """Deterministic climatology: latitude sets the mean, season and hour modulate it.

    Physically shaped rather than physically accurate — mean temperature falls
    with latitude, seasonal swing grows with latitude, the southern hemisphere is
    phase-flipped, and there is a diurnal cycle. Enough structure that a monsoon
    lane and a northern winter lane behave differently, which is what the
    simulator needs from it.

    Fully reproducible: the pseudo-noise is hashed from position and timestamp,
    so the same query always returns the same series with no seed threading.
    """

    def __init__(self, noise_c: float = 2.5) -> None:
        self.noise_c = float(noise_c)

    def hourly(self, lat: float, lon: float, start: dt.datetime, hours: int) -> np.ndarray:
        if hours <= 0:
            raise ValueError("hours must be positive")
        idx = np.arange(hours)
        stamps = [start + dt.timedelta(hours=int(i)) for i in idx]
        doy = np.array([s.timetuple().tm_yday for s in stamps], dtype=float)
        hod = np.array([s.hour for s in stamps], dtype=float)

        abs_lat = abs(lat)
        mean_c = 30.0 - 0.45 * abs_lat
        seasonal_amp = 0.30 * abs_lat
        hemisphere = 1.0 if lat >= 0 else -1.0

        seasonal = seasonal_amp * np.sin(2 * np.pi * (doy - 105.0) / 365.25) * hemisphere
        diurnal = 5.5 * np.sin(2 * np.pi * (hod - 9.0) / 24.0)

        # Deterministic pseudo-noise: stable across processes and platforms,
        # unlike hash(), which is salted per interpreter run.
        key = f"{lat:.4f},{lon:.4f},{start.isoformat()}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        wobble = np.random.default_rng(seed).normal(0.0, self.noise_c, hours)

        return mean_c + seasonal + diurnal + wobble


class ReanalysisClimate:
    """Real Open-Meteo historical reanalysis, pre-fetched per location.

    THIS IS THE ONE THE HEADLINE FIGURES RUN ON. `SyntheticClimate` exists for
    tests and offline work; it is not what the results are built from.

    WHY PRE-FETCH RATHER THAN FETCH PER SHIPMENT
    --------------------------------------------
    Shipments depart at arbitrary hours, so caching on (lat, lon, departure)
    produces a distinct key almost every time and issues tens of thousands of
    requests for a portfolio of a few thousand. Instead: fetch ONE window of
    hourly data per airport — 18 requests for the whole network — and slice it.
    A year of hourly temperatures is ~8,800 floats, about 35 KB as float32.

    THE CACHE IS COMMITTED ON PURPOSE. Reproducibility depends on it: a re-run
    that silently pulls a revised reanalysis changes every downstream number.
    It also means the build works with no network at all, which matters when the
    thing being demonstrated is a live site.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache/weather",
        start: dt.date = dt.date(2024, 1, 1),
        days: int = 366,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.start = start
        self.days = days
        self._mem: dict[tuple[float, float], np.ndarray] = {}

    # -- cache plumbing ----------------------------------------------------
    def _key(self, lat: float, lon: float) -> str:
        return f"{lat:+08.3f}_{lon:+09.3f}_{self.start:%Y%m%d}_{self.days}"

    def _path(self, lat: float, lon: float) -> Path:
        return self.cache_dir / f"{self._key(lat, lon)}.npz"

    def _load(self, lat: float, lon: float) -> np.ndarray | None:
        rounded = (round(lat, 3), round(lon, 3))
        if rounded in self._mem:
            return self._mem[rounded]
        p = self._path(*rounded)
        if p.exists():
            arr = np.load(p)["t"].astype(float)
            self._mem[rounded] = arr
            return arr
        return None

    def fetch(self, lat: float, lon: float, force: bool = False) -> np.ndarray:
        """Download and cache one window for a location. Idempotent."""
        rounded = (round(lat, 3), round(lon, 3))
        if not force:
            cached = self._load(*rounded)
            if cached is not None:
                return cached

        import requests

        end = self.start + dt.timedelta(days=self.days - 1)
        resp = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": rounded[0], "longitude": rounded[1],
                "start_date": self.start.isoformat(), "end_date": end.isoformat(),
                "hourly": "temperature_2m", "timezone": "UTC",
            },
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["hourly"]["temperature_2m"]

        arr = np.array([np.nan if v is None else v for v in raw], dtype=float)
        if np.isnan(arr).any():
            # Reanalysis gaps are rare and short; carry the last good value
            # forward rather than dropping hours and shifting the whole series.
            idx = np.where(~np.isnan(arr))[0]
            arr = np.interp(np.arange(arr.size), idx, arr[idx])

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._path(*rounded), t=arr.astype(np.float32))
        self._mem[rounded] = arr
        return arr

    # -- AmbientSource -----------------------------------------------------
    def hourly(self, lat: float, lon: float, start: dt.datetime, hours: int) -> np.ndarray:
        if hours <= 0:
            raise ValueError("hours must be positive")
        arr = self._load(round(lat, 3), round(lon, 3))
        if arr is None:
            raise FileNotFoundError(
                f"no cached reanalysis for ({lat:.3f}, {lon:.3f}). "
                "Run `python scripts/fetch_weather.py` first — the cache is committed "
                "so builds stay reproducible and work offline."
            )

        offset = int((start - dt.datetime.combine(self.start, dt.time())).total_seconds() // 3600)
        # Wrap rather than clamp. A shipment departing near the end of the window
        # continues into the same calendar period a year earlier, which preserves
        # the diurnal cycle; clamping would pin it to a constant and quietly
        # remove the variation the thermal model is driven by.
        idx = (np.arange(offset, offset + hours)) % arr.size
        return arr[idx]


class OpenMeteoArchive:
    """Open-Meteo historical archive, cached on disk.

    Free, no API key, CC-BY 4.0 (attribute in the README). Hourly data from
    1940-01-01 to yesterday. Rate limits are generous but calls are *weighted* —
    a request spanning more than 14 days or 10 variables counts as several — so
    the cache does real work.
    """

    def __init__(self, cache_dir: str = "data/cache/weather") -> None:
        self.cache_dir = cache_dir
        self._cache = None

    def _open(self):
        if self._cache is None:
            import diskcache

            self._cache = diskcache.Cache(self.cache_dir)
        return self._cache

    def hourly(self, lat: float, lon: float, start: dt.datetime, hours: int) -> np.ndarray:
        if hours <= 0:
            raise ValueError("hours must be positive")
        end = start + dt.timedelta(hours=hours - 1)
        key = f"{lat:.3f},{lon:.3f},{start:%Y-%m-%dT%H},{hours}"

        cache = self._open()
        if key in cache:
            return np.asarray(cache[key], dtype=float)

        import requests

        resp = requests.get(
            ARCHIVE_URL,
            params={
                "latitude": round(lat, 3),
                "longitude": round(lon, 3),
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "hourly": "temperature_2m",
                "timezone": "UTC",
            },
            timeout=30,
        )
        resp.raise_for_status()
        series = resp.json()["hourly"]["temperature_2m"]

        vals = np.asarray(series[start.hour : start.hour + hours], dtype=float)
        if vals.size < hours:  # short tail near the archive edge
            vals = np.pad(vals, (0, hours - vals.size), mode="edge")
        if np.isnan(vals).any():
            vals = np.where(np.isnan(vals), np.nanmean(vals), vals)

        cache[key] = vals.tolist()
        return vals
