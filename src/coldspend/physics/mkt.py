"""Mean Kinetic Temperature — USP <1079>, Haynes (1971).

MKT is the single temperature that would produce the same cumulative Arrhenius
degradation as the actual, varying temperature history over the same duration.

    T_MKT = (dH/R) / ( -ln[ ( SUM_i  t_i * exp(-dH/(R*T_i)) ) / SUM_i t_i ] )

with all temperatures in KELVIN.

THE CONSTANT
------------
The conventional activation enthalpy is dH = 83.144 kJ/mol, and it is chosen
precisely so that dH/R = 10000 K EXACTLY when R = 8.3144 J/(mol*K).

We hard-code the ratio. Deriving it from a rounded R (8.314) yields 10000.481 and
drifts every hand-computed unit test in the third decimal. This is the single most
common implementation error in MKT code.
"""

from __future__ import annotations

import numpy as np

__all__ = ["EA_OVER_R", "KELVIN", "mkt", "mkt_from_series"]

EA_OVER_R: float = 10000.0
"""dH/R in kelvin. Exact by construction of the USP <1079> convention."""

KELVIN: float = 273.15


def mkt(temps_c, weights=None) -> float:
    """Mean Kinetic Temperature in degrees Celsius.

    Parameters
    ----------
    temps_c : array-like
        Temperatures in degrees Celsius.
    weights : array-like, optional
        Time weight for each temperature (any consistent unit — only ratios
        matter). Defaults to equal weighting, which is correct for a trace
        sampled at a uniform interval.

    Notes
    -----
    MKT is bounded: ``min(temps) <= MKT <= max(temps)``, always. A trace that
    never exceeds 8 degC can never have an MKT above 8 degC. Any claim to the
    contrary is an arithmetic error, not a finding.

    MKT >= the arithmetic mean, with equality only for a constant trace —
    the convexity of exp(-EA_OVER_R/T) in T over the relevant range.
    """
    t = np.asarray(temps_c, dtype=float)
    if t.size == 0:
        raise ValueError("empty temperature trace")

    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != t.shape:
        raise ValueError(f"weights shape {w.shape} != temps shape {t.shape}")
    if np.any(w < 0):
        raise ValueError("negative time weights")
    total = w.sum()
    if total <= 0:
        raise ValueError("total weight must be positive")

    tk = t + KELVIN
    if np.any(tk <= 0):
        raise ValueError("temperature at or below absolute zero")

    # Work in log space. exp(-10000/T) underflows to 0 around T < 30 K, which is
    # far outside any cold-chain range, but the shift keeps this exact regardless.
    z = -EA_OVER_R / tk
    zmax = z.max()
    log_mean = zmax + np.log(np.sum(w * np.exp(z - zmax)) / total)
    return EA_OVER_R / (-log_mean) - KELVIN


def mkt_from_series(temps_c, timestamps_h) -> float:
    """MKT for an irregularly-sampled trace, weighting by the interval each
    reading represents (trapezoid-style midpoint durations).

    ``timestamps_h`` is elapsed hours, strictly increasing.
    """
    t = np.asarray(temps_c, dtype=float)
    ts = np.asarray(timestamps_h, dtype=float)
    if t.shape != ts.shape:
        raise ValueError("temps and timestamps must be the same length")
    if t.size == 1:
        return float(t[0])
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    edges = np.concatenate([[ts[0]], (ts[:-1] + ts[1:]) / 2.0, [ts[-1]]])
    return mkt(t, np.diff(edges))
