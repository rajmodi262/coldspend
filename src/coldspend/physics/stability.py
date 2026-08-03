"""Arrhenius stability budget.

The idea that turns a binary alarm into a continuous, actionable number.

A product's label carries a shelf life at a reference storage temperature. The
Arrhenius rate law says degradation runs faster when warmer:

    k(T) = A * exp(-Ea / (R*T))

so relative to the reference temperature the rate multiplier is

    k(T)/k(Tref) = exp( -(Ea/R) * (1/T - 1/Tref) )       [T in kelvin]

Integrating that multiplier over a temperature trace gives EQUIVALENT HOURS AT
THE REFERENCE TEMPERATURE — how much shelf life the journey actually spent. As a
fraction of the labelled shelf life, that is the STABILITY BUDGET CONSUMED.

Two shipments that both stayed inside 2-8 degC, one cruising at 3.0 degC and one at
7.5 degC, are indistinguishable to compliance and differ by 1.787x here. That
ratio is an acceptance test (tests/test_stability.py).

HONEST LIMITS — say these out loud rather than have them found:
  * Ea is FITTED and PRODUCT-SPECIFIC. The 83.144 kJ/mol default is the MKT
    convention, not a measurement of your molecule.
  * Protein aggregation is known NON-Arrhenius. For mAbs and cell therapy this
    model is an approximation whose error grows with excursion severity.
  * Cold damage (freezing) is not Arrhenius at all and is handled separately —
    see `freeze_damage`. Re-icing too hard can destroy product; the model must
    be able to say so.
"""

from __future__ import annotations

import numpy as np

from .mkt import KELVIN

__all__ = [
    "EA_DEFAULT_J_PER_MOL",
    "R_GAS",
    "rate_multiplier",
    "equivalent_hours",
    "budget_consumed",
    "freeze_damage",
]

R_GAS: float = 8.3144
"""J/(mol*K). The value for which 83144/R == 10000 exactly."""

EA_DEFAULT_J_PER_MOL: float = 83144.0
"""Default activation energy. The USP <1079> convention value — a stand-in, not
a measurement. Override per product where a fitted value exists."""


MAX_RATE_MULTIPLIER: float = 1.0e6
"""Ceiling on Arrhenius extrapolation. See `rate_multiplier`."""


def rate_multiplier(
    temps_c,
    t_ref_c: float = 5.0,
    ea_j_per_mol: float = EA_DEFAULT_J_PER_MOL,
    max_multiplier: float = MAX_RATE_MULTIPLIER,
):
    """Degradation rate relative to the reference temperature. Dimensionless.

    Returns 1.0 at ``t_ref_c``, >1 above it, <1 below.

    WHY THERE IS A CEILING
    ----------------------
    An activation energy is FITTED NEAR THE STORAGE CONDITION. Extrapolating it
    far outside that range is not conservative — it is meaningless, and it fails
    loudly: a cryogenic product referenced at -135 degC, evaluated on a +30 degC
    tarmac, spans ~165 K and yields a multiplier around 1e19. That number is an
    artefact of extrapolating a local fit across a range where the degradation
    mechanism itself has changed (and where the product has, in any case, simply
    thawed — a different failure mode entirely).

    The ceiling makes the model refuse to make claims it cannot support. It is
    set far above any legitimate 2-8 degC excursion: a 2-8 degC product taken to
    40 degC gives a multiplier of about 56, four orders of magnitude below the cap.
    If the cap ever binds for a refrigerated product, that is a bug worth
    investigating, not a number worth reporting.

    Real practice agrees: excursion assessment uses Arrhenius over modest
    excursions (Gilead 2024, AAPS J, Tier 3), not across phase changes.
    """
    t = np.asarray(temps_c, dtype=float) + KELVIN
    if np.any(t <= 0):
        raise ValueError("temperature at or below absolute zero")
    a = ea_j_per_mol / R_GAS
    exponent = -a * (1.0 / t - 1.0 / (t_ref_c + KELVIN))
    return np.minimum(np.exp(np.minimum(exponent, np.log(max_multiplier))), max_multiplier)


def equivalent_hours(
    temps_c,
    hours,
    t_ref_c: float = 5.0,
    ea_j_per_mol: float = EA_DEFAULT_J_PER_MOL,
    max_multiplier: float = MAX_RATE_MULTIPLIER,
) -> float:
    """Shelf life spent, expressed as hours at the reference temperature.

    ``hours`` is the duration each reading represents (scalar or per-reading).
    """
    t = np.asarray(temps_c, dtype=float)
    h = np.full_like(t, float(hours)) if np.isscalar(hours) else np.asarray(hours, dtype=float)
    if h.shape != t.shape:
        raise ValueError("hours must be scalar or the same shape as temps")
    if np.any(h < 0):
        raise ValueError("negative duration")
    return float(np.sum(h * rate_multiplier(t, t_ref_c, ea_j_per_mol, max_multiplier)))


def budget_consumed(
    temps_c,
    hours,
    shelf_life_hours: float,
    t_ref_c: float = 5.0,
    ea_j_per_mol: float = EA_DEFAULT_J_PER_MOL,
) -> float:
    """Fraction of the labelled stability budget consumed. 0.34 == 34% spent.

    Can exceed 1.0 — that is the model reporting the product is out of budget,
    not an error. Clamping here would hide exactly the case that matters.
    """
    if shelf_life_hours <= 0:
        raise ValueError("shelf_life_hours must be positive")
    return equivalent_hours(temps_c, hours, t_ref_c, ea_j_per_mol) / shelf_life_hours


def freeze_damage(temps_c, hours, freeze_point_c: float = -0.5) -> float:
    """Cumulative degree-hours below the freezing threshold.

    Deliberately NOT Arrhenius. Freeze damage is a distinct, often irreversible
    failure mode — WHO/PATH studies find freeze exposure in a large share of
    vaccine shipments, and openFDA recall reasons include it explicitly.

    This exists so the optimizer's `re-ice` action can be penalised for
    over-cooling. An intervention model that cannot harm is not a model.
    """
    t = np.asarray(temps_c, dtype=float)
    h = np.full_like(t, float(hours)) if np.isscalar(hours) else np.asarray(hours, dtype=float)
    return float(np.sum(np.clip(freeze_point_c - t, 0.0, None) * h))
