"""The generator — one shipment, both arms.

JOURNEY STRUCTURE
    origin dwell -> flight -> HUB DWELL -> flight -> destination dwell

The hub is the decision epoch. Interventions happen there and nowhere else,
because that is where the shipment is on the ground and reachable. This is an
operational constraint, not a modelling convenience.

THE TWO FEATURES THAT MAKE THIS DATASET USEFUL
----------------------------------------------
Both are identification prerequisites (see DECISIONS.md D10), not realism polish:

1. COMPLIANCE FRICTION. The SOP is obeyed imperfectly — gel-pack stockouts, an
   unstaffed re-ice bay, an alert landing after flight close-out. If compliance
   were perfect, P(treat) would jump 0 -> 1 at the threshold, leaving no untreated
   units above it: a positivity failure no sample size repairs, and no regression
   discontinuity to run.

2. AN UNOBSERVED CONFOUNDER. Operator judgment — product criticality, customer
   escalation, how the box looked on the ramp. It raises both the untreated burn
   AND the chance of intervening, and it never reaches the logger. Without it,
   plain covariate adjustment on the observed running variable beats RD, and the
   causal argument collapses. Crucially U is drawn independently of the running
   variable, so its distribution is CONTINUOUS at the threshold — which is
   exactly the RD assumption, and why RD survives it.

BOTH POTENTIAL OUTCOMES ARE RETAINED, simulated forward from the identical hub
state under common random numbers. `y0`/`y1`/`ite` are GROUND TRUTH: they exist
to score estimators and must never be fed to one. Column names are prefixed
`truth_` so a leak is visible in any feature list.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from ..physics.mkt import mkt
from ..physics.stability import budget_consumed, freeze_damage
from ..physics.thermal import PackagingClass, simulate
from .network import LANES, Lane, by_iata, flight_hours
from .products import CATALOGUE, Product
from .weather import AmbientSource

__all__ = ["SimConfig", "simulate_shipment", "simulate_portfolio"]

CARGO_HOLD_C = 17.0
"""Pharma-configured belly hold. Controlled, but well above 2-8 degC — which is
why passive packaging still burns budget in the air."""

TARMAC_SOLAR_LOAD_C = 12.0
"""Apron surface heating over ambient during a hold. Sun on tarmac, no airflow."""

STEPS_PER_HOUR = 6
"""10-minute simulation resolution, matching real data-logger sampling.

NOT a precision nicety — a requirement of the identification strategy, and a
subtle one. At hourly resolution the running variable (minutes above spec) can
only take values 0, 60, 120, 180 ... so it is DISCRETE, and a regression
discontinuity on a discrete running variable is a known hard case: the bandwidth
cannot shrink toward the cutoff and conventional confidence intervals undercover.
At 10 minutes the support is fine enough for the standard estimator to behave.

Weather stays hourly and piecewise-constant, so the closed-form RC solution
remains exact — we are refining the observation grid, not interpolating physics
we do not have.
"""


@dataclass(frozen=True)
class SimConfig:
    # --- the decision rule -------------------------------------------------
    sop_threshold_min: float = 300.0
    """Minutes above 8 degC triggering the re-ice SOP. 5 h mirrors the real
    ColdTrace escalation rule."""
    p_treat_below: float = 0.10
    """Discretionary re-icing below the threshold."""
    p_treat_above: float = 0.85
    """SOP compliance above it. Deliberately < 1.0 — see module docstring."""
    confounder_strength: float = 0.16

    # --- failure modes -----------------------------------------------------
    p_tarmac_hold: float = 0.18
    tarmac_hold_h: tuple[float, float] = (1.0, 6.0)
    p_customs_hold: float = 0.12
    customs_hold_h: tuple[float, float] = (2.0, 14.0)
    p_preconditioning_error: float = 0.06
    """Gel packs not pre-conditioned: the payload starts warm."""
    p_sensor_dropout: float = 0.08
    sensor_drift_c: float = 0.35

    # --- dwell times -------------------------------------------------------
    origin_dwell_h: tuple[float, float] = (2.0, 8.0)
    hub_dwell_h: tuple[float, float] = (2.0, 10.0)
    dest_dwell_h: tuple[float, float] = (3.0, 24.0)

    # --- intervention effect ----------------------------------------------
    reice_setpoint_c: float = 4.0
    """Re-icing resets the payload toward this. Note it is ABOVE freezing on
    purpose: over-cooling is a real failure mode and `freeze_damage` prices it."""

    # --- genuine post-decision uncertainty ---------------------------------
    p_missed_connection: float = 0.22
    missed_connection_h: tuple[float, float] = (3.0, 20.0)
    post_hub_weather_shock_c: float = 5.0
    handling_exposure_c: float = 3.0
    """Unforecastable things that happen AFTER the decision is made.

    These exist to stop the model being able to reconstruct the outcome from its
    own inputs. Without them the simulator is very nearly deterministic given
    decision-time features, every classifier scores ~0.99 AUC, and the honest
    reading of that number is not 'the model is excellent' but 'the model has
    rediscovered the differential equation I wrote' — which is exactly the
    circularity attack this project has to survive.

    All three are physically real and genuinely unknowable at the hub: a missed
    onward connection, weather that deviates from forecast, and handling exposure
    (door openings, time on the ramp, load position). They are drawn AFTER the
    intervention decision and applied IDENTICALLY to both arms, so they add
    irreducible noise without contaminating the treatment effect.
    """

    packaging_variability: float = 0.40
    """Log-normal spread on tau and latent capacity, unit to unit.

    NOT cosmetic realism — an IDENTIFICATION REQUIREMENT, and the subtlest one in
    this file. With a single fixed tau per class, the fleet splits into two
    discrete populations: EPS boxes that always run hot and VIP+PCM boxes that
    always stay cold. The running variable becomes bimodal with an empty valley
    where the SOP threshold sits, and a regression discontinuity has no density
    at the cutoff to fit against.

    Real shippers vary — packaging age, pre-conditioning quality, load factor,
    how well the lid was closed. Modelling that variation gives the running
    variable continuous support, which is what the design needs.
    """


@dataclass
class ShipmentResult:
    record: dict = field(default_factory=dict)
    ambient: np.ndarray = field(default_factory=lambda: np.array([]))
    trace_observed: np.ndarray = field(default_factory=lambda: np.array([]))
    trace_untreated: np.ndarray = field(default_factory=lambda: np.array([]))
    trace_treated: np.ndarray = field(default_factory=lambda: np.array([]))


def _u(rng: np.random.Generator, lo_hi: tuple[float, float]) -> float:
    return float(rng.uniform(*lo_hi))


def _minutes_above(trace: np.ndarray, dt_h: float, limit_c: float) -> float:
    """Time above a threshold, with crossings LINEARLY INTERPOLATED.

    Counting whole steps (`np.sum(trace > limit) * dt`) quantizes the result to
    multiples of the step size, which puts a large MASS POINT exactly on the SOP
    threshold. That is fatal for the regression discontinuity: the atom sits on
    the right of the boundary with full kernel weight and anchors that intercept,
    while the left side must extrapolate from one step below. The estimated jump
    inflates, and inflates MORE as the bandwidth narrows — which is the opposite
    of how RD bias is supposed to behave, and is how this was caught.

    Interpolating the crossing times makes the running variable continuous at no
    extra simulation cost, and is also simply the more accurate integral.
    """
    t = np.asarray(trace, dtype=float)
    if t.size < 2:
        return float((t > limit_c).sum() * dt_h * 60.0)

    a, b = t[:-1], t[1:]
    above_a, above_b = a > limit_c, b > limit_c
    frac = np.zeros(a.size)
    frac[above_a & above_b] = 1.0
    # Partial intervals: the share of the step spent on the far side of the line.
    rising = ~above_a & above_b
    falling = above_a & ~above_b
    with np.errstate(divide="ignore", invalid="ignore"):
        span = b - a
        frac[rising] = np.where(span[rising] != 0, (b[rising] - limit_c) / span[rising], 0.0)
        frac[falling] = np.where(span[falling] != 0, (limit_c - b[falling]) / span[falling], 0.0)
    return float(np.clip(frac, 0.0, 1.0).sum() * dt_h * 60.0)


def simulate_shipment(
    rng: np.random.Generator,
    lane: Lane,
    product: Product,
    pkg: PackagingClass,
    depart: dt.datetime,
    ambient: AmbientSource,
    cfg: SimConfig = SimConfig(),
    shipment_id: str = "",
) -> ShipmentResult:
    dt_h = 1.0 / STEPS_PER_HOUR
    origin, hub, dest = by_iata(lane.origin), by_iata(lane.hub), by_iata(lane.dest)

    # Unit-to-unit variation in thermal performance. See SimConfig.packaging_variability
    # -- without this the running variable is bimodal and the RD has no density
    # at the cutoff.
    if cfg.packaging_variability > 0:
        jitter = float(rng.lognormal(0.0, cfg.packaging_variability))
        pkg = PackagingClass(
            name=pkg.name,
            tau_h=pkg.tau_h * jitter,
            pcm_temp_c=pkg.pcm_temp_c,
            pcm_latent_k=pkg.pcm_latent_k * jitter,
            setpoint_c=pkg.setpoint_c,
            failure_rate_per_h=pkg.failure_rate_per_h,
        )

    # ---- build the ambient profile leg by leg -----------------------------
    h_org = _u(rng, cfg.origin_dwell_h)
    h_hub = _u(rng, cfg.hub_dwell_h)
    h_dst = _u(rng, cfg.dest_dwell_h)

    tarmac = rng.random() < cfg.p_tarmac_hold
    if tarmac:
        h_hub += _u(rng, cfg.tarmac_hold_h)
    customs = rng.random() < cfg.p_customs_hold
    if customs:
        h_dst += _u(rng, cfg.customs_hold_h)

    n_org, n_hub, n_dst = int(round(h_org)), int(round(h_hub)), int(round(h_dst))
    n_f1 = int(round(flight_hours(origin, hub)))
    n_f2 = int(round(flight_hours(hub, dest)))
    n_org, n_hub, n_dst = max(n_org, 1), max(n_hub, 1), max(n_dst, 1)

    amb_org = ambient.hourly(origin.lat, origin.lon, depart, n_org)
    amb_hub = ambient.hourly(
        hub.lat, hub.lon, depart + dt.timedelta(hours=n_org + n_f1), n_hub
    )
    amb_dst = ambient.hourly(
        dest.lat, dest.lon, depart + dt.timedelta(hours=n_org + n_f1 + n_hub + n_f2), n_dst
    )
    if tarmac:
        amb_hub = amb_hub + TARMAC_SOLAR_LOAD_C

    # Hourly weather held constant within the hour, then observed on the 10-minute
    # grid. np.repeat, not interpolation: Open-Meteo gives hourly values and
    # inventing sub-hourly structure would be fabricating data we do not have.
    amb_pre = np.repeat(
        np.concatenate([amb_org, np.full(n_f1, CARGO_HOLD_C), amb_hub]), STEPS_PER_HOUR
    )
    amb_post = np.repeat(
        np.concatenate([np.full(n_f2, CARGO_HOLD_C), amb_dst]), STEPS_PER_HOUR
    )
    amb_all = np.concatenate([amb_pre, amb_post])

    # ---- leg 1: origin -> hub ---------------------------------------------
    t_start = product.ref_temp_c
    if rng.random() < cfg.p_preconditioning_error:
        t_start += 6.0                     # gel packs not pre-conditioned

    pre, t_hub, latent_hub = simulate(
        amb_pre, dt_h, pkg, t_start_c=t_start, rng=rng, return_state=True
    )

    # ---- the decision epoch ------------------------------------------------
    running_var = _minutes_above(pre, dt_h, product.storage_max_c)

    # U: operator judgment. Independent of the running variable BY CONSTRUCTION,
    # so its distribution is continuous at the threshold. That continuity is the
    # RD assumption; breaking it here would invalidate the whole design.
    u = float(rng.normal(0.0, 1.0))

    # ONE uniform draw, compared against BOTH sides' treatment probabilities.
    # This is what makes principal strata computable: a unit is a COMPLIER if it
    # would be treated above the threshold and not below it, an always-taker if
    # treated either way, a never-taker if neither. Drawing separately per side
    # would make complier status unknowable, and then the RD estimate could only
    # be checked against the wrong benchmark — the unweighted local mean ITE,
    # which is a different estimand entirely.
    u_treat = float(rng.random())
    p_above_side = float(np.clip(cfg.p_treat_above + cfg.confounder_strength * u, 0.02, 0.98))
    p_below_side = float(np.clip(cfg.p_treat_below + cfg.confounder_strength * u, 0.02, 0.98))
    p_treat = p_above_side if running_var >= cfg.sop_threshold_min else p_below_side
    treated = int(u_treat < p_treat)

    d_if_above = int(u_treat < p_above_side)
    d_if_below = int(u_treat < p_below_side)
    stratum = ("complier" if (d_if_above and not d_if_below)
               else "always_taker" if d_if_below
               else "never_taker")

    # U also raises the untreated burn — confounding by indication.
    u_burn = 0.9 * u

    # ---- both arms, common random numbers ---------------------------------
    # The intervention must suit the packaging. Re-icing a passive 2-8 degC box
    # means fresh gel packs; "re-icing" a cryogenic dry shipper means topping up
    # its nitrogen charge, back to ITS setpoint. Applying a +4 degC target to a
    # -150 degC consignment does not help it, it destroys it -- and the model
    # says so, loudly, which is how this was caught.
    reice_target = pkg.setpoint_c if pkg.setpoint_c is not None else cfg.reice_setpoint_c

    # Post-decision randomness. Drawn AFTER the treatment choice and applied to
    # BOTH arms identically (common random numbers), so it adds irreducible
    # uncertainty without touching the treatment effect.
    missed = rng.random() < cfg.p_missed_connection
    if missed:
        extra = int(round(_u(rng, cfg.missed_connection_h) * STEPS_PER_HOUR))
        amb_post = np.concatenate([amb_post[:1].repeat(extra) + TARMAC_SOLAR_LOAD_C, amb_post])
    amb_post = amb_post + rng.normal(0.0, cfg.post_hub_weather_shock_c)
    amb_post = amb_post + rng.normal(0.0, cfg.handling_exposure_c, amb_post.size)
    amb_all = np.concatenate([amb_pre, amb_post])

    post_0 = simulate(amb_post, dt_h, pkg, t_start_c=t_hub, rng=rng,
                      latent_start_k=latent_hub)
    post_1 = simulate(amb_post, dt_h, pkg, t_start_c=reice_target, rng=rng,
                      latent_start_k=pkg.pcm_latent_k)   # re-ice restores latent

    full_0 = np.concatenate([pre, post_0]) + u_burn
    full_1 = np.concatenate([pre, post_1]) + u_burn

    def outcome(trace: np.ndarray) -> float:
        return 100.0 * budget_consumed(
            trace, dt_h, product.shelf_life_h, product.ref_temp_c, product.ea_j_per_mol
        )

    y0, y1 = outcome(full_0), outcome(full_1)
    observed = full_1 if treated else full_0

    # ---- the sensor sees less than the truth ------------------------------
    drift = float(rng.normal(0.0, cfg.sensor_drift_c))
    sensed = observed + drift
    if rng.random() < cfg.p_sensor_dropout:
        gap = rng.integers(2, max(3, len(sensed) // 6))
        at = int(rng.integers(0, max(1, len(sensed) - gap)))
        sensed = sensed.copy()
        sensed[at : at + gap] = np.nan

    # ---- what is knowable AT THE DECISION EPOCH ---------------------------
    # Everything below is computed from the PRE-HUB trace only. The distinction
    # matters more than any modelling choice in this project: the whole-journey
    # summaries further down (mkt_c, peak_c, budget_consumed_pct) are measured
    # AFTER arrival, and using one as a feature would be textbook leakage —
    # predicting the outcome from the outcome. Decision-time features carry the
    # `hub_` prefix so a leak is visible in any feature list.
    hub_budget = 100.0 * budget_consumed(
        pre, dt_h, product.shelf_life_h, product.ref_temp_c, product.ea_j_per_mol
    )
    post_only = full_1[len(pre):] if treated else full_0[len(pre):]

    rec = {
        "shipment_id": shipment_id,
        "lane": lane.code,
        "origin": lane.origin,
        "hub": lane.hub,
        "dest": lane.dest,
        "product": product.name,
        "packaging": pkg.name,
        "consignment_value_usd": product.consignment_value_usd,
        "depart": depart,
        "transit_h": float(len(amb_all)) * dt_h,
        "tarmac_hold": int(tarmac),
        "customs_hold": int(customs),
        # --- DECISION-TIME features: known at the hub, safe to model on -----
        "hub_budget_pct": hub_budget,
        "hub_mkt_c": mkt(pre),
        "hub_peak_c": float(np.max(pre)),
        "hub_min_c": float(np.min(pre)),
        "hub_elapsed_h": float(len(pre)) * dt_h,
        "remaining_h": float(len(amb_post)) * dt_h,
        "depart_month": depart.month,
        "tau_h": pkg.tau_h,
        "running_var_min": running_var,
        "treated": treated,
        # --- OUTCOMES: measured at arrival, never features ------------------
        "post_hub_excursion": int(np.any(post_only > product.storage_max_c)),
        "post_hub_budget_pct": 100.0 * budget_consumed(
            post_only, dt_h, product.shelf_life_h, product.ref_temp_c, product.ea_j_per_mol
        ),
        # ^ THE outcome for causal estimation, and the choice is load-bearing.
        # Whole-journey budget includes the pre-hub burn, which is IDENTICAL in
        # both arms (so contributes nothing to the effect) while varying hugely
        # across shipments (so contributes nearly all the noise). Estimating a
        # treatment effect on it is like measuring a diet on final body weight
        # instead of weight change: the signal is real and utterly swamped.
        # Using it drove the RD's signal-to-placebo ratio to 0.62 — fake cutoffs
        # producing larger jumps than the real one.
        "budget_consumed_pct": outcome(observed),
        "mkt_c": mkt(observed),
        "peak_c": float(np.max(observed)),
        "min_c": float(np.min(observed)),
        "minutes_above_spec": _minutes_above(observed, dt_h, product.storage_max_c),
        "freeze_degree_h": freeze_damage(observed, dt_h) if product.freeze_sensitive else 0.0,
        "excursion": int(np.any(observed > product.storage_max_c)),
        "destroyed": int(outcome(observed) >= 100.0),
        # ^ budget fully spent. Reported as a FLAG because the percentage itself
        # is unbounded and meaningless once past 100 -- a thawed cryogenic dose
        # is destroyed, not "412% degraded". Aggregate on this, never on the mean
        # of budget_consumed_pct, which a handful of write-offs would dominate.
        "sensor_drift_c": drift,
        # --- GROUND TRUTH: for scoring estimators, never as a feature -------
        "truth_y0_pct": y0,
        "truth_y1_pct": y1,
        "truth_ite_pct": y0 - y1,
        "truth_confounder_u": u,
        "truth_p_treat": p_treat,
        "truth_stratum": stratum,
        # ^ Principal stratum. The RD's estimand is the LATE for COMPLIERS, so
        # validating it against the mean ITE of everyone near the cutoff compares
        # two different quantities and calls the difference bias.
        "truth_post_amb_mean_c": float(np.mean(amb_post)),
        "truth_post_amb_max_c": float(np.max(amb_post)),
        # ^ The downstream weather actually realised. Unknowable at the hub — a
        # forecast is not this. These exist to build the BAYES ORACLE: a model
        # given U and the realised future defines the ceiling on achievable
        # skill, so the real model can be scored against what is ACHIEVABLE
        # rather than against a meaningless 1.0.
    }
    return ShipmentResult(rec, amb_all, sensed, full_0, full_1)


def simulate_portfolio(
    n: int,
    ambient: AmbientSource,
    seed: int = 20260803,
    cfg: SimConfig = SimConfig(),
    start: dt.datetime | None = None,
    window_days: int = 365,
    packaging: tuple[PackagingClass, ...] | None = None,
) -> list[ShipmentResult]:
    """A portfolio of shipments spread across lanes, products and a full year.

    A full year is deliberate — seasonality is one of the network-diagnostic
    findings, and it cannot appear in data drawn from a single month.
    """
    from ..physics.thermal import CRYO_DRY_SHIPPER, EPS, VIP_PCM

    pkgs = packaging or (EPS, VIP_PCM)
    rng = np.random.default_rng(seed)
    start = start or dt.datetime(2024, 1, 1)

    out = []
    for i in range(n):
        lane = LANES[int(rng.integers(len(LANES)))]
        product = CATALOGUE[int(rng.integers(len(CATALOGUE)))]
        # Packaging must be able to serve the product. A cryogenic consignment in
        # a passive cooler is not a risky shipment, it is an impossible one.
        pkg = CRYO_DRY_SHIPPER if product.cryogenic else pkgs[int(rng.integers(len(pkgs)))]
        depart = start + dt.timedelta(hours=int(rng.integers(window_days * 24)))
        out.append(
            simulate_shipment(
                rng, lane, product, pkg, depart, ambient, cfg, shipment_id=f"SHP{i:06d}"
            )
        )
    return out
