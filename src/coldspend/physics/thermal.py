"""Lumped-capacitance thermal model for a shipping container.

    dT_int/dt = (T_amb(t) - T_int) / tau

CLOSED FORM, NOT AN ODE SOLVER
------------------------------
Over a step where ambient is constant, the solution is exact:

    T(t + dt) = T_amb + (T(t) - T_amb) * exp(-dt / tau)

Ambient from a weather API IS piecewise-constant (hourly), so the closed form is
not an approximation — it is the exact solution, and it is ~100x faster than
scipy.integrate.solve_ivp. Never integrate this numerically.

PHASE-CHANGE MATERIAL
---------------------
A first-order RC model is WRONG for PCM shippers, and this is the subtlety most
implementations miss. PCM does not decay exponentially toward ambient; it holds a
PLATEAU at its phase-change temperature while latent heat is absorbed, and only
then reverts to RC decay. A pure RC curve through a PCM shipper's data will fit
badly and, worse, will fit badly in the optimistic direction.

We model latent heat as a reservoir expressed in kelvin of equivalent sensible
capacity (L / C_thermal), which keeps the arithmetic dimensionally clean without
needing C and R separately:

    while melting:   dLatent_K = (T_amb - T_pcm) / tau * dt

TAU FROM AN ADVERTISED HOLD TIME
--------------------------------
Vendors publish "96 h" or "120 h" hold times, not tau. `tau_from_hold_time`
inverts the RC solution for a stated payload limit — see its docstring for the
assumption that inversion makes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PackagingClass", "EPS", "VIP_PCM", "ACTIVE_REEFER", "tau_from_hold_time", "simulate"]


@dataclass(frozen=True)
class PackagingClass:
    """Thermal characterisation of a shipper.

    tau_h        : thermal time constant, hours
    pcm_temp_c   : phase-change plateau temperature (None = no PCM)
    pcm_latent_k : latent capacity as kelvin of equivalent sensible mass
    setpoint_c   : active units hold this regardless of ambient (None = passive)
    failure_rate_per_h : hazard of an active unit failing; ignored when passive
    """

    name: str
    tau_h: float
    pcm_temp_c: float | None = None
    pcm_latent_k: float = 0.0
    setpoint_c: float | None = None
    failure_rate_per_h: float = 0.0

    def __post_init__(self) -> None:
        if self.tau_h <= 0:
            raise ValueError("tau_h must be positive")
        if self.pcm_temp_c is not None and self.pcm_latent_k <= 0:
            raise ValueError("PCM declared with no latent capacity")


# Representative classes. tau values are order-of-magnitude anchors pending
# calibration against the WHO PQS catalogue and vendor hold-time datasheets;
# see research/05-physics-and-regulation.md.
EPS = PackagingClass("passive EPS", tau_h=8.0)
VIP_PCM = PackagingClass("VIP + PCM", tau_h=72.0, pcm_temp_c=5.0, pcm_latent_k=18.0)
ACTIVE_REEFER = PackagingClass(
    "active reefer", tau_h=200.0, setpoint_c=5.0, failure_rate_per_h=1.5e-4
)
CRYO_DRY_SHIPPER = PackagingClass(
    # LN2 vapour-phase dry shipper for cell and gene therapy. Holds vapour
    # temperature until the nitrogen charge is exhausted, then warms fast.
    # Modelled as setpoint-holding with a depletion hazard over a ~10-day charge.
    "cryo dry shipper",
    tau_h=36.0,
    setpoint_c=-150.0,
    failure_rate_per_h=4.0e-3,
)


def tau_from_hold_time(hold_time_h: float, t_amb_c: float, t_start_c: float, t_limit_c: float) -> float:
    """Invert the RC solution to recover tau from an advertised hold time.

    A vendor's "96 hours" means: starting at ``t_start_c``, in ambient
    ``t_amb_c``, the payload takes 96 h to reach ``t_limit_c``.

        tau = -hold_time / ln( (T_limit - T_amb) / (T_start - T_amb) )

    ASSUMPTION THIS MAKES: that the shipper is pure RC. For a PCM shipper the
    advertised hold time INCLUDES the latent plateau, so inverting it this way
    attributes the plateau to sensible capacity and overstates tau. Use this for
    EPS-class shippers; for PCM, fit tau and latent capacity jointly.
    """
    if not (t_start_c < t_limit_c < t_amb_c or t_amb_c < t_limit_c < t_start_c):
        raise ValueError("t_limit_c must lie strictly between t_start_c and t_amb_c")
    return -hold_time_h / np.log((t_limit_c - t_amb_c) / (t_start_c - t_amb_c))


def simulate(
    ambient_c,
    dt_h: float,
    pkg: PackagingClass,
    t_start_c: float = 5.0,
    rng: np.random.Generator | None = None,
    latent_start_k: float | None = None,
    return_state: bool = False,
):
    """Interior temperature over a piecewise-constant ambient series.

    Returns an array the same length as ``ambient_c``: the interior temperature
    at the END of each step.

    ``latent_start_k`` and ``return_state`` exist so a journey can be simulated
    in segments with state carried across them — which is what a mid-transit
    intervention requires. Re-icing at a hub restores latent capacity; NOT
    carrying the remaining latent across the hub would silently hand the
    untreated arm a free PCM recharge and destroy the counterfactual.

    With ``return_state=True`` returns ``(trace, final_temp_c, final_latent_k)``.
    """
    amb = np.asarray(ambient_c, dtype=float)
    if amb.ndim != 1 or amb.size == 0:
        raise ValueError("ambient_c must be a non-empty 1-D series")
    if dt_h <= 0:
        raise ValueError("dt_h must be positive")

    out = np.empty_like(amb)
    t = float(t_start_c)
    latent = float(pkg.pcm_latent_k if latent_start_k is None else latent_start_k)
    decay = np.exp(-dt_h / pkg.tau_h)

    # An active unit holds setpoint until it fails; after failure it decays like
    # a passive box of the same tau. Sampling the failure hour up front keeps the
    # run reproducible from the seed alone.
    fail_step = None
    if pkg.setpoint_c is not None and pkg.failure_rate_per_h > 0:
        r = rng or np.random.default_rng()
        fail_h = r.exponential(1.0 / pkg.failure_rate_per_h)
        fail_step = int(fail_h / dt_h)

    for i, a in enumerate(amb):
        if pkg.setpoint_c is not None and (fail_step is None or i < fail_step):
            t = pkg.setpoint_c
            out[i] = t
            continue

        if latent > 0 and pkg.pcm_temp_c is not None and a > pkg.pcm_temp_c:
            # Melting: the payload is pinned at the plateau while latent heat
            # absorbs the inbound flux. Partial steps are handled by spending
            # what remains and decaying for the rest of the step.
            draw = (a - pkg.pcm_temp_c) / pkg.tau_h * dt_h
            if draw <= latent:
                latent -= draw
                t = pkg.pcm_temp_c
                out[i] = t
                continue
            frac = latent / draw          # fraction of the step still plateaued
            latent = 0.0
            t = pkg.pcm_temp_c
            t = a + (t - a) * np.exp(-(1.0 - frac) * dt_h / pkg.tau_h)
            out[i] = t
            continue

        t = a + (t - a) * decay
        out[i] = t

    return (out, t, latent) if return_state else out
